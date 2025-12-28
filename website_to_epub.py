#!/usr/bin/env python3
"""
Fetch all pages from https://jax-ml.github.io/scaling-book/ and convert to EPUB.
Handles the Distill.js template used by the scaling-book website.
"""

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from ebooklib import epub
import re
import os
from urllib.parse import urljoin, urlparse
import copy
import base64
from io import BytesIO

BASE_URL = "https://jax-ml.github.io/scaling-book/"

# All pages in order
PAGES = [
    ("index", "Part 0: Introduction"),
    ("roofline", "Part 1: Intro to Rooflines"),
    ("tpus", "Part 2: All About TPUs"),
    ("sharding", "Part 3: Sharded Matmuls"),
    ("transformers", "Part 4: Transformers"),
    ("training", "Part 5: Training"),
    ("applied-training", "Part 6: Training LLaMA"),
    ("inference", "Part 7: Inference"),
    ("applied-inference", "Part 8: Serving LLaMA"),
    ("profiling", "Part 9: Profiling"),
    ("jax-stuff", "Part 10: All About JAX"),
    ("conclusion", "Part 11: Conclusions"),
    ("gpus", "Part 12: GPUs"),
]

# Cache for downloaded images
image_cache = {}

def fetch_page(page_slug, max_retries=3):
    """Fetch a page and return its HTML content with retry logic."""
    import time

    if page_slug == "index":
        url = BASE_URL
    else:
        url = urljoin(BASE_URL, page_slug)

    print(f"Fetching: {url}")

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"    Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

def make_absolute_url(url, base=BASE_URL):
    """Convert relative URLs to absolute."""
    if url.startswith(('http://', 'https://', 'data:')):
        return url
    return urljoin(base, url)

def download_image(url, max_retries=3):
    """Download an image and return its content and media type."""
    import time

    if url in image_cache:
        return image_cache[url]

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            content = response.content
            content_type = response.headers.get('content-type', 'image/png')

            # Determine file extension
            if 'png' in content_type or url.endswith('.png'):
                ext = 'png'
                media_type = 'image/png'
            elif 'gif' in content_type or url.endswith('.gif'):
                ext = 'gif'
                media_type = 'image/gif'
            elif 'webp' in content_type or url.endswith('.webp'):
                ext = 'webp'
                media_type = 'image/webp'
            elif 'svg' in content_type or url.endswith('.svg'):
                ext = 'svg'
                media_type = 'image/svg+xml'
            else:
                ext = 'jpg'
                media_type = 'image/jpeg'

            result = (content, media_type, ext)
            image_cache[url] = result
            return result

        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Warning: Failed to download image {url}")
                return None

    return None

def latex_to_unicode(latex):
    """Convert LaTeX to Unicode text representation."""
    latex = latex.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')

    subscript_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
        'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'j': 'ⱼ',
        'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ',
        'p': 'ₚ', 'r': 'ᵣ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ',
        'v': 'ᵥ', 'x': 'ₓ', '+': '₊', '-': '₋', '=': '₌',
        '(': '₍', ')': '₎',
    }

    superscript_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ', 'd': 'ᵈ', 'e': 'ᵉ',
        'f': 'ᶠ', 'g': 'ᵍ', 'h': 'ʰ', 'i': 'ⁱ', 'j': 'ʲ',
        'k': 'ᵏ', 'l': 'ˡ', 'm': 'ᵐ', 'n': 'ⁿ', 'o': 'ᵒ',
        'p': 'ᵖ', 'r': 'ʳ', 's': 'ˢ', 't': 'ᵗ', 'u': 'ᵘ',
        'v': 'ᵛ', 'w': 'ʷ', 'x': 'ˣ', 'y': 'ʸ', 'z': 'ᶻ',
        '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
        'T': 'ᵀ', '*': '∗',
    }

    greek = {
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
        r'\epsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η', r'\theta': 'θ',
        r'\iota': 'ι', r'\kappa': 'κ', r'\lambda': 'λ', r'\mu': 'μ',
        r'\nu': 'ν', r'\xi': 'ξ', r'\pi': 'π', r'\rho': 'ρ',
        r'\sigma': 'σ', r'\tau': 'τ', r'\upsilon': 'υ', r'\phi': 'φ',
        r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
        r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
        r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Phi': 'Φ',
        r'\Psi': 'Ψ', r'\Omega': 'Ω',
    }

    symbols = {
        r'\times': '×', r'\div': '÷', r'\pm': '±', r'\mp': '∓',
        r'\cdot': '·', r'\cdots': '⋯', r'\ldots': '…',
        r'\leq': '≤', r'\geq': '≥', r'\neq': '≠', r'\approx': '≈',
        r'\equiv': '≡', r'\sim': '∼', r'\propto': '∝',
        r'\infty': '∞', r'\partial': '∂', r'\nabla': '∇',
        r'\sum': '∑', r'\prod': '∏', r'\int': '∫',
        r'\sqrt': '√', r'\forall': '∀', r'\exists': '∃',
        r'\in': '∈', r'\notin': '∉', r'\subset': '⊂', r'\supset': '⊃',
        r'\cup': '∪', r'\cap': '∩', r'\emptyset': '∅',
        r'\rightarrow': '→', r'\leftarrow': '←', r'\Rightarrow': '⇒',
        r'\Leftarrow': '⇐', r'\leftrightarrow': '↔', r'\Leftrightarrow': '⇔',
        r'\uparrow': '↑', r'\downarrow': '↓',
        r'\circ': '∘', r'\bullet': '•', r'\star': '★',
        r'\land': '∧', r'\lor': '∨', r'\neg': '¬',
        r'\oplus': '⊕', r'\otimes': '⊗',
        r'\le': '≤', r'\ge': '≥', r'\ll': '≪', r'\gg': '≫',
        r'\max': 'max', r'\min': 'min',
    }

    result = latex

    for cmd, char in greek.items():
        result = result.replace(cmd, char)

    for cmd, char in symbols.items():
        result = result.replace(cmd, char)

    result = re.sub(r'\\text\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\textbf\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', result)

    result = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', result)

    def convert_subscript(match):
        content = match.group(1) or match.group(2)
        return ''.join(subscript_map.get(c, c) for c in content)

    result = re.sub(r'_\{([^}]*)\}|_([a-zA-Z0-9])', convert_subscript, result)

    def convert_superscript(match):
        content = match.group(1) or match.group(2)
        return ''.join(superscript_map.get(c, c) for c in content)

    result = re.sub(r'\^\{([^}]*)\}|\^([a-zA-Z0-9*])', convert_superscript, result)

    result = re.sub(r'\\sqrt\{([^}]*)\}', r'√(\1)', result)
    result = re.sub(r'\\[a-zA-Z]+', '', result)
    result = result.replace('{', '').replace('}', '')
    result = ' '.join(result.split())

    return result

def convert_latex_to_unicode(text):
    """Convert LaTeX math expressions to Unicode."""
    if not text:
        return text

    def replace_display_math(match):
        latex = match.group(1)
        latex = re.sub(r'\\begin\{[^}]+\}', '', latex)
        latex = re.sub(r'\\end\{[^}]+\}', '', latex)
        unicode_math = latex_to_unicode(latex)
        return f'<p class="math-block">{unicode_math}</p>'

    def replace_inline_math(match):
        latex = match.group(1)
        unicode_math = latex_to_unicode(latex)
        return f'<span class="math-inline">{unicode_math}</span>'

    text = re.sub(r'\\\[(.+?)\\\]', replace_display_math, text, flags=re.DOTALL)
    text = re.sub(r'\$\$([^$]+)\$\$', replace_display_math, text, flags=re.DOTALL)
    text = re.sub(r'\\\((.+?)\\\)', replace_inline_math, text, flags=re.DOTALL)
    text = re.sub(r'(?<!\$)\$([^$]+)\$(?!\$)', replace_inline_math, text)

    return text

def convert_latex_in_soup(soup):
    """Convert all LaTeX math in the soup to Unicode."""
    for element in list(soup.find_all(string=True)):
        if element.parent.name in ['script', 'style', 'code', 'pre']:
            continue

        text = str(element)
        if '$' in text or '\\[' in text or '\\(' in text:
            new_text = convert_latex_to_unicode(text)
            if new_text != text:
                new_soup = BeautifulSoup(new_text, 'html.parser')
                element.replace_with(new_soup)

    return soup

def fix_images_for_epub(soup, book, chapter_id):
    """Process images for EPUB - download and embed them."""
    images_added = []

    # Handle picture elements
    for picture in list(soup.find_all('picture')):
        img = picture.find('img')
        if img:
            src = img.get('src')
            if src:
                src = make_absolute_url(src)
                new_img = soup.new_tag('img')
                new_img['src'] = src
                new_img['alt'] = img.get('alt', '')
                new_img['style'] = 'max-width: 100%; height: auto;'
                picture.replace_with(new_img)
            else:
                picture.decompose()
        else:
            picture.decompose()

    # Process all img elements
    img_counter = 0
    for img in soup.find_all('img'):
        src = img.get('src')
        if not src:
            img.decompose()
            continue

        src = make_absolute_url(src)

        # Download the image
        result = download_image(src)
        if result:
            content, media_type, ext = result
            img_counter += 1
            img_filename = f"images/{chapter_id}_img_{img_counter}.{ext}"

            # Create EPUB image item
            img_item = epub.EpubImage()
            img_item.file_name = img_filename
            img_item.media_type = media_type
            img_item.content = content
            book.add_item(img_item)
            images_added.append(img_item)

            # Update img src to relative path
            img['src'] = img_filename
        else:
            # Remove broken images
            img.decompose()

    # Handle iframes (Plotly charts) - replace with placeholder
    for iframe in list(soup.find_all('iframe')):
        src = iframe.get('src', '')
        placeholder = soup.new_tag('p')
        placeholder['class'] = 'iframe-placeholder'
        placeholder.string = f"[Interactive chart: {src.split('/')[-1]}]"
        iframe.replace_with(placeholder)

    return soup, images_added

def convert_distill_elements(soup):
    """Convert Distill.js custom elements to standard HTML."""

    for elem in soup.find_all('d-title'):
        elem.name = 'div'
        elem['class'] = elem.get('class', []) + ['distill-title']

    for elem in soup.find_all('d-article'):
        elem.name = 'article'

    for elem in soup.find_all('d-contents'):
        elem.name = 'nav'
        elem['class'] = elem.get('class', []) + ['toc']

    footnote_counter = [0]
    for elem in soup.find_all('d-footnote'):
        footnote_counter[0] += 1
        marker = soup.new_tag('sup')
        marker['class'] = ['footnote-marker']
        marker.string = f"[{footnote_counter[0]}]"
        elem.replace_with(marker)

    for elem in soup.find_all('d-cite'):
        key = elem.get('key', '')
        new_span = soup.new_tag('span')
        new_span['class'] = ['citation']
        new_span.string = f"[{key}]" if key else ""
        elem.replace_with(new_span)

    for elem in soup.find_all('d-math'):
        elem.name = 'span'
        elem['class'] = elem.get('class', []) + ['math']

    for tag_name in ['d-front-matter', 'd-byline', 'd-appendix', 'd-footnote-list',
                     'd-citation-list', 'd-bibliography']:
        for elem in soup.find_all(tag_name):
            elem.decompose()

    for script in soup.find_all('script'):
        script.decompose()

    for elem in soup.find_all(['nav', 'header', 'footer']):
        elem.decompose()

    for class_name in ['navbar', 'nav-links', 'toggle-container', 'giscus']:
        for elem in soup.find_all(class_=class_name):
            elem.decompose()

    for elem in soup.find_all(id=lambda x: x and 'giscus' in x):
        elem.decompose()

    for elem in soup.find_all('progress'):
        elem.decompose()

    return soup

def fix_internal_links(soup, current_page_slug, all_page_slugs):
    """Convert internal links to EPUB internal links."""
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']

        if href.startswith(('mailto:', 'javascript:', 'data:')):
            continue

        if href.startswith(('http://', 'https://')):
            if 'jax-ml.github.io/scaling-book' in href:
                parsed = urlparse(href)
                path = parsed.path.rstrip('/')
                slug = path.split('/scaling-book/')[-1] if '/scaling-book/' in path else 'index'
                if not slug:
                    slug = 'index'
                if slug in all_page_slugs:
                    if parsed.fragment:
                        a_tag['href'] = f"chapter_{slug}.xhtml#{parsed.fragment}"
                    else:
                        a_tag['href'] = f"chapter_{slug}.xhtml"
            continue

        if href.startswith('#'):
            pass  # Keep same-page anchors
        elif href.startswith('/scaling-book/'):
            slug = href.replace('/scaling-book/', '').rstrip('/')
            if not slug:
                slug = 'index'
            if '#' in slug:
                slug_part, anchor = slug.split('#', 1)
                slug_part = slug_part or 'index'
                if slug_part in all_page_slugs:
                    a_tag['href'] = f"chapter_{slug_part}.xhtml#{anchor}"
            else:
                if slug in all_page_slugs:
                    a_tag['href'] = f"chapter_{slug}.xhtml"
        else:
            if '#' in href:
                slug_part, anchor = href.split('#', 1)
                slug_part = slug_part.rstrip('/') or current_page_slug
                if slug_part in all_page_slugs:
                    a_tag['href'] = f"chapter_{slug_part}.xhtml#{anchor}"
            else:
                slug = href.rstrip('/')
                if slug in all_page_slugs:
                    a_tag['href'] = f"chapter_{slug}.xhtml"

    return soup

def extract_content(html, page_slug, all_page_slugs):
    """Extract and clean content from a page."""
    soup = BeautifulSoup(html, 'html.parser')

    content = soup.find('d-article') or soup.find('article') or soup.find('main')
    title_elem = soup.find('d-title')

    page_soup = BeautifulSoup('<div></div>', 'html.parser')
    page_div = page_soup.div

    if title_elem:
        title_copy = copy.copy(title_elem)
        page_div.append(title_copy)

    if content:
        content_copy = copy.copy(content)
        page_div.append(content_copy)

    page_soup = convert_distill_elements(page_soup)
    page_soup = convert_latex_in_soup(page_soup)
    page_soup = fix_internal_links(page_soup, page_slug, all_page_slugs)

    return page_soup

# CSS for EPUB
EPUB_CSS = '''
body {
    font-family: Georgia, "Times New Roman", serif;
    line-height: 1.6;
    margin: 1em;
    color: #333;
}

h1 {
    font-size: 1.8em;
    margin-top: 1em;
    margin-bottom: 0.5em;
    color: #1a1a1a;
    border-bottom: 2px solid #333;
    padding-bottom: 0.3em;
}

h2 {
    font-size: 1.4em;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
    color: #2a2a2a;
}

h3 {
    font-size: 1.2em;
    margin-top: 1em;
    margin-bottom: 0.3em;
    color: #3a3a3a;
}

h4 {
    font-size: 1.1em;
    margin-top: 0.8em;
    margin-bottom: 0.3em;
}

p {
    margin: 0.8em 0;
    text-align: justify;
}

a {
    color: #0055aa;
    text-decoration: none;
}

pre, code {
    font-family: monospace;
    background-color: #f5f5f5;
    border-radius: 3px;
}

pre {
    padding: 0.8em;
    overflow-x: auto;
    font-size: 0.85em;
    border: 1px solid #ddd;
    white-space: pre-wrap;
    word-wrap: break-word;
}

code {
    padding: 0.1em 0.3em;
    font-size: 0.9em;
}

pre code {
    padding: 0;
    background: none;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1em auto;
}

figure {
    margin: 1em 0;
}

figcaption, .caption {
    font-size: 0.9em;
    color: #666;
    text-align: center;
    margin-top: 0.5em;
    font-style: italic;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 0.9em;
}

th, td {
    border: 1px solid #ddd;
    padding: 0.5em;
    text-align: left;
}

th {
    background-color: #f5f5f5;
}

ul, ol {
    margin: 0.8em 0;
    padding-left: 1.5em;
}

li {
    margin: 0.3em 0;
}

blockquote {
    border-left: 3px solid #ddd;
    margin: 1em 0;
    padding: 0.5em 0 0.5em 1em;
    color: #555;
    font-style: italic;
}

.math-inline {
    font-family: serif;
    font-style: italic;
}

.math-block {
    text-align: center;
    margin: 1em 0;
    font-family: serif;
    font-style: italic;
}

.footnote-marker {
    color: #0055aa;
    font-size: 0.8em;
}

.citation {
    color: #666;
    font-size: 0.9em;
}

.distill-title h1 {
    font-size: 1.6em;
    margin-bottom: 0.3em;
}

.distill-title p {
    font-size: 1em;
    color: #555;
}

.subtitle {
    font-size: 1.1em;
    color: #666;
    font-style: italic;
}

.announce {
    background: #fff3cd;
    border: 1px solid #ffc107;
    padding: 0.8em;
    border-radius: 4px;
    margin: 1em 0;
}

.iframe-placeholder {
    background: #f5f5f5;
    border: 1px solid #ddd;
    padding: 1em;
    text-align: center;
    color: #666;
    font-style: italic;
}

.next-section {
    margin-top: 2em;
    padding-top: 1em;
    border-top: 1px solid #eee;
}
'''

def main():
    print("=" * 60)
    print("Scaling Book Website to EPUB Converter")
    print("=" * 60)

    # Create EPUB book
    book = epub.EpubBook()
    book.set_identifier('scaling-book-2025')
    book.set_title('How to Scale Your Model: A Systems View of LLMs on TPUs')
    book.set_language('en')
    book.add_author('Jacob Austin et al.')

    # Add metadata
    book.add_metadata('DC', 'publisher', 'Google DeepMind')
    book.add_metadata('DC', 'description',
        'Training LLMs often feels like alchemy, but understanding and optimizing '
        'the performance of your models doesn\'t have to. This book demystifies the '
        'science of scaling language models.')

    all_page_slugs = [slug for slug, _ in PAGES]
    chapters = []

    # Add CSS
    css_item = epub.EpubItem(
        uid="style",
        file_name="style/main.css",
        media_type="text/css",
        content=EPUB_CSS
    )
    book.add_item(css_item)

    # Fetch and process all pages
    for page_slug, title in PAGES:
        try:
            html = fetch_page(page_slug)
            soup = extract_content(html, page_slug, all_page_slugs)

            # Process images for EPUB
            soup, images = fix_images_for_epub(soup, book, page_slug)

            # Get the content HTML
            content_div = soup.find('div')
            content_html = str(content_div) if content_div else "<p>Content not available.</p>"

            # Ensure we have some content
            if not content_html.strip() or content_html.strip() == "<div></div>":
                content_html = "<p>Content not available.</p>"

            # Create chapter
            chapter = epub.EpubHtml(
                title=title,
                file_name=f"chapter_{page_slug}.xhtml",
                lang='en'
            )

            # Set content using ebooklib's expected format
            chapter.set_content(f'''<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>{title}</title>
    <link rel="stylesheet" type="text/css" href="style/main.css"/>
</head>
<body>
    <h1>{title}</h1>
    {content_html}
</body>
</html>''')
            chapter.add_item(css_item)

            book.add_item(chapter)
            chapters.append(chapter)

            print(f"  ✓ Processed: {title} ({len(images)} images)")

        except Exception as e:
            import traceback
            print(f"  ✗ Error processing {page_slug}: {e}")
            traceback.print_exc()

    # Create table of contents
    book.toc = [(epub.Section('Chapters'), chapters)]

    # Add navigation files
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Define spine (reading order)
    book.spine = ['nav'] + chapters

    # Write EPUB file
    epub_path = "/home/user/scaling-book/scaling-book.epub"
    print(f"\nWriting EPUB to: {epub_path}")
    epub.write_epub(epub_path, book, {})

    # Get file size
    size_mb = os.path.getsize(epub_path) / (1024 * 1024)
    print(f"\n✓ Success! EPUB saved to: {epub_path}")
    print(f"  File size: {size_mb:.2f} MB")
    print(f"  Chapters: {len(chapters)}")

if __name__ == "__main__":
    main()
