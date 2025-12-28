#!/usr/bin/env python3
"""
Fetch all pages from https://jax-ml.github.io/scaling-book/ and convert to EPUB.
NARROW VERSION - Optimized for small screens with reduced math sizing.
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
import latex2mathml.converter

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

def latex_to_mathml(latex, display=False):
    """Convert LaTeX to MathML using latex2mathml."""
    # Unescape HTML entities
    latex = latex.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')

    # Remove \begin{equation}, \end{equation}, \begin{align*}, etc.
    latex = re.sub(r'\\begin\{[^}]+\}', '', latex)
    latex = re.sub(r'\\end\{[^}]+\}', '', latex)

    # Clean up alignment characters that latex2mathml doesn't handle
    latex = latex.replace('&', '')  # Remove alignment markers
    latex = re.sub(r'\\\[[\d.]*em\]', '', latex)  # Remove spacing like \\[0.5em]
    latex = latex.replace('\\\\', ' ')  # Replace line breaks with space

    # Strip whitespace
    latex = latex.strip()

    if not latex:
        return ''

    try:
        # Convert to MathML
        display_mode = 'block' if display else 'inline'
        mathml = latex2mathml.converter.convert(latex, display=display_mode)
        return mathml
    except Exception as e:
        # Fallback: return the raw LaTeX in a styled span
        escaped = latex.replace('<', '&lt;').replace('>', '&gt;')
        if display:
            return f'<p class="math-fallback" style="text-align: center; font-style: italic;">{escaped}</p>'
        else:
            return f'<span class="math-fallback" style="font-style: italic;">{escaped}</span>'

def convert_latex_to_mathml(text):
    """Convert LaTeX math expressions to MathML."""
    if not text:
        return text

    def replace_display_math(match):
        latex = match.group(1)
        mathml = latex_to_mathml(latex, display=True)
        return f'<div class="math-block">{mathml}</div>'

    def replace_inline_math(match):
        latex = match.group(1)
        mathml = latex_to_mathml(latex, display=False)
        return mathml  # MathML is already wrapped

    # Replace \[...\] display math
    text = re.sub(r'\\\[(.+?)\\\]', replace_display_math, text, flags=re.DOTALL)

    # Replace $$...$$ display math
    text = re.sub(r'\$\$([^$]+)\$\$', replace_display_math, text, flags=re.DOTALL)

    # Replace \(...\) inline math
    text = re.sub(r'\\\((.+?)\\\)', replace_inline_math, text, flags=re.DOTALL)

    # Replace inline math ($...$)
    text = re.sub(r'(?<!\$)\$([^$]+)\$(?!\$)', replace_inline_math, text)

    return text

def convert_latex_in_soup(soup):
    """Convert all LaTeX math in the soup to MathML."""
    for element in list(soup.find_all(string=True)):
        if element.parent.name in ['script', 'style', 'code', 'pre']:
            continue

        text = str(element)
        if '$' in text or '\\[' in text or '\\(' in text:
            new_text = convert_latex_to_mathml(text)
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

# CSS for EPUB - NARROW VERSION
# Key changes from regular version:
# - Reduced margins (0.5em instead of 1em)
# - Smaller base font size
# - Smaller math font size (0.75em)
# - Horizontal scroll for overflow
# - Smaller headings
EPUB_CSS_NARROW = '''
body {
    font-family: Georgia, "Times New Roman", serif;
    line-height: 1.5;
    margin: 0.5em;
    padding: 0;
    color: #333;
    font-size: 0.95em;
}

h1 {
    font-size: 1.5em;
    margin-top: 0.8em;
    margin-bottom: 0.4em;
    color: #1a1a1a;
    border-bottom: 2px solid #333;
    padding-bottom: 0.2em;
}

h2 {
    font-size: 1.2em;
    margin-top: 1em;
    margin-bottom: 0.3em;
    color: #2a2a2a;
}

h3 {
    font-size: 1.1em;
    margin-top: 0.8em;
    margin-bottom: 0.2em;
    color: #3a3a3a;
}

h4 {
    font-size: 1em;
    margin-top: 0.6em;
    margin-bottom: 0.2em;
}

p {
    margin: 0.6em 0;
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
    padding: 0.5em;
    overflow-x: auto;
    font-size: 0.75em;
    border: 1px solid #ddd;
    white-space: pre-wrap;
    word-wrap: break-word;
}

code {
    padding: 0.1em 0.2em;
    font-size: 0.85em;
}

pre code {
    padding: 0;
    background: none;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0.8em auto;
}

figure {
    margin: 0.8em 0;
}

figcaption, .caption {
    font-size: 0.85em;
    color: #666;
    text-align: center;
    margin-top: 0.4em;
    font-style: italic;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.8em 0;
    font-size: 0.8em;
}

th, td {
    border: 1px solid #ddd;
    padding: 0.4em;
    text-align: left;
}

th {
    background-color: #f5f5f5;
}

ul, ol {
    margin: 0.6em 0;
    padding-left: 1.2em;
}

li {
    margin: 0.2em 0;
}

blockquote {
    border-left: 3px solid #ddd;
    margin: 0.8em 0;
    padding: 0.4em 0 0.4em 0.8em;
    color: #555;
    font-style: italic;
}

.math-inline {
    font-family: serif;
    font-style: italic;
}

.math-block {
    text-align: center;
    margin: 0.8em 0;
    font-family: serif;
    font-style: italic;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}

.footnote-marker {
    color: #0055aa;
    font-size: 0.75em;
}

.citation {
    color: #666;
    font-size: 0.85em;
}

.distill-title h1 {
    font-size: 1.4em;
    margin-bottom: 0.2em;
}

.distill-title p {
    font-size: 0.9em;
    color: #555;
}

.subtitle {
    font-size: 1em;
    color: #666;
    font-style: italic;
}

.announce {
    background: #fff3cd;
    border: 1px solid #ffc107;
    padding: 0.6em;
    border-radius: 4px;
    margin: 0.8em 0;
    font-size: 0.9em;
}

.iframe-placeholder {
    background: #f5f5f5;
    border: 1px solid #ddd;
    padding: 0.8em;
    text-align: center;
    color: #666;
    font-style: italic;
    font-size: 0.85em;
}

.next-section {
    margin-top: 1.5em;
    padding-top: 0.8em;
    border-top: 1px solid #eee;
}

/* MathML styling - NARROW VERSION */
/* Smaller font size to fit on narrow screens */
math {
    font-family: "STIX Two Math", "Cambria Math", "Latin Modern Math", serif;
    font-size: 0.75em;
}

math[display="block"] {
    display: block;
    text-align: center;
    margin: 0.6em 0;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}

/* Wrapper for math blocks to enable scrolling */
.math-block {
    text-align: center;
    margin: 0.6em 0;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    max-width: 100%;
}

.math-block math {
    display: inline-block;
}

.math-fallback {
    font-family: monospace;
    background: #f5f5f5;
    padding: 0.1em 0.3em;
    font-size: 0.8em;
    word-break: break-all;
}

/* Force wide equations to scroll rather than overflow */
mrow, mtable, mfrac {
    max-width: 100%;
}
'''

def main():
    print("=" * 60)
    print("Scaling Book Website to EPUB Converter (NARROW VERSION)")
    print("=" * 60)

    # Create EPUB book
    book = epub.EpubBook()
    book.set_identifier('scaling-book-narrow-2025')
    book.set_title('How to Scale Your Model: A Systems View of LLMs on TPUs (Narrow)')
    book.set_language('en')
    book.add_author('Jacob Austin et al.')

    # Add metadata
    book.add_metadata('DC', 'publisher', 'Google DeepMind')
    book.add_metadata('DC', 'description',
        'Training LLMs often feels like alchemy, but understanding and optimizing '
        'the performance of your models doesn\'t have to. This book demystifies the '
        'science of scaling language models. (Narrow version optimized for small screens)')

    all_page_slugs = [slug for slug, _ in PAGES]
    chapters = []

    # Add CSS - NARROW VERSION
    css_item = epub.EpubItem(
        uid="style",
        file_name="style/main.css",
        media_type="text/css",
        content=EPUB_CSS_NARROW
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

            # Set content using ebooklib's expected format with MathML namespace
            chapter.set_content(f'''<html xmlns="http://www.w3.org/1999/xhtml" xmlns:m="http://www.w3.org/1998/Math/MathML">
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
    epub_path = "/home/user/scaling-book/scaling-book-narrow.epub"
    print(f"\nWriting EPUB to: {epub_path}")
    epub.write_epub(epub_path, book, {})

    # Get file size
    size_mb = os.path.getsize(epub_path) / (1024 * 1024)
    print(f"\n✓ Success! EPUB saved to: {epub_path}")
    print(f"  File size: {size_mb:.2f} MB")
    print(f"  Chapters: {len(chapters)}")

if __name__ == "__main__":
    main()
