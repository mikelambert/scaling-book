#!/usr/bin/env python3
"""
Fetch all pages from https://jax-ml.github.io/scaling-book/ and convert to a combined PDF
with internal links converted to PDF anchors.

Handles the Distill.js template used by the scaling-book website.
"""

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from weasyprint import HTML, CSS
import re
import os
from urllib.parse import urljoin, urlparse
import copy
# Note: latex2mathml generates MathML but WeasyPrint has limited support
# We'll use a custom Unicode-based renderer instead

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

def fetch_page(page_slug):
    """Fetch a page and return its HTML content."""
    if page_slug == "index":
        url = BASE_URL
    else:
        url = urljoin(BASE_URL, page_slug)

    print(f"Fetching: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text

def make_absolute_url(url, base=BASE_URL):
    """Convert relative URLs to absolute."""
    if url.startswith(('http://', 'https://', 'data:')):
        return url
    return urljoin(base, url)

def latex_to_unicode(latex):
    """Convert LaTeX to Unicode text representation."""
    # Unescape HTML entities
    latex = latex.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')

    # Unicode subscript and superscript maps
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

    # Greek letters
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

    # Math symbols
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
    }

    result = latex

    # Replace Greek letters
    for cmd, char in greek.items():
        result = result.replace(cmd, char)

    # Replace math symbols
    for cmd, char in symbols.items():
        result = result.replace(cmd, char)

    # Handle \text{...} - just extract the text
    result = re.sub(r'\\text\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\textbf\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', result)

    # Handle fractions: \frac{a}{b} -> a/b
    result = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', result)

    # Handle subscripts: _{...} or _x
    def convert_subscript(match):
        content = match.group(1) or match.group(2)
        return ''.join(subscript_map.get(c, c) for c in content)

    result = re.sub(r'_\{([^}]*)\}|_([a-zA-Z0-9])', convert_subscript, result)

    # Handle superscripts: ^{...} or ^x
    def convert_superscript(match):
        content = match.group(1) or match.group(2)
        return ''.join(superscript_map.get(c, c) for c in content)

    result = re.sub(r'\^\{([^}]*)\}|\^([a-zA-Z0-9*])', convert_superscript, result)

    # Handle \sqrt{...}
    result = re.sub(r'\\sqrt\{([^}]*)\}', r'√(\1)', result)

    # Clean up remaining LaTeX commands
    result = re.sub(r'\\[a-zA-Z]+', '', result)

    # Remove remaining braces
    result = result.replace('{', '').replace('}', '')

    # Clean up extra whitespace
    result = ' '.join(result.split())

    return result

def convert_latex_to_unicode(text):
    """Convert LaTeX math expressions to Unicode."""
    if not text:
        return text

    def replace_display_math(match):
        """Replace display math with styled block."""
        latex = match.group(1)
        # Remove \begin{equation}, \end{equation}, etc.
        latex = re.sub(r'\\begin\{[^}]+\}', '', latex)
        latex = re.sub(r'\\end\{[^}]+\}', '', latex)
        unicode_math = latex_to_unicode(latex)
        return f'<div class="math-block">{unicode_math}</div>'

    def replace_inline_math(match):
        """Replace $...$ with styled inline."""
        latex = match.group(1)
        unicode_math = latex_to_unicode(latex)
        return f'<span class="math-inline">{unicode_math}</span>'

    # Replace \[...\] display math (\\\[ = backslash + literal [)
    text = re.sub(r'\\\[(.+?)\\\]', replace_display_math, text, flags=re.DOTALL)

    # Replace $$...$$ display math
    text = re.sub(r'\$\$([^$]+)\$\$', replace_display_math, text, flags=re.DOTALL)

    # Replace \(...\) inline math
    text = re.sub(r'\\\((.+?)\\\)', replace_inline_math, text, flags=re.DOTALL)

    # Replace inline math ($...$)
    text = re.sub(r'(?<!\$)\$([^$]+)\$(?!\$)', replace_inline_math, text)

    return text

def convert_latex_in_soup(soup):
    """Convert all LaTeX math in the soup to Unicode."""
    # Process all text nodes
    for element in list(soup.find_all(string=True)):
        if element.parent.name in ['script', 'style', 'code', 'pre']:
            continue

        text = str(element)
        # Check for any math delimiters: $, \[, \(, or $$
        if '$' in text or '\\[' in text or '\\(' in text:
            new_text = convert_latex_to_unicode(text)
            if new_text != text:
                # Create new soup from the converted text and replace
                new_soup = BeautifulSoup(new_text, 'html.parser')
                element.replace_with(new_soup)

    return soup

def fix_images(soup):
    """Make all image sources absolute URLs and simplify picture elements."""
    # First, unwrap picture elements - replace them with just the img inside
    for picture in list(soup.find_all('picture')):
        img = picture.find('img')
        if img:
            # Get the best source URL - prefer png/jpg from img src
            best_src = None

            if img.get('src'):
                best_src = make_absolute_url(img['src'])

            # If no img src, try source srcset
            if not best_src:
                source = picture.find('source')
                if source and source.get('srcset'):
                    srcset = source['srcset']
                    # Get the first URL from srcset
                    parts = srcset.split(',')
                    if parts:
                        first_part = parts[0].strip().split()
                        if first_part:
                            best_src = make_absolute_url(first_part[0])

            if best_src:
                # Create a simple img element
                new_img = soup.new_tag('img')
                new_img['src'] = best_src
                new_img['style'] = 'max-width: 100%; height: auto;'
                if img.get('alt'):
                    new_img['alt'] = img['alt']
                picture.replace_with(new_img)
            else:
                picture.decompose()
        else:
            # No img found, just remove the picture element
            picture.decompose()

    # Fix remaining img elements
    for img in soup.find_all('img'):
        if img.get('src'):
            img['src'] = make_absolute_url(img['src'])
        # Remove srcset as it complicates things
        if img.get('srcset'):
            del img['srcset']
        # Remove loading=lazy for PDF
        if img.get('loading'):
            del img['loading']
        # Remove onerror handlers
        if img.get('onerror'):
            del img['onerror']

    # Handle iframes (e.g., Plotly charts) - replace with a placeholder
    for iframe in list(soup.find_all('iframe')):
        src = iframe.get('src', '')
        if src:
            # Create a placeholder div
            placeholder = soup.new_tag('div')
            placeholder['class'] = ['iframe-placeholder']
            placeholder['style'] = 'background: #f5f5f5; border: 1px solid #ddd; padding: 20px; text-align: center; margin: 1em 0;'
            placeholder.string = f"[Interactive chart: {src.split('/')[-1]}]"
            iframe.replace_with(placeholder)
        else:
            iframe.decompose()

    return soup

def convert_distill_elements(soup):
    """Convert Distill.js custom elements to standard HTML."""

    # Convert d-title to a div
    for elem in soup.find_all('d-title'):
        elem.name = 'div'
        elem['class'] = elem.get('class', []) + ['distill-title']

    # Convert d-article to article
    for elem in soup.find_all('d-article'):
        elem.name = 'article'

    # Convert d-contents to nav (table of contents)
    for elem in soup.find_all('d-contents'):
        elem.name = 'nav'
        elem['class'] = elem.get('class', []) + ['toc']

    # Convert d-footnote to span with a marker
    footnote_counter = [0]
    for elem in soup.find_all('d-footnote'):
        footnote_counter[0] += 1
        # Create a superscript footnote marker
        marker = soup.new_tag('sup')
        marker['class'] = ['footnote-marker']
        marker.string = f"[{footnote_counter[0]}]"

        # Create the footnote text
        footnote_text = soup.new_tag('span')
        footnote_text['class'] = ['footnote-text']
        footnote_text.string = f" [{footnote_counter[0]}]: {elem.get_text(strip=True)}"

        # Replace the element with just the marker (footnotes will be inline)
        elem.replace_with(marker)

    # Convert d-cite to a simple citation span
    for elem in soup.find_all('d-cite'):
        key = elem.get('key', '')
        new_span = soup.new_tag('span')
        new_span['class'] = ['citation']
        new_span.string = f"[{key}]" if key else ""
        elem.replace_with(new_span)

    # Convert d-math to a span (equations)
    for elem in soup.find_all('d-math'):
        elem.name = 'span'
        elem['class'] = elem.get('class', []) + ['math']

    # Remove d-front-matter, d-byline, d-appendix, d-footnote-list, d-citation-list, d-bibliography
    for tag_name in ['d-front-matter', 'd-byline', 'd-appendix', 'd-footnote-list',
                     'd-citation-list', 'd-bibliography']:
        for elem in soup.find_all(tag_name):
            elem.decompose()

    # Remove script tags
    for script in soup.find_all('script'):
        script.decompose()

    # Remove navigation elements
    for elem in soup.find_all(['nav', 'header', 'footer']):
        elem.decompose()

    # Remove elements with certain classes
    for class_name in ['navbar', 'nav-links', 'toggle-container', 'giscus']:
        for elem in soup.find_all(class_=class_name):
            elem.decompose()

    # Remove giscus div
    for elem in soup.find_all(id=lambda x: x and 'giscus' in x):
        elem.decompose()

    # Remove progress bar
    for elem in soup.find_all('progress'):
        elem.decompose()

    return soup

def fix_internal_links(soup, current_page_slug, all_page_slugs):
    """
    Convert internal links to PDF anchors.
    """
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']

        # Skip external links and special protocols
        if href.startswith(('mailto:', 'javascript:', 'data:')):
            continue

        # Check if it's an internal link to the scaling-book site
        if href.startswith(('http://', 'https://')):
            if 'jax-ml.github.io/scaling-book' in href:
                parsed = urlparse(href)
                path = parsed.path.rstrip('/')
                slug = path.split('/scaling-book/')[-1] if '/scaling-book/' in path else 'index'
                if not slug:
                    slug = 'index'
                if slug in all_page_slugs:
                    if parsed.fragment:
                        a_tag['href'] = f"#page-{slug}-{parsed.fragment}"
                    else:
                        a_tag['href'] = f"#page-{slug}"
            continue

        # Handle relative links
        if href.startswith('#'):
            # Same-page anchor
            anchor = href[1:]
            a_tag['href'] = f"#page-{current_page_slug}-{anchor}"
        elif href.startswith('/scaling-book/'):
            # Absolute path within site
            slug = href.replace('/scaling-book/', '').rstrip('/')
            if not slug:
                slug = 'index'
            if '#' in slug:
                slug_part, anchor = slug.split('#', 1)
                slug_part = slug_part or 'index'
                if slug_part in all_page_slugs:
                    a_tag['href'] = f"#page-{slug_part}-{anchor}"
            else:
                if slug in all_page_slugs:
                    a_tag['href'] = f"#page-{slug}"
        else:
            # Relative link
            if '#' in href:
                slug_part, anchor = href.split('#', 1)
                slug_part = slug_part.rstrip('/') or current_page_slug
                if slug_part in all_page_slugs:
                    a_tag['href'] = f"#page-{slug_part}-{anchor}"
            else:
                slug = href.rstrip('/')
                if slug in all_page_slugs:
                    a_tag['href'] = f"#page-{slug}"

    return soup

def prefix_ids(soup, page_slug):
    """Prefix all IDs with the page slug to avoid conflicts."""
    for elem in soup.find_all(id=True):
        old_id = elem['id']
        if not old_id.startswith('page-'):
            elem['id'] = f"page-{page_slug}-{old_id}"
    return soup

def extract_content(html, page_slug, all_page_slugs):
    """Extract and clean content from a page."""
    soup = BeautifulSoup(html, 'html.parser')

    # Find the main content - look for d-article or article or main
    content = soup.find('d-article') or soup.find('article') or soup.find('main')
    title_elem = soup.find('d-title')

    # Create new soup for this page's content
    page_soup = BeautifulSoup('<div></div>', 'html.parser')
    page_div = page_soup.div

    # Add title content if exists
    if title_elem:
        title_copy = copy.copy(title_elem)
        page_div.append(title_copy)

    # Add main content
    if content:
        content_copy = copy.copy(content)
        page_div.append(content_copy)

    # Fix images to use absolute URLs
    page_soup = fix_images(page_soup)

    # Convert Distill elements to standard HTML
    page_soup = convert_distill_elements(page_soup)

    # Convert LaTeX math to MathML
    page_soup = convert_latex_in_soup(page_soup)

    # Fix internal links
    page_soup = fix_internal_links(page_soup, page_slug, all_page_slugs)

    # Prefix IDs
    page_soup = prefix_ids(page_soup, page_slug)

    return page_soup

def create_combined_html(pages_content):
    """Combine all pages into a single HTML document."""
    html_template = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>How to Scale Your Model - Combined PDF</title>
</head>
<body>
</body>
</html>'''

    combined = BeautifulSoup(html_template, 'html.parser')
    body = combined.find('body')

    for i, (page_slug, title, soup) in enumerate(pages_content):
        # Page container with anchor
        page_container = combined.new_tag('section')
        page_container['id'] = f"page-{page_slug}"
        page_container['class'] = ['chapter']

        # Add page break (except for first page)
        if i > 0:
            page_container['style'] = 'page-break-before: always;'

        # Add chapter title
        chapter_header = combined.new_tag('div')
        chapter_header['class'] = ['chapter-header']
        chapter_title = combined.new_tag('h1')
        chapter_title['class'] = ['chapter-title']
        chapter_title.string = title
        chapter_header.append(chapter_title)
        page_container.append(chapter_header)

        # Add horizontal rule
        hr = combined.new_tag('hr')
        page_container.append(hr)

        # Add content from the page
        content_div = soup.find('div')
        if content_div:
            for child in list(content_div.children):
                if isinstance(child, Tag):
                    page_container.append(copy.copy(child))
                elif isinstance(child, NavigableString) and str(child).strip():
                    page_container.append(str(child))

        body.append(page_container)

    return combined

# CSS for proper PDF rendering
PDF_CSS = '''
@page {
    size: A4;
    margin: 2cm 2.5cm;
    @bottom-center {
        content: counter(page);
        font-size: 10pt;
        color: #666;
    }
}

* {
    box-sizing: border-box;
}

body {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #333;
    max-width: 100%;
}

.chapter {
    max-width: 100%;
}

.chapter-header {
    margin-bottom: 1.5em;
}

.chapter-title {
    font-size: 28pt;
    font-weight: bold;
    color: #1a1a1a;
    margin: 0 0 0.5em 0;
    padding-bottom: 0.3em;
}

hr {
    border: none;
    border-top: 2px solid #333;
    margin: 1em 0 2em 0;
}

h1 {
    font-size: 22pt;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    color: #1a1a1a;
    page-break-after: avoid;
}

h2 {
    font-size: 18pt;
    margin-top: 1.3em;
    margin-bottom: 0.4em;
    color: #2a2a2a;
    page-break-after: avoid;
}

h3 {
    font-size: 14pt;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
    color: #3a3a3a;
    page-break-after: avoid;
}

h4 {
    font-size: 12pt;
    margin-top: 1em;
    margin-bottom: 0.3em;
    color: #444;
    page-break-after: avoid;
}

p {
    margin: 0.8em 0;
    text-align: justify;
    hyphens: auto;
}

a {
    color: #0055aa;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Code blocks */
pre, code {
    font-family: "SF Mono", Monaco, "Cascadia Code", "Roboto Mono", Consolas, "Liberation Mono", monospace;
    background-color: #f6f8fa;
    border-radius: 4px;
}

pre {
    padding: 1em;
    overflow-x: auto;
    font-size: 9pt;
    line-height: 1.5;
    border: 1px solid #e1e4e8;
    margin: 1em 0;
    white-space: pre-wrap;
    word-wrap: break-word;
}

code {
    padding: 0.2em 0.4em;
    font-size: 10pt;
}

pre code {
    padding: 0;
    background: none;
    border: none;
    font-size: inherit;
}

/* Images and figures */
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1em auto;
}

figure {
    margin: 1.5em 0;
    page-break-inside: avoid;
}

figcaption, .caption {
    font-size: 10pt;
    color: #666;
    text-align: center;
    margin-top: 0.5em;
    font-style: italic;
}

picture {
    display: block;
}

/* Tables */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 10pt;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}

th {
    background-color: #f5f5f5;
    font-weight: bold;
}

tr:nth-child(even) {
    background-color: #fafafa;
}

/* Lists */
ul, ol {
    margin: 0.8em 0;
    padding-left: 2em;
}

li {
    margin: 0.3em 0;
}

/* Blockquotes */
blockquote {
    border-left: 4px solid #ddd;
    margin: 1em 0;
    padding: 0.5em 0 0.5em 1em;
    color: #555;
    font-style: italic;
}

/* Footnotes */
.footnote-marker {
    color: #0055aa;
    font-size: 9pt;
}

.footnote-text {
    display: block;
    font-size: 9pt;
    color: #666;
    margin-top: 0.5em;
}

/* Citations */
.citation {
    color: #666;
    font-size: 10pt;
}

/* Math - MathML styling */
.math, .math-inline, .math-block {
    font-family: "Latin Modern Math", "STIX Two Math", "DejaVu Math TeX Gyre", serif;
}

.math-inline {
    display: inline;
}

.math-block {
    display: block;
    text-align: center;
    margin: 1em 0;
}

.math-error {
    color: #666;
    font-style: italic;
}

math {
    font-family: "Latin Modern Math", "STIX Two Math", "DejaVu Math TeX Gyre", serif;
}

/* MathML specific */
mfrac {
    display: inline-block;
    vertical-align: middle;
}

msub, msup, msubsup {
    font-size: inherit;
}

/* Table of contents */
.toc {
    background: #f9f9f9;
    padding: 1em;
    margin: 1em 0;
    border-radius: 4px;
}

.toc h3 {
    margin-top: 0;
}

/* Distill-specific styles */
.distill-title {
    margin-bottom: 1.5em;
}

.distill-title h1 {
    font-size: 26pt;
    margin-bottom: 0.3em;
}

.distill-title p {
    font-size: 12pt;
    color: #555;
}

.subtitle {
    font-size: 14pt;
    color: #666;
    font-style: italic;
}

/* Announcements/callouts */
.announce {
    background: #fff3cd;
    border: 1px solid #ffc107;
    padding: 1em;
    border-radius: 4px;
    margin: 1em 0;
}

/* Next section links */
.next-section {
    font-size: 13pt;
    margin-top: 2em;
    padding-top: 1em;
    border-top: 1px solid #eee;
}

/* Hide elements that shouldn't appear */
.navbar, .nav-links, .toggle-container, nav.navbar, header nav,
.giscus, #giscus_thread, progress, .section-button {
    display: none !important;
}

/* Syntax highlighting for code blocks */
.highlight .k, .highlight .kd, .highlight .kn, .highlight .kp,
.highlight .kr, .highlight .kt { color: #d73a49; }
.highlight .s, .highlight .s1, .highlight .s2, .highlight .sr { color: #032f62; }
.highlight .c, .highlight .c1, .highlight .cm { color: #6a737d; font-style: italic; }
.highlight .n, .highlight .na, .highlight .nb, .highlight .nc,
.highlight .nd, .highlight .ne, .highlight .nf, .highlight .ni,
.highlight .nl, .highlight .nn, .highlight .no, .highlight .nt,
.highlight .nv { color: #005cc5; }
.highlight .m, .highlight .mf, .highlight .mh, .highlight .mi,
.highlight .mo { color: #005cc5; }
.highlight .o, .highlight .ow { color: #d73a49; }
'''

def main():
    print("=" * 60)
    print("Scaling Book Website to PDF Converter")
    print("=" * 60)

    all_page_slugs = [slug for slug, _ in PAGES]
    pages_content = []

    # Fetch all pages
    for page_slug, title in PAGES:
        try:
            html = fetch_page(page_slug)
            soup = extract_content(html, page_slug, all_page_slugs)
            pages_content.append((page_slug, title, soup))
            print(f"  ✓ Processed: {title}")
        except Exception as e:
            import traceback
            print(f"  ✗ Error fetching {page_slug}: {e}")
            traceback.print_exc()

    print("\nCombining pages...")
    combined_html = create_combined_html(pages_content)

    # Save combined HTML for debugging
    html_path = "/home/user/scaling-book/scaling-book-combined.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(str(combined_html.prettify()))
    print(f"Saved combined HTML to: {html_path}")

    # Generate PDF
    print("\nGenerating PDF (this may take a while)...")
    pdf_path = "/home/user/scaling-book/scaling-book.pdf"

    try:
        css = CSS(string=PDF_CSS)
        html_doc = HTML(string=str(combined_html), base_url=BASE_URL)
        html_doc.write_pdf(pdf_path, stylesheets=[css])
        print(f"\n✓ Success! PDF saved to: {pdf_path}")

        # Get file size
        size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")

    except Exception as e:
        print(f"✗ Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
