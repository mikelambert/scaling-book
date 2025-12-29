#!/usr/bin/env python3
"""
Fetch all pages from https://jax-ml.github.io/scaling-book/ and convert to EPUB.
KINDLE VERSION - Display math rendered as images using real LaTeX.
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
import hashlib
import subprocess
import tempfile
import shutil

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
# Cache for rendered math images
math_image_cache = {}
# Counter for math images
math_image_counter = [0]
# Stats
math_stats = {"success": 0, "fallback": 0}

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

def clean_latex_for_rendering(latex):
    """Clean LaTeX for rendering - handle HTML entities and common issues."""
    # Unescape HTML entities
    latex = latex.replace('&gt;', '>')
    latex = latex.replace('&lt;', '<')
    latex = latex.replace('&amp;', '&')
    latex = latex.replace('&#39;', "'")
    latex = latex.replace('&quot;', '"')

    # Remove \begin{equation}, etc. - we'll wrap in equation ourselves
    latex = re.sub(r'\\begin\{equation\*?\}', '', latex)
    latex = re.sub(r'\\end\{equation\*?\}', '', latex)

    # Handle align environments - convert to aligned
    latex = re.sub(r'\\begin\{align\*?\}', r'\\begin{aligned}', latex)
    latex = re.sub(r'\\end\{align\*?\}', r'\\end{aligned}', latex)

    # Handle gather environments
    latex = re.sub(r'\\begin\{gather\*?\}', r'\\begin{gathered}', latex)
    latex = re.sub(r'\\end\{gather\*?\}', r'\\end{gathered}', latex)

    # Remove spacing hints that might cause issues
    latex = re.sub(r'\\\[[\d.]*em\]', r'\\\\', latex)

    # Clean up extra whitespace
    latex = latex.strip()

    return latex

def render_latex_with_dvipng(latex, dpi=150):
    """Render LaTeX to PNG using real LaTeX + dvipng."""

    # Create temporary directory
    tmpdir = tempfile.mkdtemp()

    try:
        # LaTeX document template
        tex_content = r'''\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{mathtools}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}

% Define common commands that might be missing
\providecommand{\text}[1]{\textrm{#1}}
\providecommand{\operatorname}[1]{\textrm{#1}}

\pagestyle{empty}
\begin{document}
\begin{equation*}
''' + latex + r'''
\end{equation*}
\end{document}
'''

        # Write tex file
        tex_path = os.path.join(tmpdir, 'math.tex')
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(tex_content)

        # Run latex to create DVI
        result = subprocess.run(
            ['latex', '-interaction=nonstopmode', '-halt-on-error', 'math.tex'],
            cwd=tmpdir,
            capture_output=True,
            timeout=30
        )

        dvi_path = os.path.join(tmpdir, 'math.dvi')
        if not os.path.exists(dvi_path):
            return None

        # Run dvipng to create PNG
        png_path = os.path.join(tmpdir, 'math.png')
        result = subprocess.run(
            ['dvipng', '-D', str(dpi), '-T', 'tight', '-bg', 'Transparent',
             '-o', png_path, dvi_path],
            cwd=tmpdir,
            capture_output=True,
            timeout=30
        )

        if not os.path.exists(png_path):
            return None

        # Read PNG content
        with open(png_path, 'rb') as f:
            return f.read()

    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None
    finally:
        # Clean up
        shutil.rmtree(tmpdir, ignore_errors=True)

def render_latex_to_image(latex, book):
    """Render LaTeX to a PNG image and add to book."""

    # Clean the LaTeX
    clean_latex = clean_latex_for_rendering(latex)
    if not clean_latex:
        return None

    # Check cache
    cache_key = hashlib.md5(clean_latex.encode()).hexdigest()
    if cache_key in math_image_cache:
        return math_image_cache[cache_key]

    # Try to render with LaTeX
    img_content = render_latex_with_dvipng(clean_latex)

    if img_content:
        math_stats["success"] += 1

        # Create unique filename
        math_image_counter[0] += 1
        img_filename = f"images/math_{math_image_counter[0]}.png"

        # Add to book
        img_item = epub.EpubImage()
        img_item.file_name = img_filename
        img_item.media_type = 'image/png'
        img_item.content = img_content
        book.add_item(img_item)

        # Cache it
        math_image_cache[cache_key] = img_filename

        return img_filename
    else:
        math_stats["fallback"] += 1
        return None

def latex_to_unicode_inline(latex):
    """Convert inline LaTeX to Unicode approximation."""
    # Unescape HTML entities
    latex = latex.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')

    # Greek letters
    greek = {
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
        r'\epsilon': 'ε', r'\varepsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η',
        r'\theta': 'θ', r'\vartheta': 'ϑ', r'\iota': 'ι', r'\kappa': 'κ',
        r'\lambda': 'λ', r'\mu': 'μ', r'\nu': 'ν', r'\xi': 'ξ',
        r'\pi': 'π', r'\varpi': 'ϖ', r'\rho': 'ρ', r'\varrho': 'ϱ',
        r'\sigma': 'σ', r'\varsigma': 'ς', r'\tau': 'τ', r'\upsilon': 'υ',
        r'\phi': 'φ', r'\varphi': 'ϕ', r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
        r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
        r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Upsilon': 'Υ',
        r'\Phi': 'Φ', r'\Psi': 'Ψ', r'\Omega': 'Ω',
    }

    # Math symbols
    symbols = {
        r'\times': '×', r'\div': '÷', r'\pm': '±', r'\mp': '∓',
        r'\cdot': '·', r'\ast': '∗', r'\star': '⋆', r'\circ': '∘',
        r'\bullet': '•', r'\oplus': '⊕', r'\otimes': '⊗',
        r'\leq': '≤', r'\le': '≤', r'\geq': '≥', r'\ge': '≥',
        r'\neq': '≠', r'\ne': '≠', r'\approx': '≈', r'\simeq': '≃',
        r'\cong': '≅', r'\equiv': '≡', r'\propto': '∝', r'\sim': '∼',
        r'\ll': '≪', r'\gg': '≫', r'\prec': '≺', r'\succ': '≻',
        r'\infty': '∞', r'\partial': '∂', r'\nabla': '∇',
        r'\sum': '∑', r'\prod': '∏', r'\coprod': '∐',
        r'\int': '∫', r'\iint': '∬', r'\iiint': '∭', r'\oint': '∮',
        r'\rightarrow': '→', r'\to': '→', r'\leftarrow': '←',
        r'\leftrightarrow': '↔', r'\Rightarrow': '⇒', r'\Leftarrow': '⇐',
        r'\Leftrightarrow': '⇔', r'\mapsto': '↦', r'\longmapsto': '⟼',
        r'\uparrow': '↑', r'\downarrow': '↓', r'\updownarrow': '↕',
        r'\forall': '∀', r'\exists': '∃', r'\nexists': '∄',
        r'\in': '∈', r'\notin': '∉', r'\ni': '∋',
        r'\subset': '⊂', r'\supset': '⊃', r'\subseteq': '⊆', r'\supseteq': '⊇',
        r'\cup': '∪', r'\cap': '∩', r'\setminus': '∖',
        r'\emptyset': '∅', r'\varnothing': '∅',
        r'\ldots': '…', r'\cdots': '⋯', r'\dots': '…', r'\vdots': '⋮', r'\ddots': '⋱',
        r'\langle': '⟨', r'\rangle': '⟩',
        r'\lceil': '⌈', r'\rceil': '⌉', r'\lfloor': '⌊', r'\rfloor': '⌋',
        r'\neg': '¬', r'\land': '∧', r'\lor': '∨', r'\wedge': '∧', r'\vee': '∨',
        r'\top': '⊤', r'\bot': '⊥', r'\perp': '⊥',
        r'\prime': '′', r'\angle': '∠', r'\triangle': '△',
        r'\square': '□', r'\Diamond': '◇',
        r'\aleph': 'ℵ', r'\hbar': 'ℏ', r'\ell': 'ℓ', r'\wp': '℘',
        r'\Re': 'ℜ', r'\Im': 'ℑ',
        r'\dagger': '†', r'\ddagger': '‡',
        # Common operators
        r'\log': 'log', r'\ln': 'ln', r'\exp': 'exp',
        r'\sin': 'sin', r'\cos': 'cos', r'\tan': 'tan',
        r'\sec': 'sec', r'\csc': 'csc', r'\cot': 'cot',
        r'\sinh': 'sinh', r'\cosh': 'cosh', r'\tanh': 'tanh',
        r'\arcsin': 'arcsin', r'\arccos': 'arccos', r'\arctan': 'arctan',
        r'\min': 'min', r'\max': 'max', r'\sup': 'sup', r'\inf': 'inf',
        r'\lim': 'lim', r'\limsup': 'lim sup', r'\liminf': 'lim inf',
        r'\det': 'det', r'\dim': 'dim', r'\ker': 'ker', r'\hom': 'hom',
        r'\arg': 'arg', r'\deg': 'deg', r'\gcd': 'gcd', r'\mod': 'mod',
    }

    result = latex

    # Apply replacements
    for pattern, replacement in {**greek, **symbols}.items():
        result = result.replace(pattern, replacement)

    # Handle subscripts (simple cases)
    subscript_map = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
                     '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
                     'i': 'ᵢ', 'j': 'ⱼ', 'n': 'ₙ', 'm': 'ₘ', 'k': 'ₖ',
                     'a': 'ₐ', 'e': 'ₑ', 'o': 'ₒ', 'x': 'ₓ', 'r': 'ᵣ',
                     'u': 'ᵤ', 'v': 'ᵥ', 'p': 'ₚ', 's': 'ₛ', 't': 'ₜ',
                     '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎'}

    superscript_map = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                       '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
                       'i': 'ⁱ', 'j': 'ʲ', 'n': 'ⁿ', 'm': 'ᵐ', 'k': 'ᵏ',
                       'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ', 'd': 'ᵈ', 'e': 'ᵉ',
                       'f': 'ᶠ', 'g': 'ᵍ', 'h': 'ʰ', 'l': 'ˡ', 'o': 'ᵒ',
                       'p': 'ᵖ', 'r': 'ʳ', 's': 'ˢ', 't': 'ᵗ', 'u': 'ᵘ',
                       'v': 'ᵛ', 'w': 'ʷ', 'x': 'ˣ', 'y': 'ʸ', 'z': 'ᶻ',
                       '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
                       'T': 'ᵀ', 'H': 'ᴴ'}

    def replace_script(match, script_map):
        content = match.group(1)
        converted = ''
        for char in content:
            if char in script_map:
                converted += script_map[char]
            else:
                converted += char
        return converted

    result = re.sub(r'_\{([^}]+)\}', lambda m: replace_script(m, subscript_map), result)
    result = re.sub(r'_([a-zA-Z0-9])', lambda m: replace_script(m, subscript_map), result)
    result = re.sub(r'\^\{([^}]+)\}', lambda m: replace_script(m, superscript_map), result)
    result = re.sub(r'\^([a-zA-Z0-9])', lambda m: replace_script(m, superscript_map), result)

    # Handle fractions as a/b
    result = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', result)

    # Handle sqrt
    result = re.sub(r'\\sqrt\{([^}]+)\}', r'√(\1)', result)
    result = re.sub(r'\\sqrt\[([^\]]+)\]\{([^}]+)\}', r'\1√(\2)', result)

    # Handle text commands
    result = re.sub(r'\\text\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\textrm\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mathit\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\textbf\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mathcal\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mathbb\{([^}]*)\}', r'\1', result)

    # Remove remaining LaTeX commands
    result = re.sub(r'\\[a-zA-Z]+', '', result)
    result = re.sub(r'[{}]', '', result)

    # Clean up whitespace
    result = re.sub(r'\s+', ' ', result)

    return result.strip()

def convert_latex_for_kindle(text, book):
    """Convert LaTeX expressions - display math to images, inline to Unicode."""
    if not text:
        return text

    def replace_display_math(match):
        latex = match.group(1)
        # Try to render as image
        img_path = render_latex_to_image(latex, book)
        if img_path:
            return f'<div class="math-block"><img src="{img_path}" alt="equation" class="math-img"/></div>'
        else:
            # Fallback to styled text
            unicode_text = latex_to_unicode_inline(latex)
            escaped = unicode_text.replace('<', '&lt;').replace('>', '&gt;')
            return f'<div class="math-block math-fallback">{escaped}</div>'

    def replace_inline_math(match):
        latex = match.group(1)
        unicode_text = latex_to_unicode_inline(latex)
        escaped = unicode_text.replace('<', '&lt;').replace('>', '&gt;')
        return f'<span class="math-inline">{escaped}</span>'

    # Replace \[...\] display math
    text = re.sub(r'\\\[(.+?)\\\]', replace_display_math, text, flags=re.DOTALL)

    # Replace $$...$$ display math
    text = re.sub(r'\$\$([^$]+)\$\$', replace_display_math, text, flags=re.DOTALL)

    # Replace \(...\) inline math
    text = re.sub(r'\\\((.+?)\\\)', replace_inline_math, text, flags=re.DOTALL)

    # Replace inline math ($...$)
    text = re.sub(r'(?<!\$)\$([^$]+)\$(?!\$)', replace_inline_math, text)

    return text

def convert_latex_in_soup(soup, book):
    """Convert all LaTeX math in the soup."""
    for element in list(soup.find_all(string=True)):
        if element.parent.name in ['script', 'style', 'code', 'pre']:
            continue

        text = str(element)
        if '$' in text or '\\[' in text or '\\(' in text:
            new_text = convert_latex_for_kindle(text, book)
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

        # Skip math images (already added to book)
        if src.startswith('images/math_'):
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

def extract_content(html, page_slug, all_page_slugs, book):
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
    page_soup = convert_latex_in_soup(page_soup, book)
    page_soup = fix_internal_links(page_soup, page_slug, all_page_slugs)

    return page_soup

# CSS for EPUB - KINDLE VERSION
EPUB_CSS_KINDLE = '''
body {
    font-family: Georgia, "Times New Roman", serif;
    line-height: 1.5;
    margin: 0.5em;
    padding: 0;
    color: #333;
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
    font-size: 1.3em;
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
    font-size: 0.8em;
    border: 1px solid #ddd;
    white-space: pre-wrap;
    word-wrap: break-word;
}

code {
    padding: 0.1em 0.2em;
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
    font-size: 0.85em;
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

/* Math styling for Kindle */
.math-inline {
    font-family: serif;
    font-style: italic;
}

.math-block {
    text-align: center;
    margin: 1em 0;
}

.math-img {
    max-width: 100%;
    height: auto;
    display: inline-block;
}

.math-fallback {
    font-family: serif;
    font-style: italic;
    padding: 0.5em;
    background: #f9f9f9;
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
}

.iframe-placeholder {
    background: #f5f5f5;
    border: 1px solid #ddd;
    padding: 0.8em;
    text-align: center;
    color: #666;
    font-style: italic;
}

.next-section {
    margin-top: 1.5em;
    padding-top: 0.8em;
    border-top: 1px solid #eee;
}
'''

def main():
    print("=" * 60)
    print("Scaling Book Website to EPUB Converter (KINDLE VERSION)")
    print("Display math rendered as images using real LaTeX")
    print("=" * 60)

    # Check for LaTeX
    try:
        result = subprocess.run(['latex', '--version'], capture_output=True, timeout=5)
        print("✓ LaTeX available for high-quality math rendering")
    except:
        print("✗ LaTeX not found - math images will use fallback")

    # Create EPUB book
    book = epub.EpubBook()
    book.set_identifier('scaling-book-kindle-2025')
    book.set_title('How to Scale Your Model: A Systems View of LLMs on TPUs')
    book.set_language('en')
    book.add_author('Jacob Austin et al.')

    # Add metadata
    book.add_metadata('DC', 'publisher', 'Google DeepMind')
    book.add_metadata('DC', 'description',
        'Training LLMs often feels like alchemy, but understanding and optimizing '
        'the performance of your models doesn\'t have to. This book demystifies the '
        'science of scaling language models. (Kindle-optimized with math as images)')

    all_page_slugs = [slug for slug, _ in PAGES]
    chapters = []

    # Add CSS
    css_item = epub.EpubItem(
        uid="style",
        file_name="style/main.css",
        media_type="text/css",
        content=EPUB_CSS_KINDLE
    )
    book.add_item(css_item)

    # Fetch and process all pages
    for page_slug, title in PAGES:
        try:
            html = fetch_page(page_slug)
            soup = extract_content(html, page_slug, all_page_slugs, book)

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

            # Set content
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

            print(f"  ✓ Processed: {title} ({len(images)} images, {math_image_counter[0]} math images total)")

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
    epub_path = "/home/user/scaling-book/scaling-book-kindle.epub"
    print(f"\nWriting EPUB to: {epub_path}")
    epub.write_epub(epub_path, book, {})

    # Get file size
    size_mb = os.path.getsize(epub_path) / (1024 * 1024)
    print(f"\n✓ Success! EPUB saved to: {epub_path}")
    print(f"  File size: {size_mb:.2f} MB")
    print(f"  Chapters: {len(chapters)}")
    print(f"  Math images rendered: {math_stats['success']}")
    print(f"  Math fallbacks (text): {math_stats['fallback']}")

if __name__ == "__main__":
    main()
