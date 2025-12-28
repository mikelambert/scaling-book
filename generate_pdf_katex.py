#!/usr/bin/env python3
"""
Generate PDF using KaTeX for math rendering + Playwright for PDF generation.
This approach properly renders LaTeX math by using a real browser engine.
"""

import os
import re
import yaml
import subprocess
import asyncio

# Chapter files in order (section_number)
CHAPTERS = [
    ("index.md", 0),
    ("roofline.md", 1),
    ("tpus.md", 2),
    ("sharding.md", 3),
    ("transformers.md", 4),
    ("training.md", 5),
    ("applied-training.md", 6),
    ("inference.md", 7),
    ("applied-inference.md", 8),
    ("profiling.md", 9),
    ("jax-stuff.md", 10),
    ("conclusion.md", 11),
    ("gpus.md", 12),
]

# Map from various link formats to chapter IDs
CHAPTER_LINK_MAP = {
    'index': 'chapter-index',
    'roofline': 'chapter-roofline',
    'tpus': 'chapter-tpus',
    'sharding': 'chapter-sharding',
    'transformers': 'chapter-transformers',
    'training': 'chapter-training',
    'applied-training': 'chapter-applied-training',
    'inference': 'chapter-inference',
    'applied-inference': 'chapter-applied-inference',
    'profiling': 'chapter-profiling',
    'jax-stuff': 'chapter-jax-stuff',
    'conclusion': 'chapter-conclusion',
    'gpus': 'chapter-gpus',
    '../roofline': 'chapter-roofline',
    '../tpus': 'chapter-tpus',
    '../sharding': 'chapter-sharding',
    '../transformers': 'chapter-transformers',
    '../training': 'chapter-training',
    '../applied-training': 'chapter-applied-training',
    '../inference': 'chapter-inference',
    '../applied-inference': 'chapter-applied-inference',
    '../profiling': 'chapter-profiling',
    '../jax-stuff': 'chapter-jax-stuff',
    '../conclusion': 'chapter-conclusion',
    '../gpus': 'chapter-gpus',
    '..': 'chapter-index',
    '.': 'chapter-index',
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_frontmatter(content):
    """Parse YAML frontmatter from markdown content."""
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            try:
                metadata = yaml.safe_load(parts[1])
                body = parts[2]
                return metadata, body
            except yaml.YAMLError:
                pass
    return {}, content


def process_liquid_tags(content):
    """Convert Jekyll liquid tags to HTML."""
    def replace_figure(match):
        attrs = match.group(1)
        path_match = re.search(r'path\s*=\s*"([^"]+)"', attrs)
        class_match = re.search(r'class\s*=\s*"([^"]+)"', attrs)
        caption_match = re.search(r'caption\s*=\s*"([^"]+)"', attrs)

        if path_match:
            path = path_match.group(1)
            # Use relative path for HTTP serving
            img_class = class_match.group(1) if class_match else "img-fluid"
            caption = caption_match.group(1) if caption_match else ""

            html = f'<figure class="figure {img_class}">'
            html += f'<img src="{path}" class="{img_class}">'
            if caption:
                html += f'<figcaption>{caption}</figcaption>'
            html += '</figure>'
            return html
        return match.group(0)

    content = re.sub(r'\{%\s*include\s+figure\.liquid\s+([^%]+)\s*%\}', replace_figure, content)
    content = re.sub(r'\{%[^%]+%\}', '', content)
    content = re.sub(r'\{\{[^}]+\}\}', '', content)
    return content


def process_latex_for_katex(content):
    """Convert LaTeX delimiters for KaTeX auto-render."""
    # KaTeX auto-render expects $...$ for inline and $$...$$ for display
    # But we need to handle \begin{equation} etc inside $$

    # Remove equation wrappers inside $$ blocks
    content = re.sub(r'\$\$\s*\\begin\{equation\*?\}', '$$', content)
    content = re.sub(r'\\end\{equation\*?\}\s*\$\$', '$$', content)

    # Same for align environments - convert to aligned inside $$
    def fix_align(match):
        inner = match.group(1)
        return f'$$\\begin{{aligned}}{inner}\\end{{aligned}}$$'

    content = re.sub(r'\$\$\s*\\begin\{align\*?\}([\s\S]*?)\\end\{align\*?\}\s*\$\$', fix_align, content)

    return content


def process_internal_links(content):
    """Convert internal markdown links to PDF anchor links."""
    def replace_link(match):
        text = match.group(1)
        url = match.group(2)

        if url.startswith(('http://', 'https://', 'mailto:')):
            return match.group(0)

        if '#' in url:
            base_url, anchor = url.split('#', 1)
        else:
            base_url = url
            anchor = None

        chapter_id = CHAPTER_LINK_MAP.get(base_url)

        if chapter_id:
            if anchor:
                return f'[{text}](#{chapter_id}-{anchor})'
            else:
                return f'[{text}](#{chapter_id})'
        else:
            if anchor:
                return f'[{text}](#{anchor})'
            return match.group(0)

    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_link, content)
    return content


def process_distill_elements(content):
    """Convert Distill-style elements to standard HTML."""
    footnote_counter = [0]
    footnotes = []

    def replace_footnote(match):
        footnote_counter[0] += 1
        num = footnote_counter[0]
        footnote_text = match.group(1)
        footnotes.append((num, footnote_text))
        return f'<sup class="footnote-ref"><a href="#fn{num}" id="fnref{num}">[{num}]</a></sup>'

    content = re.sub(r'<d-footnote>(.*?)</d-footnote>', replace_footnote, content, flags=re.DOTALL)
    return content, footnotes


def process_citations(content):
    """Handle d-cite elements."""
    content = re.sub(r'<d-cite\s+key="([^"]+)"[^>]*></d-cite>', r'[\1]', content)
    return content


def convert_markdown_to_html(md_content):
    """Convert markdown to HTML using Python markdown library."""
    import markdown

    extensions = [
        'markdown.extensions.fenced_code',
        'markdown.extensions.codehilite',
        'markdown.extensions.tables',
        'markdown.extensions.toc',
        'markdown.extensions.sane_lists',
    ]

    extension_configs = {
        'codehilite': {
            'css_class': 'highlight',
            'guess_lang': False,
        }
    }

    md = markdown.Markdown(extensions=extensions, extension_configs=extension_configs)
    return md.convert(md_content)


def get_html_template(content):
    """Return the complete HTML with KaTeX embedded locally."""
    # Read local KaTeX files
    katex_dir = os.path.join(BASE_DIR, 'katex_assets')

    with open(os.path.join(katex_dir, 'katex.min.css'), 'r') as f:
        katex_css = f.read()
    with open(os.path.join(katex_dir, 'katex.min.js'), 'r') as f:
        katex_js = f.read()
    with open(os.path.join(katex_dir, 'auto-render.min.js'), 'r') as f:
        autorender_js = f.read()

    # Using string concatenation to avoid format() issues with JS curly braces
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>How to Scale Your Model</title>
    <style>
''' + katex_css + '''
    </style>
    <script>
''' + katex_js + '''
    </script>
    <script>
''' + autorender_js + '''
    </script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            renderMathInElement(document.body, {
                delimiters: [
                    {left: "$$", right: "$$", display: true},
                    {left: "$", right: "$", display: false}
                ],
                throwOnError: false
            });
        });
    </script>
    <style>
        @page {
            size: A4;
            margin: 2cm 2.5cm;
        }

        body {
            font-family: 'Georgia', 'Times New Roman', serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
            max-width: 100%;
        }

        h1 {
            font-size: 24pt;
            font-weight: bold;
            color: #1a1a1a;
            margin-top: 2em;
            margin-bottom: 0.5em;
            page-break-after: avoid;
            border-bottom: 2px solid #333;
            padding-bottom: 0.3em;
        }

        h2 {
            font-size: 18pt;
            font-weight: bold;
            color: #2a2a2a;
            margin-top: 1.5em;
            page-break-after: avoid;
        }

        h3 {
            font-size: 14pt;
            font-weight: bold;
            color: #3a3a3a;
            margin-top: 1.2em;
            page-break-after: avoid;
        }

        p {
            margin-bottom: 0.8em;
            text-align: justify;
        }

        a {
            color: #0066cc;
            text-decoration: none;
        }

        code {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 9pt;
            background-color: #f5f5f5;
            padding: 0.15em 0.3em;
            border-radius: 3px;
            border: 1px solid #ddd;
        }

        pre {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 9pt;
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 1em;
            overflow-x: auto;
            line-height: 1.4;
            page-break-inside: avoid;
        }

        pre code {
            background: none;
            border: none;
            padding: 0;
        }

        blockquote {
            border-left: 4px solid #0066cc;
            margin: 1em 0;
            padding: 0.5em 1em;
            background-color: #f9f9f9;
            font-style: italic;
        }

        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
            font-size: 10pt;
            page-break-inside: avoid;
        }

        th, td {
            border: 1px solid #ddd;
            padding: 0.5em;
            text-align: left;
        }

        th {
            background-color: #f0f0f0;
            font-weight: bold;
        }

        figure {
            margin: 1.5em 0;
            text-align: center;
            page-break-inside: avoid;
        }

        figure img {
            max-width: 100%;
            height: auto;
        }

        figcaption {
            font-size: 10pt;
            color: #666;
            margin-top: 0.5em;
            font-style: italic;
        }

        .chapter-header {
            page-break-before: always;
            margin-bottom: 2em;
        }

        .chapter-header:first-of-type {
            page-break-before: avoid;
        }

        .chapter-title {
            font-size: 28pt;
            color: #1a1a1a;
            border-bottom: 3px solid #0066cc;
            padding-bottom: 0.3em;
            margin-bottom: 0.3em;
        }

        .chapter-description {
            font-size: 11pt;
            color: #444;
            background-color: #f5f9ff;
            padding: 1em;
            border-radius: 5px;
            border-left: 4px solid #0066cc;
            margin-bottom: 1.5em;
        }

        .title-page {
            page-break-after: always;
            text-align: center;
            padding-top: 20%;
        }

        .title-page h1 {
            font-size: 36pt;
            border: none;
        }

        .title-page .subtitle {
            font-size: 18pt;
            color: #666;
            margin-bottom: 2em;
        }

        .title-page .authors {
            font-size: 12pt;
            color: #444;
            margin-top: 3em;
        }

        .toc {
            page-break-after: always;
        }

        .toc h2 {
            text-align: center;
            border: none;
        }

        .toc ul {
            list-style: none;
            padding-left: 0;
        }

        .toc li {
            margin: 0.5em 0;
            padding-left: 1em;
        }

        .footnote-ref {
            font-size: 8pt;
            vertical-align: super;
        }

        .footnotes {
            font-size: 9pt;
            color: #555;
            margin-top: 2em;
            padding-top: 1em;
            border-top: 1px solid #ddd;
        }

        /* KaTeX display math centering */
        .katex-display {
            margin: 1em 0;
            text-align: center;
        }
    </style>
</head>
<body>
''' + content + '''
</body>
</html>
'''


def generate_title_page(metadata):
    """Generate a title page HTML."""
    title = metadata.get('title', 'How to Scale Your Model')
    subtitle = metadata.get('subtitle', '')
    authors = metadata.get('authors', [])

    html = '<div class="title-page">\n'
    html += f'<h1>{title}</h1>\n'
    if subtitle:
        html += f'<p class="subtitle">{subtitle}</p>\n'

    if authors:
        html += '<div class="authors">\n'
        for author in authors:
            name = author.get('name', '')
            affiliation = author.get('affiliations', {}).get('name', '') if isinstance(author.get('affiliations'), dict) else ''
            html += f'<p>{name}'
            if affiliation:
                html += f' <em>({affiliation})</em>'
            html += '</p>\n'
        html += '</div>\n'

    html += '</div>\n'
    return html


def generate_toc():
    """Generate table of contents."""
    html = '<div class="toc">\n'
    html += '<h2>Table of Contents</h2>\n'
    html += '<ul>\n'

    parts = [
        ("Part 1: Preliminaries", [
            ("Chapter 1: All About Rooflines", "roofline"),
            ("Chapter 2: How to Think About TPUs", "tpus"),
            ("Chapter 3: Sharded Matrices", "sharding"),
        ]),
        ("Part 2: Transformers", [
            ("Chapter 4: Transformer Math", "transformers"),
            ("Chapter 5: Training Parallelism", "training"),
            ("Chapter 6: Training LLaMA 3", "applied-training"),
            ("Chapter 7: Inference", "inference"),
            ("Chapter 8: Serving LLaMA 3", "applied-inference"),
        ]),
        ("Part 3: Practical", [
            ("Chapter 9: Profiling", "profiling"),
            ("Chapter 10: JAX Programming", "jax-stuff"),
        ]),
        ("Part 4: Conclusions", [
            ("Chapter 11: Further Reading", "conclusion"),
            ("Chapter 12: GPUs", "gpus"),
        ]),
    ]

    for part_title, chapters in parts:
        html += f'<li><strong>{part_title}</strong>\n<ul>\n'
        for chapter_title, chapter_slug in chapters:
            html += f'<li><a href="#chapter-{chapter_slug}">{chapter_title}</a></li>\n'
        html += '</ul></li>\n'

    html += '</ul>\n</div>\n'
    return html


async def generate_pdf_with_playwright(html_path, pdf_path):
    """Use Playwright to generate PDF from HTML via local HTTP server."""
    from playwright.async_api import async_playwright
    import http.server
    import socketserver
    import threading

    # Start a simple HTTP server in the background
    html_dir = os.path.dirname(html_path)
    html_filename = os.path.basename(html_path)
    port = 8765

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=html_dir, **kwargs)
        def log_message(self, format, *args):
            pass  # Suppress logging

    httpd = socketserver.TCPServer(("", port), QuietHandler)
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    print(f"  Started HTTP server on port {port}")

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()

            # Enable console logging to debug
            page.on("console", lambda msg: print(f"  [Browser] {msg.text}"))

            # Load the HTML file via HTTP
            print("  Loading HTML via HTTP...")
            await page.goto(f'http://localhost:{port}/{html_filename}', wait_until='networkidle', timeout=60000)

            # Wait for KaTeX to load and render
            print("  Waiting for KaTeX...")
            await page.wait_for_timeout(3000)

            # Try to wait for KaTeX elements to appear
            try:
                await page.wait_for_selector('.katex', timeout=10000)
                katex_count = await page.locator('.katex').count()
                print(f"  KaTeX elements found: {katex_count}")
            except:
                print("  Warning: No .katex elements found - math may not have rendered")

            # Generate PDF
            print("  Generating PDF...")
            await page.pdf(
                path=pdf_path,
                format='A4',
                margin={'top': '2cm', 'bottom': '2cm', 'left': '2.5cm', 'right': '2.5cm'},
                print_background=True
            )

            await browser.close()
    finally:
        httpd.shutdown()


def main():
    print("Generating PDF using KaTeX + Playwright...")

    all_html = []
    first_metadata = None

    sorted_chapters = sorted(CHAPTERS, key=lambda x: x[1])

    for i, (filename, section_num) in enumerate(sorted_chapters):
        filepath = os.path.join(BASE_DIR, filename)
        print(f"  Processing {filename}...")

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        metadata, body = parse_frontmatter(content)

        if i == 0:
            first_metadata = metadata

        # Process content
        body = process_liquid_tags(body)
        body = process_internal_links(body)
        body = process_latex_for_katex(body)
        body = process_citations(body)
        body, footnotes = process_distill_elements(body)

        # Convert markdown to HTML
        chapter_html = convert_markdown_to_html(body)

        # Add footnotes if any
        if footnotes:
            chapter_html += '\n<hr class="footnotes-sep">\n<section class="footnotes">\n<ol>\n'
            for num, text in footnotes:
                chapter_html += f'<li id="fn{num}">{text} <a href="#fnref{num}">↩</a></li>\n'
            chapter_html += '</ol>\n</section>\n'

        # Add chapter header
        title = metadata.get('title', f'Chapter {section_num}')
        description = metadata.get('description', '')
        chapter_id = 'chapter-' + filename.replace('.md', '')

        header = f'<div class="chapter-header" id="{chapter_id}">\n'
        header += f'<h1 class="chapter-title">{title}</h1>\n'
        if description and section_num > 0:
            header += f'<div class="chapter-description">{description}</div>\n'
        header += '</div>\n'

        all_html.append(header + chapter_html)

    # Build complete HTML
    title_page = generate_title_page(first_metadata)
    toc = generate_toc()

    full_content = title_page + toc + ''.join(all_html)
    full_html = get_html_template(full_content)

    # Save HTML
    html_path = os.path.join(BASE_DIR, 'scaling-book-katex.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"  Saved HTML to {html_path}")

    # Generate PDF using Playwright
    print("  Generating PDF with Playwright...")
    pdf_path = os.path.join(BASE_DIR, 'scaling-book-katex.pdf')

    asyncio.run(generate_pdf_with_playwright(html_path, pdf_path))

    print(f"  Saved PDF to {pdf_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
