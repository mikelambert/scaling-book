#!/usr/bin/env python3
"""
Script to convert all chapters of the scaling book to a nicely formatted PDF.
"""

import os
import re
import yaml
import markdown
from weasyprint import HTML, CSS
from pygments.formatters import HtmlFormatter
from latex2mathml.converter import convert as latex_to_mathml

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

def process_latex_math(content):
    """Convert LaTeX math to MathML for proper rendering."""

    def convert_display_math(match):
        """Convert display math $$...$$ to MathML."""
        latex = match.group(1).strip()
        try:
            mathml = latex_to_mathml(latex, display="block")
            return f'<div class="math-display">{mathml}</div>'
        except Exception as e:
            # If conversion fails, return escaped LaTeX
            return f'<div class="math-display math-fallback">[{latex}]</div>'

    def convert_inline_math(match):
        """Convert inline math $...$ to MathML."""
        latex = match.group(1).strip()
        try:
            mathml = latex_to_mathml(latex, display="inline")
            return f'<span class="math-inline">{mathml}</span>'
        except Exception as e:
            # If conversion fails, return escaped LaTeX
            return f'<span class="math-inline math-fallback">[{latex}]</span>'

    # First, protect code blocks from math processing
    code_blocks = []
    def save_code_block(match):
        code_blocks.append(match.group(0))
        return f'<<<CODE_BLOCK_{len(code_blocks) - 1}>>>'

    # Save fenced code blocks
    content = re.sub(r'```[\s\S]*?```', save_code_block, content)
    # Save inline code
    content = re.sub(r'`[^`]+`', save_code_block, content)

    # Process display math first ($$...$$) - be careful with newlines
    content = re.sub(r'\$\$([^$]+?)\$\$', convert_display_math, content, flags=re.DOTALL)

    # Process inline math ($...$) - but not $$ which we already handled
    # Match $ followed by non-$ content, ending with $ not followed by $
    content = re.sub(r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)', convert_inline_math, content)

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        content = content.replace(f'<<<CODE_BLOCK_{i}>>>', block)

    return content

def process_liquid_tags(content):
    """Convert Jekyll liquid tags to HTML."""
    # Handle figure.liquid includes
    def replace_figure(match):
        attrs = match.group(1)
        path_match = re.search(r'path\s*=\s*"([^"]+)"', attrs)
        class_match = re.search(r'class\s*=\s*"([^"]+)"', attrs)
        caption_match = re.search(r'caption\s*=\s*"([^"]+)"', attrs)

        if path_match:
            path = path_match.group(1)
            full_path = os.path.join(BASE_DIR, path)
            img_class = class_match.group(1) if class_match else "img-fluid"
            caption = caption_match.group(1) if caption_match else ""

            html = f'<figure class="figure {img_class}">'
            html += f'<img src="file://{full_path}" class="{img_class}" style="max-width: 100%; height: auto;">'
            if caption:
                html += f'<figcaption class="figure-caption">{caption}</figcaption>'
            html += '</figure>'
            return html
        return match.group(0)

    content = re.sub(r'\{%\s*include\s+figure\.liquid\s+([^%]+)\s*%\}', replace_figure, content)

    # Remove other liquid tags we can't process
    content = re.sub(r'\{%[^%]+%\}', '', content)
    content = re.sub(r'\{\{[^}]+\}\}', '', content)

    return content

def process_distill_elements(content):
    """Convert Distill-style elements to standard HTML."""
    # Convert d-footnote to numbered footnotes
    footnote_counter = [0]
    footnotes = []

    def replace_footnote(match):
        footnote_counter[0] += 1
        num = footnote_counter[0]
        footnote_text = match.group(1)
        footnotes.append((num, footnote_text))
        return f'<sup class="footnote-ref"><a href="#fn{num}" id="fnref{num}">[{num}]</a></sup>'

    content = re.sub(r'<d-footnote>(.*?)</d-footnote>', replace_footnote, content, flags=re.DOTALL)

    # Store footnotes for later
    content = (content, footnotes)

    return content

def process_citations(content):
    """Handle d-cite elements."""
    # Simple replacement - just show as reference
    content = re.sub(r'<d-cite\s+key="([^"]+)"[^>]*></d-cite>', r'[\1]', content)
    return content

def convert_markdown_to_html(md_content, metadata):
    """Convert markdown to HTML with extensions."""
    # Process liquid tags first
    md_content = process_liquid_tags(md_content)

    # Process LaTeX math before markdown conversion
    md_content = process_latex_math(md_content)

    # Process citations
    md_content = process_citations(md_content)

    # Process distill elements (returns tuple with footnotes)
    md_content, footnotes = process_distill_elements(md_content)

    # Configure markdown extensions
    extensions = [
        'markdown.extensions.fenced_code',
        'markdown.extensions.codehilite',
        'markdown.extensions.tables',
        'markdown.extensions.toc',
        'markdown.extensions.nl2br',
        'markdown.extensions.sane_lists',
        'markdown.extensions.smarty',
    ]

    extension_configs = {
        'codehilite': {
            'css_class': 'highlight',
            'guess_lang': False,
        }
    }

    md = markdown.Markdown(extensions=extensions, extension_configs=extension_configs)
    html = md.convert(md_content)

    # Add footnotes section if any
    if footnotes:
        html += '\n<hr class="footnotes-sep">\n<section class="footnotes">\n<ol class="footnotes-list">\n'
        for num, text in footnotes:
            html += f'<li id="fn{num}" class="footnote-item"><p>{text} <a href="#fnref{num}" class="footnote-backref">↩</a></p></li>\n'
        html += '</ol>\n</section>\n'

    return html

def get_css():
    """Generate CSS for the PDF."""
    # Get pygments CSS for code highlighting
    pygments_css = HtmlFormatter(style='default').get_style_defs('.highlight')

    return f'''
@page {{
    size: A4;
    margin: 2cm 2.5cm;
    @top-center {{
        content: "How to Scale Your Model";
        font-size: 10pt;
        color: #666;
    }}
    @bottom-center {{
        content: counter(page);
        font-size: 10pt;
    }}
}}

@page :first {{
    @top-center {{ content: none; }}
}}

* {{
    box-sizing: border-box;
}}

body {{
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
    max-width: 100%;
}}

h1 {{
    font-size: 24pt;
    font-weight: bold;
    color: #1a1a1a;
    margin-top: 2em;
    margin-bottom: 0.5em;
    page-break-after: avoid;
    border-bottom: 2px solid #333;
    padding-bottom: 0.3em;
}}

h2 {{
    font-size: 18pt;
    font-weight: bold;
    color: #2a2a2a;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    page-break-after: avoid;
}}

h3 {{
    font-size: 14pt;
    font-weight: bold;
    color: #3a3a3a;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
    page-break-after: avoid;
}}

h4, h5, h6 {{
    font-size: 12pt;
    font-weight: bold;
    color: #4a4a4a;
    margin-top: 1em;
    margin-bottom: 0.3em;
}}

p {{
    margin-bottom: 0.8em;
    text-align: justify;
    hyphens: auto;
}}

a {{
    color: #0066cc;
    text-decoration: none;
}}

a:hover {{
    text-decoration: underline;
}}

code {{
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 9pt;
    background-color: #f5f5f5;
    padding: 0.15em 0.3em;
    border-radius: 3px;
    border: 1px solid #ddd;
}}

pre {{
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 9pt;
    background-color: #f8f8f8;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 1em;
    overflow-x: auto;
    line-height: 1.4;
    page-break-inside: avoid;
}}

pre code {{
    background: none;
    border: none;
    padding: 0;
}}

blockquote {{
    border-left: 4px solid #0066cc;
    margin: 1em 0;
    padding: 0.5em 1em;
    background-color: #f9f9f9;
    font-style: italic;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 10pt;
    page-break-inside: avoid;
}}

th, td {{
    border: 1px solid #ddd;
    padding: 0.5em;
    text-align: left;
}}

th {{
    background-color: #f0f0f0;
    font-weight: bold;
}}

tr:nth-child(even) {{
    background-color: #fafafa;
}}

figure {{
    margin: 1.5em 0;
    text-align: center;
    page-break-inside: avoid;
}}

figure img {{
    max-width: 100%;
    height: auto;
}}

.img-small img {{
    max-width: 60%;
}}

figcaption, .figure-caption {{
    font-size: 10pt;
    color: #666;
    margin-top: 0.5em;
    font-style: italic;
}}

ul, ol {{
    margin: 0.8em 0;
    padding-left: 2em;
}}

li {{
    margin-bottom: 0.3em;
}}

.chapter-header {{
    page-break-before: always;
    margin-bottom: 2em;
}}

.chapter-header:first-of-type {{
    page-break-before: avoid;
}}

.chapter-title {{
    font-size: 28pt;
    color: #1a1a1a;
    border-bottom: 3px solid #0066cc;
    padding-bottom: 0.3em;
    margin-bottom: 0.3em;
}}

.chapter-subtitle {{
    font-size: 14pt;
    color: #666;
    font-style: italic;
    margin-bottom: 1em;
}}

.chapter-description {{
    font-size: 11pt;
    color: #444;
    background-color: #f5f9ff;
    padding: 1em;
    border-radius: 5px;
    border-left: 4px solid #0066cc;
    margin-bottom: 1.5em;
}}

.footnote-ref {{
    font-size: 8pt;
    vertical-align: super;
}}

.footnotes {{
    font-size: 9pt;
    color: #555;
    margin-top: 2em;
    padding-top: 1em;
    border-top: 1px solid #ddd;
}}

.footnotes-list {{
    padding-left: 1.5em;
}}

.footnote-item {{
    margin-bottom: 0.5em;
}}

.footnote-backref {{
    font-size: 8pt;
}}

.footnotes-sep {{
    margin-top: 2em;
}}

.announce {{
    background-color: #e8f4e8;
    padding: 1em;
    border-radius: 5px;
    border-left: 4px solid #28a745;
    margin: 1em 0;
}}

.next-section {{
    background-color: #f0f8ff;
    padding: 1em;
    border-radius: 5px;
    text-align: center;
    margin: 2em 0;
}}

/* Title page */
.title-page {{
    page-break-after: always;
    text-align: center;
    padding-top: 20%;
}}

.title-page h1 {{
    font-size: 36pt;
    border: none;
    margin-bottom: 0.3em;
}}

.title-page .subtitle {{
    font-size: 18pt;
    color: #666;
    margin-bottom: 2em;
}}

.title-page .authors {{
    font-size: 12pt;
    color: #444;
    margin-top: 3em;
}}

.title-page .author {{
    margin: 0.3em 0;
}}

/* TOC */
.toc {{
    page-break-after: always;
}}

.toc h2 {{
    text-align: center;
    border: none;
}}

.toc ul {{
    list-style: none;
    padding-left: 0;
}}

.toc li {{
    margin: 0.5em 0;
    padding-left: 1em;
}}

.toc a {{
    color: #333;
}}

/* Math styling */
.math-display {{
    text-align: center;
    margin: 1em 0;
    overflow-x: auto;
}}

.math-inline {{
    display: inline;
}}

.math-fallback {{
    font-family: 'Consolas', 'Monaco', monospace;
    color: #666;
}}

math {{
    font-size: 1.1em;
}}

/* MathML specific styling */
mfrac {{
    vertical-align: middle;
}}

msub, msup, msubsup {{
    font-size: 0.8em;
}}

{pygments_css}
'''

def generate_title_page(first_chapter_metadata):
    """Generate a title page HTML."""
    title = first_chapter_metadata.get('title', 'How to Scale Your Model')
    subtitle = first_chapter_metadata.get('subtitle', '')
    authors = first_chapter_metadata.get('authors', [])

    html = '<div class="title-page">\n'
    html += f'<h1>{title}</h1>\n'
    if subtitle:
        html += f'<p class="subtitle">{subtitle}</p>\n'

    if authors:
        html += '<div class="authors">\n'
        for author in authors:
            name = author.get('name', '')
            affiliation = author.get('affiliations', {}).get('name', '')
            html += f'<p class="author">{name}'
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
            ("Chapter 3: Sharded Matrices and How to Multiply Them", "sharding"),
        ]),
        ("Part 2: Transformers", [
            ("Chapter 4: All the Transformer Math You Need to Know", "transformers"),
            ("Chapter 5: How to Parallelize a Transformer for Training", "training"),
            ("Chapter 6: Training LLaMA 3 on TPUs", "applied-training"),
            ("Chapter 7: All About Transformer Inference", "inference"),
            ("Chapter 8: Serving LLaMA 3 on TPUs", "applied-inference"),
        ]),
        ("Part 3: Practical Tutorials", [
            ("Chapter 9: How to Profile TPU Code", "profiling"),
            ("Chapter 10: Programming TPUs in JAX", "jax-stuff"),
        ]),
        ("Part 4: Conclusions and Bonus", [
            ("Chapter 11: Conclusions and Further Reading", "conclusion"),
            ("Chapter 12: How to Think About GPUs", "gpus"),
        ]),
    ]

    for part_title, chapters in parts:
        html += f'<li><strong>{part_title}</strong>\n<ul>\n'
        for chapter_title, _ in chapters:
            html += f'<li>{chapter_title}</li>\n'
        html += '</ul></li>\n'

    html += '</ul>\n</div>\n'
    return html

def main():
    print("Converting chapters to PDF...")

    all_html = []
    first_metadata = None

    # Sort chapters by section number
    sorted_chapters = sorted(CHAPTERS, key=lambda x: x[1])

    for i, (filename, section_num) in enumerate(sorted_chapters):
        filepath = os.path.join(BASE_DIR, filename)
        print(f"  Processing {filename}...")

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        metadata, body = parse_frontmatter(content)

        if i == 0:
            first_metadata = metadata

        # Convert to HTML
        chapter_html = convert_markdown_to_html(body, metadata)

        # Add chapter header
        title = metadata.get('title', f'Chapter {section_num}')
        description = metadata.get('description', '')

        header = f'<div class="chapter-header">\n'
        header += f'<h1 class="chapter-title">{title}</h1>\n'
        if description and section_num > 0:  # Skip description for intro as it's quite long
            header += f'<div class="chapter-description">{description}</div>\n'
        header += '</div>\n'

        all_html.append(header + chapter_html)

    # Build complete HTML document
    title_page = generate_title_page(first_metadata)
    toc = generate_toc()

    full_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>How to Scale Your Model</title>
</head>
<body>
{title_page}
{toc}
{''.join(all_html)}
</body>
</html>
'''

    # Save HTML for debugging
    html_path = os.path.join(BASE_DIR, 'scaling-book.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"  Saved HTML to {html_path}")

    # Generate PDF
    print("  Generating PDF...")
    pdf_path = os.path.join(BASE_DIR, 'scaling-book.pdf')
    css = CSS(string=get_css())
    HTML(string=full_html, base_url=BASE_DIR).write_pdf(pdf_path, stylesheets=[css])
    print(f"  Saved PDF to {pdf_path}")

    print("\nDone! Generated scaling-book.pdf")

if __name__ == '__main__':
    main()
