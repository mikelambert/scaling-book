#!/usr/bin/env python3
"""
Build a PDF from the scaling book markdown files.
Handles Jekyll/Distill syntax conversion and internal link preservation.
"""

import re
import os
import subprocess
from pathlib import Path

# Chapter order (without .md extension)
CHAPTERS = [
    "index",
    "roofline",
    "tpus",
    "sharding",
    "transformers",
    "training",
    "applied-training",
    "inference",
    "applied-inference",
    "profiling",
    "jax-stuff",
    "conclusion",
    "gpus",
]

# Chapter titles for display
CHAPTER_TITLES = {
    "index": "Introduction: How to Scale Your Model",
    "roofline": "Part 1: A Brief Intro to Roofline Analysis",
    "tpus": "Part 2: How to Think About TPUs",
    "sharding": "Part 3: Sharded Matrices and How to Multiply Them",
    "transformers": "Part 4: All the Transformer Math You Need to Know",
    "training": "Part 5: How to Parallelize a Transformer for Training",
    "applied-training": "Part 6: Training LLaMA 3 on TPUs",
    "inference": "Part 7: All About Transformer Inference",
    "applied-inference": "Part 8: Serving LLaMA 3 on TPUs",
    "profiling": "Part 9: How to Profile TPU Code",
    "jax-stuff": "Part 10: Programming TPUs in JAX",
    "conclusion": "Part 11: Conclusions and Further Reading",
    "gpus": "Part 12: How to Think About GPUs",
}

BASE_DIR = Path(__file__).parent


def strip_yaml_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown."""
    if content.startswith("---"):
        # Find the closing ---
        end = content.find("---", 3)
        if end != -1:
            return content[end + 3:].lstrip()
    return content


def convert_figure_liquid(content: str, base_dir: Path) -> str:
    """
    Convert Jekyll figure.liquid includes to markdown images.
    {% include figure.liquid path="assets/img/foo.png" class="..." caption="..." %}
    ->
    ![caption](assets/img/foo.png)
    """
    pattern = r'{%\s*include\s+figure\.liquid\s+([^%]+)%}'

    def replace_figure(match):
        attrs_str = match.group(1)

        # Extract path
        path_match = re.search(r'path="([^"]+)"', attrs_str)
        path = path_match.group(1) if path_match else ""

        # Extract caption (may contain HTML)
        caption_match = re.search(r'caption="([^"]*(?:"[^"]*"[^"]*)*)"', attrs_str)
        if not caption_match:
            # Try simpler pattern
            caption_match = re.search(r"caption=['\"]([^'\"]+)['\"]", attrs_str)

        caption = ""
        if caption_match:
            caption = caption_match.group(1)
            # Strip HTML tags from caption for alt text
            caption = re.sub(r'<[^>]+>', '', caption)
            caption = caption.strip()

        # Make path absolute for pandoc
        full_path = base_dir / path
        if full_path.exists():
            return f"\n![{caption}]({path})\n"
        else:
            return f"\n![{caption}]({path})\n"

    return re.sub(pattern, replace_figure, content, flags=re.DOTALL)


def convert_internal_links(content: str, current_chapter: str) -> str:
    """
    Convert internal links to PDF internal references.
    [text](roofline) -> [text](#roofline)
    [text](../roofline) -> [text](#roofline)
    [text](conclusion#further-reading) -> [text](#conclusion-further-reading)
    """
    # Pattern for markdown links
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'

    def replace_link(match):
        text = match.group(1)
        url = match.group(2)

        # Skip external URLs
        if url.startswith(('http://', 'https://', 'mailto:', '#')):
            return match.group(0)

        # Skip image references and other non-chapter links
        if url.endswith(('.png', '.jpg', '.gif', '.svg', '.pdf')):
            return match.group(0)

        # Remove ../ prefix if present
        url = re.sub(r'^\.\./', '', url)

        # Handle anchor links (file#section)
        if '#' in url:
            file_part, anchor = url.split('#', 1)
            # If file_part is a chapter, convert to internal link
            if file_part in CHAPTERS:
                # Combine file and anchor with hyphen for unique ID
                return f'[{text}](#{file_part}-{anchor})'
            elif file_part == '':
                # Same-file anchor
                return f'[{text}](#{current_chapter}-{anchor})'
            return match.group(0)

        # Simple chapter link
        if url in CHAPTERS:
            return f'[{text}](#{url})'

        return match.group(0)

    return re.sub(link_pattern, replace_link, content)


def convert_distill_footnotes(content: str, chapter_id: str) -> str:
    """
    Convert <d-footnote>text</d-footnote> to pandoc footnotes.
    Uses chapter-prefixed IDs to avoid duplicates across chapters.
    """
    footnote_counter = [0]  # Use list for mutability in closure
    footnotes = []

    def extract_footnote(match):
        footnote_counter[0] += 1
        note_text = match.group(1).strip()
        # Clean up the footnote text
        note_text = note_text.replace('\n', ' ').strip()
        # Use chapter-prefixed ID to avoid duplicates
        footnote_id = f'{chapter_id}-fn{footnote_counter[0]}'
        footnotes.append((footnote_id, note_text))
        return f'[^{footnote_id}]'

    # Extract all footnotes
    content = re.sub(r'<d-footnote>(.*?)</d-footnote>', extract_footnote, content, flags=re.DOTALL)

    # Append footnote definitions at the end
    if footnotes:
        content += "\n\n"
        for fn_id, text in footnotes:
            content += f'[^{fn_id}]: {text}\n\n'

    return content


def convert_distill_citations(content: str) -> str:
    """
    Convert <d-cite key="..."></d-cite> to simple bracketed references.
    """
    return re.sub(r'<d-cite\s+key="([^"]+)"></d-cite>', r'[\1]', content)


def convert_html_elements(content: str) -> str:
    """
    Convert or remove various HTML elements that don't work well in PDF.
    """
    # Convert <p markdown=1 class="...">text</p> to just text
    content = re.sub(r'<p[^>]*markdown[^>]*>(.*?)</p>', r'\1', content, flags=re.DOTALL)

    # Convert <h3 markdown=1 class="...">text</h3> to ### text
    content = re.sub(r'<h3[^>]*markdown[^>]*>(.*?)</h3>', r'### \1', content, flags=re.DOTALL)

    # Convert <sup>*</sup> to ^*^
    content = re.sub(r'<sup>([^<]+)</sup>', r'^\1^', content)

    # Remove empty divs and other structural HTML
    content = re.sub(r'<div[^>]*>\s*</div>', '', content)

    # Convert {% details %} Jekyll blocks to blockquotes
    content = re.sub(r'{%\s*details\s+([^%]+)%}', r'\n> **\1**\n>\n', content)
    content = re.sub(r'{%\s*enddetails\s*%}', r'\n', content)

    return content


def fix_latex_commands(content: str) -> str:
    """
    Fix LaTeX commands that may cause issues.
    """
    # Fix typo: \lfoor -> \lfloor
    content = content.replace('\\lfoor', '\\lfloor')

    # Replace \textnormal with \text which is more portable
    content = content.replace('\\textnormal', '\\text')

    # Remove duplicate color definitions (they're in the header now)
    color_defs = [
        r'\def \red#1{\textcolor{red}{#1}}',
        r'\def \green#1{\textcolor{green}{#1}}',
        r'\def \blue#1{\textcolor{blue}{#1}}',
        r'\def \purple#1{\textcolor{purple}{#1}}',
        r'\def \orange#1{\textcolor{orange}{#1}}',
        r'\def \gray#1{\textcolor{gray}{#1}}',
    ]
    for color_def in color_defs:
        content = content.replace(color_def, '')

    # Convert Unicode math symbols to LaTeX commands when outside math mode
    # But be careful not to double-convert
    content = re.sub(r'(?<!\$)⊗(?!\$)', r'$\\otimes$', content)

    # Fix nested equation environments: $$\begin{align*} should just be \begin{align*}
    # because $$ is already display math in pandoc
    # Handle with optional whitespace/newlines
    content = re.sub(r'\$\$\s*\\begin\{align\*?\}', r'\\begin{align*}', content)
    content = re.sub(r'\\end\{align\*?\}\s*\$\$', r'\\end{align*}', content)

    content = re.sub(r'\$\$\s*\\begin\{equation\*?\}', r'\\begin{equation*}', content)
    content = re.sub(r'\\end\{equation\*?\}\s*\$\$', r'\\end{equation*}', content)

    content = re.sub(r'\$\$\s*\\begin\{aligned\}', r'\\begin{aligned}', content)
    content = re.sub(r'\\end\{aligned\}\s*\$\$', r'\\end{aligned}', content)

    content = re.sub(r'\$\$\s*\\begin\{array\}', r'\\begin{array}', content)
    content = re.sub(r'\\end\{array\}\s*\$\$', r'\\end{array}', content)

    # Fix double-escaped braces in math mode: $\\{...\\}$ -> $\{...\}$
    content = re.sub(r'\$([^$]*?)\\\\{([^$]*?)\\\\}([^$]*?)\$', r'$\1\\{\2\\}\3$', content)

    return content


def fix_headers_for_chapter(content: str, chapter_id: str) -> str:
    """
    Add chapter ID prefix to all headers for unique anchor names.
    Also add explicit anchor IDs for pandoc.
    """
    lines = content.split('\n')
    result = []

    for line in lines:
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if header_match:
            hashes = header_match.group(1)
            title = header_match.group(2)

            # Generate anchor from title
            anchor = title.lower()
            anchor = re.sub(r'[^\w\s-]', '', anchor)
            anchor = re.sub(r'\s+', '-', anchor)
            anchor = f'{chapter_id}-{anchor}'

            # Add explicit anchor ID
            result.append(f'{hashes} {title} {{#{anchor}}}')
        else:
            result.append(line)

    return '\n'.join(result)


def process_chapter(chapter_id: str, base_dir: Path) -> str:
    """Process a single chapter file."""
    filepath = base_dir / f"{chapter_id}.md"

    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return ""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Processing pipeline
    content = strip_yaml_frontmatter(content)
    content = convert_figure_liquid(content, base_dir)
    content = convert_distill_footnotes(content, chapter_id)
    content = convert_distill_citations(content)
    content = convert_html_elements(content)
    content = fix_latex_commands(content)
    content = convert_internal_links(content, chapter_id)
    content = fix_headers_for_chapter(content, chapter_id)

    # Add chapter title and anchor
    title = CHAPTER_TITLES.get(chapter_id, chapter_id.replace('-', ' ').title())
    header = f'\n\n# {title} {{#{chapter_id}}}\n\n'

    return header + content


def build_combined_markdown(base_dir: Path) -> str:
    """Build a single combined markdown file from all chapters."""
    combined = """---
title: "How to Scale Your Model: A Systems View of LLMs on TPUs"
author: "Jacob Austin, Sholto Douglas, Roy Frostig, Anselm Levskaya, Charlie Chen, Sharad Vikram, Federico Lebron, Peter Choy, Vinay Ramasesh, Albert Webson, Reiner Pope"
documentclass: report
toc: true
toc-depth: 3
geometry: margin=1in
linkcolor: blue
urlcolor: blue
header-includes:
  - \\usepackage{bookmark}
  - \\usepackage{xcolor}
  - \\def\\red#1{\\textcolor{red}{#1}}
  - \\def\\green#1{\\textcolor{green}{#1}}
  - \\def\\blue#1{\\textcolor{blue}{#1}}
  - \\def\\purple#1{\\textcolor{purple}{#1}}
  - \\def\\orange#1{\\textcolor{orange}{#1}}
  - \\def\\gray#1{\\textcolor{gray}{#1}}
---

"""

    for chapter_id in CHAPTERS:
        print(f"Processing {chapter_id}...")
        chapter_content = process_chapter(chapter_id, base_dir)
        combined += chapter_content
        combined += "\n\n\\newpage\n\n"

    return combined


def convert_html_to_pdf_weasyprint(html_path: Path, pdf_path: Path):
    """Convert HTML to PDF using weasyprint."""
    from weasyprint import HTML, CSS

    # Custom CSS for better PDF output
    custom_css = CSS(string='''
        @page {
            size: letter;
            margin: 1in;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
        }
        h1 {
            page-break-before: always;
            font-size: 24pt;
            margin-top: 0;
        }
        h1:first-of-type {
            page-break-before: avoid;
        }
        h2 { font-size: 18pt; }
        h3 { font-size: 14pt; }
        pre, code {
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 9pt;
            background-color: #f6f8fa;
            padding: 2px 4px;
            border-radius: 3px;
        }
        pre {
            padding: 16px;
            overflow-x: auto;
        }
        pre code {
            padding: 0;
            background: none;
        }
        img {
            max-width: 100%;
            height: auto;
        }
        a {
            color: #0366d6;
            text-decoration: none;
        }
        blockquote {
            border-left: 4px solid #dfe2e5;
            padding-left: 16px;
            margin-left: 0;
            color: #6a737d;
        }
        table {
            border-collapse: collapse;
            width: 100%;
        }
        th, td {
            border: 1px solid #dfe2e5;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f6f8fa;
        }
        /* Math styling */
        .math {
            font-style: italic;
        }
    ''')

    HTML(filename=str(html_path)).write_pdf(
        str(pdf_path),
        stylesheets=[custom_css]
    )


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Build PDF from scaling book')
    parser.add_argument('--html-only', action='store_true', help='Only generate HTML, not PDF')
    parser.add_argument('--force-html', action='store_true', help='Use HTML+weasyprint even if LaTeX is available')
    args = parser.parse_args()

    base_dir = BASE_DIR
    output_md = base_dir / "scaling-book-combined.md"
    output_html = base_dir / "scaling-book.html"
    output_pdf = base_dir / "scaling-book.pdf"

    print("Building combined markdown...")
    combined = build_combined_markdown(base_dir)

    # Write intermediate markdown file
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(combined)
    print(f"Wrote combined markdown to {output_md}")

    # Convert to PDF using pandoc via pypandoc
    print("Converting to PDF with pandoc...")
    try:
        import pypandoc

        # Check if we have a LaTeX engine
        pdf_engine = None
        if not args.force_html:
            for engine in ['tectonic', 'pdflatex', 'xelatex', 'lualatex']:
                try:
                    subprocess.run([engine, '--version'], capture_output=True, check=True)
                    pdf_engine = engine
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue

        if pdf_engine and not args.html_only:
            print(f"Using {pdf_engine} for PDF generation...")

            # For tectonic, we need to set environment to continue past errors
            env = os.environ.copy()
            if pdf_engine == 'tectonic':
                env['TECTONIC_UNTRUSTED_MODE'] = 'true'

            # Use a wrapper script that tells tectonic to keep going
            if pdf_engine == 'tectonic':
                pdf_engine_args = ['--pdf-engine=tectonic', '--pdf-engine-opt=-Z', '--pdf-engine-opt=continue-on-errors']
            else:
                pdf_engine_args = [f'--pdf-engine={pdf_engine}']

            pypandoc.convert_file(
                str(output_md),
                'pdf',
                outputfile=str(output_pdf),
                extra_args=[
                    *pdf_engine_args,
                    '--toc',
                    '--toc-depth=3',
                    '-V', 'geometry:margin=1in',
                    '-V', 'linkcolor=blue',
                    '-V', 'urlcolor=blue',
                    '--resource-path', str(base_dir),
                ]
            )
            print(f"Successfully created {output_pdf}")
        else:
            # Fallback: convert to HTML first with MathJax, then to PDF via weasyprint
            print("No LaTeX engine found. Using HTML + weasyprint approach...")

            # Convert to HTML with MathJax for math rendering
            pypandoc.convert_file(
                str(output_md),
                'html',
                outputfile=str(output_html),
                extra_args=[
                    '--standalone',
                    '--toc',
                    '--toc-depth=3',
                    '--resource-path', str(base_dir),
                    '--embed-resources',
                    '--mathjax',
                    '--metadata', 'title=How to Scale Your Model',
                ]
            )
            print(f"Created HTML at {output_html}")

            # Convert HTML to PDF using weasyprint
            try:
                print("Converting HTML to PDF with weasyprint...")
                convert_html_to_pdf_weasyprint(output_html, output_pdf)
                print(f"Successfully created {output_pdf}")
            except Exception as e:
                print(f"Weasyprint conversion failed: {e}")
                print(f"HTML file is available at {output_html}")
                print("You can open it in a browser and print to PDF.")

    except Exception as e:
        print(f"Error converting to PDF: {e}")
        print(f"The combined markdown is available at {output_md}")
        raise


if __name__ == "__main__":
    main()
