# EPUB build for *How to Scale Your Model*

Builds a Kindle/e-reader-friendly EPUB 3 of the book from the chapter
markdown sources — much more pleasant on a small screen than printing the
website to PDF.

## What it does

- Orders the 13 chapters by their `section_number` frontmatter and adds a
  cover, title page, table of contents (nav + NCX fallback), and a generated
  bibliography from `assets/bibliography/main.bib`.
- **LaTeX math**: simple inline expressions (the large majority) are converted
  to real HTML (`<i>`, `<sub>`, `<sup>`, unicode symbols) so they reflow and
  scale with the reader's font size. Display equations and complex inline math
  are rendered to SVG with MathJax, sized in `em` so they track font size and
  never overflow the screen.
- **Distill/Jekyll markup**: `figure.liquid` includes become `<figure>`s,
  `<d-footnote>` becomes per-chapter endnotes with `epub:type="noteref"`
  popup-footnote markup (tap the number on Kindle/Kobo to pop up the note),
  `<d-cite>` becomes numbered links into the bibliography, and
  `{% details %}` spoilers become always-expanded boxed sections.
- **Images**: downscaled to ≤1200 px, transparency flattened to white for
  e-ink, recompressed (~10 MB total); animated GIFs become their first frame
  with a link to the animated online version. Interactive plotly iframes
  become links to the online chapter.
- Cross-chapter links are rewritten to internal EPUB links (validated at
  build time), and everything passes `epubcheck` with zero errors.

## Building

Requirements: python3 with `markdown-it-py`, `Pillow`, `PyYAML`; node 18+.

```bash
cd _scripts/epub && npm install    # installs mathjax (once)
pip install markdown-it-py Pillow PyYAML

python3 _scripts/epub/build_epub.py
```

Output: `_build/How_to_Scale_Your_Model.epub`. Send it to a Kindle via
https://www.amazon.com/sendtokindle (or copy to any e-reader).

Optional validation: `pip install epubcheck && python3 -m epubcheck _build/How_to_Scale_Your_Model.epub`
