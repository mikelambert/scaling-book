#!/usr/bin/env python3
"""Build a Kindle-friendly EPUB of "How to Scale Your Model".

Converts the Jekyll/distill markdown chapters into a clean EPUB 3 (with NCX
fallback for older readers):

  * figure.liquid includes  -> <figure> with locally processed images
  * <d-footnote>            -> per-chapter endnotes (epub:type popup footnotes)
  * <d-cite>                -> numbered links into a generated bibliography
  * {% details %} blocks    -> always-expanded, visually boxed sections
  * LaTeX math              -> simple inline math becomes real HTML
                               (<i>/<sub>/<sup>/unicode); everything else is
                               rendered to SVG via MathJax (render_math.mjs)
  * images                  -> downscaled/recompressed for e-ink; animated
                               GIFs become their first frame with a note

Usage:  python3 _scripts/epub/build_epub.py
Output: _build/How_to_Scale_Your_Model.epub
"""

import html
import io
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from markdown_it import MarkdownIt
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
BUILD = ROOT / "_build" / "epub"
OEBPS = BUILD / "OEBPS"
SITE_URL = "https://jax-ml.github.io/scaling-book/"
MAX_IMG_WIDTH = 1200
EX_TO_EM = 0.5  # MathJax SVG output assumes 1ex = 8px at em = 16px

# Chapters in reading order (section_number 0..12).
CHAPTERS = [
    "index", "roofline", "tpus", "sharding", "transformers", "training",
    "applied-training", "inference", "applied-inference", "profiling",
    "jax-stuff", "conclusion", "gpus",
]

# Private-use-area sentinels: cannot occur in the source text, survive
# markdown-it untouched, and are substituted in the final HTML pass.
MATH_O, MATH_C = "", ""
FOOT_S = ""
CITE_O, CITE_C = "", ""
CODE_O, CODE_C = "", ""
DOLLAR = ""

MD = MarkdownIt("js-default", options_update={
    "html": True, "xhtmlOut": True, "typographer": False, "breaks": False,
})


# --------------------------------------------------------------------------
# Frontmatter
# --------------------------------------------------------------------------

def split_frontmatter(text):
    m = re.match(r"\A---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}, text
    import yaml
    meta = yaml.safe_load(m.group(1)) or {}
    return meta, text[m.end():]


# --------------------------------------------------------------------------
# Simple inline math -> HTML (conservative; returns None to fall back to SVG)
# --------------------------------------------------------------------------

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ϵ",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι",
    "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π",
    "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "ϕ",
    "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ",
    "Omega": "Ω",
}

# Commands rendered as literal characters.
SIMPLE_CMDS = {
    "times": "×", "cdot": "·", "div": "÷", "pm": "±", "mp": "∓",
    "approx": "≈", "sim": "∼", "simeq": "≃", "cong": "≅", "equiv": "≡",
    "propto": "∝", "neq": "≠", "ne": "≠", "leq": "≤", "le": "≤",
    "geq": "≥", "ge": "≥", "ll": "≪", "gg": "≫",
    "rightarrow": "→", "to": "→", "leftarrow": "←", "leftrightarrow": "↔",
    "Rightarrow": "⇒", "Leftarrow": "⇐", "Leftrightarrow": "⇔",
    "mapsto": "↦", "uparrow": "↑", "downarrow": "↓",
    "infty": "∞", "partial": "∂", "nabla": "∇", "circ": "∘",
    "ldots": "…", "dots": "…", "cdots": "⋯", "vdots": "⋮",
    "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆",
    "cup": "∪", "cap": "∩", "setminus": "∖", "emptyset": "∅",
    "forall": "∀", "exists": "∃", "oplus": "⊕", "otimes": "⊗",
    "odot": "⊙", "ast": "∗", "star": "⋆", "bullet": "•", "prime": "′",
    "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉",
    "langle": "⟨", "rangle": "⟩", "mid": "∣",
    "%": "%", "{": "{", "}": "}", "_": "_", "&": "&", "#": "#",
    "$": "$", " ": " ", ",": " ", ";": " ", "!": "", "|": "‖",
    "quad": " ", "qquad": "  ",
}

# Function-style commands rendered as upright words.
WORD_CMDS = {
    "min", "max", "log", "ln", "exp", "sin", "cos", "tan", "arg",
    "gcd", "det", "dim", "lim", "sup", "inf", "argmin", "argmax", "bmod",
}

TEXT_CMDS = {"text", "textrm", "mathrm", "textit", "texttt",
             "operatorname", "textbf", "mathbf"}

PLAIN_CHARS = set("+-=()[]/.,;:!?<>|'*\" @~")


def _parse_group(s, i):
    """s[i] == '{'; return (inner, next_i) or (None, i) on unbalanced."""
    depth, k = 1, i + 1
    n = len(s)
    while k < n and depth:
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
        k += 1
    if depth:
        return None, i
    return s[i + 1:k - 1], k


def _simple_parse(s):
    """Render tex string s as HTML; return None if anything is unsupported."""
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in " \t\n":
            if out and out[-1] != " ":
                out.append(" ")
            i += 1
        elif c == "\\":
            j = i + 1
            while j < n and s[j].isalpha():
                j += 1
            if j == i + 1:  # escaped single char
                if j < n and s[j] in SIMPLE_CMDS:
                    out.append(html.escape(SIMPLE_CMDS[s[j]]))
                    i = j + 1
                else:
                    return None
            else:
                cmd = s[i + 1:j]
                if cmd in GREEK:
                    out.append("<i>%s</i>" % GREEK[cmd])
                    i = j
                elif cmd in SIMPLE_CMDS:
                    out.append(html.escape(SIMPLE_CMDS[cmd]))
                    i = j
                elif cmd in WORD_CMDS:
                    out.append(cmd)
                    i = j
                elif cmd in TEXT_CMDS:
                    while j < n and s[j] == " ":
                        j += 1
                    if j >= n or s[j] != "{":
                        return None
                    inner, k = _parse_group(s, j)
                    if inner is None or re.search(r"[\\{}^_$]", inner):
                        return None
                    esc = html.escape(inner)
                    if cmd in ("textbf", "mathbf"):
                        out.append("<b>%s</b>" % esc)
                    elif cmd == "texttt":
                        out.append("<code>%s</code>" % esc)
                    elif cmd == "textit":
                        out.append("<i>%s</i>" % esc)
                    else:
                        out.append(esc)
                    i = k
                else:
                    return None
        elif c in "_^":
            i += 1
            if i >= n:
                return None
            if s[i] == "{":
                inner, k = _parse_group(s, i)
                if inner is None:
                    return None
                inner_html = _simple_parse(inner)
                i = k
            elif s[i] == "\\":
                j = i + 1
                while j < n and s[j].isalpha():
                    j += 1
                # include a braced argument, e.g. _\text{math}
                if j < n and s[j] == "{":
                    _, j = _parse_group(s, j)
                inner_html = _simple_parse(s[i:j])
                i = j
            else:
                inner_html = _simple_parse(s[i])
                i += 1
            if inner_html is None:
                return None
            out.append("<sub>%s</sub>" % inner_html if c == "_"
                       else "<sup>%s</sup>" % inner_html)
        elif c == "{":
            inner, k = _parse_group(s, i)
            if inner is None:
                return None
            inner_html = _simple_parse(inner)
            if inner_html is None:
                return None
            out.append(inner_html)
            i = k
        elif c.isalpha():
            j = i
            while j < n and s[j].isalpha():
                j += 1
            out.append("<i>%s</i>" % s[i:j])
            i = j
        elif c.isdigit():
            j = i
            while j < n and (s[j].isdigit() or s[j] == "."):
                j += 1
            out.append(s[i:j])
            i = j
        elif c in PLAIN_CHARS:
            out.append(html.escape(c))
            i += 1
        else:
            return None
    return "".join(out)


def simple_math_to_html(tex):
    tex = tex.strip()
    if not tex or len(tex) > 250:
        return None
    result = _simple_parse(tex)
    if result is None:
        return None
    return '<span class="mth">%s</span>' % result.strip()


# --------------------------------------------------------------------------
# Math extraction
# --------------------------------------------------------------------------

@dataclass
class MathTable:
    items: dict = field(default_factory=dict)   # id -> (tex, display)
    by_key: dict = field(default_factory=dict)  # (tex, display) -> id

    def add(self, tex, display):
        key = (tex, display)
        if key in self.by_key:
            return self.by_key[key]
        mid = "m%04d" % len(self.items)
        self.items[mid] = key
        self.by_key[key] = mid
        return mid


def _kramdown_unescape(tex):
    # On the live site, single-$ math passes through kramdown *text*
    # processing, which collapses backslash escapes (e.g. \\{ -> \{).
    return re.sub(r"\\\\([\\{}\[\]])", r"\\\1", tex)


def _extract_inline_dollars(line, table):
    res = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c != "$":
            res.append(c)
            i += 1
            continue
        j = line.find("$", i + 1)
        ok = False
        while j != -1:
            content = line[i + 1:j]
            if content and not content[0].isspace() and not content[-1].isspace() \
                    and "|" not in content:
                ok = True
                break
            j = line.find("$", j + 1)
        if ok:
            mid = table.add(_kramdown_unescape(content), False)
            res.append(MATH_O + mid + MATH_C)
            i = j + 1
        else:
            res.append(c)
            i += 1
    return "".join(res)


def extract_math(text, table):
    """Replace $$...$$ and $...$ with sentinels."""

    # Literal dollars: kramdown+MathJax net effect renders \\$ and \$ as $.
    text = text.replace("\\\\$", DOLLAR).replace("\\$", DOLLAR)

    # Display blocks: $$ at line start, closing $$ at line end.
    def repl_block(m):
        mid = table.add(m.group(2).strip(), True)
        return "\n\n%s%s%s%s\n\n" % (m.group(1), MATH_O, mid, MATH_C)

    text = re.sub(r"^([ \t]*)\$\$(.+?)\$\$[ \t]*$", repl_block, text,
                  flags=re.M | re.S)

    # Inline $$...$$ (kramdown renders these as inline math).
    def repl_ii(m):
        mid = table.add(_kramdown_unescape(m.group(1).strip()), False)
        return MATH_O + mid + MATH_C

    text = re.sub(r"\$\$(.+?)\$\$", repl_ii, text, flags=re.S)

    # Inline $...$ within a single line, MathJax-style validity rules.
    text = "\n".join(_extract_inline_dollars(ln, table)
                     for ln in text.split("\n"))

    return text.replace(DOLLAR, "$")


# --------------------------------------------------------------------------
# Code protection
# --------------------------------------------------------------------------

def extract_code(text, store):
    def repl(m):
        store.append(m.group(0))
        return "%s%d%s" % (CODE_O, len(store) - 1, CODE_C)

    text = re.sub(r"^```.*?^```[ \t]*$", repl, text, flags=re.M | re.S)
    text = re.sub(r"`[^`\n]+`", repl, text)
    return text


def restore_code(text, store):
    return re.sub(CODE_O + r"(\d+)" + CODE_C,
                  lambda m: store[int(m.group(1))], text)


# --------------------------------------------------------------------------
# Liquid / distill preprocessing
# --------------------------------------------------------------------------

IMAGES_USED = {}   # source rel path -> (epub file name, was_gif)


def register_image(rel_path):
    rel_path = rel_path.strip("/")
    if rel_path in IMAGES_USED:
        return IMAGES_USED[rel_path]
    was_gif = rel_path.lower().endswith(".gif")
    stem = rel_path.replace("assets/", "").replace("/", "_")
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)
    if was_gif:
        stem = stem[:-4] + ".png"
    IMAGES_USED[rel_path] = (stem, was_gif)
    return IMAGES_USED[rel_path]


def process_inline_md(text, table, code_store):
    """Inline-render a snippet (caption, footnote, description)."""
    text = extract_math(text, table)
    text = restore_code(text, code_store)
    return MD.renderInline(text)


def figure_to_html(attrs, table, code_store):
    path = attrs.get("path", "").strip()
    if not path:
        return ""
    epub_name, was_gif = register_image(path)
    caption = attrs.get("caption", "")
    cap_html = process_inline_md(caption, table, code_store) if caption else ""
    if was_gif:
        note = ('<em>(Animation: still frame shown; see the '
                '<a href="%s">online version</a>.)</em>' % (SITE_URL + path))
        cap_html = (cap_html + " " + note) if cap_html else note
    alt_src = restore_code(caption, code_store)
    alt_src = re.sub(r"<[^>]+>", "", alt_src)
    alt_src = re.sub(r"[$\\`<>]", "", alt_src)
    alt = html.escape(alt_src[:200].strip()) or "figure"
    klass = " img-small" if "img-small" in attrs.get("class", "") else ""
    fig = ['<figure class="bookfig">',
           '<img src="images/%s" class="figimg%s" alt="%s"/>' % (epub_name, klass, alt)]
    if cap_html:
        fig.append('<figcaption>%s</figcaption>' % cap_html)
    fig.append("</figure>")
    return "\n".join(fig)


def preprocess(text, table, code_store, footnotes, cites, bib_order,
               chap_idx=0):
    # HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)

    text = extract_code(text, code_store)

    # source typo: <it> is not an HTML tag
    text = text.replace("<it>", "<i>").replace("</it>", "</i>")

    # figure.liquid includes (attribute values may contain \" escapes)
    def repl_fig(m):
        attrs = {k: v.replace('\\"', '"').replace("\\\\", "\\")
                 for k, v in re.findall(r'([\w-]+)="((?:[^"\\]|\\.)*)"', m.group(1))}
        return "\n\n" + figure_to_html(attrs, table, code_store) + "\n\n"

    text = re.sub(r"\{%\s*include figure\.liquid(.*?)%\}", repl_fig, text,
                  flags=re.S)

    # d-footnote -> noteref superscript (raw inline HTML survives markdown-it,
    # and unlike a sentinel char it doesn't break _emphasis_ flanking rules)
    def repl_fn(m):
        footnotes.append(m.group(1).strip())
        n = len(footnotes)
        return ('<sup class="fnref"><a epub:type="noteref" href="#fn%d_%d" '
                'id="fnref%d_%d">%d</a></sup>'
                % (chap_idx, n, chap_idx, n, n))

    text = re.sub(r"<d-footnote>(.*?)</d-footnote>", repl_fn, text, flags=re.S)

    # d-cite -> numbered bibliography links (order of first citation)
    def repl_cite(m):
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        refs = []
        for k in keys:
            if k not in bib_order:
                bib_order.append(k)
            refs.append('<a href="bibliography.xhtml#bib_%s">%d</a>'
                        % (k, bib_order.index(k) + 1))
        cites.append(keys)
        return '<sup class="cite">[%s]</sup>' % ", ".join(refs)

    text = re.sub(r'<d-cite key="([^"]+)"\s*>\s*</d-cite>', repl_cite, text)

    # {% details TITLE %} ... {% enddetails %}
    def repl_details(m):
        title = process_inline_md(m.group(1).strip(), table, code_store)
        return ('\n\n<div class="details">\n'
                '<div class="details-title">%s</div>\n\n' % title)

    text = re.sub(r"\{%\s*details (.*?)\s*%\}", repl_details, text)
    text = re.sub(r"\{%\s*enddetails\s*%\}", "\n\n</div>\n\n", text)

    # <p markdown=1 class="x">...</p> and <h3 markdown=1 ...>...</h3>
    def repl_mdblock(m):
        cls = m.group(2) or "takeaway"
        return '\n\n<div class="%s">\n\n%s\n\n</div>\n\n' % (cls, m.group(3).strip())

    text = re.sub(r'<(p|h3) markdown=1(?: class="([^"]*)")?>(.*?)</\1>',
                  repl_mdblock, text, flags=re.S)

    # inline elements flagged markdown=1: drop the attribute, keep the tag
    text = re.sub(r"<(b|i|em|strong|span)([^>]*?) markdown=1([^>]*)>",
                  r"<\1\2\3>", text)
    # block divs flagged markdown=1: keep tag, isolate so markdown runs inside
    text = re.sub(r"<div([^>]*?) markdown=1([^>]*)>", r"\n\n<div\1\2>\n", text)

    # iframes -> pointer to the interactive online version
    def repl_iframe(m):
        url = SITE_URL + m.group(1).strip("/")
        return ('<p class="iframe-note"><em>Interactive figure &#8212; see the '
                '<a href="%s">online version</a>.</em></p>' % url)

    text = re.sub(
        r"<iframe src=\"\{\{\s*'([^']+)'\s*\|\s*relative_url\s*\}\}\"[^>]*>\s*</iframe>",
        repl_iframe, text)

    # any leftover liquid {{ '...' | relative_url }}
    text = re.sub(r"\{\{\s*'([^']+)'\s*\|\s*relative_url\s*\}\}",
                  lambda m: SITE_URL + m.group(1).strip("/"), text)

    # math, then restore code
    text = extract_math(text, table)
    text = restore_code(text, code_store)
    return text


# --------------------------------------------------------------------------
# Heading ids (kramdown-compatible) and link rewriting
# --------------------------------------------------------------------------

def kramdown_slug(text_content, seen):
    s = re.sub(r"<[^>]+>", "", text_content)
    s = html.unescape(s)
    s = re.sub(r"[^a-zA-Z0-9 -]", "", s).strip().replace(" ", "-").lower()
    if not s or not s[0].isalpha():
        s = "section" + ("-" + s if s else "")
    base, n = s, 1
    while s in seen:
        s = "%s-%d" % (base, n)
        n += 1
    seen.add(s)
    return s


def add_heading_ids(body, seen_ids):
    def repl(m):
        tag, attrs, inner = m.group(1), m.group(2), m.group(3)
        if "id=" in attrs:
            return m.group(0)
        slug = kramdown_slug(inner, seen_ids)
        return '<%s%s id="%s">%s</%s>' % (tag, attrs, slug, inner, tag)

    return re.sub(r"<(h[1-6])([^>]*)>(.*?)</\1>", repl, body, flags=re.S)


def chapter_file(slug):
    return "ch%02d_%s.xhtml" % (CHAPTERS.index(slug), slug.replace("-", "_"))


def rewrite_links(body, current_slug):
    # normalize single-quoted hrefs from raw HTML in captions/footnotes
    body = re.sub(r"href='([^']*)'", lambda m: 'href="%s"' % m.group(1), body)

    def repl(m):
        href = m.group(1)
        target = href
        if target.startswith(SITE_URL):
            target = target[len(SITE_URL):]
        elif re.match(r"^[a-z][a-z0-9+.-]*:", href):
            return m.group(0)  # external scheme
        elif href.startswith("#"):
            return m.group(0)
        while target.startswith(("./", "../")):
            target = target[2:] if target.startswith("./") else target[3:]
        anchor = ""
        if "#" in target:
            target, anchor = target.split("#", 1)
        target = target.strip("/")
        if target == "":
            return 'href="#%s"' % anchor if anchor else m.group(0)
        if target in CHAPTERS:
            frag = ("#" + anchor) if anchor else ""
            if target == current_slug and frag:
                return 'href="%s"' % frag
            return 'href="%s%s"' % (chapter_file(target), frag)
        # unknown relative link -> point at the website
        return 'href="%s"' % (SITE_URL + target + (("#" + anchor) if anchor else ""))

    return re.sub(r'href="([^"]+)"', repl, body)


# --------------------------------------------------------------------------
# Bibliography
# --------------------------------------------------------------------------

def parse_bib(path):
    text = path.read_text()
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", text, re.S):
        body = m.group(3)
        fields = {}
        for fm in re.finditer(
                r'(\w+)\s*=\s*(?:"((?:[^"]|\n)*?)"|\{(.*?)\}|([\w.]+))\s*,?\s*\n',
                body, re.S):
            val = fm.group(2) or fm.group(3) or fm.group(4) or ""
            fields[fm.group(1).lower()] = re.sub(r"\s+", " ", val).strip()
        entries[m.group(2)] = fields
    return entries


def format_bib_entry(fields):
    names = []
    for a in re.split(r"\s+and\s+", fields.get("author", "")):
        a = a.strip().strip("”“\"{}")
        if "," in a:
            last, first = [p.strip() for p in a.split(",", 1)]
            names.append(("%s %s" % (first, last)).strip())
        elif a:
            names.append(a)
    author_str = (", ".join(names[:4]) + " et al.") if len(names) > 4 \
        else ", ".join(names)
    year = fields.get("year", "")
    title = fields.get("title", "").strip("{}")
    venue = fields.get("journal") or fields.get("booktitle") or ""
    url = fields.get("url", "")
    s = html.escape(author_str)
    if year:
        s += " (%s)" % html.escape(year)
    s += ". <i>%s</i>." % html.escape(title)
    if venue:
        s += " %s." % html.escape(venue)
    if url:
        s += ' <a href="%s">link</a>' % html.escape(url)
    return s


# --------------------------------------------------------------------------
# Sentinel substitution in final HTML
# --------------------------------------------------------------------------

def substitute_sentinels(body, math_html, bib_order, chap_idx):
    return re.sub(MATH_O + r"(m\d+)" + MATH_C,
                  lambda m: math_html[m.group(1)], body)


def math_to_html_map(table, metrics):
    out = {}
    for mid, (tex, display) in table.items.items():
        if not display:
            simple = simple_math_to_html(tex)
            if simple is not None:
                out[mid] = simple
                continue
        met = metrics.get(mid)
        if met is None:
            raise RuntimeError("math %s failed to render: %s" % (mid, tex[:120]))
        w = (met["width_ex"] or 2) * EX_TO_EM
        h = (met["height_ex"] or 2) * EX_TO_EM
        va = (met["valign_ex"] or 0) * EX_TO_EM
        alt = html.escape(tex).replace("\n", " ")[:500]
        if display:
            out[mid] = ('<img class="math-display" src="math/%s.svg" '
                        'style="width:%.2fem;" alt="%s"/>' % (mid, w, alt))
        else:
            out[mid] = ('<img class="math-inline" src="math/%s.svg" '
                        'style="height:%.2fem;vertical-align:%.2fem;" alt="%s"/>'
                        % (mid, h, va, alt))
    return out


# --------------------------------------------------------------------------
# XHTML fixes
# --------------------------------------------------------------------------

def xhtml_fix(body):
    body = re.sub(r"<(img|br|hr|source|input)((?:[^>\"']|\"[^\"]*\"|'[^']*')*?)\s*/?>",
                  r"<\1\2/>", body)
    body = body.replace("&nbsp;", "&#160;").replace("&mdash;", "&#8212;")
    body = re.sub(r"&(?![a-zA-Z]+;|#\d+;|#x[0-9a-fA-F]+;)", "&amp;", body)
    return body


# --------------------------------------------------------------------------
# EPUB assembly
# --------------------------------------------------------------------------

XHTML_TMPL = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en" xml:lang="en">
<head>
<meta charset="utf-8"/>
<title>%(title)s</title>
<link rel="stylesheet" type="text/css" href="css/book.css"/>
</head>
<body>
%(body)s
</body>
</html>
"""


def write_xhtml(name, title, body):
    (OEBPS / name).write_text(
        XHTML_TMPL % {"title": html.escape(title), "body": body})


def process_images():
    img_dir = OEBPS / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for rel, (name, was_gif) in sorted(IMAGES_USED.items()):
        src = ROOT / rel
        if not src.exists():
            print("WARNING: missing image %s" % rel)
            continue
        im = Image.open(src)
        if was_gif:
            im.seek(0)
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
        if im.width > MAX_IMG_WIDTH:
            nh = round(im.height * MAX_IMG_WIDTH / im.width)
            im = im.resize((MAX_IMG_WIDTH, nh), Image.LANCZOS)
        dest = img_dir / name
        if name.lower().endswith((".jpg", ".jpeg")):
            im.save(dest, "JPEG", quality=80, optimize=True)
        else:
            im.save(dest, "PNG", optimize=True)
            if dest.stat().st_size > 200_000:
                q = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                dither=Image.Dither.FLOYDSTEINBERG)
                buf = io.BytesIO()
                q.save(buf, "PNG", optimize=True)
                if buf.tell() < dest.stat().st_size * 0.8:
                    dest.write_bytes(buf.getvalue())
        total += dest.stat().st_size
    print("images: %d files, %.1f MB" % (len(IMAGES_USED), total / 1e6))


# --------------------------------------------------------------------------
# Cover generation
# --------------------------------------------------------------------------

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SERIF_I = "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"
FONT_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def make_cover(dest, authors):
    """Compose a portrait cover: title, dragon art band, authors."""
    import random
    if not all(os.path.exists(f) for f in
               (FONT_BOLD, FONT_SERIF_I, FONT_SANS, FONT_MONO)):
        # No fonts available: fall back to the raw dragon art.
        Image.open(ROOT / "assets/img/dragon.png").save(dest)
        return
    rnd = random.Random(7)
    W, H = 1600, 2560

    def lerp(a, b, t):
        return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))

    # night sky -> dusk pink, matching the artwork's palette
    top, mid, bot = (36, 32, 58), (66, 52, 96), (148, 78, 128)
    bg = Image.new("RGB", (W, H))
    px = bg.load()
    for y in range(H):
        t = y / (H - 1)
        c = lerp(top, mid, t / 0.6) if t < 0.6 else lerp(mid, bot, (t - 0.6) / 0.4)
        for x in range(W):
            px[x, y] = c

    d = ImageDraw.Draw(bg)

    def in_text_zone(x, y):
        return 170 < x < 1430 and 240 < y < 1030

    placed = 0
    while placed < 140:  # stars
        x, y = rnd.randint(20, W - 20), rnd.randint(20, int(H * 0.42))
        if in_text_zone(x, y):
            continue
        placed += 1
        r = rnd.choice([1, 1, 1, 2, 2, 3])
        b = rnd.randint(120, 220)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(b, b, min(255, b + 20)))
    placed = 0
    while placed < 10:  # 4-point sparkles
        x, y = rnd.randint(60, W - 60), rnd.randint(60, int(H * 0.38))
        if in_text_zone(x, y):
            continue
        placed += 1
        sz = rnd.randint(8, 18)
        d.line([x - sz, y, x + sz, y], fill=(235, 230, 250), width=2)
        d.line([x, y - sz, x, y + sz], fill=(235, 230, 250), width=2)

    glow_layer = Image.new("RGB", (W, H), (0, 0, 0))
    gd = ImageDraw.Draw(glow_layer)
    sharp = []

    def glow_text(xy, text, font, glow_color, sharp_color):
        gd.text(xy, text, font=font, fill=glow_color, anchor="mm")
        sharp.append((xy, text, font, sharp_color))

    f_title = ImageFont.truetype(FONT_BOLD, 172)
    f_small = ImageFont.truetype(FONT_BOLD, 118)
    f_sub = ImageFont.truetype(FONT_SERIF_I, 62)
    glow_text((W // 2, 360), "HOW TO", f_small, (60, 40, 90), (222, 215, 240))
    glow_text((W // 2, 545), "SCALE YOUR", f_title, (110, 60, 130), (255, 250, 240))
    glow_text((W // 2, 740), "MODEL", f_title, (110, 60, 130), (255, 250, 240))
    glow_text((W // 2, 940), "A Systems View of LLMs on TPUs", f_sub,
              (40, 70, 70), (140, 235, 215))

    art = Image.open(ROOT / "assets/img/dragon.png").convert("RGB")
    band_h = round(W * art.height / art.width)
    band_y = 1120
    art = art.resize((W, band_h), Image.LANCZOS)
    for yy, col in [(band_y - 10, (255, 95, 162)),
                    (band_y + band_h + 10, (75, 224, 200))]:
        gd.rectangle([0, yy - 3, W, yy + 3], fill=col)

    bg = ImageChops.screen(bg, glow_layer.filter(ImageFilter.GaussianBlur(18)))
    bg.paste(art, (0, band_y))
    d = ImageDraw.Draw(bg)
    d.rectangle([0, band_y - 12, W, band_y - 8], fill=(255, 120, 175))
    d.rectangle([0, band_y + band_h + 8, W, band_y + band_h + 12], fill=(96, 232, 210))
    for xy, text, font, color in sharp:
        d.text(xy, text, font=font, fill=color, anchor="mm")

    f_auth = ImageFont.truetype(FONT_SANS, 46)
    y = 2085
    for i in range(0, len(authors), 3):
        d.text((W // 2, y), "  ·  ".join(authors[i:i + 3]),
               font=f_auth, fill=(228, 222, 240), anchor="mm")
        y += 78
    f_foot = ImageFont.truetype(FONT_MONO, 40)
    d.text((W // 2, 2465), "jax-ml.github.io/scaling-book", font=f_foot,
           fill=(170, 160, 200), anchor="mm")
    bg.save(dest, "PNG", optimize=True)


def main():
    if BUILD.exists():
        shutil.rmtree(BUILD)
    OEBPS.mkdir(parents=True)
    (BUILD / "META-INF").mkdir()
    (OEBPS / "css").mkdir()

    table = MathTable()
    bib_order = []
    chapters = []

    # ---- pass 1: preprocess + markdown render (sentinels intact) --------
    for slug in CHAPTERS:
        raw = (ROOT / (slug + ".md")).read_text()
        meta, body_md = split_frontmatter(raw)
        code_store, footnotes, cites = [], [], []
        text = preprocess(body_md, table, code_store, footnotes, cites,
                          bib_order, chap_idx=CHAPTERS.index(slug))
        body = MD.render(text)
        fn_htmls = [process_inline_md(fn, table, code_store) for fn in footnotes]
        desc = meta.get("description", "") or ""
        desc_html = process_inline_md(desc.strip(), table, code_store) if desc.strip() else ""
        chapters.append({
            "slug": slug,
            "title": str(meta.get("title", slug)),
            "subtitle": str(meta.get("subtitle", "") or ""),
            "part": int(meta.get("section_number", 0)),
            "desc_html": desc_html,
            "authors": meta.get("authors", []),
            "date": str(meta.get("date", "")),
            "body": body,
            "fn_htmls": fn_htmls,
        })

    # ---- render math to SVG ---------------------------------------------
    svg_entries = [{"id": mid, "tex": tex, "display": display}
                   for mid, (tex, display) in table.items.items()
                   if display or simple_math_to_html(tex) is None]
    math_dir = OEBPS / "math"
    math_dir.mkdir()
    req = BUILD / "math_requests.json"
    req.write_text(json.dumps(svg_entries))
    print("math: %d unique expressions, %d rendered as SVG, %d as HTML"
          % (len(table.items), len(svg_entries),
             len(table.items) - len(svg_entries)))
    r = subprocess.run(
        ["node", str(SCRIPT_DIR / "render_math.mjs"), str(req), str(math_dir)],
        capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        raise RuntimeError("math rendering failed")
    metrics = json.loads((math_dir / "metrics.json").read_text())
    (math_dir / "metrics.json").unlink()
    math_html = math_to_html_map(table, metrics)

    # ---- bibliography -----------------------------------------------------
    bib = parse_bib(ROOT / "assets" / "bibliography" / "main.bib")
    for k in bib_order:
        if k not in bib:
            print("WARNING: missing bib key: %s" % k)

    # ---- pass 2: write chapters -------------------------------------------
    spine, toc = [], []
    for idx, ch in enumerate(chapters):
        seen_ids = set()
        body = add_heading_ids(ch["body"], seen_ids)
        body = rewrite_links(body, ch["slug"])
        body = substitute_sentinels(body, math_html, bib_order, idx)

        if ch["fn_htmls"]:
            notes = ['<section class="footnotes" epub:type="footnotes">',
                     "<h2>Notes</h2>"]
            for n, fn_html in enumerate(ch["fn_htmls"], 1):
                fn_html = rewrite_links(fn_html, ch["slug"])
                fn_html = substitute_sentinels(fn_html, math_html, bib_order, idx)
                notes.append(
                    '<aside id="fn%d_%d" epub:type="footnote" class="footnote">'
                    '<p><a href="#fnref%d_%d">%d.</a> %s</p></aside>'
                    % (idx, n, idx, n, n, fn_html))
            notes.append("</section>")
            body += "\n" + "\n".join(notes)

        part_label = ("Part %d" % ch["part"]) if ch["slug"] != "index" else ""
        header = ['<header class="chapter-header">']
        if part_label:
            header.append('<p class="part-label">%s</p>' % part_label)
        header.append("<h1>%s</h1>" % html.escape(ch["title"]))
        if ch["slug"] == "index" and ch["subtitle"]:
            header.append('<p class="book-subtitle">%s</p>'
                          % html.escape(ch["subtitle"]))
        if ch["desc_html"]:
            desc = rewrite_links(ch["desc_html"], ch["slug"])
            desc = substitute_sentinels(desc, math_html, bib_order, idx)
            header.append('<div class="chapter-desc">%s</div>' % desc)
        header.append("</header>")
        body = "\n".join(header) + "\n" + body

        fname = chapter_file(ch["slug"])
        title = ("%s: %s" % (part_label, ch["title"])) if part_label else ch["title"]
        write_xhtml(fname, title, xhtml_fix(body))
        spine.append(fname)
        toc.append((fname, title))

    # ---- bibliography chapter ----------------------------------------------
    if bib_order:
        items = []
        for k in bib_order:
            fields = bib.get(k)
            entry = format_bib_entry(fields) if fields else html.escape(k)
            items.append('<li id="bib_%s">%s</li>' % (k, entry))
        body = ('<header class="chapter-header"><h1>Bibliography</h1></header>'
                '<ol class="bibliography">\n%s\n</ol>' % "\n".join(items))
        write_xhtml("bibliography.xhtml", "Bibliography", xhtml_fix(body))
        spine.append("bibliography.xhtml")
        toc.append(("bibliography.xhtml", "Bibliography"))

    # ---- cover + title page --------------------------------------------------
    authors = [a.get("name", "") for a in chapters[0].get("authors", [])
               if isinstance(a, dict)]
    authors = [re.sub(r"<[^>]+>", "", a).replace("*", "").strip()
               for a in authors if a]
    tp = ['<div class="titlepage">',
          '<h1 class="book-title">How to Scale Your Model</h1>',
          '<p class="book-subtitle">A Systems View of LLMs on TPUs</p>',
          '<p class="book-authors">%s</p>' % html.escape(", ".join(authors)),
          '<p class="book-note">EPUB edition generated from the online book at '
          '<a href="%s">jax-ml.github.io/scaling-book</a>.</p>' % SITE_URL,
          "</div>"]
    write_xhtml("titlepage.xhtml", "How to Scale Your Model", "\n".join(tp))
    write_xhtml("cover.xhtml", "Cover",
                '<div class="cover"><img src="images/cover.png" '
                'alt="How to Scale Your Model"/></div>')

    process_images()
    make_cover(OEBPS / "images" / "cover.png", authors)
    shutil.copy(SCRIPT_DIR / "book.css", OEBPS / "css" / "book.css")

    # ---- nav / ncx / opf ------------------------------------------------------
    nav_items = ['<li><a href="titlepage.xhtml">Title Page</a></li>']
    ncx_points = []
    for i, (fname, title) in enumerate(toc):
        nav_items.append('<li><a href="%s">%s</a></li>'
                         % (fname, html.escape(title)))
        ncx_points.append(
            '<navPoint id="np%d" playOrder="%d"><navLabel><text>%s</text>'
            '</navLabel><content src="%s"/></navPoint>'
            % (i + 2, i + 2, html.escape(title), fname))
    nav_body = ('<nav epub:type="toc" id="toc"><h1>Contents</h1>\n<ol>\n%s\n</ol></nav>\n'
                '<nav epub:type="landmarks" hidden="hidden"><ol>'
                '<li><a epub:type="cover" href="cover.xhtml">Cover</a></li>'
                '<li><a epub:type="toc" href="nav.xhtml#toc">Table of Contents</a></li>'
                '<li><a epub:type="bodymatter" href="%s">Begin Reading</a></li>'
                '</ol></nav>' % ("\n".join(nav_items), spine[0]))
    write_xhtml("nav.xhtml", "Contents", nav_body)

    book_id = "urn:uuid:5c1f7a2e-8d3b-4f6a-9e0c-3b7d51c90a42"
    manifest = [
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
        '<item id="css" href="css/book.css" media-type="text/css"/>',
        '<item id="titlepage" href="titlepage.xhtml" media-type="application/xhtml+xml"/>',
        '<item id="coverpage" href="cover.xhtml" media-type="application/xhtml+xml"/>',
    ]
    spine_items = ['<itemref idref="coverpage"/>', '<itemref idref="titlepage"/>',
                   '<itemref idref="nav"/>']
    for i, fname in enumerate(spine):
        manifest.append('<item id="c%d" href="%s" media-type="application/xhtml+xml"/>'
                        % (i, fname))
        spine_items.append('<itemref idref="c%d"/>' % i)
    cover_item_id = "imgcover"
    manifest.append('<item id="imgcover" href="images/cover.png" '
                    'media-type="image/png" properties="cover-image"/>')
    for j, (rel, (name, _)) in enumerate(sorted(IMAGES_USED.items())):
        mt = "image/jpeg" if name.lower().endswith((".jpg", ".jpeg")) else "image/png"
        manifest.append('<item id="img%d" href="images/%s" media-type="%s"/>'
                        % (j, name, mt))
    for j, svg in enumerate(sorted(math_dir.glob("*.svg"))):
        manifest.append('<item id="math%d" href="math/%s" media-type="image/svg+xml"/>'
                        % (j, svg.name))

    creators = "\n".join('<dc:creator id="au%d">%s</dc:creator>'
                         % (i, html.escape(a)) for i, a in enumerate(authors))
    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:identifier id="bookid">%s</dc:identifier>
<dc:title>How to Scale Your Model: A Systems View of LLMs on TPUs</dc:title>
%s
<dc:language>en</dc:language>
<dc:source>%s</dc:source>
<dc:date>%s</dc:date>
<meta property="dcterms:modified">%sT00:00:00Z</meta>
<meta name="cover" content="%s"/>
</metadata>
<manifest>
%s
</manifest>
<spine toc="ncx">
%s
</spine>
</package>
""" % (book_id, creators, SITE_URL, chapters[0]["date"] or "2025-02-04",
       os.environ.get("EPUB_BUILD_DATE", "2026-07-11"), cover_item_id,
       "\n".join(manifest), "\n".join(spine_items))
    (OEBPS / "content.opf").write_text(opf)

    ncx = """<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
<head><meta name="dtb:uid" content="%s"/><meta name="dtb:depth" content="1"/>
<meta name="dtb:totalPageCount" content="0"/><meta name="dtb:maxPageNumber" content="0"/></head>
<docTitle><text>How to Scale Your Model</text></docTitle>
<navMap>
<navPoint id="np1" playOrder="1"><navLabel><text>Title Page</text></navLabel><content src="titlepage.xhtml"/></navPoint>
%s
</navMap>
</ncx>
""" % (book_id, "\n".join(ncx_points))
    (OEBPS / "toc.ncx").write_text(ncx)

    (BUILD / "META-INF" / "container.xml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>
""")

    out = ROOT / "_build" / "How_to_Scale_Your_Model.epub"
    files = sorted(p for p in BUILD.rglob("*")
                   if p.is_file() and p.name != "math_requests.json")
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("mimetype", "application/epub+zip",
                   compress_type=zipfile.ZIP_STORED)
        for f in files:
            z.write(f, str(f.relative_to(BUILD)),
                    compress_type=zipfile.ZIP_DEFLATED)
    print("wrote %s (%.1f MB)" % (out, out.stat().st_size / 1e6))

    # ---- link validation ---------------------------------------------------
    ids_by_file = {}
    for f in OEBPS.glob("*.xhtml"):
        ids_by_file[f.name] = set(re.findall(r'id="([^"]+)"', f.read_text()))
    problems = 0
    for f in sorted(OEBPS.glob("*.xhtml")):
        for href in re.findall(r'href="([^"]+)"', f.read_text()):
            if re.match(r"^[a-z][a-z0-9+.-]*:", href):
                continue
            target, _, frag = href.partition("#")
            target = target or f.name
            if target not in ids_by_file:
                if not target.endswith(".css"):
                    print("BROKEN FILE LINK in %s: %s" % (f.name, href))
                    problems += 1
            elif frag and frag not in ids_by_file[target]:
                print("BROKEN ANCHOR in %s: %s" % (f.name, href))
                problems += 1
    print("link check: %d problems" % problems)

    # leftover sentinel check
    for f in sorted(OEBPS.glob("*.xhtml")):
        t = f.read_text()
        for ch_ in (MATH_O, MATH_C, FOOT_S, CITE_O, CITE_C, CODE_O, CODE_C, DOLLAR):
            if ch_ in t:
                print("LEFTOVER SENTINEL %r in %s" % (ch_, f.name))
    # leftover liquid/distill check
    for f in sorted(OEBPS.glob("*.xhtml")):
        t = f.read_text()
        for pat in (r"\{%", r"\{\{", r"<d-", r"\$\$"):
            m = re.search(pat, t)
            if m:
                print("LEFTOVER MARKUP %s in %s at %d" % (pat, f.name, m.start()))


if __name__ == "__main__":
    main()
