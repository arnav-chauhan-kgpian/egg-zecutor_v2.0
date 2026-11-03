"""Documents the model reads in pages: web pages and the file attached to a
question. Shared by `visit_webpage`, `find_in_page` and `read_file`.

* **One store per trajectory.** A document is converted to text once, split
  into pages once, and kept on the task environment (`documents_of(env)`),
  so `find_in_page` reports the page numbers `visit_webpage` and `read_file`
  show, even if the site changes between calls.
* **Pages are bounded in tokens, not only in characters.** A page is at most
  `page_chars` *units*: one per ASCII letter or whitespace character, two per
  other ASCII symbol, three per digit and per non-ASCII character below
  U+2E80 (accented Latin, Greek, Cyrillic, ...), four from U+2E80 up (CJK and
  beyond). Prose costs Qwen's tokenizer about a token per four letters, but a
  digit is a token of its own, and symbols and non-Latin scripts sit in
  between; the weights were fitted on 636 chunks of web pages and GAIA
  attachments (2026-10-29, Qwen3-14B tokenizer) so that 6000 units stayed
  under 2,750 tokens on every chunk (median about 1,500). The tools'
  descriptions say a page has at most `page_chars` characters, and fewer for
  text with many digits, symbols or non-Latin characters.
* **HTML is read as assistive technology reads it.** Elements marked
  `aria-hidden="true"` are left out, and a link with an `aria-label` is
  shown by that name: sites print some text twice (a full and a short name,
  one hidden by CSS, which this conversion cannot evaluate), and mark the
  copy that way. Content hidden only by CSS or the `hidden` attribute is
  kept: it is often a collapsed section a reader can open.
* **HTML is cleaned for reading.** Scripts, styles, site navigation menus
  (`<nav>` and `role=navigation` outside the page's `<main>` or `<article>`,
  MediaWiki's header and page tools), the
  Internet Archive's toolbar and Wikipedia's "[edit]" links are removed;
  links are made absolute (so the model can visit them as shown) and lose
  their titles; same-page links (citations, "jump to") become plain text;
  an image becomes "[image: <alt text>]", or nothing when it has none. The
  page's other text is all kept; `visit_webpage`'s description says what is
  left out.
* **PDF text is marked by page** ("[PDF page 3 of 12]"), since questions
  cite page numbers.
"""
from __future__ import annotations

import bisect
import io
import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from .web.http import decode

DEFAULT_PAGE_CHARS = 6000

# --------------------------------------------------------------------- paging


def _ascii_unit(code: int) -> int:
    ch = chr(code)
    if ch.isalpha() or ch.isspace():
        return 1
    return 3 if ch.isdigit() else 2


_ASCII_UNITS = [_ascii_unit(i) for i in range(128)]


def _unit(ch: str) -> int:
    o = ord(ch)
    if o < 0x80:
        return _ASCII_UNITS[o]
    return 3 if o < 0x2E80 else 4


def text_units(text: str) -> int:
    return sum(map(_unit, text))


def split_pages(text: str, budget: int) -> list[str]:
    """`text` in pages of at most `budget` units (see the module doc). A page
    ends at a line break in its second half if there is one, else at a space
    there, else exactly at the budget. Joining the pages gives `text` back."""
    if budget <= 0:
        raise ValueError("page budget must be positive")
    if not text:
        return [""]
    # cumulative[i] = units of text[:i]
    cumulative = [0, *itertools.accumulate(map(_unit, text))]
    pages, start, n = [], 0, len(text)
    while start < n:
        # the largest end with cumulative[end] - cumulative[start] <= budget
        end = bisect.bisect_right(cumulative, cumulative[start] + budget, lo=start + 1) - 1
        end = max(end, start + 1)  # a single character over budget still makes progress
        if end < n:
            half = start + (end - start) // 2
            cut = text.rfind("\n", half, end)
            if cut == -1:
                cut = text.rfind(" ", half, end)
            if cut != -1:
                end = cut + 1
        pages.append(text[start:end])
        start = end
    return pages


def page_offsets(pages: list[str]) -> list[int]:
    """Start offset of each page in the joined text."""
    offsets, pos = [], 0
    for page in pages:
        offsets.append(pos)
        pos += len(page)
    return offsets


# --------------------------------------------------------------------- HTML

_BOILERPLATE = [
    "script", "style", "noscript", "template", "svg", "iframe", "object", "embed",
    "#wm-ipp-base", "#wm-ipp-print", "#donato", "#wm-ipp",  # the Internet Archive's toolbar on archived pages
    ".vector-header-container", ".vector-page-toolbar", "#p-personal",  # MediaWiki's header and page tools
    "span.mw-editsection",  # Wikipedia's "[edit]" links
    "li.interlanguage-link",  # Wikipedia's links to the same article in ~100 other languages
]


def html_to_text(body: bytes, encoding: str | None, base_url: str) -> str:
    """Readable Markdown from an HTML page (see the module doc)."""
    from bs4 import BeautifulSoup, Comment
    from markdownify import markdownify

    soup = BeautifulSoup(body, "html.parser", from_encoding=encoding)
    for selector in _BOILERPLATE:
        for tag in soup.select(selector):
            tag.decompose()
    # Site menus: navigation outside the page's main content. Navigation
    # inside it (Wikipedia's navboxes: tables of related articles) is content.
    for tag in soup.select("nav, [role=navigation]"):
        if tag.decomposed:
            continue
        if tag.find_parent(["main", "article"]) is None and tag.find_parent(attrs={"role": "main"}) is None:
            tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    # As assistive technology reads the page: what it is told to skip is left
    # out (sites mark duplicate and decorative copies of text this way).
    for tag in soup.find_all(attrs={"aria-hidden": "true"}):
        if not tag.decomposed:
            tag.decompose()
    page_url = base_url.partition("#")[0]  # the page's own address, for links to places on this same page
    base = soup.find("base", href=True)
    base_url = urljoin(base_url, base["href"]) if base is not None else base_url

    for img in soup.find_all("img"):
        alt = " ".join((img.get("alt") or "").split())
        img.replace_with(f"[image: {alt}]" if alt else "")
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#") or href.lower().startswith(("javascript:", "mailto:", "data:")) or not a.get_text(strip=True):
            a.unwrap()
            continue
        absolute = urljoin(base_url, href)
        page, _, fragment = absolute.partition("#")
        if page == page_url and fragment:
            a.unwrap()  # a link to another place on this same page
            continue
        label = " ".join((a.get("aria-label") or "").split())
        if label:
            # The link's accessible name. Sites that print a name twice (a full
            # and a short form, one hidden by CSS) give it once here.
            a.clear()
            a.string = label
        a.attrs = {"href": absolute}

    text = markdownify(str(soup), heading_style="ATX", bullets="-")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------- PDF


def pdf_to_text(body: bytes) -> str:
    """Each page's text after a "[PDF page k of n]" line. Raises ValueError
    when the file is not a readable PDF."""
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(body))
        count = len(reader.pages)
        parts = []
        for k, page in enumerate(reader.pages, 1):
            parts.append(f"[PDF page {k} of {count}]\n{(page.extract_text() or '').strip()}")
    except Exception as exc:  # noqa: BLE001 - pypdf raises many types for a damaged file
        raise ValueError(f"it is a PDF file whose text could not be read ({type(exc).__name__})") from None
    return "\n\n".join(parts).strip()


# --------------------------------------------------------------------- spreadsheets


def _column_letter(index: int) -> str:
    letters = ""
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _cell_text(value) -> str:
    if value is None:
        return ""
    text = repr(value) if isinstance(value, float) else str(value)  # a float exactly as stored: 3.0, 0.1
    return " ".join(text.split()).replace("|", "\\|")


def xlsx_to_text(body: bytes) -> str:
    """Every sheet as a Markdown table whose header row gives the column
    letters and whose first column gives the row numbers, so a cell can be
    named as in Excel (B7). Cell values are the values stored in the file;
    for a formula, the value Excel last computed. Empty rows are left out."""
    from openpyxl import load_workbook

    try:
        workbook = load_workbook(io.BytesIO(body), read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001 - openpyxl raises many types for a damaged file
        raise ValueError(f"it is an Excel file that could not be read ({type(exc).__name__})") from None
    sections = []
    try:
        for ws in workbook.worksheets:
            rows = []
            width = 0
            for r, row in enumerate(ws.iter_rows(values_only=True), 1):
                cells = [_cell_text(v) for v in row]
                while cells and not cells[-1]:
                    cells.pop()
                if cells:
                    rows.append((r, cells))
                    width = max(width, len(cells))
            state = "" if ws.sheet_state == "visible" else f" ({ws.sheet_state})"
            if not rows:
                sections.append(f'## Sheet "{ws.title}"{state}\n\n(empty)')
                continue
            header = "| row | " + " | ".join(_column_letter(c) for c in range(1, width + 1)) + " |"
            rule = "|---|" + "---|" * width
            lines = [header, rule]
            for r, cells in rows:
                lines.append(f"| {r} | " + " | ".join(cells + [""] * (width - len(cells))) + " |")
            sections.append(f'## Sheet "{ws.title}"{state}: rows {rows[0][0]}-{rows[-1][0]}, columns A-{_column_letter(width)}\n\n'
                            + "\n".join(lines))
    finally:
        workbook.close()
    return "\n\n".join(sections)


# --------------------------------------------------------------------- files


def normalize_newlines(text: str) -> str:
    """Windows (\\r\\n) and old Mac (\\r) line endings as \\n: a raw \\r in an
    observation is invisible noise to the model."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


TEXT_SUFFIXES = {".txt", ".py", ".json", ".jsonld", ".pdb", ".xml", ".md", ".csv", ".tsv"}
FILE_SUFFIXES = {".xlsx", ".pdf"} | TEXT_SUFFIXES


def describe_supported() -> str:
    return "Excel (.xlsx), PDF, and text files (" + ", ".join(sorted(TEXT_SUFFIXES)) + ")"


def file_to_text(name: str, body: bytes) -> str:
    """A file's text by its suffix. Raises ValueError for a type this module
    cannot show, or a file it cannot read."""
    suffix = Path(name).suffix.lower()
    if suffix == ".xlsx":
        return xlsx_to_text(body)
    if suffix == ".pdf":
        return pdf_to_text(body)
    if suffix in TEXT_SUFFIXES:
        return normalize_newlines(decode(body, None))  # CSV too: its raw text is exact, and compact
    raise ValueError(f"its type ({suffix or 'no suffix'}) is not one this tool can show as text")


# --------------------------------------------------------------------- store


@dataclass(frozen=True)
class Attachment:
    """The file attached to a question: the name the model is told, and where it is."""

    name: str
    path: Path

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()


def attachment_of(env) -> Attachment | None:
    return getattr(env, "attachment", None)


@dataclass
class Document:
    key: str  # the URL as requested, or the attached file's name
    label: str  # how observations name it: the final URL, or "the attached file <name>"
    pages: list[str]
    note: str | None = None  # appended to the last page (e.g. a download cut at max_bytes)
    private: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(self.pages)


class DocumentStore(dict):
    """key -> Document, for one trajectory."""


def documents_of(env) -> DocumentStore:
    """The trajectory's store, created on the environment the first time any
    tool asks, so tools built for one trajectory share it."""
    store = getattr(env, "documents", None)
    if store is None:
        store = DocumentStore()
        setattr(env, "documents", store)
    return store


def normalize_url(url: str) -> str:
    """For matching a URL against text the model has seen: no scheme, no
    `www.`, no trailing slash, no fragment, lower-case host."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()
    host = parts.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    path = parts.path.rstrip("/")
    return host + path + (f"?{parts.query}" if parts.query else "")


def page_observation(doc: Document, number: int, what: str) -> str:
    """'Page k of n of <label>', the page, and the document's note on its last page."""
    body = doc.pages[number - 1].strip() or ("(this page has no text)" if len(doc.pages) > 1 else f"(the {what} has no text)")
    head = f"Page {number} of {len(doc.pages)} of {doc.label}"
    note = f"\n\n{doc.note}" if doc.note and number == len(doc.pages) else ""
    return f"{head}\n\n{body}{note}"
