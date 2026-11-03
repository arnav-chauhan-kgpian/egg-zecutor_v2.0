"""`fetch_webpage` — a web page as text, in numbered pages. It plays the
role of smolagents' VisitWebpageTool in `mistral(2).ipynb`, with two
differences, both so that nothing is cut without the model being told:
smolagents keeps the head and tail of a long page and drops its middle; here
the text is split into pages, each headed "Page k of n", and any page can be
read. A page's text is kept for the rest of the trajectory, so page 2 is
cut from the same text as page 1 even if the site changes in between.

HTML is converted to Markdown with `markdownify` (as smolagents does), after
scripts and styles are removed; PDF text comes from `pypdf`; plain text,
JSON and XML are shown as they are. Anything else is stated as a type this
tool cannot show.
"""
from __future__ import annotations

import io
import re
import time

from ..core.types import ToolResult
from .base import Tool, ToolArgumentError
from .web.http import BLOCKED_SITES, UrlProblem, blocked, decode, fetch, page_choice, split_pages

DEFAULT_PAGE_CHARS = 10_000
DEFAULT_MAX_BYTES = 10 << 20
TEXT_TYPES = ("application/json", "application/xml", "application/javascript", "application/x-yaml", "application/yaml")


def html_to_markdown(body: bytes, encoding: str | None) -> str:
    """Without a charset from the server, BeautifulSoup finds the page's
    own (a BOM or a <meta> tag), else guesses."""
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    soup = BeautifulSoup(body, "html.parser", from_encoding=encoding)
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
    text = markdownify(str(soup), heading_style="ATX")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def pdf_to_text(body: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(body))
    return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()


def to_text(content_type: str, body: bytes, encoding: str | None) -> str:
    """The body as text; raises `UrlProblem` for a type that has none."""
    if content_type in ("text/html", "application/xhtml+xml") or (not content_type and body.lstrip()[:1] == b"<"):
        return html_to_markdown(body, encoding)
    if content_type == "application/pdf" or body[:5] == b"%PDF-":
        try:
            return pdf_to_text(body)
        except Exception:  # noqa: BLE001 - pypdf raises many types for a damaged file
            raise UrlProblem("it is a PDF file whose text could not be read") from None
    if content_type.startswith("text/") or content_type in TEXT_TYPES or content_type.endswith(("+json", "+xml")):
        return decode(body, encoding).strip()
    raise UrlProblem(f"its content type is {content_type or 'not stated'}, which this tool cannot show as text")


class FetchWebpageTool(Tool):
    name = "fetch_webpage"
    allowed_params: set[str] = {"page_chars", "timeout_s", "max_bytes", "blocked_url_prefixes"}

    def __init__(self, env, **params):
        super().__init__(env, **params)
        self._pages: dict[str, tuple[str, list[str], bool]] = {}  # requested URL -> (final URL, pages, body truncated)

    @property
    def page_chars(self) -> int:
        return int(self.params.get("page_chars", DEFAULT_PAGE_CHARS))

    @property
    def description(self) -> str:
        text = (
            "Fetches a web page by its URL and returns its text: HTML converted to Markdown, the text of a PDF, or "
            f"plain text, JSON or XML as it is. The text is split into pages of at most {self.page_chars} "
            'characters. The result begins with "Page k of n of <URL>", where the URL is the page\'s address after '
            "any redirects; to read another page, call the tool again with the same URL and that page number."
        )
        if self.params.get("blocked_url_prefixes"):
            text += f" Pages of {BLOCKED_SITES}, cannot be read."
        return text

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The address of the page, starting with http:// or https://."},
                "page": {"type": "integer", "description": "Which page of the text to return, counting from 1. Default 1."},
            },
            "required": ["url"],
        }

    def call(self, url: str = "", page: int | None = None, **unexpected) -> ToolResult:
        if unexpected:
            raise ToolArgumentError(f"fetch_webpage got unexpected argument(s): {', '.join(sorted(unexpected))}")
        url = url.strip() if isinstance(url, str) else ""
        if not url:
            return ToolResult(observation="The URL is empty; nothing was fetched.", private={"empty_url": True})
        private: dict = {"url": url}
        if blocked(url, self.params.get("blocked_url_prefixes") or []):
            private["blocked"] = True
            return ToolResult(observation=f"{url} cannot be read in this task: it is a page of one of the {BLOCKED_SITES}.",
                              private=private)

        private["cached"] = url in self._pages
        if url not in self._pages:
            started = time.perf_counter()
            try:
                got = fetch(url, timeout=float(self.params.get("timeout_s", 20)),
                            max_bytes=int(self.params.get("max_bytes", DEFAULT_MAX_BYTES)))
                private.update(final_url=got.url, status=got.status, content_type=got.content_type, bytes=len(got.body),
                               seconds=round(time.perf_counter() - started, 3))
                if got.status >= 400:
                    raise UrlProblem(f"the server answered with HTTP status {got.status}" + (f" ({got.reason})" if got.reason else ""))
                if blocked(got.url, self.params.get("blocked_url_prefixes") or []):
                    private["blocked"] = True
                    return ToolResult(observation=f"{url} redirects to {got.url}, which cannot be read in this task: it is a "
                                                  f"page of one of the {BLOCKED_SITES}.", private=private)
                text = to_text(got.content_type, got.body, got.encoding)
            except UrlProblem as problem:
                private["problem"] = str(problem)
                return ToolResult(observation=f"{url} could not be fetched: {problem}.", private=private)
            self._pages[url] = (got.url, split_pages(text, self.page_chars), got.truncated)

        final_url, pages, truncated = self._pages[url]
        number, problem = page_choice(page, len(pages))
        private.update(pages=len(pages), page=number)
        if problem:
            return ToolResult(observation=f"{problem} ({final_url})", private=private)
        body = pages[number - 1].strip() or "(this page has no text)"
        head = f"Page {number} of {len(pages)} of {final_url}"
        note = ""
        if truncated and number == len(pages):
            limit_mb = int(self.params.get("max_bytes", DEFAULT_MAX_BYTES)) / (1 << 20)
            note = f"\n\n(Only the first {limit_mb:g} MB of this page were downloaded; any text after that is missing.)"
        return ToolResult(observation=f"{head}\n\n{body}{note}", private=private)
