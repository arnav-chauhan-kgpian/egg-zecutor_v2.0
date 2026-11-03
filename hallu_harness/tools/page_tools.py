"""`visit_webpage`, `find_in_page` and `read_file`: reading web pages and the
attached file in bounded pages, and searching them.

They share one document store per trajectory (`documents.documents_of`), so
a page is fetched and split once, and `find_in_page` names the pages the
other two show. Every observation is bounded: a page is at most
`page_chars` units (`documents.split_pages`); a search shows at most
`max_matches` snippets of bounded length and states how many it left out.

Failures follow `tools/web/http.py`: what the model asked for (an HTTP
error, a host that does not exist, a type that has no text, a page number
out of range) is stated in the observation; this machine being offline is
raised, so the trajectory is recorded as crashed and re-run.
"""
from __future__ import annotations

import bisect
import re
import time

from ..core.types import ToolResult
from .base import Tool, ToolArgumentError
from .documents import (
    DEFAULT_PAGE_CHARS,
    Document,
    attachment_of,
    describe_supported,
    documents_of,
    file_to_text,
    html_to_text,
    normalize_newlines,
    page_observation,
    split_pages,
)
from .web.http import BLOCKED_SITES, UrlProblem, blocked, decode, fetch, page_choice

DEFAULT_MAX_BYTES = 20 << 20
TEXT_TYPES = ("application/json", "application/ld+json", "application/xml", "application/javascript", "application/x-yaml",
              "application/yaml")
DEFAULT_MAX_MATCHES = 10
DEFAULT_SNIPPET_CHARS = 150  # on each side of a match
MAX_QUERY_CHARS = 200
ARCHIVE_NOTE = (
    "To read a page as it was on an earlier date, visit https://web.archive.org/web/<YYYYMMDD>/<URL of the page>: "
    "the Internet Archive answers with its saved copy closest to that date, and its URL shows the copy's date and time."
)


def _check_unexpected(tool: str, unexpected: dict) -> None:
    if unexpected:
        raise ToolArgumentError(f"{tool} got unexpected argument(s): {', '.join(sorted(unexpected))}")


def _page_budget(params: dict) -> int:
    return int(params.get("page_chars", DEFAULT_PAGE_CHARS))


def pages_sentence(page_chars: int) -> str:
    return (f"The text is split into pages of at most {page_chars} characters (fewer for text with many digits, symbols "
            "or non-Latin characters)")


# --------------------------------------------------------------------- loading


def web_text(content_type: str, body: bytes, encoding: str | None, url: str) -> str:
    """A fetched body as text; raises `UrlProblem` for a type that has none."""
    from .documents import pdf_to_text

    if content_type in ("text/html", "application/xhtml+xml") or (not content_type and body.lstrip()[:1] == b"<"):
        return html_to_text(body, encoding, url)
    if content_type == "application/pdf" or body[:5] == b"%PDF-":
        try:
            return pdf_to_text(body)
        except ValueError as exc:
            raise UrlProblem(str(exc)) from None
    if content_type.startswith("text/") or content_type in TEXT_TYPES or content_type.endswith(("+json", "+xml")):
        return normalize_newlines(decode(body, encoding)).strip()
    raise UrlProblem(f"its content type is {content_type or 'not stated'}, which this tool cannot show as text")


def load_web(env, url: str, params: dict) -> tuple[Document | None, str | None, dict]:
    """The document for `url` from the trajectory's store, fetching it the
    first time. Returns (document, None, private) or (None, the observation
    that says why there is none, private)."""
    store = documents_of(env)
    private: dict = {"url": url, "cached": url in store}
    prefixes = params.get("blocked_url_prefixes") or []
    if blocked(url, prefixes):
        private["blocked"] = True
        return None, f"{url} cannot be read in this task: it is a page of one of the {BLOCKED_SITES}.", private
    if url in store:
        return store[url], None, private
    started = time.perf_counter()
    max_bytes = int(params.get("max_bytes", DEFAULT_MAX_BYTES))
    try:
        got = fetch(url, timeout=float(params.get("timeout_s", 20)), max_bytes=max_bytes)
        private.update(final_url=got.url, status=got.status, content_type=got.content_type, bytes=len(got.body),
                       seconds=round(time.perf_counter() - started, 3))
        if got.status >= 400:
            raise UrlProblem(f"the server answered with HTTP status {got.status}" + (f" ({got.reason})" if got.reason else ""))
        if not got.body.strip():
            raise UrlProblem(f"the server answered with HTTP status {got.status} and an empty page")
        if blocked(got.url, prefixes):
            private["blocked"] = True
            return None, (f"{url} redirects to {got.url}, which cannot be read in this task: it is a page of one of the "
                          f"{BLOCKED_SITES}."), private
        text = web_text(got.content_type, got.body, got.encoding, got.url)
    except UrlProblem as problem:
        private["problem"] = str(problem)
        return None, f"{url} could not be fetched: {problem}.", private
    note = None
    if got.truncated:
        note = f"(Only the first {max_bytes / (1 << 20):g} MB of this page were downloaded; any text after that is missing.)"
    doc = Document(key=url, label=got.url, pages=split_pages(text, _page_budget(params)), note=note,
                   private={"final_url": got.url, "content_type": got.content_type, "chars": len(text)})
    store[url] = doc
    private["chars"] = len(text)
    return doc, None, private


def load_attachment(env, params: dict) -> tuple[Document | None, str | None, dict]:
    attachment = attachment_of(env)
    if attachment is None:
        return None, "This question has no attached file.", {"attachment": None}
    store = documents_of(env)
    private: dict = {"attachment": attachment.name, "cached": attachment.name in store}
    if attachment.name not in store:
        try:
            text = file_to_text(attachment.name, attachment.read_bytes())
        except ValueError as problem:
            private["problem"] = str(problem)
            return None, f"The attached file {attachment.name} cannot be shown: {problem}.", private
        store[attachment.name] = Document(key=attachment.name, label=f"the attached file {attachment.name}",
                                          pages=split_pages(text, _page_budget(params)), private={"chars": len(text)})
    doc = store[attachment.name]
    private["chars"] = doc.private["chars"]
    return doc, None, private


# --------------------------------------------------------------------- visit_webpage


class VisitWebpageTool(Tool):
    name = "visit_webpage"
    allowed_params: set[str] = {"page_chars", "timeout_s", "max_bytes", "blocked_url_prefixes"}

    @property
    def description(self) -> str:
        text = (
            "Fetches a web page by its URL and returns one page of its text: HTML converted to Markdown, the text of "
            "a PDF (each PDF page headed \"[PDF page k of n]\"), or plain text, JSON or XML as it is. From HTML, "
            "scripts, the site's navigation menus and elements marked as hidden from screen readers (aria-hidden) are "
            "left out; links are shown with absolute URLs, by their accessible name (aria-label) when they have one; "
            "and images as [image: <their alt text>]; images themselves cannot be seen. "
            f"{pages_sentence(_page_budget(self.params))}. The result begins with \"Page k of n of <URL>\", where the "
            "URL is the page's address after any redirects; to read another page, call the tool again with the same "
            f"URL and that page number. {ARCHIVE_NOTE}"
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
        _check_unexpected(self.name, unexpected)
        url = url.strip() if isinstance(url, str) else ""
        if not url:
            return ToolResult(observation="The URL is empty; nothing was fetched.", private={"empty_url": True})
        doc, problem, private = load_web(self.env, url, self.params)
        if doc is None:
            return ToolResult(observation=problem, private=private)
        number, page_problem = page_choice(page, len(doc.pages))
        private.update(pages=len(doc.pages), page=number)
        if page_problem:
            return ToolResult(observation=f"{page_problem} ({doc.label})", private=private)
        return ToolResult(observation=page_observation(doc, number, "web page"), private=private)


# --------------------------------------------------------------------- read_file


class ReadFileTool(Tool):
    name = "read_file"
    allowed_params: set[str] = {"page_chars"}

    @property
    def description(self) -> str:
        attachment = attachment_of(self.env)
        pages = pages_sentence(_page_budget(self.params))
        kinds = (
            f"It can show {describe_supported()}. An Excel file is shown sheet by sheet as tables whose header row "
            "gives the column letters and whose first column gives the row numbers, so B7 is column B of row 7; a "
            "cell holds the value stored in the file (for a formula, the value Excel last computed). A PDF's pages "
            "are each headed \"[PDF page k of n]\"."
        )
        if attachment is None:
            return ("Returns one page of the text of the file attached to a question. The current question has no "
                    "attached file, so this tool returns only that statement.")
        return (f"Returns one page of the text of the file attached to the question, {attachment.name}. {kinds} "
                f"{pages}. The result begins with \"Page k of n\"; to read another page, call the tool again with "
                "that page number.")

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"page": {"type": "integer", "description": "Which page of the text to return, counting from 1. Default 1."}},
            "required": [],
        }

    def call(self, page: int | None = None, **unexpected) -> ToolResult:
        _check_unexpected(self.name, unexpected)
        doc, problem, private = load_attachment(self.env, self.params)
        if doc is None:
            return ToolResult(observation=problem, private=private)
        number, page_problem = page_choice(page, len(doc.pages))
        private.update(pages=len(doc.pages), page=number)
        if page_problem:
            return ToolResult(observation=f"{page_problem} ({doc.label})", private=private)
        return ToolResult(observation=page_observation(doc, number, "file"), private=private)


# --------------------------------------------------------------------- find_in_page


def _query_pattern(text: str) -> re.Pattern:
    """Case-insensitive; any run of whitespace in the query matches any run
    of whitespace in the document."""
    words = text.split()
    return re.compile(r"\s+".join(re.escape(w) for w in words), re.IGNORECASE)


def _snippet(text: str, start: int, end: int, side: int) -> str:
    """Up to `side` characters on each side of a match, its line breaks kept
    (a row of a table or a PDF stays on its own line), other runs of spaces
    collapsed; the match marked «like this». Lines after the first are
    indented to sit under it in a numbered list."""
    left, right = max(0, start - side), min(len(text), end + side)

    def tidy(part: str) -> str:
        return re.sub(r"\n\s*\n+", "\n", re.sub(r"[ \t\r\f\v]+", " ", part))

    body = (("…" if left > 0 else "") + tidy(text[left:start]) + "«" + " ".join(text[start:end].split()) + "»"
            + tidy(text[end:right]) + ("…" if right < len(text) else ""))
    return "\n   ".join(line.strip() for line in body.strip().split("\n"))


class FindInPageTool(Tool):
    """Uses `visit_webpage`'s and `read_file`'s parameters for fetching and
    paging (the dataset sets them on those tools), so a document is split into
    the same pages whichever tool loads it first."""

    name = "find_in_page"
    allowed_params: set[str] = {"max_matches", "snippet_chars"}

    def _source_params(self, kind: str) -> dict:
        all_params = getattr(self.env, "tool_params", None) or {}
        return all_params.get("visit_webpage" if kind == "web" else "read_file", {})

    @property
    def description(self) -> str:
        attachment = attachment_of(self.env)
        what = "a web page (by its URL)"
        if attachment is not None:
            what += f" or the attached file (by its name, {attachment.name})"
        limit = int(self.params.get("max_matches", DEFAULT_MAX_MATCHES))
        return (
            f"Searches the whole text of {what} for a word or phrase, matched as plain text (not as a pattern), "
            f"ignoring letter case and treating any run of spaces and line breaks as one space. Returns how many times it occurs and, for the first {limit} "
            "occurrences, the page number on which each begins, with some text around it (its line breaks kept); the "
            "match itself is marked «like this». The page numbers are those the page or file is shown with when you read it. A web "
            "page not read before is fetched first."
        )

    @property
    def parameters(self) -> dict:
        attachment = attachment_of(self.env)
        source = "The URL of the web page" + (f", or {attachment.name} for the attached file" if attachment else "") + "."
        return {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": source},
                "text": {"type": "string", "description": f"The word or phrase to look for (at most {MAX_QUERY_CHARS} characters)."},
            },
            "required": ["source", "text"],
        }

    def call(self, source: str = "", text: str = "", **unexpected) -> ToolResult:
        _check_unexpected(self.name, unexpected)
        source = source.strip() if isinstance(source, str) else ""
        query = " ".join(text.split()) if isinstance(text, str) else ""
        if not source:
            return ToolResult(observation="The source is empty; nothing was searched.", private={"empty_source": True})
        if not query:
            return ToolResult(observation="The text to look for is empty; nothing was searched.", private={"empty_query": True})
        if len(query) > MAX_QUERY_CHARS:
            return ToolResult(observation=f"The text to look for has {len(query)} characters; at most {MAX_QUERY_CHARS} are "
                                          "allowed. Nothing was searched.", private={"query_too_long": len(query)})
        attachment = attachment_of(self.env)
        if attachment is not None and source == attachment.name:
            doc, problem, private = load_attachment(self.env, self._source_params("file"))
        else:
            doc, problem, private = load_web(self.env, source, self._source_params("web"))
        private["query"] = query
        if doc is None:
            return ToolResult(observation=problem, private=private)

        full = doc.text
        starts, pos = [], 0
        for page in doc.pages:
            starts.append(pos)
            pos += len(page)
        matches = list(_query_pattern(query).finditer(full))
        limit = int(self.params.get("max_matches", DEFAULT_MAX_MATCHES))
        side = int(self.params.get("snippet_chars", DEFAULT_SNIPPET_CHARS))
        private.update(matches=len(matches), pages=len(doc.pages))
        if not matches:
            return ToolResult(observation=f'"{query}" does not occur in {doc.label} ({len(doc.pages)} '
                                          f'page{"s" if len(doc.pages) != 1 else ""}).', private=private)

        lines = []
        for k, m in enumerate(matches[:limit], 1):
            page = bisect.bisect_right(starts, m.start())
            lines.append(f"{k}. page {page}: {_snippet(full, m.start(), m.end(), side)}")
        pages_hit = sorted({bisect.bisect_right(starts, m.start()) for m in matches})
        private["pages_with_matches"] = pages_hit
        n = len(doc.pages)
        count = (f'"{query}" occurs {len(matches)} time{"s" if len(matches) != 1 else ""} in {doc.label} '
                 f'({n} page{"s" if n != 1 else ""})')
        if len(matches) > limit:
            shown = ", ".join(map(str, pages_hit[:40])) + (" …" if len(pages_hit) > 40 else "")
            count += f". The first {limit} are shown; occurrences begin on pages {shown}"
        return ToolResult(observation=count + ":\n\n" + "\n".join(lines), private=private)
