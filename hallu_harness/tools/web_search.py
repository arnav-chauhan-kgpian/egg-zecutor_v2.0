"""`web_search` — a web search through `ddgs`, the library behind
smolagents' DuckDuckGoSearchTool (the search tool of `mistral(2).ipynb`).

`ddgs` reports "No results found." both for a query that has none and for a
search engine that refused or failed to answer; from thor most of its engines
answer every query that way (only `auto` and `yahoo` worked on 2026-10-27).
So an empty answer is checked with a probe query that always has results: if
the probe is empty too, the search engine is down, which is the harness's
failure and is raised; otherwise the model is told its query found nothing.
"""
from __future__ import annotations

import time

from ..core.types import ToolResult
from .base import Tool, ToolArgumentError
from .web.http import BLOCKED_SITES, NetworkDown, blocked, require_network

DEFAULT_MAX_RESULTS = 10
PROBE_QUERY = "python programming language"


class WebSearchTool(Tool):
    name = "web_search"
    allowed_params: set[str] = {"max_results", "backend", "region", "timeout_s", "blocked_url_prefixes"}

    @property
    def max_results(self) -> int:
        return int(self.params.get("max_results", DEFAULT_MAX_RESULTS))

    @property
    def description(self) -> str:
        text = (
            f"Searches the web and returns up to {self.max_results} results, each with its title, URL and a short "
            "snippet of the page. The results come from a public search engine and show the web as it is today."
        )
        if self.params.get("blocked_url_prefixes"):
            text += f" Pages of {BLOCKED_SITES}, are left out of the results."
        return text

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for."}},
            "required": ["query"],
        }

    def _search(self, query: str) -> list[dict]:
        """The engine's results; [] if it found nothing. Raises when it fails."""
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

        backend = self.params.get("backend", "auto")
        region = self.params.get("region", "us-en")
        timeout = float(self.params.get("timeout_s", 10))
        for attempt in range(3):
            try:
                return DDGS(timeout=timeout).text(query, region=region, max_results=self.max_results, backend=backend)
            except (RatelimitException, TimeoutException) as exc:
                failure = exc
            except DDGSException as exc:
                if "no results" in str(exc).lower():
                    return []
                failure = exc
            time.sleep(2 * (attempt + 1))
        require_network("web search")
        raise RuntimeError(f"the search engine failed three times: {type(failure).__name__}: {failure}")

    def call(self, query: str = "", **unexpected) -> ToolResult:
        if unexpected:
            raise ToolArgumentError(f"web_search got unexpected argument(s): {', '.join(sorted(unexpected))}")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(observation="The query is empty; nothing was searched.", private={"empty_query": True})
        started = time.perf_counter()
        raw = self._search(query.strip())
        if not raw:
            # An engine that is down answers "no results" to everything.
            require_network("web search")
            if not self._search(PROBE_QUERY):
                raise NetworkDown(f"the search engine found nothing for the probe query {PROBE_QUERY!r} either")
        prefixes = self.params.get("blocked_url_prefixes") or []
        kept = [r for r in raw if not blocked(r.get("href", ""), prefixes)]
        private = {
            "backend": self.params.get("backend", "auto"),
            "raw_results": len(raw),
            "blocked": len(raw) - len(kept),
            "urls": [r.get("href") for r in kept],
            "seconds": round(time.perf_counter() - started, 3),
        }
        if not kept:
            return ToolResult(observation="No results were found for this query.", private=private)
        lines = []
        for n, r in enumerate(kept, 1):
            lines.append(f"{n}. {(r.get('title') or '(no title)').strip()}\n   {r.get('href', '')}\n   {(r.get('body') or '').strip()}")
        return ToolResult(observation="\n\n".join(lines), private=private)
