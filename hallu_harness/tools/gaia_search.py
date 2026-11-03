"""`web_search` for GAIA (registered as `gaia_web_search`): web search with
every result cut to a bounded length, through one of two backends:

* `serper` (the default): Google's results through the Serper API
  (https://serper.dev), with the key in the environment variable named by
  `api_key_env` (default `SERPER_API_KEY`; never in a config). On 2026-10-29
  it answered in about 2 s per query.
* `ddgs`: free public engines through `ddgs`, tried in turn (below). Slower
  (up to 25 s when engines fail over) and different from run to run.

The HALoGEN `web_search` (`tools/web_search.py`) is kept as it is, since
recorded HALoGEN runs depend on its exact text, but it does not fit GAIA:

* it asks one `ddgs` backend (`auto`). On 2026-10-29 from thor that answered
  "No results found." for 9 of 15 ordinary queries, and the tool told the
  model its query had found nothing whenever a probe query still worked;
* it prints titles and snippets uncut: one query returned 13,000 characters
  (8,269 tokens) of snippets.

With `ddgs`, the engines in `engines` are asked in order until one returns results
(each up to `attempts` times), and the observation names nothing about
engines: it lists results. If none returns any, a probe query that always
has results is sent to the same engines; if the probe also gets nothing,
the search engines are unreachable, which is the harness's failure (the
trajectory is recorded as crashed and re-run). Otherwise the model is told
that the search engines returned no results for its query, which is what
happened.

With `serper`, the API says explicitly whether it failed, so an empty result
list is the search's answer and the model is told so. A rejected key, a rate
limit that outlasts the retries, or an unreachable API is the harness's
failure: raised, so the trajectory is recorded as crashed and re-run on
resume, never shown to the model.

Spent Serper credits fall back to `ddgs` (`fallback: ddgs`, the default;
`fallback: none` raises instead). A 400/402/403 answer is taken as spent
credits when Serper's account endpoint then reports a balance of 0 or less
(the error's own wording is not relied on). From then on, for the rest of the
process, searches go to `ddgs` without asking Serper again. The description
says this can happen, and every fallback result begins with a note saying
the results come from public search engines, so the model is never told
"Google" for results that are not. `private.backend` records which backend
answered each call.

The tool's name is `web_search`, the name models know from smolagents and
Qwen-Agent; its registry name is `gaia_web_search` because the HALoGEN tool
holds `web_search`. The runner hands a tool the parameters stored under its
model-facing name, so this tool reads its own from the task environment
(`env.tool_params["gaia_web_search"]`), which the GAIA benchmark provides.
"""
from __future__ import annotations

import time

from ..core.types import ToolResult
from .base import Tool, ToolArgumentError
from .web.http import BLOCKED_SITES, NetworkDown, blocked, require_network

REGISTRY_NAME = "gaia_web_search"
DEFAULT_ENGINES = ("yahoo", "bing", "google", "yandex")  # 7, 5, 4, 6 of 12 queries on 2026-10-29; mojeek, brave, duckduckgo 0
DEFAULT_MAX_RESULTS = 8
DEFAULT_TITLE_CHARS = 150
DEFAULT_SNIPPET_CHARS = 300
PROBE_QUERY = "python programming language"
SERPER_URL = "https://google.serper.dev/search"
SERPER_ACCOUNT_URL = "https://google.serper.dev/account"
FALLBACK_NOTE = ("Note: the Serper API's search credits are used up, so these results come from public search engines "
                 "instead of Google.")
DEFAULT_KEY_ENV = "SERPER_API_KEY"
BACKENDS = ("serper", "ddgs")


class SerperCreditsExhausted(RuntimeError):
    """The Serper account has no credits left."""


def cut(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class GaiaWebSearchTool(Tool):
    name = "web_search"
    # Set once Serper reports spent credits; later searches in this process go
    # straight to the fallback.
    serper_exhausted = False
    allowed_params: set[str] = {"backend", "fallback", "api_key_env", "gl", "hl", "engines", "attempts", "max_results", "timeout_s",
                                "region", "title_chars", "snippet_chars", "blocked_url_prefixes"}

    def __init__(self, env, **params):
        own = (getattr(env, "tool_params", None) or {}).get(REGISTRY_NAME, {})
        super().__init__(env, **{**own, **params})

    @property
    def max_results(self) -> int:
        return int(self.params.get("max_results", DEFAULT_MAX_RESULTS))

    @property
    def backend(self) -> str:
        backend = self.params.get("backend", "serper")
        if backend not in BACKENDS:
            raise RuntimeError(f"gaia_web_search: unknown backend {backend!r}; known: {', '.join(BACKENDS)}")
        return backend

    @property
    def fallback(self) -> str | None:
        fallback = self.params.get("fallback", "ddgs")
        if fallback not in ("ddgs", "none", None):
            raise RuntimeError(f"gaia_web_search: unknown fallback {fallback!r}; known: ddgs, none")
        return None if fallback in ("none", None) else fallback

    @property
    def description(self) -> str:
        snippet = int(self.params.get("snippet_chars", DEFAULT_SNIPPET_CHARS))
        title = int(self.params.get("title_chars", DEFAULT_TITLE_CHARS))
        if self.backend != "serper":
            source = "The results come from public search engines"
        elif self.fallback:
            source = ("The results are Google's, obtained through the Serper API (if its search credits run out, they "
                      "come from public search engines instead, and the result begins with a note saying so),")
        else:
            source = "The results are Google's, obtained through the Serper API,"
        text = (
            f"Searches the web and returns up to {self.max_results} results, each with its title, URL and a short "
            f"snippet of the page. A title longer than {title} characters or a snippet longer than {snippet} is cut, and "
            f"ends with \"…\". {source} and show the web as it is today. To read a result, visit its URL."
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

    @classmethod
    def check_environment(cls, params: dict) -> str | None:
        import os

        if params.get("backend", "serper") == "serper":
            name = params.get("api_key_env", DEFAULT_KEY_ENV)
            if not os.environ.get(name):
                return (f"gaia_web_search uses Serper but ${name} is not set; every search will end its trajectory as "
                        "crashed (set it, or use backend: ddgs)")
            if params.get("fallback", "ddgs") != "ddgs":
                return None
        try:
            import ddgs  # noqa: F401
        except ImportError:
            return "gaia_web_search needs the ddgs package (pip install ddgs); every search will end its trajectory as crashed"
        return None

    # -- serper ----------------------------------------------------------------

    def _post(self, payload: dict, key: str):
        import requests

        return requests.post(SERPER_URL, json=payload, headers={"X-API-KEY": key, "Content-Type": "application/json"},
                             timeout=float(self.params.get("timeout_s", 20)))

    def _balance(self, key: str) -> float | None:
        """The Serper account's remaining credits, or None if it cannot be read."""
        import requests

        try:
            response = requests.get(SERPER_ACCOUNT_URL, headers={"X-API-KEY": key}, timeout=float(self.params.get("timeout_s", 20)))
            return float(response.json()["balance"]) if response.status_code == 200 else None
        except (requests.RequestException, ValueError, KeyError, TypeError):
            return None

    def _serper(self, query: str) -> tuple[list[dict], dict]:
        """Organic results as {title, href, body}, and what to record. Raises
        for anything that is not the search's own answer."""
        import os

        import requests

        name = self.params.get("api_key_env", DEFAULT_KEY_ENV)
        key = os.environ.get(name)
        if not key:
            raise RuntimeError(f"the Serper API key is not set (${name})")
        payload = {"q": query, "num": max(10, self.max_results), "gl": self.params.get("gl", "us"), "hl": self.params.get("hl", "en")}
        tries = max(3, int(self.params.get("attempts", 2)) + 1)
        attempts: list[dict] = []
        for attempt in range(tries):
            started = time.perf_counter()
            try:
                response = self._post(payload, key)
            except requests.RequestException as exc:
                attempts.append({"error": f"{type(exc).__name__}: {exc}"[:200], "seconds": round(time.perf_counter() - started, 3)})
                if attempt + 1 < tries:
                    time.sleep(2 ** attempt)
                    continue
                require_network("web search (Serper)")
                raise RuntimeError(f"the Serper API did not answer: {attempts[-1]['error']}") from None
            attempts.append({"status": response.status_code, "seconds": round(time.perf_counter() - started, 3)})
            if response.status_code == 200:
                data = response.json()
                results = [{"title": r.get("title"), "href": r.get("link"),
                            "body": " — ".join(x for x in (r.get("date"), r.get("snippet")) if x)}
                           for r in data.get("organic", []) if r.get("link")]
                return results, {"backend": "serper", "attempts": attempts, "credits": data.get("credits")}
            message = response.text[:200]
            if response.status_code in (400, 402, 403):
                balance = self._balance(key)
                if balance is not None and balance <= 0:
                    raise SerperCreditsExhausted(f"the Serper account has no credits left (balance {balance:g}; HTTP "
                                                 f"{response.status_code}: {message})")
            if response.status_code in (401, 403):
                raise RuntimeError(f"Serper rejected the key in ${name} (HTTP {response.status_code}): {message}")
            if response.status_code in (429, 500, 502, 503, 504) and attempt + 1 < tries:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(f"the Serper API answered HTTP {response.status_code}: {message}")
        raise AssertionError("unreachable")

    # -- ddgs ------------------------------------------------------------------

    def _engine(self, engine: str, query: str) -> tuple[list[dict], str | None]:
        """(results, None), or ([], why it gave none)."""
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException

        try:
            return DDGS(timeout=float(self.params.get("timeout_s", 10))).text(
                query, region=self.params.get("region", "us-en"), max_results=self.max_results, backend=engine) or [], None
        except DDGSException as exc:
            return [], f"{type(exc).__name__}: {exc}"

    def _search(self, query: str) -> tuple[list[dict], str | None, list[dict]]:
        """Results from the first engine that returns any, that engine, and every attempt made."""
        attempts = []
        for round_ in range(int(self.params.get("attempts", 2))):
            for engine in self.params.get("engines", DEFAULT_ENGINES):
                started = time.perf_counter()
                results, problem = self._engine(engine, query)
                attempts.append({"engine": engine, "round": round_, "results": len(results), "problem": problem,
                                 "seconds": round(time.perf_counter() - started, 3)})
                if results:
                    return results, engine, attempts
            if round_ + 1 < int(self.params.get("attempts", 2)):
                time.sleep(2)
        return [], None, attempts

    def call(self, query: str = "", **unexpected) -> ToolResult:
        if unexpected:
            raise ToolArgumentError(f"web_search got unexpected argument(s): {', '.join(sorted(unexpected))}")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(observation="The query is empty; nothing was searched.", private={"empty_query": True})
        query = " ".join(query.split())
        note = ""
        use_serper = self.backend == "serper" and not (self.fallback and GaiaWebSearchTool.serper_exhausted)
        exhausted_now = None
        if use_serper:
            try:
                raw, private = self._serper(query)
            except SerperCreditsExhausted as exc:
                if not self.fallback:
                    raise
                GaiaWebSearchTool.serper_exhausted = True
                use_serper, exhausted_now = False, str(exc)
        if use_serper:
            private["raw_results"] = len(raw)
            if not raw:
                return ToolResult(observation="The search engine returned no results for this query.", private=private)
        else:
            raw, engine, attempts = self._search(query)
            private = {"backend": "ddgs", "engine": engine, "attempts": attempts, "raw_results": len(raw)}
            if self.backend == "serper":
                note = FALLBACK_NOTE + "\n\n"
                private["fallback_from"] = "serper"
                if exhausted_now:
                    private["serper_error"] = exhausted_now
        result = self._finish(query, raw, private)
        if note:
            result.observation = note + result.observation
        return result

    def _finish(self, query: str, raw: list[dict], private: dict) -> ToolResult:
        if not raw:
            require_network("web search")
            probe, probe_engine, probe_attempts = self._search(PROBE_QUERY)
            private["probe"] = {"engine": probe_engine, "attempts": probe_attempts}
            if not probe:
                raise NetworkDown(f"no search engine returned results, not even for the probe query {PROBE_QUERY!r}")
            return ToolResult(observation="The search engines returned no results for this query.", private=private)

        prefixes = self.params.get("blocked_url_prefixes") or []
        kept = [r for r in raw if not blocked(r.get("href", ""), prefixes)][: self.max_results]
        private.update(blocked=len(raw) - len(kept), urls=[r.get("href") for r in kept])
        if not kept:
            source = "search engine" if private.get("backend") == "serper" else "search engines"
            return ToolResult(observation=f"The {source} returned no results for this query that can be shown in this task.",
                              private=private)
        title_chars = int(self.params.get("title_chars", DEFAULT_TITLE_CHARS))
        snippet_chars = int(self.params.get("snippet_chars", DEFAULT_SNIPPET_CHARS))
        lines = []
        for n, r in enumerate(kept, 1):
            url = r.get("href") or ""  # never cut: a cut URL cannot be visited
            lines.append(f"{n}. {cut(r.get('title') or '(no title)', title_chars)}\n   {url}\n   "
                         f"{cut(r.get('body') or '', snippet_chars) or '(no snippet)'}")
        return ToolResult(observation="\n\n".join(lines), private=private)
