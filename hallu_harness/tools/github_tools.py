"""The GitHub tools of `mistral(2).ipynb` — repository search, code search,
reading a file — on GitHub's REST API.

What changed from the notebook:

* The token comes from an environment variable (`token_env`, default
  `GITHUB_TOKEN`), never from the code. Code search refuses anonymous
  requests; the other two work without a token, at a low rate limit.
* Results are numbered text, not a Python list's repr.
* A file longer than `page_chars` is shown in numbered pages; a directory
  is listed; a binary file is stated as such instead of decoded to noise.
* A query GitHub rejects (HTTP 422) or a path that does not exist is told
  to the model; a rate limit is waited out or, if it resets too late, raised
  as the harness's failure, with a rejected token or GitHub being down
  (`web/http.py`).
* `blocked_repos` leaves repositories out of results and refuses to read
  them; the descriptions say so when the list is not empty.
* A repository description longer than `description_chars` is cut, and the
  result says so. GitHub's API returns descriptions of any length, and spam
  repositories with 26k-230k-character descriptions match many queries: one
  such result overflowed the model's context window.
"""
from __future__ import annotations

import base64
import time
from urllib.parse import quote

from ..core.errors import ConfigError
from ..core.types import ToolResult
from .base import Tool, ToolArgumentError
from .web.http import BLOCKED_REPOS, REPO_NAME, GitHubAPI, UrlProblem, fetch, page_choice, split_pages

DEFAULT_MAX_RESULTS = 10
DEFAULT_PAGE_CHARS = 10_000
DEFAULT_MAX_BYTES = 10 << 20
DEFAULT_DESCRIPTION_CHARS = 500  # GitHub's site allows 350; real descriptions seen run to ~1.2k
COMMON_PARAMS = {"token_env", "timeout_s", "max_wait_s", "blocked_repos"}


def cut_description(text: str, limit: int) -> str:
    """`text` cut at a space at most `limit` characters in, marked as cut."""
    if len(text) <= limit:
        return text
    end = text.rfind(" ", limit // 2, limit)
    return f"{text[:end if end >= 0 else limit].rstrip()} … (cut; {len(text)} characters in all)"


class _GitHubTool(Tool):
    requires_token = False

    @classmethod
    def check_environment(cls, params: dict) -> str | None:
        """Called by `plugins.check_config` before a run: raises if the tool
        cannot work, returns a warning if it will work poorly."""
        import os

        env = params.get("token_env", "GITHUB_TOKEN")
        if os.environ.get(env):
            return None
        if cls.requires_token:
            raise ConfigError(f"{cls.name} needs a GitHub token in ${env}: GitHub's code search API refuses anonymous requests")
        return (f"{cls.name}: ${env} is not set, so GitHub allows 60 requests an hour (10 searches a minute); a longer run "
                "will stop on the rate limit (those trajectories are recorded as crashed and re-run on resume)")

    def api(self) -> GitHubAPI:
        return GitHubAPI(self.params.get("token_env", "GITHUB_TOKEN"), float(self.params.get("timeout_s", 30)),
                         float(self.params.get("max_wait_s", 120)))

    def _blocked(self, full_name: str) -> bool:
        return full_name.lower() in {r.lower() for r in self.params.get("blocked_repos") or []}

    def _with_blocklist(self, text: str, effect: str = "are left out of the results") -> str:
        return text + (f" {BLOCKED_REPOS[0].upper()}{BLOCKED_REPOS[1:]}, {effect}." if self.params.get("blocked_repos") else "")

    @staticmethod
    def _rejected(response) -> str:
        try:
            body = response.json()
        except ValueError:
            return f"HTTP {response.status_code}"
        details = [e.get("message") for e in body.get("errors", []) if isinstance(e, dict) and e.get("message")]
        return "; ".join([str(body.get("message", f"HTTP {response.status_code}")), *details])

    def _search(self, kind: str, query, unexpected: dict) -> tuple[ToolResult | None, dict | None, dict]:
        if unexpected:
            raise ToolArgumentError(f"{self.name} got unexpected argument(s): {', '.join(sorted(unexpected))}")
        private: dict = {"query": query}
        if not isinstance(query, str) or not query.strip():
            return ToolResult(observation="The query is empty; nothing was searched.", private=private), None, private
        api = self.api()
        started = time.perf_counter()
        response = api.get(f"/search/{kind}", {"q": query.strip(), "per_page": int(self.params.get("max_results", DEFAULT_MAX_RESULTS))})
        private.update(status=response.status_code, rate_limit_wait_s=round(api.waited_s, 1), seconds=round(time.perf_counter() - started, 3))
        if response.status_code in (400, 422):
            return ToolResult(observation=f"GitHub rejected the query: {self._rejected(response)}.", private=private), None, private
        if response.status_code >= 400:
            raise RuntimeError(f"GitHub search answered HTTP {response.status_code}: {self._rejected(response)}")
        return None, response.json(), private


class GithubRepositorySearchTool(_GitHubTool):
    name = "github_repository_search"
    allowed_params: set[str] = COMMON_PARAMS | {"max_results", "description_chars"}

    @property
    def description_chars(self) -> int:
        return int(self.params.get("description_chars", DEFAULT_DESCRIPTION_CHARS))

    @property
    def description(self) -> str:
        n = int(self.params.get("max_results", DEFAULT_MAX_RESULTS))
        return self._with_blocklist(
            f"Searches public GitHub repositories with GitHub's repository search and returns up to {n} of the best "
            "matches, each with its full name (owner/name), its number of stars, its description (cut at "
            f"{self.description_chars} characters) and its URL. The query uses GitHub's search syntax; for example, "
            "dicom viewer language:python finds Python repositories about DICOM viewers."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The repository search query."}},
            "required": ["query"],
        }

    def call(self, query: str = "", **unexpected) -> ToolResult:
        early, data, private = self._search("repositories", query, unexpected)
        if early is not None:
            return early
        items = data.get("items", [])
        kept = [r for r in items if not self._blocked(r.get("full_name", ""))]
        private.update(total_count=data.get("total_count"), returned=len(items), blocked=len(items) - len(kept),
                       repositories=[r.get("full_name") for r in kept], cut_descriptions={})
        if not kept:
            return ToolResult(observation="GitHub found no repositories matching this query.", private=private)
        lines = [f"GitHub reports {data.get('total_count', len(kept))} matching repositories. The best matches:"]
        for n, r in enumerate(kept, 1):
            description = (r.get("description") or "").strip() or "(no description)"
            if len(description) > self.description_chars:
                private["cut_descriptions"][r.get("full_name")] = len(description)  # full length, for analysis
                description = cut_description(description, self.description_chars)
            lines.append(f"{n}. {r.get('full_name')} ({r.get('stargazers_count', 0)} stars)\n   {description}\n   {r.get('html_url')}")
        return ToolResult(observation="\n\n".join(lines), private=private)


class GithubCodeSearchTool(_GitHubTool):
    name = "github_code_search"
    allowed_params: set[str] = COMMON_PARAMS | {"max_results"}
    requires_token = True

    @property
    def description(self) -> str:
        n = int(self.params.get("max_results", DEFAULT_MAX_RESULTS))
        return self._with_blocklist(
            "Searches the contents of files in public GitHub repositories with GitHub's code search and returns up "
            f"to {n} matching files, each with its repository, its path and its URL. Only each repository's default "
            "branch is searched. The query uses GitHub's code search syntax; for example, import pydicom language:python "
            "finds Python files that import pydicom. The qualifiers repo:, path:, filename:, extension: and language: "
            "narrow a search."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The code search query."}},
            "required": ["query"],
        }

    def call(self, query: str = "", **unexpected) -> ToolResult:
        early, data, private = self._search("code", query, unexpected)
        if early is not None:
            return early
        items = data.get("items", [])
        kept = [i for i in items if not self._blocked((i.get("repository") or {}).get("full_name", ""))]
        private.update(total_count=data.get("total_count"), returned=len(items), blocked=len(items) - len(kept),
                       files=[f"{(i.get('repository') or {}).get('full_name')}/{i.get('path')}" for i in kept])
        if not kept:
            return ToolResult(observation="GitHub found no files matching this query.", private=private)
        lines = [f"GitHub reports {data.get('total_count', len(kept))} matching files. The best matches:"]
        for n, i in enumerate(kept, 1):
            lines.append(f"{n}. {(i.get('repository') or {}).get('full_name')}: {i.get('path')}\n   {i.get('html_url')}")
        return ToolResult(observation="\n\n".join(lines), private=private)


class GithubGetFileTool(_GitHubTool):
    name = "github_get_file"
    allowed_params: set[str] = COMMON_PARAMS | {"page_chars", "max_bytes"}

    def __init__(self, env, **params):
        super().__init__(env, **params)
        self._pages: dict[tuple[str, str, str], tuple[list[str], bool]] = {}  # (pages, cut at max_bytes), kept like fetch_webpage's

    @property
    def page_chars(self) -> int:
        return int(self.params.get("page_chars", DEFAULT_PAGE_CHARS))

    @property
    def description(self) -> str:
        return self._with_blocklist(
            "Reads a file from the default branch of a public GitHub repository and returns its text, split into "
            f'pages of at most {self.page_chars} characters; the result begins with "Page k of n". If the path is a '
            "directory, it returns the directory's entries instead (an empty path lists the repository's top "
            "directory). Binary files are not shown.",
            effect="cannot be read",
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "The repository's owner (user or organization)."},
                "repo": {"type": "string", "description": "The repository's name."},
                "path": {"type": "string", "description": "The file's path in the repository, e.g. src/module.py."},
                "page": {"type": "integer", "description": "Which page of the file to return, counting from 1. Default 1."},
            },
            "required": ["owner", "repo", "path"],
        }

    def call(self, owner: str = "", repo: str = "", path: str = "", page: int | None = None, **unexpected) -> ToolResult:
        if unexpected:
            raise ToolArgumentError(f"github_get_file got unexpected argument(s): {', '.join(sorted(unexpected))}")
        owner, repo = str(owner or "").strip(), str(repo or "").strip()
        path = str(path or "").strip().strip("/")
        where = f"{owner}/{repo}" + (f"/{path}" if path else "")
        private: dict = {"owner": owner, "repo": repo, "path": path}
        if not REPO_NAME.match(owner) or not REPO_NAME.match(repo):
            return ToolResult(observation=f"{owner!r}/{repo!r} is not a valid GitHub owner and repository name.", private=private)
        if self._blocked(f"{owner}/{repo}"):
            private["blocked"] = True
            return ToolResult(observation=f"{owner}/{repo} cannot be read in this task: it is one of the {BLOCKED_REPOS}.",
                              private=private)

        key = (owner.lower(), repo.lower(), path)
        private["cached"] = key in self._pages
        if key not in self._pages:
            api = self.api()
            response = api.get(f"/repos/{owner}/{repo}/contents/{quote(path, safe='/')}")
            private.update(status=response.status_code, rate_limit_wait_s=round(api.waited_s, 1))
            if response.status_code == 404:
                return ToolResult(observation=f"There is no file or directory {path or '/'} in {owner}/{repo}, or the "
                                              "repository does not exist or is not public.", private=private)
            if response.status_code >= 400:
                raise RuntimeError(f"GitHub contents API answered HTTP {response.status_code}: {self._rejected(response)}")
            data = response.json()
            if isinstance(data, list):
                entries = sorted(data, key=lambda e: (e.get("type") != "dir", e.get("name", "").lower()))
                listing = "\n".join(f"- {e.get('name')}{'/' if e.get('type') == 'dir' else ''}" for e in entries)
                private["entries"] = len(entries)
                return ToolResult(observation=f"{where} is a directory with {len(entries)} entries:\n{listing}", private=private)
            kind = data.get("type")
            if kind != "file":
                target = data.get("target") or data.get("submodule_git_url")
                return ToolResult(observation=f"{where} is a {kind}" + (f" pointing to {target}" if target else "") + ", not a file.",
                                  private=private)
            body = self._file_bytes(data, private)
            if body is None:
                return ToolResult(observation=f"{where} could not be downloaded from GitHub.", private=private)
            if b"\x00" in body[:8000]:
                private["binary"] = True
                return ToolResult(observation=f"{where} is a binary file of {data.get('size', len(body))} bytes; its content is not shown.",
                                  private=private)
            self._pages[key] = (split_pages(body.decode("utf-8", errors="replace"), self.page_chars), bool(private.get("truncated")))

        pages, truncated = self._pages[key]
        number, problem = page_choice(page, len(pages))
        private.update(pages=len(pages), page=number)
        if problem:
            return ToolResult(observation=f"{problem} ({where})", private=private)
        note = ""
        if truncated and number == len(pages):
            limit_mb = int(self.params.get("max_bytes", DEFAULT_MAX_BYTES)) / (1 << 20)
            note = f"\n\n(Only the first {limit_mb:g} MB of this file were downloaded; the rest is missing.)"
        return ToolResult(observation=f"Page {number} of {len(pages)} of {where}\n\n{pages[number - 1]}{note}", private=private)

    def _file_bytes(self, data: dict, private: dict) -> bytes | None:
        """The file's content: inline (base64) up to 1 MB; larger files have
        none inline and are downloaded from their raw URL."""
        if data.get("encoding") == "base64" and data.get("content"):
            return base64.b64decode(data["content"])
        if data.get("size") == 0:
            return b""
        url = data.get("download_url")
        if not url:
            return None
        try:
            got = fetch(url, timeout=float(self.params.get("timeout_s", 30)), max_bytes=int(self.params.get("max_bytes", DEFAULT_MAX_BYTES)))
        except UrlProblem as problem:
            private["problem"] = str(problem)
            return None
        private.update(raw_status=got.status, truncated=got.truncated)
        return got.body if got.status < 400 else None
