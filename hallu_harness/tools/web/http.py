"""HTTP for the web and GitHub tools, and the rule that separates the
model's mistakes from the harness's failures.

A request that fails is one of two things:

* a fact about what the model asked for: an HTTP error status, a host that
  does not exist, a server that does not answer in time. The tool states it
  in its observation (`UrlProblem`).
* a failure of this machine's network, or of a service's quota or
  credentials. The model did not cause it, so it is raised; `Tool.run` turns
  it into `HarnessAbort`, the trajectory is recorded as crashed and re-run on
  resume, and the model never sees it.

After a connection failure or a timeout, two well-known hosts are probed
(`network_up`). If neither answers, the network is down, and the failure is
the harness's.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

# The tools say what they are; they do not pose as a browser. Sites that
# refuse scripted clients (Stack Overflow answers 403, pypi.org project
# pages serve a JavaScript challenge) are reported to the model as they answer.
USER_AGENT = "hallu_harness/1.0 (research agent harness; python-requests)"
PROBE_URLS = ("https://www.python.org/", "https://github.com/")

GITHUB_API = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"


class NetworkDown(RuntimeError):
    """This machine cannot reach the internet: a harness failure."""


class UrlProblem(Exception):
    """A fact about the requested URL, stated to the model."""


def network_up(timeout: float = 10.0) -> bool:
    import requests

    for url in PROBE_URLS:
        try:
            requests.head(url, timeout=timeout, allow_redirects=True, headers={"User-Agent": USER_AGENT})
            return True
        except requests.RequestException:
            continue
    return False


def require_network(what: str) -> None:
    if not network_up():
        raise NetworkDown(f"{what}: this machine cannot reach the internet ({', '.join(PROBE_URLS)} do not answer)")


# ----------------------------------------------------------------- fetching


@dataclass
class Fetched:
    url: str  # after redirects
    status: int
    reason: str
    content_type: str  # media type, lower case, without parameters
    encoding: str | None  # the charset the Content-Type header names; None if it names none
    body: bytes
    truncated: bool  # the body was cut at max_bytes
    seconds: float


def _is_dns_failure(exc: Exception) -> bool:
    text = repr(exc)
    return any(s in text for s in ("NameResolutionError", "Name or service not known", "nodename nor servname",
                                   "No address associated with hostname", "getaddrinfo failed"))


def fetch(url: str, *, timeout: float, max_bytes: int, headers: dict | None = None) -> Fetched:
    """GET `url`, following redirects, reading at most `max_bytes` of the body.

    Raises `UrlProblem` for what the model should be told and `NetworkDown`
    when this machine is offline.
    """
    import requests

    if urlsplit(url).scheme not in ("http", "https"):
        raise UrlProblem("only http and https URLs can be fetched")
    started = time.perf_counter()
    try:
        with requests.get(url, timeout=timeout, stream=True, allow_redirects=True,
                          headers={"User-Agent": USER_AGENT, **(headers or {})}) as response:
            chunks, size, truncated = [], 0, False
            for chunk in response.iter_content(1 << 16):
                chunks.append(chunk)
                size += len(chunk)
                if size >= max_bytes:
                    truncated = True
                    break
            body = b"".join(chunks)[:max_bytes]
            media, *parameters = (response.headers.get("content-type") or "").split(";")
            # Not `response.encoding`: for text/* without a charset requests
            # assumes ISO-8859-1, which turns UTF-8 pages into mojibake.
            charset = next((value.strip().strip("\"'") for key, _, value in (p.partition("=") for p in parameters)
                            if key.strip().lower() == "charset"), None) or None
            return Fetched(url=response.url, status=response.status_code, reason=response.reason or "",
                           content_type=media.strip().lower(), encoding=charset, body=body, truncated=truncated,
                           seconds=time.perf_counter() - started)
    except (requests.exceptions.InvalidURL, requests.exceptions.MissingSchema, requests.exceptions.InvalidSchema):
        raise UrlProblem("it is not a valid http or https URL") from None
    except requests.exceptions.TooManyRedirects:
        raise UrlProblem("it redirects too many times") from None
    except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ContentDecodingError):
        require_network(f"fetching {url}")
        raise UrlProblem("the server's response was broken off or could not be decoded") from None
    except requests.exceptions.SSLError:
        require_network(f"fetching {url}")
        raise UrlProblem("the server's TLS certificate could not be verified") from None
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        require_network(f"fetching {url}")
        if _is_dns_failure(exc):
            raise UrlProblem(f"the host {urlsplit(url).hostname} could not be found") from None
        if isinstance(exc, requests.exceptions.Timeout) or "timed out" in repr(exc).lower():
            raise UrlProblem(f"the server did not answer within {timeout:g} seconds") from None
        raise UrlProblem("the connection to the server failed") from None


def decode(body: bytes, encoding: str | None) -> str:
    try:
        return body.decode(encoding or "utf-8", errors="replace")
    except LookupError:  # an encoding name Python does not know
        return body.decode("utf-8", errors="replace")


# ----------------------------------------------------------------- paging


def split_pages(text: str, page_chars: int) -> list[str]:
    """`text` in pages of at most `page_chars` characters, each ending at a
    line break when one falls in the page's second half."""
    pages, start = [], 0
    while start < len(text):
        end = min(start + page_chars, len(text))
        if end < len(text):
            cut = text.rfind("\n", start + page_chars // 2, end)
            if cut != -1:
                end = cut + 1
        pages.append(text[start:end])
        start = end
    return pages or [""]


def page_choice(page, count: int) -> tuple[int | None, str | None]:
    """The 1-based page asked for, or a message saying why there is none."""
    if page is None or page == "":
        return 1, None
    try:
        number = int(page)
    except (TypeError, ValueError):
        return None, f"The page number must be a whole number; got {page!r}."
    if number < 1 or number > count:
        return None, f"There {'is' if count == 1 else 'are'} {count} page{'' if count == 1 else 's'}; page {number} does not exist."
    return number, None


# ----------------------------------------------------------------- blocklists


def _host(netloc: str) -> str:
    host = netloc.lower().split("@")[-1].split(":")[0]
    return host[4:] if host.startswith("www.") else host


def blocked(url: str, prefixes: list[str]) -> bool:
    """Whether `url` starts with any of `prefixes`: same host (ignoring
    `www.` and case), and a path that starts with the prefix's path,
    ignoring case."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return False
    host, path = _host(parts.netloc), parts.path.lower()
    for prefix in prefixes:
        p = urlsplit(prefix if "://" in prefix else "https://" + prefix)
        if host == _host(p.netloc) and path.startswith(p.path.lower().rstrip("/")):
            return True
    return False


# What a blocklist keeps out, as the tool descriptions and observations say it.
BLOCKED_SITES = "sites that publish this task set, or other models' answers to it"
BLOCKED_REPOS = "repositories that publish this task set, or other models' answers to it"


# ----------------------------------------------------------------- GitHub


class GitHubAPI:
    """GitHub's REST API, with the token taken from an environment variable.

    Waiting out a rate limit is the harness's business: a limit that resets
    within `max_wait_s` is waited for (the wait is recorded); a longer one,
    a rejected token, or GitHub being unreachable is raised.
    """

    def __init__(self, token_env: str | None, timeout: float, max_wait_s: float):
        import os

        self.token_env = token_env
        self.token = os.environ.get(token_env) if token_env else None
        self.timeout = timeout
        self.max_wait_s = max_wait_s
        self.waited_s = 0.0

    def get(self, path: str, params: dict | None = None):
        import requests

        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": GITHUB_API_VERSION, "User-Agent": "hallu_harness"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        for attempt in range(4):
            try:
                response = requests.get(GITHUB_API + path, params=params, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                require_network("calling the GitHub API")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"the GitHub API did not answer: {type(exc).__name__}: {exc}") from None
            if response.status_code == 401:
                if not self.token:
                    raise RuntimeError(f"GitHub requires a token for {path}; set ${self.token_env}")
                raise RuntimeError(f"GitHub rejected the token in ${self.token_env} (HTTP 401)")
            wait = self._rate_limit_wait(response)
            if wait is None:
                return response
            if wait > self.max_wait_s or attempt == 3:
                raise RuntimeError(f"GitHub API rate limit reached; it resets in {wait:.0f} s (more than the "
                                   f"{self.max_wait_s:g} s this tool waits)")
            time.sleep(wait)
            self.waited_s += wait
        raise AssertionError("unreachable")

    @staticmethod
    def _rate_limit_wait(response) -> float | None:
        """Seconds to wait before retrying, or None if not rate-limited."""
        if response.status_code not in (403, 429):
            return None
        headers = response.headers
        message = ""
        try:
            message = str(response.json().get("message", ""))
        except ValueError:
            pass
        limited = (response.status_code == 429 or headers.get("x-ratelimit-remaining") == "0"
                   or "rate limit" in message.lower() or "retry-after" in {k.lower() for k in headers})
        if not limited:
            return None
        if headers.get("retry-after", "").isdigit():
            return float(headers["retry-after"]) + 1
        if headers.get("x-ratelimit-reset", "").isdigit():
            return max(float(headers["x-ratelimit-reset"]) - time.time(), 0.0) + 1
        return 60.0


REPO_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
