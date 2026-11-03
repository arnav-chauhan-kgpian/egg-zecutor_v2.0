"""The data this benchmark reads, each pinned and checked.

* **Prompts.** HALoGEN's `prompts/prompts_code.csv` at a fixed commit of
  github.com/AbhilashaRavichander/HALoGEN, checked against its sha256 and
  cached under `~/.cache/hallu_harness/halogen_code_imports/`. The repository
  states no license, so the file is downloaded, not copied into this repo.
* **PyPI names** (`pypi_names.txt.gz`). Every project name on PyPI's simple
  index, normalized as PEP 503 does, with a header line saying when it was
  fetched. HALoGEN asked PyPI live, per package, at grading time; a snapshot
  makes grading the same on every machine and every day (names registered
  later, including squats on names models invent, do not change old labels).
* **Import map** (`import_map.json`). Import names that differ from the
  PyPI project that provides them (`cv2` is `opencv-python`), from pipreqs'
  `mapping` file at a fixed commit (Apache-2.0), plus `SUPPLEMENT` below.
  HALoGEN checked the import name itself against PyPI, so `import cv2`
  counted as a hallucination.

`python -m hallu_harness halogen-data` writes the last two into this folder.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from ...core.config import resolve_repo_path
from ...core.errors import ConfigError

HERE = Path(__file__).parent
PYPI_NAMES_PATH = HERE / "pypi_names.txt.gz"
IMPORT_MAP_PATH = HERE / "import_map.json"
CACHE_DIR = Path.home() / ".cache" / "hallu_harness" / "halogen_code_imports"

HALOGEN_REPO = "AbhilashaRavichander/HALoGEN"
HALOGEN_COMMIT = "8b7dd77e9602af0b4c9d1430e60f474ac1b4eaa4"
PROMPTS_FILE = "prompts/prompts_code.csv"
PROMPTS_SHA256 = "0ac989c9f03157937a1f79221cfd68e51d590747dab802637cc614017d7340c4"
PROMPTS_URL = f"https://raw.githubusercontent.com/{HALOGEN_REPO}/{HALOGEN_COMMIT}/{PROMPTS_FILE}"

PIPREQS_COMMIT = "48dbafd39003b9de177b8d314795e45797555850"
PIPREQS_URL = f"https://raw.githubusercontent.com/bndr/pipreqs/{PIPREQS_COMMIT}/pipreqs/mapping"
PIPREQS_SHA256 = "bc02385c473b7ccfdc2c67697df423feaa394023f5f1d59693c86a5a0dcd7e50"

PYPI_SIMPLE_URL = "https://pypi.org/simple/"

# Import names pipreqs' mapping (at the commit above) lacks, each provided by
# the named PyPI project. Kept apart (`source: supplement` in the map) so an
# analysis can drop them.
SUPPLEMENT = {
    "RPi": "RPi.GPIO",
    "adafruit_dht": "adafruit-circuitpython-dht",
    "aws_cdk": "aws-cdk-lib",
    "board": "adafruit-blinka",
    "busio": "adafruit-blinka",
    "digitalio": "adafruit-blinka",
    "docx": "python-docx",
    "gi": "PyGObject",
    "googlesearch": "googlesearch-python",
    "mpl_toolkits": "matplotlib",
    "pkg_resources": "setuptools",
    "pythoncom": "pywin32",
    "skimage": "scikit-image",
    "smb": "pysmb",
    "vlc": "python-vlc",
    "win32api": "pywin32",
    "win32clipboard": "pywin32",
    "win32com": "pywin32",
    "win32con": "pywin32",
    "win32gui": "pywin32",
    "win32process": "pywin32",
    "win32ui": "pywin32",
    "wx": "wxPython",
}


def normalize(name: str) -> str:
    """A PyPI project name as PEP 503 compares them."""
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(url: str, timeout: float = 120.0, headers: dict | None = None) -> bytes:
    import requests

    response = requests.get(url, timeout=timeout, headers={"User-Agent": "hallu_harness", **(headers or {})})
    response.raise_for_status()
    return response.content


# ----------------------------------------------------------------- prompts


@lru_cache(maxsize=1)
def load_prompts() -> tuple[str, ...]:
    """HALoGEN's 1085 code prompts, in file order."""
    path = CACHE_DIR / HALOGEN_COMMIT[:12] / "prompts_code.csv"
    if not path.exists():
        try:
            body = _download(PROMPTS_URL)
        except Exception as exc:  # noqa: BLE001 - any failure means the prompts are unavailable
            raise ConfigError(f"could not download HALoGEN's code prompts from {PROMPTS_URL}: {type(exc).__name__}: {exc}") from None
        if hashlib.sha256(body).hexdigest() != PROMPTS_SHA256:
            raise ConfigError(f"{PROMPTS_URL} does not have the expected sha256 {PROMPTS_SHA256}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(body)
        tmp.replace(path)
    elif sha256_file(path) != PROMPTS_SHA256:
        raise ConfigError(f"{path} does not have the expected sha256 {PROMPTS_SHA256}; delete it to download it again")
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"), newline="")))
    return tuple(row["prompt"] for row in rows)


def prompts_info() -> dict:
    return {"source": f"https://github.com/{HALOGEN_REPO}", "commit": HALOGEN_COMMIT, "file": PROMPTS_FILE,
            "sha256": PROMPTS_SHA256}


# ----------------------------------------------------------------- PyPI names


@lru_cache(maxsize=2)
def _load_pypi_names(path: str) -> tuple[str, frozenset[str]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header, names = "", set()
        for line in handle:
            if line.startswith("#"):
                header = header or line[1:].strip()
            elif line.strip():
                names.add(line.strip())
    return header, frozenset(names)


def load_pypi_names(path: str | Path) -> tuple[str, frozenset[str]]:
    """(header, normalized names)."""
    resolved = resolve_repo_path(path)
    if not resolved.exists():
        raise ConfigError(f"PyPI name snapshot {path} not found; run `python -m hallu_harness halogen-data`")
    return _load_pypi_names(str(resolved))


def build_pypi_names(out: Path) -> str:
    body = _download(PYPI_SIMPLE_URL, timeout=300, headers={"Accept": "application/vnd.pypi.simple.v1+json"})
    index = json.loads(body)
    names = sorted({normalize(p["name"]) for p in index["projects"]})
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (f"# PyPI simple index ({PYPI_SIMPLE_URL}, PEP 691 JSON) fetched {stamp}, last-serial "
              f"{index.get('meta', {}).get('_last-serial')}, {len(names)} projects, names normalized per PEP 503\n")
    data = (header + "\n".join(names) + "\n").encode("utf-8")
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        handle.write(data)
    tmp.replace(out)
    return header.strip()


# ----------------------------------------------------------------- import map


@lru_cache(maxsize=2)
def _load_import_map(path: str) -> dict:
    return json.loads(Path(path).read_text())


def load_import_map(path: str | Path) -> dict:
    """{"sources": ..., "map": {import name: {"projects": [...], "source": ...}}}"""
    resolved = resolve_repo_path(path)
    if not resolved.exists():
        raise ConfigError(f"import map {path} not found; run `python -m hallu_harness halogen-data`")
    return _load_import_map(str(resolved))


def build_import_map(out: Path) -> int:
    body = _download(PIPREQS_URL)
    if hashlib.sha256(body).hexdigest() != PIPREQS_SHA256:
        raise ConfigError(f"{PIPREQS_URL} does not have the expected sha256 {PIPREQS_SHA256}")
    # Some import names have several lines (several projects provide them);
    # all are kept. (pipreqs itself keeps the last, to suggest one to install.)
    mapping: dict[str, dict] = {}
    for line in body.decode("utf-8").splitlines():
        if ":" in line:
            module, project = line.strip().split(":", 1)
            projects = mapping.setdefault(module, {"projects": [], "source": "pipreqs"})["projects"]
            if project not in projects:
                projects.append(project)
    for module, project in SUPPLEMENT.items():
        mapping.setdefault(module, {"projects": [project], "source": "supplement"})
    document = {
        "sources": {
            "pipreqs": {"url": PIPREQS_URL, "commit": PIPREQS_COMMIT, "sha256": PIPREQS_SHA256, "license": "Apache-2.0"},
            "supplement": "hallu_harness/benchmarks/halogen_code_imports/data.py: SUPPLEMENT",
        },
        "map": dict(sorted(mapping.items())),
    }
    out.write_text(json.dumps(document, indent=1, sort_keys=False) + "\n")
    return len(mapping)


def main(out_dir: str | None = None) -> int:
    """`python -m hallu_harness halogen-data`: (re)write both files."""
    folder = Path(out_dir) if out_dir else HERE
    started = time.perf_counter()
    print(f"[halogen-data] import map: {build_import_map(folder / IMPORT_MAP_PATH.name)} import names -> {folder / IMPORT_MAP_PATH.name}")
    print(f"[halogen-data] PyPI names: {build_pypi_names(folder / PYPI_NAMES_PATH.name)} -> {folder / PYPI_NAMES_PATH.name}")
    print(f"[halogen-data] HALoGEN prompts: {len(load_prompts())} (cached under {CACHE_DIR})")
    print(f"[halogen-data] done in {time.perf_counter() - started:.0f} s")
    return 0
