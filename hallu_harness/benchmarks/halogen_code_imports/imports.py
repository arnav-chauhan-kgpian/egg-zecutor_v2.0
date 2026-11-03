"""Which modules a program imports, and whether each exists.

**Extraction.** The program is parsed, and every `import`, `from ... import`
and literal `__import__("x")` / `importlib.import_module("x")` anywhere in it
(inside functions and `try` blocks too) is recorded with its line. A program
that does not parse falls back to HALoGEN's regexes. `halogen_units` applies
those regexes, unchanged, to a whole reply, as HALoGEN did, for comparison;
they miss `from a.b.c import x` and every name after the first in
`import a, b`.

**Resolution** of a top-level module name, in order:

* `stdlib`: in `sys.stdlib_module_names` or `sys.builtin_module_names` of the
  harness's interpreter, or in `LATER_STDLIB`;
* `pypi_mapped`: the import map names a project in the PyPI snapshot that
  provides it (`cv2` -> opencv-python);
* `pypi_name`: a project of this name (normalized) is in the PyPI snapshot —
  HALoGEN's test. The snapshot is PyPI's simple index, which lists every
  registered name, including some with no release at all (`cv2` and `yaml`
  are listed, while PyPI's JSON API, which HALoGEN asked, answers 404 for
  both), and names registered after a model was trained, such as squats on
  names models invent;
* `relative`: a relative import (`from . import x`), not checked;
* `unresolved`: none of these.

`unresolved` is a candidate hallucination, not a verdict: modules of
ecosystems outside PyPI (ROS's `rospy`, AWS Glue's `awsglue`) and a program's
own placeholders (`your_module`) land there too. Every fact used is recorded
per module, so an analysis can draw the line elsewhere.
"""
from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass

from .data import normalize

# Standard-library modules added after the interpreter this harness runs on
# (3.10), with the version that added them.
LATER_STDLIB = {"tomllib": "3.11", "annotationlib": "3.14", "compression": "3.14"}
STDLIB = frozenset(sys.stdlib_module_names) | frozenset(sys.builtin_module_names) | frozenset(LATER_STDLIB)

# HALoGEN's verifier (verifiers/code_packages.py), verbatim.
HALOGEN_IMPORT = re.compile(r'(^|\n)\s*import\s+(\w+)(?:\s+as\s+\w+)?\s*;?', re.IGNORECASE | re.MULTILINE)
HALOGEN_FROM_IMPORT = re.compile(r'(^|\n)\s*from\s+(\w+)(?:\.\w+)?\s+import\s+\w+;?', re.IGNORECASE | re.MULTILINE)

# `pip install` as a command: at the start of a line (after `$`, `!`, `>` or a
# comment's `#`) or inside backticks. In prose ("make sure to pip install the
# package") the words that follow are not package names.
_PIP = r"(?:python3?\s+-m\s+)?pip3?\s+install\s+"
PIP_INSTALL = (re.compile(r"(?:^|\n)[ \t]*(?:[$!>#][ \t]*)?" + _PIP + r"([^\n`#]+)"), re.compile(r"`" + _PIP + r"([^`\n]+)`"))


@dataclass
class ImportRecord:
    module: str  # top-level name ("" for a relative import of the package itself)
    statement: str  # the dotted name as written
    line: int | None
    form: str  # import | from | dynamic | regex
    level: int = 0  # dots of a relative import

    def to_dict(self) -> dict:
        return {"module": self.module, "statement": self.statement, "line": self.line, "form": self.form, "level": self.level}


def _dynamic_target(node: ast.Call) -> str | None:
    func = node.func
    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
    if name not in ("__import__", "import_module") or not node.args:
        return None
    first = node.args[0]
    return first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None


def extract_imports(code: str) -> tuple[list[ImportRecord], bool]:
    """(imports in source order, whether the program parsed)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return [ImportRecord(module=m, statement=m, line=None, form="regex") for m in halogen_units(code)], False
    found: list[tuple[int, int, int, ImportRecord]] = []  # (line, column, name's place in its statement, record)
    for node in ast.walk(tree):  # breadth-first; sorted into source order below
        if isinstance(node, ast.Import):
            for k, alias in enumerate(node.names):
                found.append((node.lineno, node.col_offset, k, ImportRecord(alias.name.split(".")[0], alias.name, node.lineno, "import")))
        elif isinstance(node, ast.ImportFrom):
            dotted = node.module or ""
            top = dotted.split(".")[0] if node.level == 0 else ""
            found.append((node.lineno, node.col_offset, 0, ImportRecord(top, "." * node.level + dotted, node.lineno, "from", node.level)))
        elif isinstance(node, ast.Call):
            target = _dynamic_target(node)
            if target:
                found.append((node.lineno, node.col_offset, 0, ImportRecord(target.split(".")[0], target, node.lineno, "dynamic")))
    found.sort(key=lambda item: item[:3])
    return [record for *_, record in found], True


def halogen_units(text: str) -> list[str]:
    """HALoGEN's atomic units for a reply: the set of names its two regexes
    capture, in first-seen order."""
    seen: dict[str, None] = {}
    for pattern in (HALOGEN_IMPORT, HALOGEN_FROM_IMPORT):
        for match in pattern.findall(text):
            seen.setdefault(match[1], None)
    return list(seen)


def pip_install_names(text: str) -> list[str]:
    """Project names after a `pip install` command in a reply (`PIP_INSTALL`),
    in order, without options, versions, extras or quotes."""
    found = sorted((m.start(1), m.group(1)) for pattern in PIP_INSTALL for m in pattern.finditer(text))
    names: dict[str, None] = {}
    for _, match in found:
        for token in match.split():
            token = token.strip("'\"")
            if token.startswith("-") or "/" in token or ":" in token or token in (".", ".."):
                continue
            name = re.split(r"[\[<>=!~;@,]", token, maxsplit=1)[0].strip()
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
                names.setdefault(name, None)
    return list(names)


class Resolver:
    def __init__(self, pypi_names: frozenset[str], import_map: dict[str, dict]):
        self.pypi_names = pypi_names
        self.import_map = import_map

    def on_pypi(self, project: str) -> bool:
        return normalize(project) in self.pypi_names

    def module(self, name: str) -> dict:
        stdlib = name in STDLIB
        pypi_name = self.on_pypi(name)
        mapped = self.import_map.get(name)
        projects = mapped["projects"] if mapped else []
        on_pypi = [p for p in projects if self.on_pypi(p)]
        mapped_on_pypi = bool(on_pypi)
        if stdlib:
            resolution = "stdlib"
        elif mapped_on_pypi:
            resolution = "pypi_mapped"
        elif pypi_name:
            resolution = "pypi_name"
        else:
            resolution = "unresolved"
        return {
            "module": name,
            "resolution": resolution,
            "stdlib": stdlib,
            "stdlib_since": LATER_STDLIB.get(name),
            "pypi_name": pypi_name,
            "mapped_projects": projects,
            "mapped_projects_on_pypi": on_pypi,
            "mapped_source": mapped["source"] if mapped else None,
        }
