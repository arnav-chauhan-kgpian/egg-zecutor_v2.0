"""Visible tests for HumanEval: the examples written in each docstring.

HumanEval docstrings state examples in many forms (counted over the 164
prompts, DESIGN.md §8.2). Supported here:

    >>> f(1, 2)                      doctest: the next line(s) are the value;
    3                                no line means the call returns None
    >>> round(f(x), 2)               any expression containing a call of f
    >>> f(1, 2) == 3                 a doctest written as an assertion
    f(1, 2) == 3   f(1, 2) ==> 3   f(1, 2) => 3   f(1, 2) ➞ 3   f(1, 2) -> 3
    f(1, 2) should return 3          f(1, 2) returns 3      f(1, 2)  # returns 3
    Input: [4, 2, 3] / Output: [2, 1]          (single-parameter functions)
    For num = "AB" the output should be 1.    (single-parameter functions)

An example is kept only if its call parses as Python and contains a call of
the entry point, and its expected value parses as a Python literal (`true` /
`false` are read as booleans). Anything else that looks like an example is
returned as `unparsed`, for the audit to list for manual review: an example
the model reads but the harness cannot check could still contradict the
tests. The audit then runs every kept example against EvalPlus's reference
solution, so a mis-extraction shows up as a conflict rather than silently.
"""
from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass

_SEPARATORS = [
    ("==>", re.compile(r"^\s*==>\s*")),
    ("==", re.compile(r"^\s*==\s*")),
    ("➞", re.compile(r"^\s*➞\s*")),
    ("=>", re.compile(r"^\s*=>\s*")),
    ("->", re.compile(r"^\s*->\s*")),
    ("# returns", re.compile(r"^\s*#\s*returns?\s+", re.IGNORECASE)),
    ("should return", re.compile(r"^\s*,?\s*should\s+return\s+", re.IGNORECASE)),
    ("returns", re.compile(r"^\s*,?\s*returns?\s+", re.IGNORECASE)),
]
# Not supported, deliberately: "f(1) = 3". Docstrings use it for definitions
# ("tri(1) = 3", "tri(n) = 1 + n / 2"), not for what the function returns.
_LIT_ERRORS = (ValueError, SyntaxError, TypeError, MemoryError, RecursionError)


@dataclass
class Example:
    call: str  # an expression containing a call of the entry point, e.g. "add(2, 3)"
    expected: str  # a Python literal, as written (true/false normalised)
    form: str
    line: int  # 1-based line in the prompt


@dataclass
class Unparsed:
    text: str
    line: int
    reason: str


def _strip_comment(source: str) -> str:
    try:
        tokens = [t for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type != tokenize.COMMENT]
        return tokenize.untokenize(tokens).strip()
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source.strip()


def _is_entry_call(node: ast.AST, entry_point: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == entry_point


def _literal_arguments(source: str) -> bool:
    try:
        call = ast.parse(source.strip(), mode="eval").body
    except SyntaxError:
        return False
    if not isinstance(call, ast.Call):
        return False
    try:
        for arg in [*call.args, *(k.value for k in call.keywords)]:
            ast.literal_eval(arg)
    except _LIT_ERRORS:
        return False
    return True


def _calls_entry(source: str, entry_point: str) -> bool:
    """A call of the entry point, or a call whose arguments include one
    (`round(f(x), 2)`, `sorted(f(x))`). Anything else — an attribute or
    subscript of the result, arithmetic on it — is not read as an example,
    because a malformed docstring line (`f(x) [1, 2]`, missing its `==`)
    would otherwise parse as one."""
    try:
        tree = ast.parse(source.strip(), mode="eval").body
    except SyntaxError:
        return False
    if _is_entry_call(tree, entry_point):
        return True
    return isinstance(tree, ast.Call) and any(
        _is_entry_call(arg, entry_point) for arg in [*tree.args, *(k.value for k in tree.keywords)]
    )


def _literal(source: str) -> str | None:
    """The value, if the whole of it is a Python literal — after removing
    trailing sentence punctuation and a trailing `# comment`. No prefix is
    ever taken: "19 - 5 - 6 = 8" is prose about 8, not the value 19."""
    text = _strip_comment(source.strip())
    text = text.rstrip(".,;:").strip()
    if re.fullmatch(r"(?i)(true|false)", text):
        return "True" if text.lower() == "true" else "False"
    if not text:
        return None
    try:
        ast.literal_eval(text)
        return text
    except _LIT_ERRORS:
        return None


def _balanced(text: str, start: int) -> int | None:
    """Index just past the bracketed group opening at `start`, respecting
    nesting and string literals; None if it does not close."""
    depth, i, quote = 0, start, None
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _single_parameter(prompt: str, entry_point: str) -> str | None:
    try:
        tree = ast.parse(prompt)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            args = node.args
            if len(args.args) == 1 and not args.vararg and not args.kwarg and not args.kwonlyargs:
                return args.args[0].arg
    return None


def extract(prompt: str, entry_point: str) -> tuple[list[Example], list[Unparsed]]:
    lines = prompt.splitlines()
    examples: list[Example] = []
    unparsed: list[Unparsed] = []
    seen: set[tuple[str, str]] = set()
    param = _single_parameter(prompt, entry_point)

    def add(call: str, expected: str, form: str, line: int) -> None:
        key = (call.strip(), expected.strip())
        if key not in seen:
            seen.add(key)
            examples.append(Example(call.strip(), expected.strip(), form, line))

    def inline(text: str, line: int) -> bool:
        """`f(...) <sep> literal` anywhere in `text`; True if any call of f was found."""
        found = False
        position = 0
        while True:
            k = text.find(entry_point + "(", position)
            if k < 0:
                return found
            if k > 0 and (text[k - 1].isalnum() or text[k - 1] in "_."):
                position = k + 1
                continue
            found = True
            end = _balanced(text, k + len(entry_point))
            if end is None:
                unparsed.append(Unparsed(text.strip(), line, "unbalanced call"))
                return found
            call, rest = text[k:end], text[end:]
            for form, pattern in _SEPARATORS:
                m = pattern.match(rest)
                if m:
                    literal = _literal(rest[m.end():])
                    if _calls_entry(call, entry_point) and _literal_arguments(call) and literal is not None:
                        add(call, literal, form, line)
                    else:
                        unparsed.append(Unparsed(text.strip(), line, f"'{form}' example whose call or value is not a literal"))
                    break
            position = end

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith(("def ", "from ", "import ")) or not stripped:
            i += 1
            continue

        if stripped.startswith(">>>"):
            source = _strip_comment(stripped[3:])
            if entry_point + "(" not in source:
                i += 1
                continue
            expected_lines, j = [], i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith(">>>") and not lines[j].strip().startswith('"""'):
                expected_lines.append(lines[j].strip())
                j += 1
            expected = " ".join(expected_lines)
            try:
                tree = ast.parse(source, mode="eval").body
            except SyntaxError:
                tree = None
            is_assertion = isinstance(tree, ast.Compare) and len(tree.ops) == 1 and isinstance(tree.ops[0], ast.Eq)
            if is_assertion and not expected:
                inline(source, i + 1)  # ">>> f(x) == y" written as an assertion
            elif tree is not None and _calls_entry(source, entry_point):
                literal = _literal(expected) if expected else "None"
                if literal is not None:
                    add(source, literal, ">>>", i + 1)
                else:
                    unparsed.append(Unparsed(f"{stripped} | {expected}", i + 1, "doctest value does not parse"))
            else:
                unparsed.append(Unparsed(stripped, i + 1, "doctest call does not parse"))
            i = j
            continue

        # A call split over several lines: join until its brackets balance.
        text, j = stripped, i
        k = text.find(entry_point + "(")
        if k >= 0 and _balanced(text, k + len(entry_point)) is None:
            while j + 1 < len(lines) and j - i < 15 and _balanced(text, k + len(entry_point)) is None:
                j += 1
                text += " " + lines[j].strip()
        if entry_point + "(" in text:
            inline(text, i + 1)
            i = j + 1
            continue

        if param is not None:
            m = re.match(r"(?i)^input\s*:\s*(.+)$", stripped)
            if m and i + 1 < len(lines):
                out = re.match(r"(?i)^output\s*:\s*(.+)$", lines[i + 1].strip())
                arg, value = _literal(m.group(1)), _literal(out.group(1)) if out else None
                if out and arg is not None and value is not None:
                    add(f"{entry_point}({arg})", value, "Input/Output", i + 1)
                    i += 2
                    continue
                if out:
                    unparsed.append(Unparsed(f"{stripped} | {lines[i + 1].strip()}", i + 1, "Input/Output values do not parse"))
            m = re.match(rf"(?i)^for\s+{re.escape(param)}\s*=\s*(.+?)\s+the output should be\s+(.+)$", stripped)
            if m:
                arg, value = _literal(m.group(1)), _literal(m.group(2))
                if arg is not None and value is not None:
                    add(f"{entry_point}({arg})", value, "For ... the output should be", i + 1)
                else:
                    unparsed.append(Unparsed(stripped, i + 1, "'For ... the output should be' values do not parse"))
        i += 1
    return examples, unparsed
