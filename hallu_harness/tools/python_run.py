"""`python` — run a Python program in the sandbox, with the question's
attached file in its working directory.

`python_exec` (the coding benchmarks' scratch tool) says "a program run only
with this tool is never taken as your solution", which reads as nonsense on
a question-answering task, takes a `stdin` the model never needs, and cannot
see an attachment. This tool runs through the same sandbox
(`exec_lab_sandbox.run_program`: ExecLab container, no network,
read-only system and site-packages, CPU-time and memory limits) with a small
wrapper around the model's code:

* the attachment arrives on standard input (as base64) and is written to
  the working directory under its own name before the code runs. The code
  then sees an empty standard input;
* the code runs as the module `__main__` from a file named `main.py`, so a
  traceback names `main.py` and the code's own line numbers, and shows no
  launcher frame;
* stdout is cut at `max_output_chars` (its beginning is kept), stderr at the
  same length (its end is kept, where a traceback's error is), and the result
  says so.

ExecLab runs every submission in an isolated container, so the sandbox is
always available.
"""
from __future__ import annotations

import base64

from ..core.types import ToolResult
from .base import Tool, ToolArgumentError
from .documents import attachment_of
from ..exec_lab_sandbox import run_program

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MEMORY_MB = 4096
DEFAULT_MAX_OUTPUT_CHARS = 4000
SCRIPT_NAME = "main.py"
# ExecLab runs each submission in an isolated container; source travels in the
# JSON body so there is no 128 KiB command-line argument cap. These limits
# guard against absurdly large submissions.
MAX_PROGRAM_BYTES = 120_000
MAX_SOURCE_CHARS = 100_000
# Checked by a test to import inside the sandbox; the description names them.
LIBRARIES = ("numpy", "pandas", "openpyxl", "pypdf", "Bio")  # Bio: Biopython (a GAIA question asks for it by name)


def build_code_with_attachment(
    code: str, attachment_name: str | None, attachment_size: int | None
) -> str:
    """Prepend attachment-loading code so the file is on disk before user code runs.

    The attachment travels as base64 on standard input; this code decodes it
    and writes it to `attachment_name`, then points stdin at /dev/null so the
    user program sees empty input.
    """
    if attachment_name is None:
        return code
    return (
        "import base64, sys, os\n"
        "_data = base64.b64decode(sys.stdin.buffer.read())\n"
        f"with open({attachment_name!r}, 'wb') as _f:\n"
        "    _f.write(_data)\n"
        "del _data\n"
        "sys.stdin.close()\n"
        "sys.stdin = open(os.devnull)\n"
        + code
    )


def cut_head(text: str, limit: int) -> tuple[str, bool]:
    return (text, False) if len(text) <= limit else (text[:limit], True)


def cut_tail(text: str, limit: int) -> tuple[str, bool]:
    return (text, False) if len(text) <= limit else (text[-limit:], True)


class PythonTool(Tool):
    name = "python"
    allowed_params: set[str] = {"timeout_s", "memory_mb", "max_output_chars"}

    @property
    def timeout(self) -> float:
        return float(self.params.get("timeout_s", DEFAULT_TIMEOUT_S))

    @property
    def max_output(self) -> int:
        return int(self.params.get("max_output_chars", DEFAULT_MAX_OUTPUT_CHARS))

    @property
    def description(self) -> str:
        attachment = attachment_of(self.env)
        file_note = (f" The file attached to the question is in the working directory as {attachment.name}."
                     if attachment is not None else "")
        return (
            f"Runs a Python 3 program in an isolated sandbox without network access, for at most {self.timeout:g} "
            "seconds of CPU time, and returns its exit code, what it printed to standard output, and any standard "
            f"error. Each call starts afresh: nothing is kept between calls.{file_note} Besides the standard library, "
            f"{', '.join(LIBRARIES[:-1])} and Bio (Biopython) can be imported; no other package can be installed. "
            f"Standard input is empty. Only the first {self.max_output} "
            "characters of standard output and the last "
            f"{self.max_output} of standard error are returned, and the result says when either was cut, so print "
            "only what you need."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "The complete Python 3 program to run."}},
            "required": ["code"],
        }

    @classmethod
    def check_environment(cls, params: dict) -> str | None:
        # ExecLab always runs in an isolated container; no jail check needed.
        return None

    def call(self, code: str = "", **unexpected) -> ToolResult:
        if unexpected:
            raise ToolArgumentError(f"python got unexpected argument(s): {', '.join(sorted(unexpected))}")
        if not isinstance(code, str) or not code.strip():
            return ToolResult(observation="The code is empty; nothing was run.", private={"empty_code": True})

        attachment = attachment_of(self.env)
        data = attachment.read_bytes() if attachment is not None else b""
        code_with_attach = build_code_with_attachment(
            code, attachment.name if attachment else None, len(data) if attachment else None
        )
        if len(code_with_attach.encode("utf-8")) > MAX_PROGRAM_BYTES:
            return ToolResult(observation=f"The code is {len(code)} characters long, more than can be run; it must be at "
                                          f"most about {MAX_SOURCE_CHARS} characters. Nothing was run.",
                              private={"code_too_long": len(code)})
        result = run_program(
            code_with_attach,
            base64.b64encode(data).decode("ascii"),
            timeout=self.timeout,
            memory_mb=float(self.params.get("memory_mb", DEFAULT_MEMORY_MB)),
            wall_timeout=2 * self.timeout + 5,
        )

        stdout, out_cut = cut_head(result.stdout, self.max_output)
        stderr, err_cut = cut_tail(result.stderr.rstrip("\n"), self.max_output)
        parts = [f"exit code: {result.exit_code}" + (" (time limit exceeded)" if result.timed_out else "")]
        parts.append("stdout:\n" + (stdout if stdout else "(empty)"))
        if out_cut or result.output_limit_hit:
            parts.append(f"(standard output cut: the program printed {len(result.stdout)}"
                         f"{' or more' if result.output_limit_hit else ''} characters; only the first {self.max_output} are shown)")
        if stderr.strip():
            parts.append("stderr:\n" + stderr)
            if err_cut:
                parts.append(f"(standard error cut: {len(result.stderr)} characters; only the last {self.max_output} are shown)")
        return ToolResult(
            observation="\n".join(parts),
            private={"exit_code": result.exit_code, "timed_out": result.timed_out, "seconds": round(result.seconds, 3),
                     "stdout_chars": len(result.stdout), "stderr_chars": len(result.stderr),
                     "attachment": attachment.name if attachment else None},
        )
