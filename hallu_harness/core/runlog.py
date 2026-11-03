"""The run log: everything `run` prints, also written to a file (DESIGN.md §12).

    run.log_dir: logs     ->  logs/<framework>/<dataset>/<model>/<YYYY-MM-DD_HH-MM-SS>.log
    run.log_file: PATH    ->  PATH, appended to by every run

With `log_dir`, every run gets a new file, named by the date and time it
started on the machine's clock. A second run that starts in the same second
(an array job on hosts sharing the disk) gets `_2`, then `_3`, and so on. The
log's first lines name the run id, the run's directory under `output_dir`.

Standard output and standard error are both copied, for the whole command:
the resolved config, model loading, library warnings, `[skip]` lines, the
console record of every step, and the traceback if the run dies. The file is
line-buffered, so a run killed outright (a host-RAM OOM) loses at most a
partial line. Output written below Python, straight to file descriptors 1
and 2 by C extensions or child processes, reaches the terminal only.
"""
from __future__ import annotations

import itertools
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, TextIO

from .config import ResolvedConfig
from .trajectory import run_labels


class _Tee:
    """A text stream that writes to the terminal stream and the log. Anything
    else (isatty, fileno, encoding) is the terminal stream's, so libraries
    that check for a terminal behave as they would without the log."""

    def __init__(self, terminal: TextIO, log: TextIO):
        self.terminal = terminal
        self.log = log

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, name: str):
        return getattr(self.terminal, name)


def claim_log_path(cfg: ResolvedConfig, started: datetime) -> Path | None:
    """The run's log file; with `log_dir`, created here, so no other run can
    take the same name."""
    if cfg.run.log_file:
        return Path(cfg.run.log_file)
    if not cfg.run.log_dir:
        return None
    folder = Path(cfg.run.log_dir).joinpath(*run_labels(cfg))
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{started.astimezone():%Y-%m-%d_%H-%M-%S}"
    for n in itertools.count(1):
        path = folder / (f"{stem}.log" if n == 1 else f"{stem}_{n}.log")
        try:
            path.open("x").close()
        except FileExistsError:
            continue
        return path


@contextmanager
def run_log(path: Path | None) -> Iterator[Path | None]:
    """Copies sys.stdout and sys.stderr to `path` while the block runs."""
    if path is None:
        yield None
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    log = open(path, "a", encoding="utf-8", buffering=1)
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(stdout, log), _Tee(stderr, log)
    try:
        yield path
    except BaseException:
        # Python prints an uncaught exception only after this block has put
        # the streams back, so the log gets its own copy here.
        traceback.print_exc(file=log)
        raise
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        log.close()
