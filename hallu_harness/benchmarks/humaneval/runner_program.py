"""The program that runs inside the sandbox to test a HumanEval solution.

One sandboxed process per grading call runs every test (DESIGN.md §8.2):
with ~775 HumanEval+ inputs per task, a process per input would cost minutes.
The program arrives on stdin (some HumanEval+ test sources are 500 KB, far
over Linux's 128 KB limit on one command-line argument) and:

1. executes the task's `prompt` (its imports and helper functions, which some
   targets call — HumanEval/10, /38, /50), then the solution, with
   `__name__ != "__main__"` so a solution's own demo block does not run;
2. checks that the solution actually defines the entry point (the prompt's
   docstring-only stub does not count);
3. runs each test under its own time limit, in CPU time (`ITIMER_PROF`),
   with the same real-time backstop as every sandboxed program
   (`ITIMER_REAL` at `wall_limit(limit)`, for code that waits), giving the
   candidate deep copies of its arguments so a solution that mutates them
   cannot affect later tests or the reference function. Both timers fire
   again every 50 ms, in case the code swallows the first interrupt
   (`except:`), and a test that went over either limit is
   TIME_LIMIT_EXCEEDED however it ended;
4. classifies each test: an exception raised inside the candidate is a
   RUNTIME_ERROR (MEMORY_LIMIT_EXCEEDED for MemoryError), a timer expiry is
   TIME_LIMIT_EXCEEDED, a failed assertion — or any other error raised while
   *comparing* the result — is WRONG_ANSWER;
5. reports results through a duplicate of the original stdout descriptor,
   each line prefixed with a random nonce, so neither the solution's prints
   nor a reassigned `sys.stdout` can forge or swallow them.

Hidden tests are executed exactly as EvalPlus's `check()` would execute them:
its module-level helpers (`assertion`, `ref_func`, `_poly`) and its `inputs`
/ `results` assignments are run as written, and the body of its single loop is
run once per input with the loop variables bound as in `check()`.
"""
from __future__ import annotations

import ast
import json
import secrets
from dataclasses import dataclass

# ExecLab handles wall-clock limits internally; these values are placed in the
# payload for compatibility with the HARNESS code but are not enforced locally.
WALL_TIME_FACTOR = 1.0
WALL_TIME_SLACK_S = 0.0

LOADER = "import sys; __harness_src = sys.stdin.read(); exec(compile(__harness_src, '<harness>', 'exec'))"

# The longest repr of a visible example's return value the sandbox sends
# back. Longer ones are cut and flagged (`repr_cut`), so a cut is never
# silent, whatever the display rule.
REPR_CAP_CHARS = 20000

HARNESS = r'''
import ast as _ast, copy as _copy, json as _json, os as _os, signal as _signal, sys as _sys, time as _time, traceback as _tb
_P = _json.loads(__PAYLOAD__)
_FD = _os.dup(1)

def _emit(obj):
    _os.write(_FD, (_P["nonce"] + _json.dumps(obj) + "\n").encode("utf-8", "replace"))

class _Timeout(BaseException):
    pass

class _CandidateError(Exception):
    def __init__(self, exc):
        self.exc = exc

# Time limits are on CPU time (ITIMER_PROF: this process's user and system
# time), with a real-time backstop (ITIMER_REAL) for code that waits. After
# a limit, its timer fires again every 50 ms, in case the code under test
# swallows an interrupt. `_armed` makes the handler a no-op once disarming
# has begun, so no interrupt escapes a handler.
_armed = [False]
_wall = _time.perf_counter

def _cpu():
    """(thread, process) CPU clocks. The process clock covers every thread
    but, while a process-wide CPU timer is armed (RLIMIT_CPU always is),
    Linux only updates it at scheduler events, so short intervals read as
    0; the thread clock is exact for this (single) thread."""
    return (_time.thread_time(), _time.process_time())

def _cpu_used(t0, t1=None):
    t1 = _cpu() if t1 is None else t1
    return max(t1[0] - t0[0], t1[1] - t0[1])

def _wall_limit(seconds):
    return _P["wall_factor"] * seconds + _P["wall_slack"]

def _on_timer(signum, frame):
    if _armed[0]:
        raise _Timeout()

_signal.signal(_signal.SIGPROF, _on_timer)
_signal.signal(_signal.SIGALRM, _on_timer)

def _arm(seconds):
    _armed[0] = True
    _signal.setitimer(_signal.ITIMER_PROF, max(float(seconds), 0.001), 0.05)
    _signal.setitimer(_signal.ITIMER_REAL, max(float(_wall_limit(seconds)), 0.001), 0.05)

def _disarm():
    _armed[0] = False
    _signal.setitimer(_signal.ITIMER_PROF, 0, 0)
    _signal.setitimer(_signal.ITIMER_REAL, 0, 0)

def _over(t0, w0, seconds, t1=None, w1=None):
    """Whether a test that ran from (t0, w0) to (t1, w1) — default: now —
    went over its limits."""
    w1 = _wall() if w1 is None else w1
    return _cpu_used(t0, t1) > seconds or w1 - w0 > _wall_limit(seconds)

def _tle(t0, seconds, t1=None):
    """What a test that went over its limits is told: which limit it hit."""
    if _cpu_used(t0, t1) > seconds:
        return "time limit exceeded: used more than %gs of CPU time" % seconds
    return "time limit exceeded: still running after %gs of real time (the limit is %gs of CPU time)" % (
        _wall_limit(seconds), seconds)

def _short_tb(exc, limit=40):
    lines = _tb.format_exception(type(exc), exc, exc.__traceback__)
    text = "".join(lines)
    keep = [l for l in text.splitlines() if '"<harness>"' not in l and "_harness_" not in l]
    return "\n".join(keep[-limit:])

def _is_floats(x):
    try:
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return x.dtype == _np.float64 or x.dtype == _np.float32
    except Exception:
        pass
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)):
        return all(isinstance(i, float) for i in x)
    return False

def _match(out, exp, atol=0):
    """EvalPlus's `assertion`, returning a bool instead of raising."""
    try:
        exact = out == exp
        exact = bool(exact)
    except Exception:
        exact = False
    if atol == 0 and _is_floats(exp):
        atol = 1e-6
    if not exact and atol != 0:
        try:
            import numpy as _np
            return bool(_np.allclose(out, exp, rtol=1e-07, atol=atol))
        except Exception:
            return False
    return exact

_g = {"__name__": "__solution__", "__builtins__": __builtins__}
try:
    exec(compile(_P["prompt"], "<prompt>", "exec"), _g)
except BaseException as _e:
    _emit({"load_error": "the task prompt failed to execute: " + _short_tb(_e)})
    _sys.exit(0)
_stub = _g.get(_P["entry_point"])
_t0, _w0 = _cpu(), _wall()
try:
    _arm(_P["load_limit"])
    try:
        exec(compile(_P["solution"], "<solution>", "exec"), _g)
    finally:
        _disarm()
    if _over(_t0, _w0, _P["load_limit"]):
        raise _Timeout()
except _Timeout:
    _disarm()
    _emit({"load_error": "defining your code: " + _tle(_t0, _P["load_limit"])})
    _sys.exit(0)
except BaseException as _e:
    _disarm()
    _emit({"load_error": _short_tb(_e)})
    _sys.exit(0)
_real = _g.get(_P["entry_point"])
if _real is None or _real is _stub or not callable(_real):
    _emit({"load_error": "your code does not define the function %s" % _P["entry_point"]})
    _sys.exit(0)

def candidate(*args, **kwargs):
    args = _copy.deepcopy(args)
    kwargs = _copy.deepcopy(kwargs)
    try:
        return _real(*args, **kwargs)
    except _Timeout:
        raise
    except BaseException as exc:
        raise _CandidateError(exc) from None

_emit({"loaded": True})

if _P["mode"] == "visible":
    _scope = dict(_g)
    _scope[_P["entry_point"]] = candidate
    for _ex in _P["examples"]:
        _rec = {"i": _ex["i"]}
        _lim = _P["limits"][_ex["i"] - 1]
        # (_t1, _w1): when the call returned or raised; the comparison and
        # repr() below are the harness's work and are not timed.
        _t0, _w0 = _cpu(), _wall()
        _t1 = _w1 = None
        try:
            _arm(_lim)
            try:
                _out = eval(compile(_ex["call"], "<example>", "eval"), _scope)
            finally:
                _disarm()
                _t1, _w1 = _cpu(), _wall()
            if _over(_t0, _w0, _lim, _t1, _w1):
                raise _Timeout()
            _exp = _ast.literal_eval(_ex["expected"])
            _rec["ok"] = _match(_out, _exp)
            try:
                _r = repr(_out)
            except Exception as _e:
                _r = "<unrepresentable %s>" % type(_out).__name__
            _rec["repr"] = _r[: _P["repr_cap"]]
            _rec["repr_cut"] = len(_r) > _P["repr_cap"]
        except _Timeout:
            _disarm()
            _rec.update(ok=False, err=_tle(_t0, _lim, _t1))
        except _CandidateError as _ce:
            _disarm()
            _rec.update(ok=False, err=_short_tb(_ce.exc), mle=isinstance(_ce.exc, MemoryError))
        except BaseException as _e:
            _disarm()
            _rec.update(ok=False, err=_short_tb(_e))
        if _t1 is None:
            _t1, _w1 = _cpu(), _wall()
        if _over(_t0, _w0, _lim, _t1, _w1):  # a judge would have stopped it at the limit, whatever came after
            _rec.update(ok=False, err=_tle(_t0, _lim, _t1))
        _rec["t"] = round(_cpu_used(_t0, _t1), 6)
        _rec["w"] = round(_w1 - _w0, 6)
        _emit(_rec)
    _emit({"done": True})
    _sys.exit(0)

# hidden / timing: EvalPlus's check(), one loop iteration per input
_c = {"__name__": "__checker__", "__builtins__": __builtins__}
exec(compile(_P["check_prelude"], "<checker>", "exec"), _c)
_loop = compile(_P["loop_body"], "<checker>", "exec")
_inputs = _c["inputs"]
_results = _c.get("results")
_limits = _P["limits"]
for _i, _inp in enumerate(_inputs):
    _env = dict(_c)
    _env["candidate"] = candidate
    _env["i"] = _i
    _env["inp"] = _inp
    if _results is not None:
        _env["exp"] = _results[_i]
    _lim = _limits[_i] if _limits is not None else 3600
    _t0, _w0 = _cpu(), _wall()
    try:
        _arm(_lim)
        try:
            exec(_loop, _env)
        finally:
            _disarm()
        _v = "ACCEPTED"
    except _Timeout:
        _disarm()
        _v = "TIME_LIMIT_EXCEEDED"
    except _CandidateError as _ce:
        _disarm()
        _v = "MEMORY_LIMIT_EXCEEDED" if isinstance(_ce.exc, MemoryError) else "RUNTIME_ERROR"
    except MemoryError:
        _disarm()
        _v = "MEMORY_LIMIT_EXCEEDED"
    except BaseException:
        _disarm()
        _v = "WRONG_ANSWER"
    if _over(_t0, _w0, _lim):  # a judge would have stopped it at the limit, whatever came after
        _v = "TIME_LIMIT_EXCEEDED"
    _emit({"i": _i, "v": _v, "t": round(_cpu_used(_t0), 6), "w": round(_wall() - _w0, 6)})
_emit({"done": True})
'''


@dataclass
class CheckProgram:
    """EvalPlus's `check()` decomposed: the code to run once, the loop body to
    run per input, and how many inputs there are."""

    prelude: str
    loop_body: str
    input_count: int
    has_results: bool


def decompose_check(test_source: str) -> CheckProgram:
    """Splits an EvalPlus test file into (module helpers + `check`'s
    assignments) and the body of `check`'s single loop.

    Every one of the 164 HumanEval+ tasks has one of two shapes, verified
    here rather than assumed:

        for i, (inp, exp) in enumerate(zip(inputs, results)): <one statement>
        for i, inp in enumerate(inputs):                       <one statement>
    """
    tree = ast.parse(test_source)
    check = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check"), None)
    if check is None:
        raise ValueError("test source has no check()")
    module = [n for n in tree.body if n is not check]
    assigns = [n for n in check.body if isinstance(n, ast.Assign)]
    loops = [n for n in check.body if isinstance(n, ast.For)]
    if len(loops) != 1 or len(assigns) + 1 != len(check.body):
        raise ValueError("check() is not assignments followed by one loop")
    loop = loops[0]
    target, iterator = ast.unparse(loop.target), ast.unparse(loop.iter)
    if (target, iterator) == ("(i, (inp, exp))", "enumerate(zip(inputs, results))"):
        has_results = True
    elif (target, iterator) == ("(i, inp)", "enumerate(inputs)"):
        has_results = False
    else:
        raise ValueError(f"unexpected loop: for {target} in {iterator}")
    if len(loop.body) != 1:
        raise ValueError("check()'s loop has more than one statement")

    segments = [ast.get_source_segment(test_source, n) for n in module]
    # The assignments are inside check(); dedent them to module level.
    for node in assigns:
        segment = ast.get_source_segment(test_source, node)
        segments.append(segment)
    prelude = "\n\n".join(s for s in segments if s)
    namespace: dict = {}
    count_tree = ast.parse("\n".join(ast.get_source_segment(test_source, n) for n in assigns if any(
        isinstance(t, ast.Name) and t.id == "inputs" for t in n.targets)))
    inputs_value = count_tree.body[0].value
    count = len(inputs_value.elts) if isinstance(inputs_value, (ast.List, ast.Tuple)) else len(eval(compile(ast.Expression(inputs_value), "<inputs>", "eval"), namespace))
    return CheckProgram(prelude=prelude, loop_body=ast.unparse(loop.body[0]), input_count=count, has_results=has_results)


def build_program(payload: dict) -> tuple[str, str]:
    """(stdin program, nonce) for one sandboxed run."""
    nonce = "@@HE" + secrets.token_hex(12) + "@@"
    payload = {**payload, "nonce": nonce, "wall_factor": WALL_TIME_FACTOR, "wall_slack": WALL_TIME_SLACK_S}
    return HARNESS.replace("__PAYLOAD__", repr(json.dumps(payload)), 1), nonce


def parse_output(stdout: str, nonce: str) -> list[dict]:
    """The harness's records, in order. A solution's own output is ignored,
    including output written without a trailing newline before a record."""
    records = []
    for line in stdout.splitlines():
        k = line.find(nonce)
        if k < 0:
            continue
        try:
            records.append(json.loads(line[k + len(nonce):]))
        except json.JSONDecodeError:
            continue
    return records
