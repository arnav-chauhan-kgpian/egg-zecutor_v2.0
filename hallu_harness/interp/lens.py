"""Logit-lens readouts, four per decoder layer, for every recorded token (DESIGN.md §11.5).

    python -m hallu_harness lens --run trajectories/<run_id> [--out DIR] [--top-k 5] [--chunk 128] \\
        [--tolerance 2.0] [--max-fraction 0.05] [--device-map auto] [--limit N] [--overwrite] [--force]

The lens is LogitLens4LLMs' (Wang 2025, arXiv 2503.11667; code:
github.com/zhenyu-02/LogitLens4LLMs, `model_helper/qwen_helper.py`). Each
decoder layer gives four vectors, each decoded with the model's own final norm
and LM head, p = softmax(lm_head(final_norm(v))):

    attention_mechanism    the attention sublayer's output (after o_proj, before the residual add)
    intermediate_residual  the layer's input plus that output: the residual stream between attention and MLP
    mlp_output             the MLP sublayer's output
    block_output           the layer's output: the standard logit lens (nostalgebraist 2020)

Their wrapper decodes only the last position and runs the MLP a second time
to get its output. Here forward hooks read all four vectors from the one
forward pass, every position is decoded, and the identities
intermediate_residual = input + attention_mechanism and
block_output = intermediate_residual + mlp_output are checked at every layer,
so a model whose layers are built differently is refused rather than
misread. The attention and MLP readouts apply the final norm to a sublayer's
output on its own, which rescales it to unit size: they show which tokens that
component points towards, not how confident the model is.

What is stored, per readout at each (position, layer): the `--top-k` most
probable token ids (int32), their probabilities (float16), and the entropy
of the full distribution in nats (float16). Logits are computed in the
model's dtype, as its own head computes them; the softmax, top-k and entropy
in float32. float16 keeps probabilities to 0.05% above 6e-5, and to within
3e-8 below that.

Positions: every context and generated token of every step, each stored once.
The readout at position t is the prediction of token t+1, so the reply's
first token is predicted at the last input position.
- Step k stores positions from `start` on, where `start` is where its token ids
  first differ from those of its `base`: the latest earlier step that was
  written. Steps that only append to the history therefore store just the new
  feedback and reply. The rest is the base's (causal attention makes the
  readouts of a shared prefix identical).
- The opening that every trajectory of the run shares (system prompt and the
  start of the first user message) is stored once, in
  `opening-<hash>.lens.npz`, and is the base of each first step.

Every step is replayed in one teacher-forced forward pass and checked against
the log-probs recorded at generation time, as `replay` does (`--tolerance`,
`--max-fraction`; `extract.logprob_check`). A step that fails is not written
and is listed in the file's metadata and the index; `lens` tries it again
the next time it runs.

Files, in `--out` (default `<run>/lens/`), compressed npz (zlib; 62% of the
raw size on a 7B trajectory, mostly from the token ids):

    <trajectory>.lens.npz     step_<k>/top_ids    int32   [positions, layers, 4, top_k]
                              step_<k>/top_probs  float16 [positions, layers, 4, top_k]
                              step_<k>/entropy    float16 [positions, layers, 4]
                              meta                JSON string (readouts, model, per-step start/end/base, checks)
    opening-<hash>.lens.npz   the same three arrays for the shared opening, its ids, and meta
    index.jsonl               one line per step

Read a step back, every position from 0, with `load_lens(path).step(k)`.

During a run with `run.lens.enabled`, `OnlineLens` writes the same files as
the run goes: each step is computed as soon as its model call returns, with
the model the run has loaded, and `<trajectory>.partial.lens.npz`
(uncompressed) is rewritten; `<trajectory>.lens.npz` is written with the
trajectory. Its opening is the prefix the run's first two trajectories share,
taken from the first one's readouts (`lens` takes it from the first
trajectory it writes). A step it could not compute (out of GPU memory, or the disk nearly full) is
marked `not_computed` in the metadata; `lens` computes a trajectory with
such a step, or a rejected one, again, and skips the others.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..models.hf_local import execution_device

READOUTS = ("attention_mechanism", "intermediate_residual", "mlp_output", "block_output")
FORMAT = "hallu_harness.logit_lens/1"
DEFINITION = "softmax(lm_head(final_norm(v))) for v in " + ", ".join(READOUTS) + " of every decoder layer"


class LensStructureError(RuntimeError):
    """The decoder layers are not pre-norm attention-then-MLP blocks, so the
    four readouts would not be the vectors they are named after."""


def _ids_sha(ids: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()


def _common_prefix(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    differ = np.flatnonzero(a[:n] != b[:n])
    return int(differ[0]) if len(differ) else n


class LogitLens:
    """The four readouts of every decoder layer of a loaded causal LM."""

    def __init__(self, model, top_k: int = 5, chunk: int = 128):
        decoder = model.get_decoder()
        self.model = model
        self.layers = decoder.layers
        self.norm = decoder.norm
        self.lm_head = model.get_output_embeddings()
        self.top_k = top_k
        self.chunk = chunk
        for i, layer in enumerate(self.layers):
            for name in ("self_attn", "post_attention_layernorm", "mlp"):
                if not hasattr(layer, name):
                    raise LensStructureError(f"decoder layer {i} ({type(layer).__name__}) has no `{name}`")

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def compute(self, ids: np.ndarray, start: int, n_output: int = 0) -> dict[str, np.ndarray]:
        """Readouts at positions start .. len(ids)-1 of one forward pass over
        `ids`, plus `replayed`: the model's log-probs of the last `n_output`
        ids (the step's reply), for checking against the recorded ones."""
        import torch

        end = len(ids)
        if not 0 <= start < end:
            raise ValueError(f"start {start} outside 0..{end - 1}")
        n, k = end - start, self.top_k
        out = {
            "top_ids": np.empty((n, self.n_layers, len(READOUTS), k), np.int32),
            "top_probs": np.empty((n, self.n_layers, len(READOUTS), k), np.float16),
            "entropy": np.empty((n, self.n_layers, len(READOUTS)), np.float16),
        }
        seen: dict[str, torch.Tensor] = {}

        def keep(t):
            return t[0, start:end]

        def first(args, kwargs, name):
            return args[0] if args else kwargs[name]

        handles = []
        for index, layer in enumerate(self.layers):
            handles += [
                layer.register_forward_pre_hook(
                    lambda m, a, kw: seen.__setitem__("x", keep(first(a, kw, "hidden_states"))), with_kwargs=True),
                layer.self_attn.register_forward_hook(
                    lambda m, a, o: seen.__setitem__("a", keep(o[0] if isinstance(o, tuple) else o))),
                layer.post_attention_layernorm.register_forward_pre_hook(
                    lambda m, a, kw: seen.__setitem__("u", keep(first(a, kw, "hidden_states"))), with_kwargs=True),
                layer.mlp.register_forward_hook(lambda m, a, o: seen.__setitem__("m", keep(o))),
                layer.register_forward_hook(
                    lambda m, a, o, index=index: self._decode_layer(index, seen, keep(o if torch.is_tensor(o) else o[0]), out)),
            ]
        try:
            device = execution_device(self.model.get_input_embeddings())
            with torch.inference_mode():
                logits = self.model(input_ids=torch.tensor(ids[None, :], dtype=torch.long, device=device),
                                    logits_to_keep=n_output + 1, use_cache=False).logits[0, :n_output]
                # In chunks: an 8k-token reply's float32 log-softmax is ~10 GB at a 150k vocabulary.
                reply = torch.tensor(ids[end - n_output:], dtype=torch.long, device=logits.device)
                replayed = torch.empty(n_output, dtype=torch.float32, device=logits.device)
                for lo in range(0, n_output, self.chunk):
                    rows = slice(lo, min(lo + self.chunk, n_output))
                    replayed[rows] = torch.log_softmax(logits[rows].float(), -1).gather(1, reply[rows, None])[:, 0]
        finally:
            for h in handles:
                h.remove()
        out["replayed"] = replayed.cpu().numpy()
        return out

    def _decode_layer(self, index: int, seen: dict, h, out: dict) -> None:
        import torch

        x, a, u, m = (seen.pop(key) for key in ("x", "a", "u", "m"))
        x = x.to(a.device)  # with a multi-GPU device_map, the hook sees the input before it is moved
        for name, got, want in (("input + attention", x + a, u), ("intermediate residual + MLP", u + m, h)):
            if not torch.equal(got, want):
                gap = float((got.float() - want.float()).abs().max() / want.float().abs().max().clamp_min(1e-12))
                if gap > 1e-3:
                    raise LensStructureError(
                        f"layer {index}: {name} does not equal the next residual (relative gap {gap:.2e}); "
                        "these layers are not the pre-norm attention-then-MLP blocks the readouts assume")
        vectors = torch.stack((a, u, m, h))  # READOUTS order
        head_device = execution_device(self.lm_head)  # not lm_head.weight.device: `meta` when offloaded
        for lo in range(0, vectors.shape[1], self.chunk):
            hi = min(lo + self.chunk, vectors.shape[1])
            z = self.lm_head(self.norm(vectors[:, lo:hi].to(head_device))).float()  # [4, c, vocab]
            logp = torch.log_softmax(z, dim=-1)
            del z
            top, top_ids = torch.topk(logp, self.top_k, dim=-1)
            entropy = torch.special.entr(logp.exp()).sum(-1)  # as the generation-time recorder computes it
            out["top_ids"][lo:hi, index] = top_ids.permute(1, 0, 2).cpu().numpy()
            out["top_probs"][lo:hi, index] = top.exp().permute(1, 0, 2).cpu().numpy()
            out["entropy"][lo:hi, index] = entropy.T.cpu().numpy()
            del logp


def _save(path: Path, arrays: dict[str, np.ndarray], meta: dict, compress: bool = True) -> None:
    from ..core.trajectory import _atomic_write_bytes

    buffer = io.BytesIO()
    (np.savez_compressed if compress else np.savez)(buffer, meta=np.array(json.dumps(meta)), **arrays)
    _atomic_write_bytes(path, buffer.getvalue())


def _read_meta(path: Path) -> dict:
    with np.load(path) as data:
        return json.loads(str(data["meta"]))


class LensWriter:
    """Writes the lens files of one run: the shared opening and one file per trajectory."""

    def __init__(self, lens: LogitLens, out_dir: Path, model_info: dict, tolerance: float = 2.0, max_fraction: float = 0.05,
                 computed: str = "offline"):
        self.lens = lens
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model_info = model_info
        self.tolerance = tolerance
        self.max_fraction = max_fraction
        self.computed = computed
        self.opening_ids: np.ndarray | None = None
        self.opening_file: str | None = None

    def _common_meta(self) -> dict:
        import torch
        import transformers

        return {
            "format": FORMAT, "definition": DEFINITION, "readouts": list(READOUTS), "top_k": self.lens.top_k,
            "n_layers": self.lens.n_layers,
            "arrays": {"top_ids": "int32 [positions, layers, readouts, top_k]",
                       "top_probs": "float16 [positions, layers, readouts, top_k]",
                       "entropy": "float16 [positions, layers, readouts], nats"},
            "alignment": "row r of a block is position start + r; the readout at position t predicts token t + 1",
            "model": self.model_info,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "computed": self.computed,  # "online": during the run (run.lens); "offline": by `lens`
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def set_opening(self, first_steps: list[np.ndarray]) -> int:
        """The token ids every trajectory's first step starts with, if there
        are at least two trajectories. Returns its length (0 = none)."""
        if len(first_steps) < 2:
            return 0
        length = min(len(ids) for ids in first_steps)
        for ids in first_steps[1:]:
            length = min(length, _common_prefix(first_steps[0], ids))
        if length == 0:
            return 0
        self.opening_ids = np.asarray(first_steps[0][:length], dtype=np.int32)
        key = json.dumps([FORMAT, self.lens.top_k, self.model_info, _ids_sha(self.opening_ids)], sort_keys=True)
        self.opening_file = f"opening-{hashlib.sha256(key.encode()).hexdigest()[:12]}.lens.npz"
        return length

    def _opening_ready(self) -> bool:
        return self.opening_file is not None and (self.out_dir / self.opening_file).exists()

    def _opening_start(self, ids: np.ndarray) -> tuple[int, bool]:
        """Where a trajectory's first step starts after the shared opening
        (0: it does not use it), and whether the opening file is to be
        written from this step's readouts, which needs all of the opening."""
        if self.opening_ids is None:
            return 0, False
        shared = min(_common_prefix(self.opening_ids, ids), len(ids) - 1)
        if self._opening_ready():
            return shared, False
        if shared == len(self.opening_ids):
            return shared, True
        return 0, False

    def _write_opening(self, readouts: dict) -> None:
        length = len(self.opening_ids)
        arrays = {name: readouts[name][:length] for name in ("top_ids", "top_probs", "entropy")}
        arrays["ids"] = self.opening_ids
        meta = {**self._common_meta(), "kind": "opening", "length": length, "ids_sha256": _ids_sha(self.opening_ids)}
        _save(self.out_dir / self.opening_file, arrays, meta)

    def begin(self, name: str, about: dict | None = None, index=None) -> "TrajectoryLens":
        return TrajectoryLens(self, name, about, index)

    def trajectory(self, name: str, steps: list[dict], about: dict | None = None, index=None) -> dict:
        """Write `<name>.lens.npz`. `steps` holds, in order, dicts with
        `step`, `input_ids`, `output_ids` and the recorded `logprobs` (or None).
        Returns the file's metadata."""
        lens = self.begin(name, about, index)
        for step in steps:
            lens.add(step)
        return lens.save()


class TrajectoryLens:
    """One trajectory's lens file, built a step at a time."""

    def __init__(self, writer: LensWriter, name: str, about: dict | None = None, index=None):
        self.writer = writer
        self.name = name
        self.about = about or {}
        self.index = index
        self.arrays: dict[str, np.ndarray] = {}
        self.written: list[tuple[int, np.ndarray]] = []  # (step, ids) of steps with a block
        self.records: list[dict] = []

    @property
    def path(self) -> Path:
        return self.writer.out_dir / f"{self.name}.lens.npz"

    @property
    def partial_path(self) -> Path:
        return self.writer.out_dir / f"{self.name}.partial.lens.npz"

    def add(self, step: dict) -> dict:
        """Compute, check and keep one step (a dict as `LensWriter.trajectory`
        takes). Returns its record."""
        from .extract import logprob_check

        w = self.writer
        n_input, n_output = len(step["input_ids"]), len(step["output_ids"])
        ids = np.concatenate([step["input_ids"], step["output_ids"]]).astype(np.int64)
        record = {"step": step["step"], "n_input": n_input, "n_output": n_output, "end": len(ids), "ids_sha256": _ids_sha(ids)}
        base, start, compute_from, new_opening = None, 0, 0, False
        if self.written:
            base, prev = self.written[-1]
            start = compute_from = min(_common_prefix(prev, ids), len(ids) - 1)
        else:
            shared, new_opening = w._opening_start(ids)
            if shared:
                base, start = "opening", shared
                compute_from = 0 if new_opening else start
        began = time.perf_counter()
        readouts = w.lens.compute(ids, compute_from, n_output)
        fields, rejected = logprob_check(readouts["replayed"], step.get("logprobs"), w.tolerance, w.max_fraction)
        record.update(fields)
        record["seconds"] = round(time.perf_counter() - began, 2)
        if not rejected:
            if new_opening:
                w._write_opening(readouts)
            skip = start - compute_from
            for key in ("top_ids", "top_probs", "entropy"):
                self.arrays[f"step_{step['step']}/{key}"] = readouts[key][skip:]
            record.update(base=base, start=start, positions=len(ids) - start)
            self.written.append((step["step"], ids))
        self._keep(record)
        return record

    def skip(self, step: int, reason: str, n_input: int | None = None, ids: np.ndarray | None = None) -> dict:
        """A step whose readouts were not computed, listed with the reason
        like a rejected step. `lens` recomputes the trajectory later."""
        record: dict = {"step": step}
        if ids is not None:
            record.update(n_input=n_input, n_output=len(ids) - n_input, end=len(ids), ids_sha256=_ids_sha(ids))
        record.update(rejected=reason, not_computed=True)
        self._keep(record)
        return record

    def _keep(self, record: dict) -> None:
        self.records.append(record)
        if self.index is not None:
            self.index.write(json.dumps({"trajectory": self.name, **self.about, **record}) + "\n")
            self.index.flush()

    def meta(self) -> dict:
        w = self.writer
        meta = {**w._common_meta(), "kind": "trajectory", "trajectory": self.about, "steps": self.records, "opening": None}
        if any(r.get("base") == "opening" for r in self.records):
            meta["opening"] = {"file": w.opening_file, "length": len(w.opening_ids), "ids_sha256": _ids_sha(w.opening_ids)}
        return meta

    def save(self, final: bool = True) -> dict:
        """Write `<name>.lens.npz`, compressed, and remove the partial file;
        or, with `final=False`, `<name>.partial.lens.npz`, uncompressed, so
        that rewriting it after every step stays cheap. Returns the metadata."""
        meta = self.meta()
        target = self.path if final else self.partial_path
        if self.arrays:
            _save(target, self.arrays, meta, compress=final)
        elif target.exists():
            target.unlink()  # every step was rejected: drop the file an earlier pass wrote
        if final and self.partial_path.exists():
            self.partial_path.unlink()
        return meta


def _free_gpu_memory() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class OnlineLens:
    """`run.lens`: the readouts computed during a run, by the model the run
    has loaded, as soon as each model call returns (DESIGN.md §11.5).

    A recorder listener: on every `model_call` the step is computed, checked
    against the log-probs just recorded, and the trajectory's partial file is
    rewritten. `finish` writes the final file with the trajectory. The files
    are the ones `lens` writes, in `<run>/lens/`.

    The shared opening is the prefix common to the first two trajectories'
    first steps, written, as `lens` writes it, from the first one's
    readouts; a later trajectory uses as much of it as it shares.

    Nothing here reaches the framework. A step whose readouts fail (a CUDA
    out-of-memory error, say), or that would be written to a disk with less
    than `min_free_gb` free, is listed as not computed, and `lens` computes
    that trajectory again later. The forward pass consumes no random numbers,
    and every generation seeds its own, so the trajectory is the same with
    or without it.
    """

    def __init__(self, model, out_dir: Path, model_info: dict, settings, note=print):
        lens = LogitLens(model, settings.top_k, settings.chunk)
        lens.compute(np.arange(4), 0)  # refuse layers the readouts do not fit now, not mid-trajectory
        self.writer = LensWriter(lens, out_dir, model_info, settings.tolerance, settings.max_fraction, computed="online")
        self.min_free_bytes = settings.min_free_gb * 1e9
        self.note = note
        self.index = (self.writer.out_dir / "index.jsonl").open("a", encoding="utf-8")
        self.trajectory: TrajectoryLens | None = None
        self.first_ids: np.ndarray | None = None  # the run's first first step, and its readouts
        self.first_readouts: dict[str, np.ndarray] | None = None

    def start(self, name: str, about: dict) -> None:
        self.trajectory = self.writer.begin(name, about, self.index)

    def __call__(self, event: str, recorder, payload) -> None:
        if event == "model_call" and self.trajectory is not None:
            self.add(payload)

    def add(self, step) -> dict:
        import shutil

        t, g = self.trajectory, step.generation
        if g.input_token_ids is None or g.output_token_ids is None:
            record = t.skip(step.number, "not computed: no token ids were recorded")
            self.note(f"[lens] step {step.number}: {record['rejected']}")
            return record
        input_ids = np.asarray(g.input_token_ids, dtype=np.int64)
        output_ids = np.asarray(g.output_token_ids, dtype=np.int64)
        ids = np.concatenate([input_ids, output_ids])
        first_step = not t.records
        run_first = first_step and self.first_ids is None
        if run_first:
            self.first_ids = ids
        free = shutil.disk_usage(self.writer.out_dir).free
        if free < self.min_free_bytes:
            record = t.skip(step.number, f"not computed: {free / 1e9:.1f} GB free on the lens directory's disk, "
                            f"below run.lens.min_free_gb {self.min_free_bytes / 1e9:g}", len(input_ids), ids)
        else:
            logprobs = None if g.output_logprobs is None else np.asarray(g.output_logprobs, dtype=np.float32)
            try:
                if first_step and not run_first and self.writer.opening_ids is None:
                    # Without the first one's readouts, the opening is written from this step's.
                    if self.writer.set_opening([self.first_ids, ids]) and self.first_readouts is not None:
                        self.writer._write_opening(self.first_readouts)
                record = t.add({"step": step.number, "input_ids": input_ids, "output_ids": output_ids, "logprobs": logprobs})
                if run_first and "rejected" not in record:  # every row from 0: it had no base
                    self.first_readouts = {key: t.arrays[f"step_{step.number}/{key}"] for key in ("top_ids", "top_probs", "entropy")}
            except Exception as exc:  # noqa: BLE001 - must not end the trajectory; listed, and redone by `lens`
                _free_gpu_memory()
                record = t.skip(step.number, f"not computed: {type(exc).__name__}: {exc}", len(input_ids), ids)
        try:
            t.save(final=False)
        except Exception as exc:  # noqa: BLE001 - same
            self.note(f"[lens] could not write {t.partial_path.name}: {exc}")
        if "rejected" in record:
            self.note(f"[lens] step {step.number}: {record['rejected']}")
        else:
            base = {None: "none", "opening": "the shared opening"}.get(record["base"], f"step {record['base']}")
            self.note(f"[lens] step {step.number}: {record['positions']:,} positions from {record['start']:,} "
                      f"(before that: {base}), {record['seconds']:.1f}s, max |replayed - recorded| log-prob "
                      f"{record.get('max_abs_logprob_diff')}")
        return record

    def finish(self) -> None:
        """Write the trajectory's final file. Never raises: the trajectory is
        recorded whatever happens here."""
        t, self.trajectory = self.trajectory, None
        if t is None or not t.records:
            return
        try:
            t.save()
        except Exception as exc:  # noqa: BLE001
            self.note(f"[lens] could not write {t.path.name}: {exc}")
            return
        kept = [r for r in t.records if "rejected" not in r]
        size = f", {t.path.stat().st_size / 1e6:.1f} MB" if kept else ""
        self.note(f"[lens] {t.path.name if kept else t.name}: {len(kept)} of {len(t.records)} step(s), "
                  f"{sum(r['positions'] for r in kept):,} positions{size}")

    def close(self) -> None:
        self.index.close()


class LensFile:
    """A trajectory's lens file. `step(k)` returns that step's readouts at
    every position from 0 (input then reply), assembled from its blocks."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta = _read_meta(self.path)
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"{self.path} is not a {FORMAT} file")
        self.readouts = tuple(self.meta["readouts"])
        self._records = {r["step"]: r for r in self.meta["steps"]}
        self._cache: dict = {}

    @property
    def steps(self) -> list[int]:
        """Steps with readouts (rejected steps are listed in `meta` only)."""
        return [r["step"] for r in self.meta["steps"] if "rejected" not in r]

    def record(self, step: int) -> dict:
        return self._records[step]

    def _arrays(self, path: Path, prefix: str = "") -> dict[str, np.ndarray]:
        with np.load(path) as data:
            return {key: data[prefix + key] for key in ("top_ids", "top_probs", "entropy")}

    def _base(self, base) -> dict[str, np.ndarray]:
        if base is None:
            return {}
        if base == "opening":
            if "opening" not in self._cache:
                opening = self.meta["opening"]
                path = self.path.with_name(opening["file"])
                if _read_meta(path)["ids_sha256"] != opening["ids_sha256"]:
                    raise ValueError(f"{path} is not the opening {self.path.name} was written against")
                self._cache["opening"] = self._arrays(path)
            return self._cache["opening"]
        return self.step(base)

    def step(self, step: int) -> dict[str, np.ndarray]:
        record = self._records[step]
        if "rejected" in record:
            raise KeyError(f"step {step} has no readouts: {record['rejected']}")
        if step not in self._cache:
            block = self._arrays(self.path, f"step_{step}/")
            base = self._base(record["base"])
            self._cache[step] = {key: np.concatenate([base[key][: record["start"]], block[key]]) if base else block[key]
                                 for key in block}
        return self._cache[step]

    def check_ids(self, step: int, input_ids: np.ndarray, output_ids: np.ndarray) -> None:
        """Raise unless the step was computed on exactly these token ids."""
        if _ids_sha(np.concatenate([input_ids, output_ids])) != self._records[step]["ids_sha256"]:
            raise ValueError(f"{self.path.name} step {step} was computed on different token ids")


def load_lens(path: str | Path) -> LensFile:
    return LensFile(path)


def already_written(target: Path, steps: list[dict]) -> bool:
    """Whether the lens file `target` holds readouts for exactly these steps,
    on these token ids. A step without readouts — not computed during the
    run, or rejected by the replay check, perhaps under an older, stricter
    rule — is not covered, so `lens` tries it again."""
    if not target.exists():
        return False
    try:
        done = {r["step"]: r["ids_sha256"] for r in _read_meta(target)["steps"] if "rejected" not in r}
    except Exception:  # noqa: BLE001 - unreadable or foreign file: recompute it
        return False
    return done == {s["step"]: _ids_sha(np.concatenate([s["input_ids"], s["output_ids"]])) for s in steps}


def main(args) -> int:
    from ..core.trajectory import load_tokens
    from .extract import Replayer, _check_metadata, _trajectory_files

    run = Path(args.run)
    files = _trajectory_files(run)
    if args.limit:
        files = files[: args.limit]
    out_dir = Path(args.out) if args.out else (run if run.is_dir() else run.parent.parent) / "lens"

    work = []
    for path in files:
        doc = json.loads(path.read_text())
        try:
            tokens = load_tokens(path)
        except FileNotFoundError:
            print(f"skip {path.name}: no token sidecar", file=sys.stderr)
            continue
        steps = []
        for s in doc["steps"]:
            t = tokens.get(f"step_{s['step']}")
            if t and "input_ids" in t and "output_ids" in t:
                steps.append({"step": s["step"], "input_ids": t["input_ids"], "output_ids": t["output_ids"],
                              "logprobs": t.get("logprobs")})
        if steps:
            work.append((path, doc, steps))
    if not work:
        print("no trajectories with recorded token ids", file=sys.stderr)
        return 1

    model_cfg = work[0][1]["harness"]["config"]["model"]
    for path, doc, _ in work:
        if doc["harness"]["config"]["model"] != model_cfg:
            raise SystemExit(f"{path.name} was run with a different model config than {work[0][0].name}; lens one run at a time")
    replayer = Replayer(model_cfg, args.device_map)
    for warning in _check_metadata(work[0][1], replayer.backend, args.force):
        print(f"WARNING (forced): {warning}", file=sys.stderr)
    info = replayer.backend.info
    model_info = {key: info.get(key) for key in ("id", "revision", "dtype", "quantization")}
    writer = LensWriter(LogitLens(replayer.model, args.top_k, args.chunk), out_dir, model_info, args.tolerance, args.max_fraction)
    opening = writer.set_opening([np.concatenate([s[0]["input_ids"], s[0]["output_ids"]]) for _, _, s in work])
    print(f"{len(work)} trajectories, {writer.lens.n_layers} layers x {len(READOUTS)} readouts, top-{args.top_k}; "
          f"shared opening {opening} tokens; writing to {out_dir}", flush=True)

    with (out_dir / "index.jsonl").open("a", encoding="utf-8") as index:
        for path, doc, steps in work:
            name = path.name[: -len(".json")]
            target = out_dir / f"{name}.lens.npz"
            if not args.overwrite and already_written(target, steps):
                print(f"{name}: already written", flush=True)
                continue
            about = {"trajectory_id": doc["trajectory_id"], "file": path.name}
            meta = writer.trajectory(name, steps, about, index)
            kept = [r for r in meta["steps"] if "rejected" not in r]
            size = target.stat().st_size / 1e6 if target.exists() and kept else 0.0
            rejected = len(meta["steps"]) - len(kept)
            print(f"{name}: {len(kept)} step(s), {sum(r['positions'] for r in kept)} positions, {size:.1f} MB"
                  + (f", {rejected} step(s) rejected" if rejected else ""), flush=True)
    return 0
