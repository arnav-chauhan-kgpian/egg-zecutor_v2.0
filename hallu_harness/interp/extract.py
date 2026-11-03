"""Offline activation extraction by replay (DESIGN.md §11.4, Phase 4).

    python -m hallu_harness replay --run trajectories/<run_id> --out activations/<name> \
        [--layers quarters|all|8,16,24] [--segments observation,think,tool_call|all] \
        [--tokens all|last] [--lookback] [--tolerance 2.0] [--max-fraction 0.05] [--force]

For every recorded step, the step's exact input ids followed by its output
ids are run through the same model in one teacher-forced forward pass, with
hooks on the chosen decoder layers. Before anything is saved, the replay
checks itself against the log-probs recorded at generation time. They do not
agree exactly — one bf16 forward pass and token-by-token bf16 decoding with a
KV cache round differently — so the check is on the distribution of the
differences (`logprob_check`):

    tokens with |replayed - recorded| > 0.1 <= max(--max-fraction x tokens, 2)   (default 0.05)
    max |replayed - recorded|               <= --tolerance                       (default 2.0)

Measured on a real 7B bf16 trajectory (5 steps, 2,000+ tokens): median 0,
90th percentile < 0.007, max 0.12, and under 1% of tokens above 0.1. On 107
Qwen3-14B bf16 steps: 0.7% of tokens above 0.1 (at most 2.9% in a step), and
a step's worst token up to 0.73. The same 7B trajectory replayed with the
wrong model (0.5B): max 7–11 and ~20% of tokens above 0.5. A step that fails
is not written; it is listed in the index with its statistics. Dtype,
quantization, weights revision and library versions are also checked up
front against the trajectory's metadata.

Saved per trajectory, one safetensors file:

    step_<n>/positions            int32  token positions kept (into input ++ output)
    step_<n>/layer_<L>            fp16   [positions, hidden] — the output of decoder layer L
                                         (1-indexed; the residual stream after that layer,
                                         before the final norm)
    step_<n>/lookback             fp16   [output tokens, layers, heads] (with --lookback)

and one line per step in `index.jsonl` (trajectory, step, kept segments,
layers, positions, verification).

Lookback ratios (Chuang et al. 2024): for each generated token, layer and
head, the attention mass on the input (context) tokens divided by the mass
on context plus already-generated tokens. Computed with eager attention,
whose full attention matrix for one layer must fit in memory, so it is
skipped (and said so in the index) for sequences longer than
`--lookback-max-tokens`.

Reminder for readers: the hidden state at position t encodes token t and is
what predicts token t+1; to study why the model emitted token t, look at t-1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ..core.config import ModelConfig, RecordingSettings
from ..core.errors import ReplayMismatch
from ..core.trajectory import load_tokens


def _layers_for(spec: str, n_layers: int) -> list[int]:
    if spec == "all":
        return list(range(1, n_layers + 1))
    if spec == "quarters":
        return sorted({max(1, round(n_layers * f)) for f in (0.25, 0.5, 0.75, 1.0)})
    layers = sorted({int(x) for x in spec.split(",") if x.strip()})
    if not layers or layers[0] < 1 or layers[-1] > n_layers:
        raise SystemExit(f"--layers must be within 1..{n_layers}")
    return layers


def _positions(step: dict, kinds: set[str] | None, mode: str, n_input: int) -> list[int]:
    positions: list[int] = []
    for seg in step["model_call"].get("segments") or []:
        if kinds is not None and seg["kind"] not in kinds:
            continue
        offset = 0 if seg["where"] == "input" else n_input
        span = list(range(offset + seg["start"], offset + seg["end"]))
        if mode == "last" and span:
            span = span[-1:]
        positions.extend(span)
    return sorted(set(positions))


# However short the reply, this many tokens may differ by more than 0.1 nats
# (see `logprob_check`).
MIN_TOKENS_OVER = 2


def logprob_check(replayed: np.ndarray, recorded: np.ndarray | None, tolerance: float, max_fraction: float) -> tuple[dict, bool]:
    """Index fields comparing replayed with recorded log-probs, and whether the
    step is rejected (shared by `replay` and `lens`).

    The check is on the distribution of the differences, not on the worst
    token. A replay that does not reproduce the generation (other weights,
    other token ids, other quantization) moves many tokens: about 20% by more
    than 0.5 nats in our measurement. bf16 rounding moves few, and moves single
    tokens a long way: on 107 Qwen3-14B bf16 steps, 0.7% of tokens (at most
    2.9% in a step) differed by more than 0.1 nats, while the worst token of a
    step reached 0.73. A rule on the worst token would reject such steps, and
    reject most often the steps holding improbable tokens. So a step is
    rejected only if more than max(`max_fraction` of its tokens,
    MIN_TOKENS_OVER) differ by more than 0.1 nats — the floor keeps a short
    reply from failing over one or two rounding outliers — or if any token
    differs by more than `tolerance`, a coarse guard against a gross local
    error.
    """
    if recorded is None or not len(recorded):
        return {"max_abs_logprob_diff": None, "note": "no recorded log-probs to verify against"}, False
    d = np.abs(replayed - recorded)
    over = int(np.sum(d > 0.1))
    stats = {"max": float(d.max()), "median": float(np.median(d)), "p99": float(np.percentile(d, 99)),
             "fraction_over_0.1": float(over / len(d))}
    fields = {"logprob_check": {**{k: round(v, 6) for k, v in stats.items()}, "tokens_over_0.1": over, "tokens": len(d)},
              "max_abs_logprob_diff": round(stats["max"], 6)}
    allowed = max(max_fraction * len(d), MIN_TOKENS_OVER)
    if over > allowed or stats["max"] > tolerance:
        fields["rejected"] = (
            f"replay does not reproduce the recorded log-probs ({over} of {len(d)} tokens differ by more than 0.1, "
            f"{allowed:g} allowed; max |diff| {stats['max']:.4f}, {tolerance:g} allowed)"
        )
        return fields, True
    return fields, False


def _trajectory_files(run: Path) -> list[Path]:
    if run.is_file():
        return [run]
    return sorted(p for p in (run / "tasks").glob("*.json") if not p.name.endswith(".partial.json"))


def _check_metadata(doc: dict, backend, force: bool) -> list[str]:
    import torch
    import transformers

    problems = []
    recorded = doc["model"]
    if recorded.get("backend") != "hf_local" and recorded.get("backend") != "vllm":
        problems.append(f"recorded with backend {recorded.get('backend')!r}, which records no token ids")
    if recorded.get("revision") and recorded.get("revision") != backend.info.get("revision"):
        problems.append(f"weights revision {recorded.get('revision')} != loaded {backend.info.get('revision')}")
    for key in ("dtype", "quantization"):
        if recorded.get(key) and recorded.get(key) != backend.info.get(key):
            problems.append(f"{key} {recorded.get(key)} != loaded {backend.info.get(key)}")
    versions = recorded.get("versions") or {}
    if versions.get("transformers") and versions["transformers"] != transformers.__version__:
        problems.append(f"transformers {versions['transformers']} != {transformers.__version__}")
    if versions.get("torch") and versions["torch"] != torch.__version__:
        problems.append(f"torch {versions['torch']} != {torch.__version__}")
    if problems and not force:
        raise ReplayMismatch("refusing to replay: " + "; ".join(problems) + " (use --force to override)")
    return problems


class Replayer:
    def __init__(self, model_cfg: dict, device_map: str):
        from ..models.hf_local import HFLocalBackend

        cfg = ModelConfig(**{**model_cfg, "load": {**model_cfg.get("load", {}), "device_map": device_map}})
        self.backend = HFLocalBackend(cfg, RecordingSettings(token_ids=False, logprobs=False, top_logprobs=0, segments=False))
        self.model = self.backend.model
        self.layers_module = self.model.get_decoder().layers

    def run_step(self, input_ids: np.ndarray, output_ids: np.ndarray, layers: list[int], positions: list[int], lookback: bool):
        import torch

        ids = np.concatenate([input_ids, output_ids]).astype(np.int64)
        n_in, n_out = len(input_ids), len(output_ids)
        device = self.backend._input_device()
        tensor = torch.tensor(ids[None, :], device=device)
        pos = torch.tensor(positions, dtype=torch.long)
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for layer in layers:
            def hook(module, args, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer] = hidden[0].index_select(0, pos.to(hidden.device)).to("cpu", torch.float16)
            handles.append(self.layers_module[layer - 1].register_forward_hook(hook))
        lookback_rows = None
        try:
            kwargs = {"input_ids": tensor, "logits_to_keep": n_out + 1}
            if lookback:
                lookback_rows = []
                attn_handles = []
                for module in self.layers_module:
                    def attn_hook(mod, args, output, store=lookback_rows):
                        weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                        if weights is None:
                            store.append(None)
                            return
                        rows = weights[0, :, n_in - 1 : n_in - 1 + n_out, :].float()  # heads, out, keys
                        context = rows[..., :n_in].sum(-1)
                        total = rows.sum(-1)
                        store.append((context / total.clamp_min(1e-12)).transpose(0, 1).to("cpu", torch.float16))
                    attn_handles.append(module.self_attn.register_forward_hook(attn_hook))
                handles.extend(attn_handles)
                kwargs["output_attentions"] = True
            with torch.no_grad():
                logits = self.model(**kwargs).logits[0].float()
        finally:
            for h in handles:
                h.remove()
        logprobs = torch.log_softmax(logits[:n_out], dim=-1)
        replayed = logprobs.gather(1, torch.tensor(output_ids, device=logprobs.device, dtype=torch.long)[:, None])[:, 0].cpu().numpy()
        lookback_array = None
        if lookback_rows and all(r is not None for r in lookback_rows):
            lookback_array = torch.stack(lookback_rows, dim=1).numpy()  # out, layers, heads
        return captured, replayed, lookback_array


def main(args) -> int:
    from safetensors.numpy import save_file

    files = _trajectory_files(Path(args.run))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("no trajectories found", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kinds = None if args.segments == "all" else set(args.segments.split(","))

    first = json.loads(files[0].read_text())
    replayer = Replayer(first["harness"]["config"]["model"], args.device_map)
    if args.lookback:
        replayer.model.set_attn_implementation("eager")
    n_layers = len(replayer.layers_module)
    layers = _layers_for(args.layers, n_layers)
    index = (out / "index.jsonl").open("a", encoding="utf-8")
    meta_checked = False

    for path in files:
        doc = json.loads(path.read_text())
        if doc["harness"]["config"]["model"] != first["harness"]["config"]["model"]:
            print(f"skip {path.name}: a different model config than the first trajectory", file=sys.stderr)
            continue
        if not meta_checked:
            warnings = _check_metadata(doc, replayer.backend, args.force)
            for w in warnings:
                print(f"WARNING (forced): {w}", file=sys.stderr)
            meta_checked = True
        try:
            tokens = load_tokens(path)
        except FileNotFoundError:
            print(f"skip {path.name}: no token sidecar", file=sys.stderr)
            continue
        arrays: dict[str, np.ndarray] = {}
        for step in doc["steps"]:
            key = f"step_{step['step']}"
            t = tokens.get(key)
            if not t or "input_ids" not in t or "output_ids" not in t:
                continue
            n_in = len(t["input_ids"])
            positions = _positions(step, kinds, args.tokens, n_in)
            if not positions:
                continue
            seq = n_in + len(t["output_ids"])
            want_lookback = args.lookback and seq <= args.lookback_max_tokens
            captured, replayed, lookback = replayer.run_step(t["input_ids"], t["output_ids"], layers, positions, want_lookback)
            row = {"trajectory_id": doc["trajectory_id"], "file": path.name.replace(".json", ".activations.safetensors"),
                   "step": step["step"], "kind": step["kind"], "layers": layers, "positions": len(positions),
                   "segments": sorted({s["kind"] for s in step["model_call"].get("segments") or [] if kinds is None or s["kind"] in kinds})}
            fields, rejected = logprob_check(replayed, t.get("logprobs"), args.tolerance, args.max_fraction)
            row.update(fields)
            if rejected:
                index.write(json.dumps(row) + "\n")
                continue
            arrays[f"{key}/positions"] = np.asarray(positions, dtype=np.int32)
            for layer, tensor in captured.items():
                arrays[f"{key}/layer_{layer}"] = tensor.numpy()
            if lookback is not None:
                arrays[f"{key}/lookback"] = lookback.astype(np.float16)
            elif args.lookback:
                row["lookback"] = f"skipped: {seq} tokens exceeds --lookback-max-tokens {args.lookback_max_tokens}"
            index.write(json.dumps(row) + "\n")
            index.flush()
        if arrays:
            save_file(arrays, str(out / path.name.replace(".json", ".activations.safetensors")))
        print(f"{path.name}: {len([k for k in arrays if k.endswith('/positions')])} step(s) written", flush=True)
    index.close()
    return 0
