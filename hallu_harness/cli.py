"""Command line.

    python -m hallu_harness run    --config configs/frameworks/smolagents.yaml [--set KEY=VALUE ...]
    python -m hallu_harness check  --config configs/frameworks/qwen_agent.yaml [--set ...] [--count-tasks]
    python -m hallu_harness diff   CONFIG_A CONFIG_B
    python -m hallu_harness list   tools|benchmarks|frameworks|backends
    python -m hallu_harness audit-humaneval [--out PATH]
    python -m hallu_harness export --run RUN_DIR --format v1|agenthallu --out DIR
    python -m hallu_harness replay --run RUN_DIR [--layers 16,32] [--segments task,observation] ...
    python -m hallu_harness lens   --run RUN_DIR [--out DIR] [--top-k 5] ...
    python -m hallu_harness halogen-data [--out DIR]
    python -m hallu_harness gaia-data [--revision SHA]

`--set` values are parsed as YAML: `--set model.generation.temperature=0.7`,
`--set dataset=humaneval` (swap a preset), `--set dataset.selection.ids=[2019/A]`.
"""
from __future__ import annotations

import argparse
import json
import sys

import yaml

from . import plugins  # registers the HALoGEN benchmark, the ungraded frameworks and the web/GitHub tools
from . import qa_plugins  # registers GAIA, the QA frameworks and the reading, search and python tools
from .core import registry
from .core.config import diff_configs, load_config
from .core.errors import ConfigError


def _load_checked(args):
    """`load_config`, then the plug-ins' cross-checks (`plugins.check_config`)."""
    cfg = load_config(args.config, args.sets, args.configs_root)
    qa_plugins.check_config(cfg)
    for warning in plugins.check_config(cfg):
        print(f"[warning] {warning}", file=sys.stderr, flush=True)
    return cfg


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="A framework config (configs/frameworks/*.yaml).")
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--configs-root", default=None, help="Where models/ and datasets/ presets live.")


def cmd_run(args) -> int:
    from datetime import datetime, timezone

    from .core.runlog import claim_log_path, run_log
    from .core.trajectory import new_run_id
    from .runner import Runner

    cfg = _load_checked(args)
    started = datetime.now(timezone.utc)
    run_id = new_run_id(cfg, started)
    # Opened before the model loads, so the log has everything the terminal shows.
    with run_log(claim_log_path(cfg, started)) as log:
        print(f"[config] {cfg.framework} | model {cfg.model_preset} ({cfg.model.id}) | dataset {cfg.dataset_preset} | sha256 {cfg.sha256[:12]}", flush=True)
        if log is not None:
            print(f"[log] {log} (run {run_id})", flush=True)
        root = Runner(cfg, run_id=run_id).run()
        print(f"[done] {root}", flush=True)
    return 0


def cmd_check(args) -> int:
    cfg = _load_checked(args)
    print(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True))
    print(f"config sha256: {cfg.sha256}")
    print(f"tools: {', '.join(cfg.tool_names)}")
    if args.count_tasks:
        benchmark = registry.get("benchmark", cfg.dataset.benchmark)(cfg.dataset)
        ids = [task.task_id for task in benchmark.load()]
        print(f"tasks: {len(ids)} x {cfg.run.samples_per_task} sample(s)")
        print("first: " + ", ".join(ids[:10]) + (" ..." if len(ids) > 10 else ""))
    return 0


def cmd_diff(args) -> int:
    a = load_config(args.a, args.sets, args.configs_root)
    b = load_config(args.b, args.sets, args.configs_root)
    diffs = diff_configs(a, b)
    if not diffs:
        print("identical")
        return 0
    confounds = [s for s in ("model", "dataset", "run") if s in diffs]
    for section, rows in diffs.items():
        print(f"[{section}]")
        for key, va, vb in rows:
            print(f"  {key}: {json.dumps(va, default=str)}  !=  {json.dumps(vb, default=str)}")
    if confounds:
        print(f"\nWARNING: {', '.join(confounds)} differ — a framework comparison between these configs is confounded.")
        return 2
    return 0


def cmd_list(args) -> int:
    for name in registry.names(args.kind.rstrip("s")):
        print(name)
    return 0


def cmd_audit(args) -> int:
    from .benchmarks.humaneval.audit import main as audit_main

    return audit_main(args.out, args.workers)


def cmd_export(args) -> int:
    if args.format == "v1":
        from .exporters.v1_cf_agent import export_run
    else:
        from .exporters.agenthallu import export_run
    out = export_run(args.run, args.out)
    print(f"exported to {out}")
    return 0


def cmd_replay(args) -> int:
    from .interp.extract import main as replay_main

    return replay_main(args)


def cmd_lens(args) -> int:
    from .interp.lens import main as lens_main

    return lens_main(args)


def cmd_halogen_data(args) -> int:
    from .benchmarks.halogen_code_imports.data import main as halogen_data_main

    return halogen_data_main(args.out)


def cmd_gaia_data(args) -> int:
    from .benchmarks.gaia.report import main as gaia_data_main

    return gaia_data_main(args.revision)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hallu_harness", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run a framework config")
    _add_config_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("check", help="resolve, validate and print a config; loads no model")
    _add_config_args(p)
    p.add_argument("--count-tasks", action="store_true", help="also load the dataset and count tasks")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("diff", help="what differs between two framework configs")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--set", dest="sets", action="append", default=[])
    p.add_argument("--configs-root", default=None)
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("list", help="registered plug-ins")
    p.add_argument("kind", choices=["tools", "benchmarks", "frameworks", "backends"])
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("audit-humaneval", help="build the HumanEval exclusion list (DESIGN.md §8.2)")
    p.add_argument("--out", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("export", help="convert a run's trajectories to another schema")
    p.add_argument("--run", required=True)
    p.add_argument("--format", choices=["v1", "agenthallu"], required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("replay", help="recompute activations offline from recorded tokens (DESIGN.md §11.4)")
    p.add_argument("--run", required=True, help="a run directory (or one trajectory .json)")
    p.add_argument("--out", required=True, help="where to write activation shards and their index")
    p.add_argument("--layers", default="quarters", help="'quarters' (25/50/75/100%%), 'all', or a list like 8,16,24")
    p.add_argument("--segments", default="all", help="segment kinds to keep, e.g. observation,think,tool_call; or 'all'")
    p.add_argument("--tokens", choices=["all", "last"], default="all", help="every token of a kept segment, or only its last")
    p.add_argument("--lookback", action="store_true", help="also compute per-head lookback ratios (Chuang et al. 2024)")
    p.add_argument("--lookback-max-tokens", type=int, default=8192, help="skip lookback for longer sequences (eager attention memory)")
    p.add_argument("--tolerance", type=float, default=2.0, help="reject a step if any token's |replayed - recorded| log-prob exceeds this")
    p.add_argument("--max-fraction", type=float, default=0.05,
                   help="reject a step if more than this fraction of its tokens (and more than 2) differ by > 0.1")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--force", action="store_true", help="replay despite a version or revision mismatch")
    p.add_argument("--limit", type=int, default=None, help="only the first N trajectories")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("lens", help="logit-lens readouts, 4 per layer, for every recorded token (DESIGN.md §11.5)")
    p.add_argument("--run", required=True, help="a run directory (or one trajectory .json)")
    p.add_argument("--out", default=None, help="where to write the lens files (default: <run>/lens)")
    p.add_argument("--top-k", type=int, default=5, help="most probable tokens kept per readout")
    p.add_argument("--chunk", type=int, default=128, help="positions decoded at once (memory: 4 x chunk x vocab floats)")
    p.add_argument("--tolerance", type=float, default=2.0, help="reject a step if any token's |replayed - recorded| log-prob exceeds this")
    p.add_argument("--max-fraction", type=float, default=0.05,
                   help="reject a step if more than this fraction of its tokens (and more than 2) differ by > 0.1")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--force", action="store_true", help="run despite a version or revision mismatch")
    p.add_argument("--limit", type=int, default=None, help="only the first N trajectories")
    p.add_argument("--overwrite", action="store_true", help="recompute trajectories whose lens file already exists")
    p.set_defaults(func=cmd_lens)

    p = sub.add_parser("halogen-data", help="write the PyPI name snapshot and import map the HALoGEN benchmark grades with")
    p.add_argument("--out", default=None, help="folder to write to (default: the benchmark's own folder)")
    p.set_defaults(func=cmd_halogen_data)

    p = sub.add_parser("gaia-data", help="download GAIA's validation split (gated) and count the questions the text tools can serve")
    p.add_argument("--revision", default=None, help="dataset commit (default: the one the benchmark pins)")
    p.set_defaults(func=cmd_gaia_data)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
