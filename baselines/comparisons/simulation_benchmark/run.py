"""Run ONE official external method in its own environment; never trains DRAFT."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from common import (
    METHODS,
    PROJECT,
    acquire_lock,
    candidate_edge_split,
    check_dataset_paths,
    completed,
    digest,
    evaluate,
    fingerprint,
    load_inputs,
    load_module,
    load_truth,
    mark_complete,
    read_json,
    resolve_data_root,
    write_inputs,
    write_json,
)


def update_context(args, **details):
    job = os.environ.get("SIMULATION_JOB_DIR")
    if job:
        write_json(
            Path(job) / "active_run.json",
            {
                "method": args.method,
                "config": str(args.config),
                "started_at": time.time(),
                **details,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Existing simulation YAML, read only. Uses datasets section.",
    )
    parser.add_argument(
        "--settings", type=Path, help="Explicit baseline-only settings JSON"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override datasets.data_dir, containing seed_<d>/",
    )
    parser.add_argument("--data-seeds", type=int, nargs="+")
    parser.add_argument(
        "--data-project",
        type=Path,
        help="Project holding the input data/ directory; defaults to SIMULATION_DATA_PROJECT or this checkout",
    )
    parser.add_argument("--model-seeds", type=int, nargs="+")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--run-label",
        help="Separate output condition under this method's runs/; data/config unchanged",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short software check, tagged separately; not benchmark results",
    )
    args = parser.parse_args()
    update_context(args)
    if args.cpus < 1:
        parser.error("--cpus must be positive")
    if args.run_label is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_label
    ):
        parser.error("--run-label must be a single safe directory name")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    import multiome_inputs

    use_multiome = multiome_inputs.enabled(cfg)
    if use_multiome:
        multiome_inputs.reference(cfg)
    datasets, draft = cfg["datasets"], cfg["draft"]
    times = list(draft["time_points"])
    smoothness_metric = str(draft.get("smoothness_metric", "frobenius"))
    if smoothness_metric not in {"frobenius", "squared_frobenius"}:
        raise ValueError(
            "draft.smoothness_metric must be frobenius or squared_frobenius"
        )
    if len(times) < 2 or len(set(times)) != len(times):
        raise ValueError("Expected at least two distinct ordered time points")
    root = resolve_data_root(
        cfg, data_root=args.data_root, data_project=args.data_project
    )
    data_seeds = check_dataset_paths(cfg, root, data_seeds=args.data_seeds)
    method_dir = PROJECT / "comparisons" / f"simulation_{args.method}"
    settings_path = args.settings or method_dir / "settings.json"
    settings = read_json(settings_path)
    adapter = load_module(f"_simulation_{args.method}", method_dir / "adapter.py")
    software = adapter.software(method_dir)
    software["packages"] = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    software["python"] = sys.version
    # Include shared/helper changes in cache identity, not just framework parameters.
    code = {
        str(p.relative_to(PROJECT)): digest(p)
        for p in sorted([*method_dir.glob("*.py"), *Path(__file__).parent.glob("*.py")])
    }
    rows = []
    for data_seed in data_seeds:
        update_context(args, data_seed=data_seed)
        data_dir = root / f"seed_{data_seed}"
        if use_multiome:
            data, sources = multiome_inputs.load_inputs(data_dir, cfg)
        else:
            data, sources = load_inputs(
                data_dir, times, int(draft["num_tf"]), int(draft["num_tg"])
            )
        task = draft.get("task", "truth_union")
        candidate_task = task in {"candidate_dynamic", "prior_reconstruction"}
        heldout_candidate_task = task == "candidate_dynamic"
        evaluation_mask = None
        if heldout_candidate_task:
            _, evaluation_mask = candidate_edge_split(
                data.prior,
                train_fraction=float(draft.get("candidate_train_fraction", 0.7)),
                seed=int(draft.get("split_seed", 2026)),
            )
        condition = args.run_label or root.name
        for seed in args.model_seeds or datasets["model_seeds"]:
            identity = {
                "method": args.method,
                "data_seed": data_seed,
                "data_directory": str(data_dir.resolve()),
                "model_seed": seed,
                "time_points": times,
                "num_tf": len(data.tfs),
                "num_tg": len(data.targets),
                "sources": sources,
                "count_gene_policy": "select explicit IDs 1..num_tf+num_tg; ignore additional background rows; keep all cells",
                "settings": settings,
                "software": software,
                "adapter_code": code,
                "cpus": args.cpus,
                "device": args.device,
                "validation_only": args.smoke,
                "task": task,
                "smoothness_metric": smoothness_metric,
                "candidate_train_fraction": (
                    float(draft.get("candidate_train_fraction", 0.7))
                    if heldout_candidate_task
                    else None
                ),
                "split_seed": (
                    int(draft.get("split_seed", 2026))
                    if heldout_candidate_task
                    else None
                ),
                "prior_policy": (
                    "candidate_prior.csv; full fixed support supplied to models; evaluation uses DRAFT's fixed held-out candidate split"
                    if heldout_candidate_task
                    else "candidate_prior.csv; full fixed support supplied to models and used for within-prior evaluation"
                    if candidate_task
                    else "one fixed truth union across all time points"
                ),
            }
            if use_multiome:
                identity["input_protocol"] = multiome_inputs.PROTOCOL
                identity["input_policy"] = multiome_inputs.POLICIES[args.method]
                identity["reference_config_sha256"] = digest(
                    PROJECT / cfg["benchmark"]["reference_config"]
                )
            if args.run_label is not None:
                identity["run_label"] = args.run_label
            tag = "smoke" if args.smoke else "benchmark"
            run = (
                method_dir
                / "runs"
                / condition
                / f"seed_{data_seed}"
                / f"model_{seed}_{tag}_{fingerprint(identity)[:16]}"
            )
            run.mkdir(parents=True, exist_ok=True)
            update_context(
                args,
                data_seed=data_seed,
                model_seed=seed,
                run_directory=str(run.resolve()),
            )
            lock = acquire_lock(run)
            try:
                if completed(run, identity):
                    print(f"Verified completed run: {run}", flush=True)
                else:
                    write_json(run / "manifest.json", identity)
                    write_inputs(data, run / "inputs")
                    print(
                        f"START {args.method} data={data_seed} model={seed}: {run}",
                        flush=True,
                    )
                    scores, provenance = adapter.fit(
                        data, settings, run, seed, args.cpus, args.device, args.smoke
                    )
                    # Verify shared input files were not changed during a long fit.
                    for name, expected in sources.items():
                        if digest(data_dir / name) != expected:
                            raise ValueError(
                                f"Source changed during fitting: {data_dir / name}"
                            )
                    if use_multiome:
                        truth, same = multiome_inputs.truth_and_modules(data_dir, data)
                        metrics = multiome_inputs.evaluate(
                            scores,
                            truth,
                            data,
                            same,
                            multiome_inputs.reference(cfg)["multiome"]["top_k"],
                        )
                    else:
                        truth = load_truth(
                            data_dir, times, len(data.tfs), len(data.targets)
                        )
                        metrics = evaluate(
                            data,
                            truth,
                            scores,
                            seed,
                            evaluation_mask=evaluation_mask,
                            smoothness_metric=smoothness_metric,
                        )
                    metrics.update(
                        {
                            "validation_only": args.smoke,
                            "method": args.method,
                            "provenance": provenance,
                        }
                    )
                    np.savez_compressed(
                        run / "relations.npz",
                        scores=scores,
                        time_points=data.times,
                        tfs=data.tfs,
                        targets=data.targets,
                    )
                    for time, matrix in zip(data.times, scores):
                        pd.DataFrame(
                            matrix, index=data.targets, columns=data.tfs
                        ).to_csv(run / f"scores_time_{time}.tsv", sep="\t")
                    write_json(run / "scores.json", metrics)
                    mark_complete(run, identity)
                metrics = read_json(run / "scores.json")
                reported = (
                    {
                        key: metrics[key]
                        for key in (
                            "candidate_auroc",
                            "same_module_auroc",
                            "temporal_continuity",
                            "change_auroc",
                            "f1_at_k",
                        )
                    }
                    if use_multiome
                    else {
                        "auroc_mean": metrics["auroc_mean"],
                        "auroc_within_union_mean": metrics["auroc_within_union_mean"],
                        "auroc_heldout_candidate_mean": metrics.get(
                            "auroc_heldout_candidate_mean"
                        ),
                        "smoothness": metrics["smoothness"],
                    }
                )
                rows.append(
                    {
                        "data_seed": data_seed,
                        "model_seed": seed,
                        "method": args.method,
                        "validation_only": args.smoke,
                        **reported,
                        "run": str(run.relative_to(PROJECT)),
                    }
                )
            finally:
                lock.unlink()
    # An invocation-specific summary cannot overwrite results from another setting or subset.
    summary = method_dir / "runs" / condition / f"summary_{fingerprint(rows)[:16]}.tsv"
    pd.DataFrame(rows).to_csv(summary, sep="\t", index=False)
    print(f"Summary: {summary}", flush=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
