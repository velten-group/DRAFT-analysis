"""Official scMTNI on the branched simulation (simulation_gnn.branch), with the lineage tree in place of the chain.

The official adapter (comparisons/simulation_scmtni/adapter.py), binary and settings are used unchanged, except that tree.txt lists
each state with its parent (T1 <- T0, A2 and B2 <- T1, ...) instead of a chain. scMTNI reads this tree natively: one parent per
child, any number of children, as in its own lineage inputs. Inputs, priors and the score (selection frequency over 50 half-cell
subsamples) are those of the main study.

scMTNI must be fitted on the same data as the internal models, so on the same machine: scMultiSim does not give identical RNA
across platforms for the same seed. Fitting checks the generation receipt and inputs as run.py does; the summary scores the
internal fits found here (simulation_gnn.branch) and requires every internal and scMTNI fit to record this data's receipt.

    # simulation_scmtni environment
    python comparisons/simulation_branch/scmtni_tree.py run [--workers 9]
    python comparisons/simulation_branch/scmtni_tree.py fit --data 101 --seed 0
    # project environment, PYTHONPATH=src, after python -m simulation_gnn.branch run
    python comparisons/simulation_branch/scmtni_tree.py summary
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "comparisons/simulation_benchmark"))

import numpy as np
import multiome_inputs
import yaml
from common import digest, load_module, read_json, write_json

VERSION = "simulation_branch_v1"  # simulation_gnn.branch, which needs Torch and is not imported here
STATES = ("T0", "T1", "A2", "A3", "A4", "B2", "B3", "B4")
PARENTS = (-1, 0, 1, 2, 3, 1, 5, 6)
DATA_SEEDS = (101, 102, 103)
MODEL_SEEDS = (0, 1, 2)
DATA = PROJECT / "data" / VERSION
RESULT = PROJECT / "result" / VERSION / "scmtni"
REPORT = PROJECT / "comparisons/simulation_benchmark/runs" / VERSION
METHOD_DIR = PROJECT / "comparisons/simulation_scmtni"
SETTINGS = os.environ.get("SCMTNI_SETTINGS", str(METHOD_DIR / "selected_settings.json"))
GENERATION_CODE = ("branch.py", "prior_m3_design.py", "R/prior_branch.R", "R/prior_trial_context.R", "R/positive_kinetics.R")


def receipt(seed):
    """The data's generation receipt: its file digests, and the digest of the receipt itself."""
    path = DATA / f"seed_{seed}" / "generation_complete.json"
    return read_json(path)["files"], digest(path)


def same_data(sources, files, marker):
    """multiome_inputs records the receipt's files plus the receipt itself as an official fit's sources."""
    return sources == dict(files, **{"generation_complete.json": marker})


def tree_lines(labels, gain, loss):
    """scMTNI's lineage file: child, parent, gain and loss probabilities, children in topological order."""
    return [f"{labels[child]}\t{labels[parent]}\t{gain}\t{loss}" for child, parent in enumerate(PARENTS) if parent >= 0]


def config():
    """The branch simulation's config (branch.config) in the form multiome_inputs.load_inputs reads."""
    cfg = yaml.safe_load((PROJECT / "configs/simulation_main_attention.yaml").read_text(encoding="utf-8"))
    params = dict(cfg["simulate"]["params"], times=len(STATES))
    return {
        "benchmark": {"allow_generator_line_ending_equivalence": True},
        "simulate": {"params": params},
        "draft": {"num_tf": cfg["draft"]["num_tf"], "num_tg": cfg["draft"]["num_tg"], "time_points": list(range(len(STATES)))},
    }


def generation_identity(cfg, seed):
    """The identity branch.ensure_data records with the data."""
    return {
        "version": VERSION,
        "params": dict(cfg["simulate"]["params"], seed=seed),
        "tree": [list(STATES), list(PARENTS)],
        "code": {f"src/simulation_gnn/{name}": digest(PROJECT / "src/simulation_gnn" / name) for name in GENERATION_CODE},
    }


def load(seed):
    folder = DATA / f"seed_{seed}"
    if not (folder / "generation_complete.json").is_file():
        raise FileNotFoundError(f"No branch data at {folder}; generate it with simulation_gnn.branch generate {seed}")
    cfg = config()
    with patch.object(multiome_inputs, "reference", lambda c: c), patch.object(multiome_inputs, "generation_identity", generation_identity):
        return multiome_inputs.load_inputs(folder, cfg)


def run_dir(seed, model_seed):
    return RESULT / f"seed_{seed}" / f"model_{model_seed}"


def fit(seed, model_seed):
    adapter = load_module("_branch_scmtni_adapter", METHOD_DIR / "adapter.py")
    settings = read_json(SETTINGS)
    data, sources = load(seed)
    labels = [f"time{t}" for t in data.times]
    identity = {
        "version": VERSION,
        "method": "scmtni",
        "data_seed": seed,
        "model_seed": model_seed,
        "settings": settings,
        "software": adapter.software(METHOD_DIR),
        "code": {path.relative_to(PROJECT).as_posix(): digest(path) for path in (METHOD_DIR / "adapter.py", Path(__file__))},
        "sources": sources,
        "tree": tree_lines(labels, settings["gain"], settings["loss"]),
    }
    run = run_dir(seed, model_seed)
    receipt = run / "complete.json"
    if receipt.is_file():
        record = read_json(receipt)
        if record["identity"] != json.loads(json.dumps(identity)) or digest(run / "relations.npz") != record["prediction_sha256"]:
            raise ValueError(f"Completed branch scMTNI run differs: {run}; use a new directory")
        print(f"[REUSE] scmtni data={seed} model={model_seed}", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "manifest.json", identity)
    original = adapter.prepare

    def prepare(data, settings, folder, cell_indices):
        original(data, settings, folder, cell_indices)
        adapter.lines(folder / "tree.txt", tree_lines(labels, settings["gain"], settings["loss"]))

    with patch.object(adapter, "prepare", prepare):
        scores, provenance = adapter.fit(data, settings, run, model_seed, 1, "cpu", False)
    for name, expected in sources.items():
        if digest(DATA / f"seed_{seed}" / name) != expected:
            raise ValueError(f"Source changed during fitting: {name}")
    provenance["lineage"] = "branched tree T0-T1, then A2-A4 and B2-B4 from T1; gain/loss are the prespecified constants"
    np.savez_compressed(run / "relations.npz", scores=scores, time_points=data.times, tfs=data.tfs, targets=data.targets)
    write_json(receipt, {"identity": identity, "provenance": provenance, "prediction_sha256": digest(run / "relations.npz")})
    print(f"[RESULT] scmtni data={seed} model={model_seed}", flush=True)


def run_all(workers):
    for seed in DATA_SEEDS:
        load(seed)  # stop before any fit if the data are missing or changed
    logs = REPORT / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def one(job):
        seed, model_seed = job
        with (logs / f"scmtni__{seed}_{model_seed}.out").open("w") as log:
            command = [sys.executable, "-u", str(Path(__file__)), "fit", "--data", str(seed), "--seed", str(model_seed)]
            return job, subprocess.run(command, check=False, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT).returncode

    jobs = [(seed, model_seed) for seed in DATA_SEEDS for model_seed in MODEL_SEEDS]
    with ThreadPoolExecutor(workers) as pool:
        failed = [job for job, code in pool.map(one, jobs) if code]
    if failed:
        raise RuntimeError(f"scMTNI fits failed: {failed}; see {logs}")
    print(f"[END] {len(jobs)} scMTNI fits complete; run the summary in the project environment", flush=True)


def summary():
    """Score the internal fits and scMTNI found here with the branch simulation's tree measures, after checking that all used this data."""
    import pandas as pd

    from simulation_gnn import branch
    from simulation_gnn import main_m3_study as study

    if (branch.STATES, branch.PARENTS) != (STATES, PARENTS):
        raise ValueError("Tree differs from simulation_gnn.branch")
    trial = study.load_study()[0].trial
    if branch.summary() is None:
        raise RuntimeError("No internal fits here; run python -m simulation_gnn.branch run first")
    internal = pd.read_csv(REPORT / "runs.tsv", sep="\t")
    rows, missing = [], []
    for seed in DATA_SEEDS:
        folder = DATA / f"seed_{seed}"
        files, marker = receipt(seed)
        for method, model_seed in internal.loc[internal.data.eq(seed), ["method", "seed"]].itertuples(index=False):
            if read_json(branch.run_dir(seed, model_seed, method) / "complete.json")["identity"]["inputs"] != files:
                raise ValueError(f"{method} data={seed} model={model_seed} was fitted on other data than {folder}")
        data = trial.load_inputs(folder, branch.config())
        truth, same = trial.m.load_truth(folder, data)
        for model_seed in MODEL_SEEDS:
            run = run_dir(seed, model_seed)
            if not (run / "complete.json").is_file():
                missing.append(f"{seed}_{model_seed}")
                continue
            record = read_json(run / "complete.json")
            if digest(run / "relations.npz") != record["prediction_sha256"]:
                raise ValueError(f"Saved prediction changed: {run}")
            if not same_data(record["identity"]["sources"], files, marker):
                raise ValueError(f"scMTNI data={seed} model={model_seed} was fitted on other data than {folder}")
            with np.load(run / "relations.npz", allow_pickle=False) as stored:
                scores = stored["scores"]
            rows.append({"method": "scmtni", "data": seed, "seed": model_seed, **branch.score(scores, scores, truth, same, data.candidate)})
    print(f"Missing {len(missing)}: {missing}", flush=True)
    if not rows:
        return None
    runs = pd.concat([internal, pd.DataFrame(rows)], ignore_index=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    runs.to_csv(REPORT / "runs_with_scmtni.tsv", sep="\t", index=False)
    metrics = ["candidate_auroc", "trunk_auroc", "branch_states_auroc", "branch_auroc", "change_auroc", "mean_continuity", "same_module_auroc", "true_vs_outside_auroc"]
    per_data = runs.groupby(["method", "data"])[metrics].mean()
    table = per_data.groupby(level=0).agg(["mean", "std"])
    table.columns = [f"{a}{'' if b == 'mean' else '_sd_between_data'}" for a, b in table.columns]
    table.insert(0, "runs", runs.groupby("method").size())
    table = table.loc[[m for m in (*study.INTERNAL, "scmtni") if m in table.index]]
    table.round(4).to_csv(REPORT / "summary_with_scmtni.tsv", sep="\t")
    pd.set_option("display.width", 240)
    print(table[["runs", *metrics]].round(3).rename(index=study.LABELS).to_string(), flush=True)
    paired = runs.pivot_table(index=["data", "seed"], columns="method", values=["candidate_auroc", "branch_auroc"]).dropna()
    for metric in ("candidate_auroc", "branch_auroc"):
        print(f"DRAFT above scMTNI, {metric}: {int((paired[metric]['draft'] > paired[metric]['scmtni']).sum())}/{len(paired)}", flush=True)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("fit", "run", "summary"))
    parser.add_argument("--data", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=len(DATA_SEEDS) * len(MODEL_SEEDS))
    args = parser.parse_args()
    if args.action == "fit":
        fit(args.data, args.seed)
    elif args.action == "run":
        run_all(args.workers)
    else:
        summary()


if __name__ == "__main__":
    main()
