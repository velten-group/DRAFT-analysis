"""Stability of DRAFT LR predictions across germline seeds, with random controls.

Reads the per-seed calls of scripts/list_ccc_peak_shifts.py and, for every germline-receiver direction, reports
(1) early/late calls: pairwise agreement between seeds and Fleiss' kappa over all seeds (pairs timed in every seed),
and how many pairs keep the same class in 5, 4 or 3 seeds; (2) peak positions: mean distance between the peaks of two
seeds in germline stages and the share within 2 stages; (3) ranking: Spearman correlation of the earliness score
between seeds; and, for reference, the agreement of each seed with the expression product, CellChat and LIANA+.

Two random controls for (1) and (2), 1,000 simulations each: uniform (every seed puts each pair's peak on a uniformly
random stage) and permutation (each seed's own predictions shuffled across pairs, keeping its distribution of
peaks).

    python scripts/ccc_seed_stability.py [--simulations 1000]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ccc_unified import LAST_EARLY, path

DIRECTIONS = ["hub_to_germline", "soma_to_germline"]
BASELINES = {"expression": "LR expression", "CellChat": "CellChat", "LIANA": "LIANA+"}
TIMED = ["early", "late"]


def fleiss_kappa(early: np.ndarray) -> float:
    """early: pairs x seeds, boolean early calls."""
    k = early.shape[1]
    counts = np.stack([early.sum(axis=1), k - early.sum(axis=1)], axis=1)
    p_j = counts.sum(axis=0) / counts.sum()
    p_i = (counts * (counts - 1)).sum(axis=1) / (k * (k - 1))
    p_e = (p_j ** 2).sum()
    return float((p_i.mean() - p_e) / (1 - p_e)) if p_e < 1 else np.nan


def call_metrics(early: np.ndarray, peak: np.ndarray) -> dict[str, float]:
    """early: pairs x seeds booleans; peak: pairs x seeds germline stage indices."""
    k = early.shape[1]
    combos = list(itertools.combinations(range(k), 2))
    agree = np.mean([(early[:, a] == early[:, b]).mean() for a, b in combos])
    dist = np.mean([np.abs(peak[:, a] - peak[:, b]).mean() for a, b in combos])
    within = np.mean([(np.abs(peak[:, a] - peak[:, b]) <= 2).mean() for a, b in combos])
    majority = np.maximum(early.sum(axis=1), k - early.sum(axis=1))
    out = {"agreement": agree, "kappa": fleiss_kappa(early), "distance": dist, "within2": within}
    for m in range(k, (k - 1) // 2, -1):
        out[f"same_{m}"] = float((majority == m).sum())
    return out


def summarise(sims: list[dict[str, float]]) -> dict[str, tuple[float, float, float]]:
    frame = pd.DataFrame(sims)
    return {c: (frame[c].mean(), frame[c].quantile(0.025), frame[c].quantile(0.975)) for c in frame.columns}


def span(values: list[float], fmt: str) -> str:
    v = np.asarray([x for x in values if np.isfinite(x)])
    return "n/a" if v.size == 0 else f"{fmt.format(v.mean())} ({fmt.format(v.min())}-{fmt.format(v.max())})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/fly/ccc_unified.yaml")
    parser.add_argument("--simulations", type=int, default=1000)
    args = parser.parse_args()
    cfg = yaml.safe_load(path(args.config).read_text(encoding="utf-8"))
    gate, lig = float(cfg["receptor_gate"]), float(cfg["ligand_min_frac"])
    tag = f"gate{gate:g}" + (f"_lig{lig:g}" if abs(lig - 0.10) > 1e-9 else "")
    run = path(cfg["output_dir"]) / f"peak_shifts_{tag}"
    calls = pd.read_csv(run / "calls.tsv", sep="\t", dtype={"seed": str, "peak_stage": str})
    labels = json.loads(path(cfg["germ_stage_labels"]).read_text(encoding="utf-8"))["labels"]
    stage_index = {s: i for i, s in enumerate(labels)}
    lit = pd.read_csv(path(cfg["literature_pairs"]), sep="\t", dtype=str)
    lit_keys = {(d, a.lower(), b.lower()) for d, a, b in zip(lit.direction, lit.ligand, lit.receptor)}
    calls["lit"] = [(d, a.lower(), b.lower()) in lit_keys for d, a, b in zip(calls.direction, calls.ligand, calls.receptor)]
    rng = np.random.default_rng(2025)

    rows, controls, per_pair = [], [], []
    for direction in DIRECTIONS:
        part = calls[calls.direction == direction]
        draft = part[part.readout == "DRAFT_A"]
        seeds = sorted(draft.seed.unique())
        if len(seeds) < 2:
            print(f"{direction}: fewer than two germline seeds in {run / 'calls.tsv'}")
            continue
        status = draft.pivot_table(index=["ligand", "receptor"], columns="seed", values="status", aggfunc="first")
        peak = draft.pivot_table(index=["ligand", "receptor"], columns="seed", values="peak_stage", aggfunc="first").apply(lambda c: c.map(stage_index))
        early_score = draft.pivot_table(index=["ligand", "receptor"], columns="seed", values="earliness", aggfunc="first")
        is_lit = draft.drop_duplicates(["ligand", "receptor"]).set_index(["ligand", "receptor"]).lit.reindex(status.index)
        base = {name: part[part.readout == name].set_index(["ligand", "receptor"]).status.reindex(status.index) for name in BASELINES}
        for scope, mask in (("all candidates", np.ones(len(status), bool)), ("literature", is_lit.to_numpy())):
            st, pk, ea = status[mask], peak[mask], early_score[mask]
            timed_all = st.isin(TIMED).all(axis=1).to_numpy()
            e = (st[timed_all] == "early").to_numpy()
            p = pk[timed_all].to_numpy(dtype=float)
            observed = call_metrics(e, p)
            n_stage = len(labels)
            uniform, permuted = [], []
            for _ in range(args.simulations):
                u = rng.integers(0, n_stage, size=p.shape).astype(float)
                uniform.append(call_metrics(u <= LAST_EARLY["germline"], u))
                order = np.stack([rng.permutation(len(p)) for _ in seeds], axis=1)
                permuted.append(call_metrics(np.take_along_axis(e, order, axis=0), np.take_along_axis(p, order, axis=0)))
            for name, sims in (("DRAFT", None), ("Uniform random", uniform), ("Permuted DRAFT", permuted)):
                rec = {"direction": direction, "scope": scope, "pairs": int(timed_all.sum()), "model": name}
                if sims is None:
                    rec.update({k: v for k, v in observed.items()})
                else:
                    rec.update({k: f"{m:.3g} ({lo:.3g}-{hi:.3g})" for k, (m, lo, hi) in summarise(sims).items()})
                controls.append(rec)
            agree, dist, within, rho = [], [], [], []
            for a, b in itertools.combinations(seeds, 2):
                both = st[a].isin(TIMED) & st[b].isin(TIMED)
                agree.append((st.loc[both, a] == st.loc[both, b]).mean())
                d = (pk.loc[both, a] - pk.loc[both, b]).abs()
                dist.append(d.mean())
                within.append((d <= 2).mean())
                rho.append(spearmanr(ea[a], ea[b], nan_policy="omit")[0])
            vs_base = {}
            for name, label in BASELINES.items():
                other = base[name][mask]
                vals = []
                for s in seeds:
                    both = st[s].isin(TIMED) & other.isin(TIMED)
                    vals.append((st.loc[both, s] == other[both]).mean() if both.any() else np.nan)
                vs_base[label] = vals
            majority = np.maximum(e.sum(axis=1), e.shape[1] - e.sum(axis=1))
            rows.append({
                "direction": direction, "scope": scope, "pairs": int(mask.sum()), "timed in all seeds": int(timed_all.sum()),
                "seed-seed agreement": span([x * 100 for x in agree], "{:.0f}%"),
                "Fleiss kappa": f"{observed['kappa']:.2f}",
                "same class in 5/4/3 seeds": "/".join(str(int((majority == k).sum())) for k in (5, 4, 3)) if len(seeds) == 5 else "",
                "peak distance (stages)": span(dist, "{:.1f}"),
                "peaks within 2 stages": span([x * 100 for x in within], "{:.0f}%"),
                "earliness Spearman": span(rho, "{:.2f}"),
                **{f"DRAFT vs {label}": span([x * 100 for x in v], "{:.0f}%") for label, v in vs_base.items()},
            })
        per_pair.append(pd.DataFrame({"direction": direction, "literature": is_lit, "seeds_early": (status == "early").sum(axis=1),
                                      "seeds_timed": status.isin(TIMED).sum(axis=1),
                                      "peak_min": peak.min(axis=1), "peak_max": peak.max(axis=1)}).reset_index())
    table = pd.DataFrame(rows)
    table.to_csv(run / "seed_stability.tsv", sep="\t", index=False)
    control_table = pd.DataFrame(controls)
    control_table.to_csv(run / "seed_stability_controls.tsv", sep="\t", index=False)
    if per_pair:
        pd.concat(per_pair).to_csv(run / "seed_stability_pairs.tsv", sep="\t", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 40)
    print("mean (min-max) over the seed pairs; DRAFT vs baselines: mean (min-max) over seeds")
    print(table.T.to_string(header=False))
    print(f"\nrandom controls, mean (95% interval) over {args.simulations} simulations")
    print(control_table.T.to_string(header=False))
    print(f"\nwritten: {run / 'seed_stability.tsv'}, {run / 'seed_stability_controls.tsv'}, {run / 'seed_stability_pairs.tsv'}")


if __name__ == "__main__":
    main()
