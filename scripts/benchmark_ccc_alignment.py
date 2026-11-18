"""Benchmark the soma-germline time alignments of the CCC analysis against the literature pairs.

The alignment decides which sender stage is paired with each receiver stage, so it matters only for soma -> germline
(21 literature pairs) and germline -> soma (one literature pair, Spi-Egfr). Alignments: linear, continuous monotone
warp and DTW (configs/fly/ccc_frozen_*; the DTW path is the one used everywhere else), two alignments built
only from the published cyst-state correspondence (Raz et al. 2023; tj and eya readings of the lSCC groups), plus two
controls:
reversed (linear path with soma time reversed, a deliberately wrong pairing) and none (sender expression averaged
over all sender groups, no timing). For each alignment and method (DRAFT readout A with the 5% receptor gate, the LR
expression product, CellChat, LIANA+; same gates and candidates as run_ccc_unified.py) the script reports the
AUROC of the earliness score of literature pairs against the other candidates, the literature pairs placed early,
and the mean distance of their peaks from GSC in germline stages.

    python scripts/benchmark_ccc_alignment.py [--germ-seed 2025 result/ccc_full_scores/germ ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ccc_unified import (
    LAST_EARLY,
    full_scores,
    group_expression,
    native_scores,
    path,
    soma_alignment,
    soma_time_points,
    to_soma_nodes,
)

from drosophila_gnn.checkpoint import save_checkpoint_metadata

PATHS = {"DTW": "configs/fly/ccc_frozen_dtw_soma_to_germ_parallel.tsv", "Warping": "configs/fly/ccc_frozen_warp_soma_to_germ_parallel.tsv"}
SEQUENTIAL_PATHS = {"DTW, SdA then SdB (original)": "configs/fly/ccc_frozen_dtw_soma_to_germ.tsv"}
METHODS = ["DRAFT", "LR expression", "CellChat", "LIANA+"]
NEAR = 2  # peak-offset rule: within 2 germline stages of GSC counts as correct


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/fly/ccc_unified.yaml")
    parser.add_argument("--germ-seed", nargs=2, action="append", metavar=("SEED", "FULL_DIR"),
                        help="germline full-score export(s) for DRAFT (default: full_scores.germline of the config, labelled 2025)")
    args = parser.parse_args()
    cfg = yaml.safe_load(path(args.config).read_text(encoding="utf-8"))
    gate_value, min_range = float(cfg["receptor_gate"]), float(cfg["dynamic_range_min"])
    lig = float(cfg["ligand_min_frac"])
    tag = f"gate{gate_value:g}" + (f"_lig{lig:g}" if abs(lig - 0.10) > 1e-9 else "")
    out_dir = path(cfg["output_dir"]) / f"alignment_benchmark_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    germ_seeds = args.germ_seed or [("2025", cfg["full_scores"]["germline"])]

    germ_labels = json.loads(path(cfg["germ_stage_labels"]).read_text(encoding="utf-8"))["labels"]
    soma_groups = json.loads(path(cfg["soma_stage_labels"]).read_text(encoding="utf-8"))["groups"]
    groups = {"soma": soma_groups, "germline": [f"Germline_{i}" for i in range(len(germ_labels))]}
    n_germ = len(germ_labels)

    # Receiver stage -> sender position (1-based, on the sender's time points) for each alignment. Soma time points
    # follow the config: the parallel SdA and SdB branches share one time point (their mean). The original sequential
    # DTW path (SdA then SdB) is kept as one row for comparison.
    positions: dict[str, dict] = {}
    for name, file in PATHS.items():
        a = soma_alignment(cfg, soma_groups, soma_groups, n_germ, file)
        positions[name] = {"soma_at_germ": a["soma_at_germ"], "germ_at_soma": a["germ_at_soma"], "nodes": a["nodes"]}
    sequential_cfg = {k: v for k, v in cfg.items() if k != "soma_parallel_branches"}
    for name, file in SEQUENTIAL_PATHS.items():
        a = soma_alignment(sequential_cfg, soma_groups, soma_groups, n_germ, file)
        positions[name] = {"soma_at_germ": a["soma_at_germ"], "germ_at_soma": a["germ_at_soma"], "nodes": a["nodes"]}
    nodes, _ = soma_time_points(cfg, soma_groups, soma_groups)
    n_nodes = len(nodes)
    group_node = np.array([next(i for i, node in enumerate(nodes) if g in node) + 1 for g in soma_groups], dtype=float)
    lin_s = 1 + (np.arange(1, n_germ + 1) - 1) * (n_nodes - 1) / (n_germ - 1)
    lin_g = 1 + (group_node - 1) * (n_germ - 1) / (n_nodes - 1)
    positions["Linear"] = {"soma_at_germ": lin_s, "germ_at_soma": lin_g, "nodes": nodes}
    # Alignments built only from the published cyst-state correspondence (Raz et al. 2023, eLife 12:e82201): CySC with
    # GSC, cyst cells enclosing spermatogonia with Gb to Sp-16, cyst cells enclosing spermatocytes (the SdA/SdB time
    # point: tj off, geko on) with the spermatocyte stages. The lSCC groups keep tj (marker of cyst cells enclosing
    # spermatogonia) but already express eya (marker of cyst cells enclosing spermatocytes): the tj reading pairs them
    # with spermatogonia, the eya reading with spermatocytes.
    first_sc = LAST_EARLY["germline"] + 2
    g_anchor = np.array([1, 2, first_sc - 1, first_sc, n_germ], dtype=float)
    for name, s_anchor, branch_germ in (
        ("Anatomical, tj reading", [1, 2, n_nodes - 1, n_nodes, n_nodes], (first_sc + n_germ) / 2),
        ("Anatomical, eya reading", [1, 2, 4, 5, n_nodes], float(n_germ)),
    ):
        s_anchor = np.array(s_anchor, dtype=float)
        inc = np.r_[True, np.diff(s_anchor) > 0]
        germ_at_soma = np.interp(group_node, s_anchor[inc], g_anchor[inc])
        germ_at_soma[group_node == n_nodes] = branch_germ
        positions[name] = {"soma_at_germ": np.interp(np.arange(1, n_germ + 1), g_anchor, s_anchor), "germ_at_soma": germ_at_soma,
                           "nodes": nodes}
    positions["Reversed (control)"] = {"soma_at_germ": n_nodes + 1 - lin_s, "germ_at_soma": n_germ + 1 - lin_g, "nodes": nodes}
    positions["None (control)"] = {"soma_at_germ": None, "germ_at_soma": None, "nodes": nodes}

    run = path(cfg["output_dir"]) / f"{Path(cfg['receptor_readout']).stem}_gate{gate_value:g}" / ""
    run = run.with_name(run.name + (f"_lig{lig:g}" if abs(lig - 0.10) > 1e-9 else ""))
    pairs = pd.read_csv(run / "pairs.tsv", sep="\t")
    lit = pd.read_csv(path(cfg["literature_pairs"]), sep="\t", dtype=str)
    lit_keys = {(d, a.lower(), b.lower()) for d, a, b in zip(lit.direction, lit.ligand, lit.receptor)}
    tg = pd.read_csv(path(cfg["data"]["tg_list"]), sep="\t", dtype=str)
    sym2fb = dict(zip(tg["gene_name"], tg["FBgn_ID"]))
    directions = {"soma_to_germline": ("soma", "germline"), "germline_to_soma": ("germline", "soma")}
    cand = {d: pairs[(pairs.direction == d) & (pairs.readout == "primary")][["ligand", "receptor"]].reset_index(drop=True) for d in directions}
    fbgns = sorted({sym2fb[x] for d in cand for x in set(cand[d].ligand) | set(cand[d].receptor)})
    all_groups = [cfg["hub_group"], *soma_groups, *groups["germline"]]
    mean, frac, _ = group_expression(cfg, fbgns, all_groups)
    natives = {"CellChat": native_scores(cfg["baselines"]["CellChat"], all_groups), "LIANA+": native_scores(cfg["baselines"]["LIANA"], all_groups)}
    draft = {"soma": [("soma", full_scores(path(cfg["full_scores"]["soma"]), cfg["receptor_readout"])[0])],
             "germline": [(seed, full_scores(path(d), cfg["receptor_readout"])[0]) for seed, d in germ_seeds]}

    rows = []
    for direction, (sender, receiver) in directions.items():
        c = cand[direction]
        label = np.array([(direction, a.lower(), b.lower()) in lit_keys for a, b in zip(c.ligand, c.receptor)])
        split = LAST_EARLY[receiver] + 1
        where_key = "soma_at_germ" if receiver == "germline" else "germ_at_soma"
        for align, pos in positions.items():
            where = pos[where_key]

            def on_sender_axis(values: np.ndarray, nodes: list = pos["nodes"], side: str = sender) -> np.ndarray:
                return to_soma_nodes(values, soma_groups, nodes) if side == "soma" else values

            curves: dict[str, list[np.ndarray]] = {}
            for a, b in zip(c.ligand, c.receptor):
                l_fb, r_fb = sym2fb[a], sym2fb[b]
                gate = (frac.loc[r_fb, groups[receiver]] >= gate_value).to_numpy(float)
                ligand = mean.loc[l_fb, groups[sender]].to_numpy(float)
                ligand_curve = np.full(len(gate), ligand.mean()) if where is None else np.interp(where, np.arange(1, len(on_sender_axis(ligand)) + 1), on_sender_axis(ligand))
                curves.setdefault("LR expression", []).append(ligand_curve * mean.loc[r_fb, groups[receiver]].to_numpy(float) * gate)
                for seed, table in draft[receiver]:
                    curves.setdefault(f"DRAFT|{seed}", []).append(ligand_curve * table.loc[r_fb].to_numpy(float) * gate)
                key = f"{a.lower()}__{b.lower()}"
                for name, native in natives.items():
                    if key not in native:
                        curves.setdefault(name, []).append(np.zeros(len(gate)))
                        continue
                    v = native[key].loc[groups[sender], groups[receiver]].to_numpy(float)
                    if where is None:
                        term = v.mean(axis=0)
                    else:
                        v = on_sender_axis(v)
                        term = np.array([np.interp(where[j], np.arange(1, v.shape[0] + 1), v[:, j]) for j in range(v.shape[1])])
                    curves.setdefault(name, []).append(term * gate)
            for name, cs in curves.items():
                method, _, seed = name.partition("|")
                score, early, peak, near = [], [], [], []
                for s in cs:
                    top = s.max()
                    score.append(0.0 if top <= 0 else (s[:split].max() - s[split:].max()) / top)
                    near.append(0.0 if top <= 0 else (s[:NEAR + 1].max() - s[NEAR + 1:].max()) / top)
                    timed = top > 0 and (top - s.min()) / top >= min_range
                    early.append(timed and int(np.argmax(s)) < split)
                    peak.append(int(np.argmax(s)) if timed else np.nan)
                score, early, peak, near = np.array(score), np.array(early), np.array(peak, dtype=float), np.array(near)
                within = int(np.sum(peak[label] <= NEAR)) if receiver == "germline" else np.nan
                if receiver == "germline":
                    dist = np.nanmean(peak[label])
                else:
                    germ_pos = (positions["DTW"]["germ_at_soma"] - 1.0)
                    dist = np.nanmean([germ_pos[int(p)] for p in peak[label] if np.isfinite(p)]) if label.any() else np.nan
                rows.append({"direction": direction, "alignment": align, "method": method, "model": seed or "-",
                             "literature": int(label.sum()), "found": int(early[label].sum()), "auroc": auroc(score, label),
                             "mean_distance_to_GSC": dist, "within_2_of_GSC": within,
                             "auroc_near_GSC": auroc(near, label) if receiver == "germline" else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "alignment_benchmark.tsv", sep="\t", index=False)
    save_checkpoint_metadata(out_dir / "summary.json", {"germline_seeds": [s for s, _ in germ_seeds]})

    main_dir = table[table.direction == "soma_to_germline"]
    view = main_dir.groupby(["alignment", "method"], sort=False).agg(
        found=("found", "mean"), auroc=("auroc", "mean"), auroc_min=("auroc", "min"), auroc_max=("auroc", "max"), dist=("mean_distance_to_GSC", "mean")).reset_index()
    view["cell"] = [f"{f:.1f}/21 ({a:.2f})" if lo == hi else f"{f:.1f}/21 ({a:.2f}; {lo:.2f}-{hi:.2f})" for f, a, lo, hi in zip(view.found, view.auroc, view.auroc_min, view.auroc_max)]
    wide = view.pivot(index="alignment", columns="method", values="cell").reindex(index=list(positions), columns=METHODS)
    pd.set_option("display.width", 220)
    print("soma -> germline: literature pairs placed early / 21 (AUROC)")
    print(wide.to_string())
    near = main_dir.groupby(["alignment", "method"], sort=False).agg(w=("within_2_of_GSC", "mean"), a=("auroc_near_GSC", "mean")).reset_index()
    near["cell"] = [f"{w:.1f}/21 ({a:.2f})" for w, a in zip(near.w, near.a)]
    print()
    print("soma -> germline, peak-offset rule: literature peaks within 2 stages of GSC / 21 (AUROC of the GSC-Sp-2 window score)")
    print(near.pivot(index="alignment", columns="method", values="cell").reindex(index=list(positions), columns=METHODS).to_string())
    dist = view.pivot(index="alignment", columns="method", values="dist").reindex(index=list(positions), columns=METHODS)
    print("\nsoma -> germline: mean distance of literature peaks from GSC (germline stages)")
    print(dist.round(1).to_string())
    g = table[table.direction == "germline_to_soma"].groupby(["alignment", "method"], sort=False).found.mean().unstack().reindex(index=list(positions), columns=METHODS)
    print("\ngermline -> soma: Spi-Egfr placed early (1 = yes)")
    print(g.to_string())
    print(f"\nwritten: {out_dir / 'alignment_benchmark.tsv'}")


if __name__ == "__main__":
    main()
