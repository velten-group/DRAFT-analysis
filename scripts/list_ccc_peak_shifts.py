"""List LR pairs whose DRAFT peak class differs from the expression product, CellChat and LIANA+.

Rule (configs/fly/ccc_unified.yaml): ligand = mean expression in the sender
cells, receptor = readout A (mean percentile over the active TFs, scored for every receptor) gated at the config's
receptor fraction (5%), LR score on the receiver's integer stages, peak class early/late. A pair is shifted in a
seed if the DRAFT class differs from the class of every baseline that times the pair. Germline receivers are
scored in each germline seed: shifted in >= 4 of 5 seeds is reproducible, in 2-3 less stable. Soma receivers have
one model and are listed separately.

    python scripts/list_ccc_peak_shifts.py
    python scripts/list_ccc_peak_shifts.py --germ-seed 2025 result/ccc_full_scores/germ --soma-full-dir result/ccc_full_scores/soma

Without --germ-seed, seeds 2024-2028 are read from result/ccc_full_scores/germ_seed<seed>. Baselines are recomputed
where their outputs exist and otherwise read from docs/reference/ccc_baselines_<tag>/baseline_calls.tsv; the
recomputed expression calls are checked against that file. Writes to <output_dir>/peak_shifts_<tag>, where <tag> is
gate<receptor gate> plus _lig<sender gate> when the sender gate is not 10%.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ccc_unified import (
    LAST_EARLY,
    call,
    full_scores,
    group_expression,
    native_scores,
    path,
    soma_alignment,
    to_soma_nodes,
)

from drosophila_gnn.checkpoint import save_checkpoint_metadata

SEEDS = ["2024", "2025", "2026", "2027", "2028"]
DIRECTIONS = {
    "hub_to_germline": ("hub", "germline"),
    "soma_to_germline": ("soma", "germline"),
    "hub_to_soma": ("hub", "soma"),
    "germline_to_soma": ("germline", "soma"),
}
BASELINES = ["expression", "CellChat", "LIANA"]
REPRODUCIBLE, LESS_STABLE = 4, 2
READOUT_FILE = "receptor_input_ungated.tsv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/fly/ccc_unified.yaml")
    parser.add_argument("--germ-seed", nargs=2, action="append", metavar=("SEED", "FULL_DIR"))
    parser.add_argument("--soma-full-dir", default=None, help="soma full-score export (default full_scores.soma of the config)")
    args = parser.parse_args()
    cfg = yaml.safe_load(path(args.config).read_text(encoding="utf-8"))
    gate_value = float(cfg["receptor_gate"])
    min_range = float(cfg["dynamic_range_min"])
    lig = float(cfg["ligand_min_frac"])
    tag = f"gate{gate_value:g}" + (f"_lig{lig:g}" if abs(lig - 0.10) > 1e-9 else "")
    out_dir = path(cfg["output_dir"]) / f"peak_shifts_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    germ_seeds = args.germ_seed or [(seed, f"result/ccc_full_scores/germ_seed{seed}") for seed in SEEDS]
    soma_dir = path(args.soma_full_dir or cfg["full_scores"]["soma"])

    germ_labels = json.loads(path(cfg["germ_stage_labels"]).read_text(encoding="utf-8"))["labels"]
    soma_meta = json.loads(path(cfg["soma_stage_labels"]).read_text(encoding="utf-8"))
    labels = {"germline": germ_labels, "soma": [f"{g} ({x})" for g, x in zip(soma_meta["groups"], soma_meta["labels"])]}
    groups = {"hub": [cfg["hub_group"]], "soma": soma_meta["groups"], "germline": [f"Germline_{i}" for i in range(len(germ_labels))]}
    all_groups = groups["hub"] + groups["soma"] + groups["germline"]

    frozen_path = path(f"docs/reference/ccc_baselines_{tag}/baseline_calls.tsv")
    frozen = pd.read_csv(frozen_path, sep="\t", dtype={"peak_stage": str})
    candidates = frozen[frozen.readout == "expression"][["direction", "ligand", "receptor"]].drop_duplicates()
    tg = pd.read_csv(path(cfg["data"]["tg_list"]), sep="\t", dtype=str)
    sym2fb = dict(zip(tg["gene_name"], tg["FBgn_ID"]))
    lit = pd.read_csv(path(cfg["literature_pairs"]), sep="\t", dtype=str)
    lit_keys = {(d, a.lower(), b.lower()) for d, a, b in zip(lit.direction, lit.ligand, lit.receptor)}

    fbgns = sorted({sym2fb[x] for x in set(candidates.ligand) | set(candidates.receptor)})
    mean, frac, _ = group_expression(cfg, fbgns, all_groups)
    align = soma_alignment(cfg, groups["soma"], labels["soma"], len(groups["germline"]))
    soma_at_germ, germ_at_soma = align["soma_at_germ"], align["germ_at_soma"]
    where = {"soma": soma_at_germ, "germline": germ_at_soma}
    # Window around the literature stage for the peak-offset rule: within 2 germline stages of GSC (germline) or of
    # CySC placed on the germline axis by the alignment (soma).
    near = {"germline": np.arange(len(groups["germline"])) <= 2, "soma": (germ_at_soma - 1.0) <= 2}
    split_of = {r: LAST_EARLY[r] + 1 for r in ("germline", "soma")}

    def scores(curve: np.ndarray, receiver: str) -> dict[str, float]:
        top = curve.max()
        if top <= 0:
            return {"earliness": 0.0, "near_score": 0.0}
        k = split_of[receiver]
        w = near[receiver]
        return {"earliness": float((curve[:k].max() - curve[k:].max()) / top), "near_score": float((curve[w].max() - curve[~w].max()) / top)}
    gate = {side: (frac[groups[side]] >= gate_value).astype(float) for side in ("germline", "soma")}

    def sender_values(values: np.ndarray, sender: str) -> np.ndarray:
        return to_soma_nodes(values, groups["soma"], align["nodes"]) if sender == "soma" else values

    def ligand_term(fb: str, sender: str, receiver: str) -> np.ndarray:
        if sender == "hub":
            return np.full(len(groups[receiver]), float(mean.loc[fb, groups["hub"][0]]))
        values = sender_values(mean.loc[fb, groups[sender]].to_numpy(dtype=float), sender)
        return np.interp(where[sender], np.arange(1, len(values) + 1), values)

    def native_term(table: pd.DataFrame, sender: str, receiver: str) -> np.ndarray:
        if sender == "hub":
            return table.loc[groups["hub"][0], groups[receiver]].to_numpy(dtype=float)
        values = sender_values(table.loc[groups[sender], groups[receiver]].to_numpy(dtype=float), sender)
        x = np.arange(1, values.shape[0] + 1)
        return np.array([np.interp(where[sender][j], x, values[:, j]) for j in range(len(groups[receiver]))])

    # Baselines: recomputed where possible, checked against the frozen calls.
    natives = {name: native_scores(spec, all_groups) for name, spec in cfg.get("baselines", {}).items() if path(spec["edges"]).exists()}
    base_rows, mismatches = [], Counter()
    frozen_idx = frozen.set_index(["direction", "ligand", "receptor", "readout"])
    for item in candidates.itertuples(index=False):
        sender, receiver = DIRECTIONS[item.direction]
        r_fb, key = sym2fb[item.receptor], f"{item.ligand.lower()}__{item.receptor.lower()}"
        curves = {"expression": ligand_term(sym2fb[item.ligand], sender, receiver) * mean.loc[r_fb, groups[receiver]].to_numpy(dtype=float)}
        for name, native in natives.items():
            curves[name] = native_term(native[key], sender, receiver) if key in native else None
        for name in BASELINES:
            ref = frozen_idx.loc[(item.direction, item.ligand, item.receptor, name)]
            extra: dict[str, float] = {}
            if name in curves:
                gated = None if curves[name] is None else curves[name] * gate[receiver].loc[r_fb].to_numpy()
                got = {"status": f"not in {name}"} if gated is None else call(gated, receiver, labels[receiver], min_range)
                extra = scores(gated, receiver) if gated is not None else {"earliness": 0.0, "near_score": 0.0}
                if got["status"] != ref["status"] or str(got.get("peak_stage", "nan")) != str(ref["peak_stage"]):
                    mismatches[name] += 1
                source = "recomputed"
            else:
                got, source = {"status": ref["status"], "peak_stage": ref["peak_stage"]}, "frozen"
                extra = {c: float(ref[c]) for c in ("earliness", "near_score") if c in ref.index}
            base_rows.append({"direction": item.direction, "ligand": item.ligand, "receptor": item.receptor, "readout": name,
                              "status": got["status"], "peak_stage": got.get("peak_stage"), "source": source, **extra})
    baselines = pd.DataFrame(base_rows)
    sources = baselines.groupby("readout")["source"].first().to_dict()
    print("baselines:", ", ".join(f"{k} {v}" for k, v in sources.items()))
    if mismatches:
        print(f"WARNING: recomputed baseline calls differ from {frozen_path} for {dict(mismatches)} pairs; inputs differ from the frozen run")

    # DRAFT readout A per model.
    models = {"germline": [(seed, path(d)) for seed, d in germ_seeds], "soma": [("soma", soma_dir)] if (soma_dir / READOUT_FILE).exists() else []}
    if not models["soma"]:
        print(f"soma full-score export not found ({soma_dir}); soma receivers are skipped")
    draft_rows = []
    for receiver, model_list in models.items():
        for seed, directory in model_list:
            readout = full_scores(directory, READOUT_FILE)[0]
            for item in candidates.itertuples(index=False):
                sender, recv = DIRECTIONS[item.direction]
                if recv != receiver:
                    continue
                r_fb = sym2fb[item.receptor]
                curve = ligand_term(sym2fb[item.ligand], sender, receiver) * readout.loc[r_fb].to_numpy(dtype=float) * gate[receiver].loc[r_fb].to_numpy()
                draft_rows.append({"direction": item.direction, "ligand": item.ligand, "receptor": item.receptor, "seed": seed,
                                   **scores(curve, receiver), **call(curve, receiver, labels[receiver], min_range)})
            print(f"{receiver} model {seed} scored", flush=True)
    draft = pd.DataFrame(draft_rows)
    calls = pd.concat([baselines.assign(seed="all"), draft.assign(readout="DRAFT_A")], ignore_index=True)
    calls.to_csv(out_dir / "calls.tsv", sep="\t", index=False)

    timed = {"early", "late"}
    base_wide = baselines.assign(cell=baselines.status.str.replace("early", "E").str.replace("late", "L") + "(" + baselines.peak_stage.fillna("").astype(str) + ")")
    base_wide = base_wide.pivot_table(index=["direction", "ligand", "receptor"], columns="readout", values="cell", aggfunc="first")
    base_class = baselines[baselines.status.isin(timed)].groupby(["direction", "ligand", "receptor"])["status"].agg(lambda s: set(s))
    rows = []
    for (direction, ligand, receptor), part in draft.groupby(["direction", "ligand", "receptor"]):
        classes = base_class.get((direction, ligand, receptor))
        record = {"direction": direction, "pair": f"{ligand}-{receptor}", "literature": (direction, ligand.lower(), receptor.lower()) in lit_keys,
                  "models": len(part), "draft_timed": int(part.status.isin(timed).sum())}
        if not classes:
            record["baseline_class"] = "no baseline timed"
        elif len(classes) > 1:
            record["baseline_class"] = "baselines disagree"
        else:
            base = next(iter(classes))
            shifted = part[part.status.isin(timed) & (part.status != base)]
            record.update({"baseline_class": base, "shifted": len(shifted),
                           "shift": "DRAFT earlier" if base == "late" else "DRAFT later",
                           "draft_peaks": ", ".join(f"{s} x{n}" for s, n in Counter(part.peak_stage.dropna()).most_common())})
        record.update({f"{name}": base_wide.loc[(direction, ligand, receptor), name] if name in base_wide.columns else "-" for name in BASELINES})
        rows.append(record)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "pair_summary.tsv", sep="\t", index=False)

    germ = table[table.direction.str.endswith("germline") & table.shifted.notna()]
    n_germ = germ.models.max() if len(germ) else 0
    tiers = {
        f"reproducible: shifted in >= {REPRODUCIBLE} of {n_germ} germline seeds": germ[germ.shifted >= REPRODUCIBLE],
        f"less stable: shifted in {LESS_STABLE}-{REPRODUCIBLE - 1} of {n_germ} germline seeds": germ[(germ.shifted >= LESS_STABLE) & (germ.shifted < REPRODUCIBLE)],
        "soma receivers, one model: shifted": table[table.direction.str.endswith("soma") & (table.shifted == 1)],
    }
    cols = ["direction", "pair", "literature", "shift", "shifted", "models", "draft_peaks", *BASELINES]
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 400)
    pd.set_option("display.max_colwidth", 48)
    summary = {"receptor_gate": gate_value, "germline_seeds": [s for s, _ in germ_seeds], "baseline_sources": sources,
               "baseline_mismatches": dict(mismatches), "pairs": {}}
    for name, (title, part) in zip(["reproducible", "less_stable", "soma_single_model"], tiers.items()):
        part = part.sort_values(["direction", "shift", "shifted", "pair"], ascending=[True, True, False, True])
        part[cols].to_csv(out_dir / f"shifts_{name}.tsv", sep="\t", index=False)
        summary["pairs"][name] = {"n": len(part), "literature": int(part.literature.sum()),
                                  "DRAFT earlier": int((part["shift"] == "DRAFT earlier").sum()), "DRAFT later": int((part["shift"] == "DRAFT later").sum())}
        print(f"\n== {title}: {len(part)} pairs ({int(part.literature.sum())} literature)")
        print(part[cols].to_string(index=False) if len(part) else "none")
    excluded = table.baseline_class.isin(["baselines disagree", "no baseline timed"])
    summary["not_compared"] = table[excluded].groupby(["direction", "baseline_class"]).size().rename("n").reset_index().to_dict("records")
    print("\nnot compared (baselines disagree among themselves or none timed):",
          table[excluded].groupby(["direction", "baseline_class"]).size().to_dict())
    save_checkpoint_metadata(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
