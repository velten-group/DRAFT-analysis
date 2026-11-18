"""Supplementary Figs. 12-14 (cell-cell communication timing) from the ccc and ccc_supp stages.

12  agreement of early/late calls between methods, four directions pooled; the published matrix was drawn from the
    sequential alignment (SdA then SdB, rounded to whole percent), the per-model values use the current alignment
13  stability of DRAFT calls across the five germline models: sequential alignment (as drawn), its random controls, and
    the current alignment (scripts/ccc_seed_stability.py)
14  literature AUROC of soma-to-germline pairs under four alignments (scripts/benchmark_ccc_alignment.py)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import paths
from ..fly import ccc
from . import save

K = "supplementary"
SEEDS = ["2024", "2025", "2026", "2027", "2028"]
BASELINES = (("CellChat", "CellChat"), ("LIANA", "LIANA+"), ("expression", "LR expression"))
TIMED = ("early", "late")


def _calls(sequential: bool) -> pd.DataFrame:
    run = paths.FLY / (ccc.SEQUENTIAL if sequential else "ccc_unified") / ccc.PEAK_SHIFTS
    return pd.read_csv(run / "calls.tsv", sep="\t", dtype=str)


def draft_agreement(calls: pd.DataFrame) -> pd.DataFrame:
    """Per germline model: DRAFT (germline receivers from that model, soma receivers from the soma model) vs each
    baseline, percentage of pairs timed by both with the same call."""
    base = calls[calls.seed.eq("all")].set_index(["direction", "ligand", "receptor", "readout"]).status
    rows = []
    for s in SEEDS:
        d = calls[calls.readout.eq("DRAFT_A") & ((calls.seed.eq(s) & calls.direction.str.endswith("germline")) |
                                                 (calls.seed.eq("soma") & calls.direction.str.endswith("soma")))]
        row = {"germline_model_seed": int(s), "candidate_pairs": len(d)}
        for b, label in BASELINES:
            both = [(r.status, base.get((r.direction, r.ligand, r.receptor, b))) for r in d.itertuples()]
            both = [(x, y) for x, y in both if x in TIMED and y in TIMED]
            row[f"DRAFT vs {label}: agreement_percent"] = 100 * np.mean([x == y for x, y in both])
            row[f"DRAFT vs {label}: pairs timed by both"] = len(both)
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_agreement(calls: pd.DataFrame, a: str, b: str) -> float:
    wide = calls[calls.seed.eq("all")].pivot_table(index=["direction", "ligand", "receptor"], columns="readout",
                                                   values="status", aggfunc="first")
    both = wide[a].isin(TIMED) & wide[b].isin(TIMED)
    return 100 * float((wide.loc[both, a] == wide.loc[both, b]).mean())


def supp12() -> None:
    seq, cur = draft_agreement(_calls(True)), draft_agreement(_calls(False))
    label = dict(BASELINES)
    rows = [(label[b], "DRAFT", seq[f"DRAFT vs {label[b]}: agreement_percent"].mean(),
             cur[f"DRAFT vs {label[b]}: agreement_percent"].mean()) for b, _ in BASELINES]
    calls = _calls(True)
    rows += [(label[b], label[a], baseline_agreement(calls, a, b), np.nan)
             for a, b in (("CellChat", "LIANA"), ("CellChat", "expression"), ("LIANA", "expression"))]
    t = pd.DataFrame(rows, columns=["method_1", "method_2", "sequential", "current_value (mean over germline models)"])
    t.insert(2, "agreement_percent (plotted)", t.pop("sequential").round().astype(int))
    save(K, "supp12_Supp_Fig_12_0.tsv", t)
    save(K, "supp12_Supp_Fig_12_1.tsv", cur)


def supp13() -> None:
    seq = paths.FLY / ccc.SEQUENTIAL / ccc.PEAK_SHIFTS
    cur = paths.FLY / "ccc_unified" / ccc.PEAK_SHIFTS
    save(K, "supp13_Supp_Fig_13_0.tsv", pd.read_csv(seq / "seed_stability.tsv", sep="\t"))
    save(K, "supp13_Supp_Fig_13_1.tsv", pd.read_csv(seq / "seed_stability_controls.tsv", sep="\t"))
    save(K, "supp13_Supp_Fig_13_2.tsv", pd.read_csv(cur / "seed_stability.tsv", sep="\t"))


def supp14() -> None:
    ab = pd.read_csv(paths.FLY / "ccc_unified" / "alignment_benchmark_gate0.05_lig0.05" / "alignment_benchmark.tsv",
                     sep="\t")
    ab = ab[ab.direction.eq("soma_to_germline") & ab.literature.eq(21) &
            ab.alignment.isin(["DTW", "Linear", "Reversed (control)", "None (control)"])]
    save(K, "supp14_Supp_Fig_14_0.tsv",
         ab[["alignment", "method", "model", "literature", "found", "auroc"]].reset_index(drop=True))


def build() -> None:
    supp12()
    supp13()
    supp14()
