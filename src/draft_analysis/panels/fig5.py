"""Fig. 5 panel tables (Mondo). Panels b and d-f are experimental and copied from the authors' workbook.

a  network centrality of Mondo per stage in five germline models (seeds 2024-2028): sum of the selected outgoing TF-TG
   probabilities in the dense network export of each checkpoint (network and seeds stages; scripts/seed_replicates.py).
   Expression: mean log-normalized Mondo expression of the paired cells per stage, rounded to 3 decimals as drawn.
c  early germline Mondo eRegulon (see fly.mondo_regulon)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .. import paths
from ..fly import numbering
from . import save

K = "main"
SEEDS = (2024, 2025, 2026, 2027, 2028)


def _sr():
    sys.path.insert(0, str(paths.REPO / "scripts"))
    import seed_replicates

    return seed_replicates


def mondo_centrality(seed: int, tfs, aliases) -> np.ndarray:
    sr = _sr()
    dense = (paths.FLY / "germ" / "network_dense" if seed == 2025
             else paths.FLY / f"germ_seed{seed}" / "network_dense")
    if not any(dense.glob("TF_TG_dense_t*.tsv")):
        raise FileNotFoundError(f"{dense}: run `draft-analysis checkpoint --stages network seeds` first")
    return sr.network_profiles(dense, tfs, aliases)[:, aliases["Mondo"]]


def panel_a() -> None:
    sr = _sr()
    stages = numbering.germline_labels()
    tfs, aliases = sr.load_tfs(paths.data("fly", "raw", "TF_filtered.tsv"))
    raw = {s: mondo_centrality(s, tfs, aliases) for s in SEEDS}
    sm = pd.DataFrame({s: sr.smooth(raw[s]) for s in SEEDS})
    expression = sr.expression_profiles(paths.DATA / "fly" / "raw", tfs)[:, aliases["Mondo"]]
    a = pd.DataFrame({"stage_index": range(15), "stage": stages})
    for s in SEEDS:
        a[f"centrality_seed_{s}_raw"] = raw[s]
    for s in SEEDS:
        a[f"centrality_seed_{s}_smoothed"] = sm[s].to_numpy()
    a["centrality_mean_of_5_seeds (line)"] = sm.mean(1).to_numpy()
    a["centrality_min (band)"] = sm.min(1).to_numpy()
    a["centrality_max (band)"] = sm.max(1).to_numpy()
    a["mean_normalized_expression"] = np.round(expression, 3)
    peaks = pd.DataFrame({"seed": list(SEEDS), "stage_of_maximal_centrality": [stages[int(np.argmax(sm[s]))] for s in SEEDS],
                          "expression_peak": stages[int(np.argmax(a.mean_normalized_expression))]})
    save(K, "fig5_Fig_5a_0.tsv", a)
    save(K, "fig5_Fig_5a_1.tsv", peaks)


def build() -> None:
    panel_a()
    from ..fly import mondo_regulon

    mondo_regulon.panel_c()
