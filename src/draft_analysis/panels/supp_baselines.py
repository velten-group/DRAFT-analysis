"""Supplementary Fig. 7: temporal smoothness of the baseline TF-TG networks (no DRAFT model involved).

Stage networks of each baseline, as deposited (data/fly/external):
- SCENIC+: published dAWA stage eRegulons (DAR = yes, DEG = yes), germline and soma. The table has no TF-TG score, so
  a TF-TG pair takes the maximum R2G_importance over its regions; pairs absent at a stage score 0.
- scDoRI: TF-TG path scores of the official v1.0.0 stage networks (germline: default settings; soma: GRN phase with
  patience 50).
- Random: per stage 50,000 unique pairs of the 761 x 23,121 TF-TG pairs, seed 12345 per lineage.
Distances are unsquared Frobenius distances between stages; smoothness = 1 - mean adjacent distance / largest
distance, for the soma along each branch (SdA, SdB).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config, paths
from . import save

K = "supplementary"
SOMA = ["CySC", "eSCC-1", "eSCC-2", "eSCC-3", "lSCC-1", "lSCC-2", "lSCC-3", "SdA", "SdB"]
N_TF, N_TG, EDGES, SEED = 761, 23121, 50_000, 12345


def distances(scores: np.ndarray) -> np.ndarray:
    flat = np.asarray(scores, dtype=np.float64)
    out = np.zeros((len(flat), len(flat)))
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            out[i, j] = out[j, i] = np.linalg.norm(flat[i] - flat[j])
    return out


def scdori(folder: str) -> np.ndarray:
    d = paths.data("fly", "external", "scdori_stage_networks", folder)
    keys, rows = None, []
    for t in range(len(list(d.glob("t*_TF_TG.tsv.gz")))):
        frame = pd.read_csv(d / f"t{t:02d}_TF_TG.tsv.gz", sep="\t")
        if keys is None:
            keys = frame[["TF_id", "TG_id"]]
        elif not keys.equals(frame[["TF_id", "TG_id"]]):
            raise ValueError(f"TF-TG row order differs at stage {t}")
        rows.append(frame.path_score.to_numpy(dtype=np.float32))
    return distances(np.stack(rows))


def scenic(lineage: str, n_stages: int) -> np.ndarray:
    table = pd.read_csv(paths.data("fly", "external", "scenicplus_published_eregulons",
                                   f"{lineage}_stage_eregulons.tsv.gz"), sep="\t")
    assert sorted(table.time_idx.unique()) == list(range(n_stages))
    x = table.groupby(["time_idx", "TF", "Gene"]).R2G_importance.max().unstack("time_idx").fillna(0.0).to_numpy().T
    return np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))


def _sample(k: int, rng: np.random.Generator) -> set:
    pairs: set = set()
    while len(pairs) < k:
        b = max(200_000, int((k - len(pairs)) * 1.5))
        codes = rng.integers(0, N_TF, size=b, dtype=np.int64) * N_TG + rng.integers(0, N_TG, size=b, dtype=np.int64)
        pairs.update(np.unique(codes).tolist())
    return set(rng.choice(np.fromiter(pairs, dtype=np.int64), size=k, replace=False).tolist())


def random(n_stages: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    sets = [_sample(EDGES, rng) for _ in range(n_stages)]
    d = np.zeros((n_stages, n_stages))
    for i in range(n_stages):
        for j in range(i + 1, n_stages):
            d[i, j] = d[j, i] = np.sqrt(2 * EDGES - 2 * len(sets[i] & sets[j]))
    return d


def smooth_along(d: np.ndarray, order: list[int]) -> float:
    sub = d[np.ix_(order, order)]
    return float(1 - np.diag(sub, 1).mean() / sub.max())


def supp7() -> None:
    germ = json.loads((config.CONFIGS / "fly" / "germline_stage_labels.json").read_text(encoding="utf-8"))["labels"]
    mats = {("SCENIC+", "Germline"): scenic("germline", 15), ("SCENIC+", "Soma"): scenic("soma", 9),
            ("scDoRI", "Germline"): scdori("germline"), ("scDoRI", "Soma"): scdori("soma_patience50"),
            ("Random", "Germline"): random(15), ("Random", "Soma"): random(9)}
    scores, blocks = [], []
    for (method, lineage), d in mats.items():
        labels = germ if lineage == "Germline" else SOMA
        assert d.shape == (len(labels), len(labels))
        if lineage == "Germline":
            scores.append({"method": method, "path": "Germline", "smoothness": smooth_along(d, list(range(len(d))))})
        else:
            scores += [{"method": method, "path": "Soma, SdA branch", "smoothness": smooth_along(d, list(range(8)))},
                       {"method": method, "path": "Soma, SdB branch", "smoothness": smooth_along(d, [*range(7), 8])}]
        blocks.append(pd.DataFrame(d / d.max(), index=labels, columns=labels).rename_axis("stage").reset_index())
    save(K, "supp07_Supp_Fig_7_0.tsv", pd.DataFrame(scores))
    for i, b in enumerate(blocks, 1):
        save(K, f"supp07_Supp_Fig_7_{i}.tsv", b)


def build() -> None:
    supp7()
