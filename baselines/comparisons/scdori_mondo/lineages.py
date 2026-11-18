"""Lineage stages for scDoRI runs. Germline stays the default; soma uses its own config and run names."""

from __future__ import annotations

import json
from pathlib import Path

SOMA_GROUPS = [f"Soma_{i}" for i in range(7)] + ["Soma_7a", "Soma_7b"]
SOMA_LABELS = [
    "CySC",
    "eSCC-1",
    "eSCC-2",
    "eSCC-3",
    "lSCC-1",
    "lSCC-2",
    "lSCC-3",
    "SdA",
    "SdB",
]


def stage_labels(lineage="germline"):
    if lineage == "soma":
        return list(SOMA_LABELS)
    if lineage != "germline":
        raise ValueError(f"Unknown lineage: {lineage}")
    path = Path(__file__).resolve().parents[2] / "configs/germline_stage_labels.json"
    return json.loads(path.read_text(encoding="utf-8"))["labels"]


def select_annotation(ann, lineage="germline"):
    """Rows of one lineage with an integer stage column; every stage must have cells."""
    groups = ann.GEX_latent_time_bins_annotated
    if lineage == "germline":
        ann = ann.loc[groups.str.fullmatch(r"Germline_\d+")].copy()
        ann["stage"] = ann.GEX_latent_time_bins_annotated.str.extract(r"(\d+)$")[
            0
        ].astype(int)
        expected = 15
    elif lineage == "soma":
        ann = ann.loc[groups.isin(SOMA_GROUPS)].copy()
        order = {name: i for i, name in enumerate(SOMA_GROUPS)}
        ann["stage"] = ann.GEX_latent_time_bins_annotated.map(order).astype(int)
        expected = len(SOMA_GROUPS)
    else:
        raise ValueError(f"Unknown lineage: {lineage}")
    if set(ann.stage) != set(range(expected)):
        raise ValueError(f"Expected all {expected} {lineage} stages.")
    return ann
