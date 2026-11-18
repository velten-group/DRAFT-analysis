"""Locations used by every stage.

Inputs are read from DATA (default ``<repo>/data``, override with the environment variable DRAFT_DATA).
Everything computed goes to RESULTS (default ``<repo>/results``, override with DRAFT_RESULTS), one folder per line:

    results/quick/        figures and Source Data workbooks drawn from the frozen Source Data (no inputs needed)
    results/checkpoint/   stage outputs (simulation/, fly/, mouse/), panel tables, comparison with the Source Data
                          and figures, all computed from the deposited checkpoints, saved predictions and inputs

The frozen Source Data tables that accompany the manuscript are in SOURCE_DATA and are never written to.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("DRAFT_DATA", REPO / "data")).resolve()
RESULTS = Path(os.environ.get("DRAFT_RESULTS", REPO / "results")).resolve()
SOURCE_DATA = REPO / "source_data"
MANIFEST = REPO / "data" / "manifest.tsv"

QUICK = RESULTS / "quick"
CHECKPOINT = RESULTS / "checkpoint"
FLY = CHECKPOINT / "fly"
# panel tables of the checkpoint line, in exactly the format of the frozen Source Data tables
PANELS = {"main": CHECKPOINT / "panels" / "main", "supplementary": CHECKPOINT / "panels" / "supplementary",
          "tables": CHECKPOINT / "panels" / "tables"}  # recomputed cells of the Supplementary Tables


def data(*parts: str) -> Path:
    """Path of an input file; raises when it is missing."""
    p = DATA.joinpath(*parts)
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing. Run `draft-analysis data` to see which inputs to download.")
    return p


def result(*parts: str, mkdir: bool = True) -> Path:
    """Path under results/checkpoint/."""
    p = CHECKPOINT.joinpath(*parts)
    if mkdir:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p
