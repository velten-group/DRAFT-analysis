"""Mouse reprogramming analysis (Fig. 3, Supp. Figs. 5-6) from the unpaired DRAFT checkpoint grn_best.pt.

This package calls the manuscript's mouse analysis code in scripts/mouse with the deposited inputs under data/mouse
and writes to results/checkpoint/mouse.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .. import paths

SCRIPTS = paths.REPO / "scripts" / "mouse"
CLUSTERS = [4, 8, 7, 3, 2, 1, 5]


def scripts_on_path() -> None:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))


def out(*parts: str) -> Path:
    return paths.CHECKPOINT.joinpath("mouse", *parts)


def native(method: str) -> Path:
    """Native edge scores per cluster of a baseline ("dictys", "scenic"), or their common feature axes ("features")."""
    return paths.data("mouse", "native_scores", method)
