"""Display names of the fifteen germline stages."""

from __future__ import annotations

import json
from pathlib import Path


def germline_stage_labels() -> list[str]:
    path = Path(__file__).resolve().parents[3] / "configs/germline_stage_labels.json"
    return json.loads(path.read_text(encoding="utf-8"))["labels"]


def set_germline_stage_axis(ax, *, fontsize: float = 10, rotation: float = 45) -> None:
    labels = germline_stage_labels()
    ax.set_xticks(
        range(len(labels)), labels, rotation=rotation, ha="right", fontsize=fontsize
    )
