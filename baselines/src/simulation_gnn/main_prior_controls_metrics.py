"""Original main-panel evaluation plus a fixed union containing every new wrong edge."""

from __future__ import annotations

import numpy as np

try:  # Official environments import this file without loading Torch or the package.
    from .controlled_prior_metrics import auc, mean
    from .controlled_prior_metrics import evaluate as original
except ImportError:
    from controlled_prior_metrics import auc, mean
    from controlled_prior_metrics import evaluate as original


def evaluate(scores, truth, prior, mask, same, logits=None, *, union_mask):
    if mask.sum() != 1800 or not np.all(mask.sum(1) == 12):
        raise ValueError("Primary panel must be the original 1800 main candidates")
    if (mask & ~union_mask).any() or (prior & ~union_mask).any():
        raise ValueError(
            "Fixed union must include the main panel and every condition's prior"
        )
    out = original(scores, truth, prior, mask, same, logits=logits)
    full = original(scores, truth, prior, union_mask, same, logits=logits)
    for name in (
        "fixed_panel_auroc",
        "fixed_panel_auprc",
        "mean_continuity",
        "change_auroc",
    ):
        out[
            name.replace("fixed_panel_", "union_")
            if name.startswith("fixed_panel_")
            else "union_" + name
        ] = full[name]
    for name, selected in (("main", mask), ("union", union_mask)):
        out[f"prior_indicator_{name}_auroc"] = mean(
            [auc(y[selected], prior[selected].astype(float)) for y in truth]
        )
    out["per_time_union_auroc"] = full["per_time_auroc"]
    return out
