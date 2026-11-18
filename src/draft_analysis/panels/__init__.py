"""Panel tables: from the pipeline results to the tables drawn in each figure panel (= Source Data sheets).

Every builder writes TSVs with exactly the names and columns of the frozen tables in source_data/, so a regenerated
table can be compared with the published one (`draft-analysis compare`).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import paths


def save(kind: str, name: str, df: pd.DataFrame) -> Path:
    out = paths.PANELS[kind] / name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    return out
