"""Inputs of the checkpoint line (data/manifest.tsv): path, figures that use it, origin, size and SHA-256.

The quick line needs none of them. `check()` reports which inputs are present and, with verify=True, whether their
SHA-256 match. The two third-party files (the official scMTNI evaluator) are not deposited: `download()` fetches them
from their repository at a fixed commit and checks their SHA-256.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

from . import paths

_SCMTNI = "https://raw.githubusercontent.com/Roy-lab/scMTNI/c4d4850629f3529cc7a8b4dae54d91daf0702809/Scripts/Evaluation/"
THIRD_PARTY = {  # path under DATA -> download URL (level "third-party" in the manifest)
    "mouse/official_aupr/source/auc.jar": _SCMTNI + "auc.jar",
    "mouse/official_aupr/source/prepareAUClistfile_intersect.py": _SCMTNI + "prepareAUClistfile_intersect.py",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download() -> int:
    """Fetch the third-party tools that are not in the deposit; keep a file only if its SHA-256 matches."""
    table = pd.read_csv(paths.MANIFEST, sep="\t", dtype=str).fillna("")
    for r in table[table.level.eq("third-party")].itertuples():
        target = paths.DATA / r.path
        if target.exists() and _sha256(target) == r.sha256:
            continue
        with urllib.request.urlopen(THIRD_PARTY[r.path], timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != r.sha256:
            raise ValueError(f"{THIRD_PARTY[r.path]}: SHA-256 differs from data/manifest.tsv")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f"downloaded {r.path}")
    return 0


def check(verify: bool = False) -> int:
    table = pd.read_csv(paths.MANIFEST, sep="\t", dtype=str).fillna("")
    missing, bad = [], []
    for r in table.itertuples():
        p = paths.DATA / r.path
        if not p.exists():
            missing.append(r)
        elif verify and r.sha256 and _sha256(p) != r.sha256:
            bad.append(r.path)
    for group, rows in table.groupby("group", sort=False):
        n_miss = sum(m.group == group for m in missing)
        size = pd.to_numeric(rows.bytes, errors="coerce").sum() / 1e9
        print(f"{group:12s} {len(rows) - n_miss:4d}/{len(rows)} files present ({size:.2f} GB listed)")
    for m in missing:
        print(f"  missing: {m.path}  [{m.used_by}]  {m.origin}")
    for b in bad:
        print(f"  SHA-256 differs: {b}")
    return 1 if (missing or bad) else 0
