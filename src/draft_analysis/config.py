"""Load a configuration from configs/ and point its paths at this repository's data/ and results/ folders.

The YAML files in configs/fly are the configurations used for the manuscript, with the manuscript's relative paths
(data/raw/..., configs/..., result/...). ``load`` rewrites them:

    data/raw/<f>             -> DATA/fly/raw/<f>
    configs/<f>              -> configs/fly/<f>
    result/<external>/...    -> DATA/fly/external/<external>/...   (outputs of other software, deposited as inputs)
    result/<other>/...       -> results/checkpoint/fly/<other>/...  (computed here)

All other parameters are left as they are.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from . import paths

CONFIGS = paths.REPO / "configs"
# result/ sub-folders that hold outputs of other software (official baseline runs); they are inputs here
EXTERNAL_RESULTS = ("ccc_regulatory_benchmark",)


def rewrite_path(value: str) -> str:
    v = value.replace("\\", "/")
    if v.startswith("data/raw/"):
        return str(paths.DATA / "fly" / "raw" / v[len("data/raw/"):])
    if v.startswith("configs/"):
        return str(CONFIGS / "fly" / v[len("configs/"):])
    if v.startswith("result/"):
        rest = v[len("result/"):]
        if rest.split("/", 1)[0] in EXTERNAL_RESULTS:
            return str(paths.DATA / "fly" / "external" / rest)
        return str(paths.FLY / rest)
    return value


def _rewrite(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _rewrite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite(v) for v in obj]
    if isinstance(obj, str) and "/" in obj:
        return rewrite_path(obj)
    return obj


def load(name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """configs/fly/<name>.yaml with paths rewritten, then `overrides` merged in (nested dicts are merged)."""
    raw = yaml.safe_load((CONFIGS / "fly" / f"{name}.yaml").read_text(encoding="utf-8"))
    cfg = _rewrite(raw)
    cfg["base_dir"] = str(paths.REPO)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            merged = copy.deepcopy(cfg[key])
            merged.update(value)
            cfg[key] = merged
        else:
            cfg[key] = value
    return cfg


def write(cfg: dict[str, Any], name: str) -> Path:
    """Save the effective configuration next to the results, so every run records what it used."""
    out = paths.result("fly", "configs", f"{name}.yaml")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def checkpoint(lineage: str, seed: int = 2025) -> Path:
    """Checkpoint files as deposited: germ_seed<seed>.pt for the germline, soma.pt (+ soma_graphs.pt) for the soma."""
    name = f"germ_seed{seed}.pt" if lineage == "germ" else "soma.pt"
    return paths.data("fly", "checkpoints", name)
