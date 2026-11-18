"""Checkpoint metadata helpers."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any

import torch


def save_checkpoint_metadata(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def torch_load_compat(path: str | Path, map_location: Any = None) -> Any:
    checkpoint_path = Path(path)
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def find_latest_checkpoint(directory: str | Path, pattern: str = "*.pt") -> Path:
    model_dir = Path(directory)
    paths = sorted(glob.glob(str(model_dir / pattern)))
    if not paths:
        raise FileNotFoundError(f"No checkpoint matching {pattern!r} was found under {model_dir}.")
    paths = sorted(paths, key=os.path.getmtime)
    return Path(paths[-1]).resolve()
