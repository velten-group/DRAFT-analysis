"""New run that reuses a finished phase 1 and repeats only the GRN phase with a longer validation patience.

The source run is not changed. Prepared data and the phase 1 checkpoint are copied and verified; the new name never
matches latest-native, so germline comparisons cannot select it by accident.
"""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from scdori_native import check_outputs, read_json, safe_run_path, write_json


def derive_grn_patience(source: Path, patience: int) -> Path:
    source = safe_run_path(source)
    manifest = read_json(source / "manifest.json")
    cfg = manifest["configuration"]
    current = cfg["training"]["grn_val_patience"]
    if patience <= current:
        raise ValueError(f"Patience must exceed the source value {current}.")
    record = read_json(source / "phase1_complete.json")
    check_outputs(record, source)
    lineage = cfg.get("lineage", "germline")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = safe_run_path(
        source.parent
        / f"native_correlation_{lineage}_patience{patience}_s{cfg['seed']}_{stamp}_{uuid.uuid4().hex[:8]}"
    )
    run.mkdir(parents=False, exist_ok=False)
    shutil.copytree(source / "data", run / "data")
    shutil.copytree(
        source / Path(record["checkpoint"]).parts[0],
        run / Path(record["checkpoint"]).parts[0],
    )
    for name in [
        "phase1_complete.json",
        "runtime.json",
        "mondo_preprocessing_report.json",
    ]:
        shutil.copy2(source / name, run / name)
    cfg["training"]["grn_val_patience"] = patience
    manifest["grn_patience"] = {
        "source_run": str(source),
        "source_grn_val_patience": current,
        "grn_val_patience": patience,
        "changed": "grn_val_patience only; prepared data and phase 1 checkpoint are copied unchanged",
    }
    write_json(run / "manifest.json", manifest)
    check_outputs(read_json(run / "phase1_complete.json"), run)
    print(f"GRN patience run: {run}", flush=True)
    return run
