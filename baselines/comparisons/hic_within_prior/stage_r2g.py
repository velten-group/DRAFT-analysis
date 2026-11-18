"""Score the per-stage SCENIC+ R2G fits (comparisons/scenicplus_r2g/stage_fit.py) on the frozen Hi-C pairs. Nothing is trained.

Fixed before results: the fits enter Fig. 4b if all stages finished and the mean nonzero share of Hi-C pairs is at least 5% in
both assays, whatever the AUROC.

    python comparisons/hic_within_prior/stage_r2g.py --evaluation EVAL --data-root RAW
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scdori_mondo"))
from compare_stage_hic import digest, read_json

sys.path.insert(0, str(HERE))
from filtered_r2g import write_results
from published_scenic import load_inputs
from workflow import STAGE_LABELS

MODEL = "SCENIC+ R2G per stage"
OUTPUT = "stage_fit_v1"


def stage_scores(directory, codes, features):
    """(stage, candidate) importance; a candidate lost in preprocessing stays zero."""
    report = read_json(directory / "stage_fit_report.json")
    genes = {name: j for j, name in enumerate(features["genes"])}
    peaks = {name: j for j, name in enumerate(features["peaks"])}
    scores = np.zeros((len(STAGE_LABELS), len(codes)), dtype=np.float64)
    if len(report["files_sha256"]) != len(STAGE_LABELS):
        raise ValueError("Per-stage R2G must cover all fifteen stages.")
    for name, expected in report["files_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Per-stage R2G output changed: {name}")
        table = pd.read_csv(directory / name, sep="\t")
        pair = (
            table.region.map(peaks).to_numpy() * len(genes)
            + table.target.map(genes).to_numpy()
        )
        positions = np.searchsorted(codes, pair)
        inside = positions < len(codes)
        inside[inside] &= codes[positions[inside]] == pair[inside]
        if not inside.all():
            raise ValueError("Per-stage R2G scores a pair outside the candidates.")
        scores[int(name[1:3]), positions] = table.importance.to_numpy()
    return scores, report


def run(evaluation, data_root=None):
    evaluation = Path(evaluation).expanduser().resolve()
    report, cache, features, _, _ = load_inputs(evaluation, data_root)
    native = Path(read_json(cache / "cache_report.json")["scenic_run"])
    with np.load(cache / "scores.npz", allow_pickle=False) as a:
        arrays = {name: a[name] for name in a.files}
    fitted, fit_report = stage_scores(native / OUTPUT, arrays["pair_codes"], features)
    if fit_report["validation_only"] != report["validation_only"]:
        raise ValueError("Software-only fits cannot enter a biological comparison.")
    out = evaluation / f"stage_r2g_{uuid.uuid4().hex[:10]}"
    out.mkdir()
    write_results(evaluation, report, arrays, {MODEL: fitted}, out)
    (out / "stage_r2g_report.json").write_text(
        json.dumps(
            {
                "evaluation": str(evaluation),
                "validation_only": bool(report["validation_only"]),
                "stage_fit_report_sha256": digest(
                    native / OUTPUT / "stage_fit_report.json"
                ),
                "code_sha256": digest(Path(__file__)),
                "adoption_rule": "enters Fig. 4b if all stages finished and the mean nonzero share of Hi-C pairs is at least 5% in both assays, whatever the AUROC; fixed before results",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Per-stage R2G: {out}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    run(args.evaluation, args.data_root)
