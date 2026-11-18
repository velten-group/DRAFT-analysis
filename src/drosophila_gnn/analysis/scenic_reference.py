"""Target counts of the SCENIC+ eRegulons from the dAWA table."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

DAWA_URL = "https://dawa.cos.uni-heidelberg.de/eregulon"


def summarize_scenic_records(
    records_by_stage: dict[int, list[dict[str, Any]]],
    gene: str,
    dar: str,
    deg: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count each target gene once, regardless of the number of linked regions."""
    if set(records_by_stage) != set(range(15)):
        raise ValueError(
            "SCENIC+ extraction requires completed queries for all G0-G14 stages."
        )
    metrics, targets = [], []
    for time in range(15):
        records = records_by_stage[time]
        if not isinstance(records, list):
            raise TypeError(f"Invalid SCENIC+ table response for G{time}.")
        genes = set()
        for record in records:
            if str(record["TF"]).casefold() != gene.casefold():
                raise ValueError(f"Unexpected TF in {gene} query at G{time}.")
            target = record["Gene"]
            if not isinstance(target, str) or not target.strip():
                raise ValueError(f"Invalid target gene in SCENIC+ G{time} response.")
            genes.add(target)
        metrics.append(
            {
                "time": time,
                "stage": f"Germline_{time}",
                "subset_dar": dar,
                "subset_deg": deg,
                "target_count": len(genes),
                "tf_region_gene_rows": len(records),
            }
        )
        targets.extend(
            {
                "time": time,
                "stage": f"Germline_{time}",
                "subset_dar": dar,
                "subset_deg": deg,
                "Gene": target,
            }
            for target in sorted(genes)
        )
    return pd.DataFrame(metrics), pd.DataFrame(
        targets,
        columns=[
            "time",
            "stage",
            "subset_dar",
            "subset_deg",
            "Gene",
        ],
    )


def fetch_scenic_reference(
    output_root: str | Path, *, gene: str = "Mondo", dar: str = "yes", deg: str = "yes"
) -> dict[str, Any]:
    """Download the same public table queried by the site's eRegulon Table tab."""
    if dar not in {"yes", "no"} or deg not in {"yes", "no"}:
        raise ValueError("DAR and DEG selections must be yes or no.")
    directory = (
        Path(output_root)
        / f"{gene.lower()}_dar-{dar}_deg-{deg}_{uuid.uuid4().hex[:12]}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    records, snapshots = {}, {}
    for time in range(15):
        values = {
            "GeneN": gene,
            "rows-per-page": 50,
            "ClusterN": f"Germline_{time}",
            "SubsetDARN": dar,
            "SubsetDEGN": deg,
        }
        payload = {
            "output": "..table.data...table.page_size..",
            "outputs": [
                {"id": "table", "property": "data"},
                {"id": "table", "property": "page_size"},
            ],
            "inputs": [
                {"id": key, "property": "value", "value": value}
                for key, value in values.items()
            ],
            "state": [],
            "changedPropIds": ["GeneN.value", "ClusterN.value"],
        }
        request = Request(
            DAWA_URL.rsplit("/", 1)[0] + "/_dash-update-component",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "DRAFT-MONDO-reference",
            },
        )
        # Failed requests must raise; they must never become biological zeros.
        with urlopen(request, timeout=30) as response:
            raw = response.read()
        data = json.loads(raw)
        records[time] = data["response"]["table"]["data"]
        name = f"G{time}_response.json"
        (directory / name).write_bytes(raw)
        snapshots[name] = hashlib.sha256(raw).hexdigest()
    metrics, targets = summarize_scenic_records(records, gene, dar, deg)
    for name, table in [
        ("mondo_scenic_stage_metrics.tsv", metrics),
        ("mondo_scenic_stage_targets.tsv", targets),
    ]:
        table.to_csv(directory / name, sep="\t", index=False, lineterminator="\n")
    manifest = {
        "gene": gene,
        "source_url": DAWA_URL,
        "metric": "number of unique Gene entries after the website DAR/DEG selections",
        "stage_mapping": "Germline_0 through Germline_14 map to G0 through G14, without offset",
        "available_filters": [{"subset_dar": dar, "subset_deg": deg}],
        "snapshot_sha256": snapshots,
        "table_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in [
                "mondo_scenic_stage_metrics.tsv",
                "mondo_scenic_stage_targets.tsv",
            ]
        },
    }
    (directory / "reference_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {"directory": str(directory), "metrics": metrics, "manifest": manifest}


def load_scenic_profile(
    directory: Path, *, gene: str, dar: str, deg: str, stage_times: np.ndarray
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the raw count profile before it enters a supplementary figure."""
    manifest = json.loads(
        (directory / "reference_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["gene"].casefold() != gene.casefold():
        raise ValueError("SCENIC+ reference is for a different TF.")
    for name, digest in manifest["table_sha256"].items():
        raw = (directory / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"SCENIC+ reference checksum differs: {name}")
    metrics = pd.read_csv(directory / "mondo_scenic_stage_metrics.tsv", sep="\t")
    profile = (
        metrics.loc[metrics.subset_dar.eq(dar) & metrics.subset_deg.eq(deg)]
        .sort_values("time")
        .copy()
    )
    if (
        profile.empty
        or profile.time.duplicated().any()
        or not np.array_equal(profile.time, stage_times)
        or not np.isfinite(profile.target_count).all()
        or (profile.target_count < 0).any()
        or not np.array_equal(profile.target_count, profile.target_count.astype(int))
    ):
        raise ValueError(
            "SCENIC+ stages/counts are missing or invalid for the selected DAR/DEG settings."
        )
    targets = pd.read_csv(directory / "mondo_scenic_stage_targets.tsv", sep="\t")
    sub = targets.loc[targets.subset_dar.eq(dar) & targets.subset_deg.eq(deg)]
    counts = (
        sub.groupby("time")
        .Gene.nunique()
        .reindex(profile.time, fill_value=0)
        .to_numpy()
    )
    if not np.array_equal(counts, profile.target_count):
        raise ValueError("SCENIC+ counts disagree with the saved unique target lists.")
    peak_stages = (
        profile.loc[profile.target_count.eq(profile.target_count.max()), "time"]
        .astype(int)
        .tolist()
    )
    return profile, {
        "directory": str(directory),
        "source_url": manifest["source_url"],
        "subset_dar": dar,
        "subset_deg": deg,
        "metric": manifest["metric"],
        "peak_stage": peak_stages[0] if len(peak_stages) == 1 else None,
        "peak_stages": peak_stages,
        "target_counts": profile.target_count.astype(int).tolist(),
        "table_sha256": manifest["table_sha256"],
    }
