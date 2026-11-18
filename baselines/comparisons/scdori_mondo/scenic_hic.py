"""Read-only dAWA stage eRegulons and native SCENIC+ RE–TG scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from hic_common import region_key

HERE = Path(__file__).resolve().parent
BASE = "https://dawa.cos.uni-heidelberg.de"
CACHE = HERE / "scenic_reference" / "dawa_dar_yes_deg_yes"
GLOBAL_CACHE = HERE / "scenic_reference" / "dawa_dar_no_deg_no"
COLS = ["time_idx", "TF", "Region", "Gene", "R2G_importance"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_bytes(path, value):
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
    temporary.write_bytes(value)
    temporary.replace(path)


def request_json(payload):
    request = Request(
        BASE + "/_dash-update-component",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "DRAFT-HiC-comparison",
        },
    )
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def table_layout_payload():
    return {
        "output": "tabmeta_EREG.children",
        "outputs": {"id": "tabmeta_EREG", "property": "children"},
        "inputs": [{"id": "tabmeta", "property": "value", "value": "tab-3"}],
        "state": [],
        "changedPropIds": ["tabmeta.value"],
    }


def tf_options(layout):
    found = []

    def walk(value):
        if isinstance(value, dict):
            props = value.get("props", {})
            if props.get("id") == "GeneN":
                found.append([option["value"] for option in props["options"]])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(layout)
    if len(found) != 1 or not found[0] or len(set(found[0])) != len(found[0]):
        raise ValueError("Cannot identify the complete dAWA TF selector.")
    return found[0]


def query_payload(tf, stage):
    values = {
        "GeneN": tf,
        "rows-per-page": 50,
        "ClusterN": f"Germline_{stage}",
        "SubsetDARN": "yes",
        "SubsetDEGN": "yes",
    }
    return {
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


def parse_records(raw, tf, stage):
    records = json.loads(raw)["response"]["table"]["data"]
    if not isinstance(records, list):
        raise TypeError(f"Invalid SCENIC+ response: {tf}, stage {stage}.")
    for record in records:
        if record["TF"] != tf:
            raise ValueError(f"Unexpected TF in SCENIC+ response: {tf}.")
        for key in ["Gene", "Region"]:
            if not isinstance(record[key], str) or not record[key].strip():
                raise ValueError(f"Missing SCENIC+ {key}: {tf}, stage {stage}.")
        score = float(record["R2G_importance"])
        if not np.isfinite(score) or score < 0:
            raise ValueError(f"Invalid SCENIC+ R2G_importance: {tf}, stage {stage}.")
    return records


def fetch_reference(
    directory=CACHE, *, requester=request_json, workers=2, validation_only=False
):
    """Cache every TF/stage response, resume partial downloads, never invent zeros."""
    directory = Path(directory).expanduser().resolve()
    if (directory / "reference_manifest.json").is_file():
        load_reference(directory)
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    layout_path = directory / "table_layout.json"
    if not layout_path.exists():
        write_bytes(layout_path, requester(table_layout_payload()))
    tfs = tf_options(json.loads(layout_path.read_bytes()))
    if workers not in [1, 2]:
        raise ValueError("Use one or two download workers for the public server.")
    records, sources, counts = [], {"table_layout.json": sha256(layout_path)}, []

    def query(job):
        index, tf, stage = job
        name = f"t{stage:02d}_tf{index:03d}.json"
        path = directory / name
        raw = (
            path.read_bytes() if path.exists() else requester(query_payload(tf, stage))
        )
        rows = parse_records(raw, tf, stage)
        if not path.exists():
            write_bytes(path, raw)
        return name, rows

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for stage in range(15):
            stage_count = 0
            for name, rows in pool.map(
                query, [(i, tf, stage) for i, tf in enumerate(tfs)]
            ):
                sources[name] = sha256(directory / name)
                stage_count += len(rows)
                records.extend({"time_idx": stage, **row} for row in rows)
            counts.append(stage_count)
            print(
                f"SCENIC+ Germline_{stage}: {stage_count} rows, {len(tfs)} TF queries complete",
                flush=True,
            )
    table = pd.DataFrame(records, columns=COLS)
    if table.empty:
        raise ValueError("All SCENIC+ stage networks are empty; cannot compare.")
    table.to_csv(directory / "stage_eregulons.tsv.gz", sep="\t", index=False)
    manifest = {
        "source_url": BASE + "/eregulon",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "validation_only": bool(validation_only),
        "subset_dar": "yes",
        "subset_deg": "yes",
        "tfs": tfs,
        "stages": list(range(15)),
        "stage_rows": counts,
        "completed_queries": 15 * len(tfs),
        "snapshot_sha256": sources,
        "table_sha256": sha256(directory / "stage_eregulons.tsv.gz"),
        "score": "native R2G_importance; duplicate RE–TG links across TFs use maximum, never sum",
        "scope": "published dAWA DAR=yes/DEG=yes stage-filtered eRegulons, not a new shared-prior SCENIC+ fit; absence means no reported edge, not absent source feature",
    }
    write_bytes(
        directory / "reference_manifest.json", json.dumps(manifest, indent=2).encode()
    )
    load_reference(directory)
    print(f"SCENIC+ reference: {directory}", flush=True)
    return directory


def load_reference(directory):
    directory = Path(directory).expanduser().resolve()
    manifest = json.loads(
        (directory / "reference_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest["stages"] != list(range(15))
        or not manifest["tfs"]
        or len(set(manifest["tfs"])) != len(manifest["tfs"])
        or manifest["completed_queries"] != 15 * len(manifest["tfs"])
        or (manifest["subset_dar"], manifest["subset_deg"]) != ("yes", "yes")
    ):
        raise ValueError(
            "SCENIC+ reference is not a completed fifteen-stage all-TF extraction."
        )
    expected_names = {"table_layout.json"} | {
        f"t{stage:02d}_tf{tf:03d}.json"
        for stage in range(15)
        for tf in range(len(manifest["tfs"]))
    }
    if set(manifest["snapshot_sha256"]) != expected_names:
        raise ValueError("SCENIC+ extraction has missing TF/stage snapshots.")
    for name, expected in manifest["snapshot_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"SCENIC+ snapshot changed: {name}")
    if sha256(directory / "stage_eregulons.tsv.gz") != manifest["table_sha256"]:
        raise ValueError("SCENIC+ network table changed.")
    table = pd.read_csv(directory / "stage_eregulons.tsv.gz", sep="\t")
    if not set(COLS).issubset(table):
        raise ValueError("SCENIC+ network table lacks required columns.")
    counts = table.groupby("time_idx").size().reindex(range(15), fill_value=0).tolist()
    if counts != manifest["stage_rows"] or not table.time_idx.isin(range(15)).all():
        raise ValueError(
            "SCENIC+ table/stage counts disagree with extraction provenance."
        )
    if not np.isfinite(table.R2G_importance).all() or (table.R2G_importance < 0).any():
        raise ValueError("Invalid native SCENIC+ scores.")
    return table, {"directory": str(directory), **manifest}


def fetch_global_reference(
    directory=GLOBAL_CACHE, *, requester=request_json, workers=2, validation_only=False
):
    """Retrieve published eRegulons without DAR/DEG filters, as a static baseline.

    This is still eRegulon-selected output, NOT the full native R2G search space.
    Check both endpoint queries for every TF; never invent stage-specific scores.
    """
    directory = Path(directory).expanduser().resolve()
    if (directory / "global_reference_manifest.json").is_file():
        load_global_reference(directory)
        return directory
    if workers not in [1, 2]:
        raise ValueError("Use one or two download workers for the public server.")
    directory.mkdir(parents=True, exist_ok=True)
    layout_path = directory / "table_layout.json"
    if not layout_path.exists():
        write_bytes(layout_path, requester(table_layout_payload()))
    tfs = tf_options(json.loads(layout_path.read_bytes()))
    records, sources = [], {"table_layout.json": sha256(layout_path)}

    def query(index_tf):
        index, tf = index_tf
        endpoint_rows, hashes = [], {}
        for stage in [0, 14]:
            name = f"global_t{stage:02d}_tf{index:03d}.json"
            path = directory / name
            payload = query_payload(tf, stage)
            for item in payload["inputs"]:
                if item["id"] in ["SubsetDARN", "SubsetDEGN"]:
                    item["value"] = "no"
            raw = path.read_bytes() if path.exists() else requester(payload)
            rows = parse_records(raw, tf, stage)
            if not path.exists():
                write_bytes(path, raw)
            endpoint_rows.append(rows)
            hashes[name] = sha256(path)

        def canonical(rows):
            return sorted(json.dumps(r, sort_keys=True) for r in rows)

        if canonical(endpoint_rows[0]) != canonical(endpoint_rows[1]):
            raise ValueError(
                f"DAR=no/DEG=no endpoints differ for {tf}; cannot treat as global reference."
            )
        return endpoint_rows[0], hashes

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (rows, hashes) in enumerate(pool.map(query, enumerate(tfs)), 1):
            records.extend(rows)
            sources.update(hashes)
            if index % 20 == 0 or index == len(tfs):
                print(f"SCENIC+ global reference: {index}/{len(tfs)} TFs", flush=True)
    table = pd.DataFrame(records, columns=COLS[1:])
    if table.empty:
        raise ValueError("The unfiltered published eRegulon reference is empty.")
    table.to_csv(directory / "global_eregulons.tsv.gz", sep="\t", index=False)
    manifest = {
        "source_url": BASE + "/eregulon",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "validation_only": bool(validation_only),
        "subset_dar": "no",
        "subset_deg": "no",
        "tfs": tfs,
        "completed_queries": 2 * len(tfs),
        "checked_stages": [0, 14],
        "source_triple_rows": len(table),
        "snapshot_sha256": sources,
        "table_sha256": sha256(directory / "global_eregulons.tsv.gz"),
        "temporal_scope": "global_static",
        "scope": "published eRegulon-selected links with DAR/DEG filters disabled; endpoint invariance checked for every TF; constant reference across stages, not independent stage observations and not complete native R2G scores",
    }
    write_bytes(
        directory / "global_reference_manifest.json",
        json.dumps(manifest, indent=2).encode(),
    )
    load_global_reference(directory)
    return directory


def load_global_reference(directory):
    directory = Path(directory).expanduser().resolve()
    manifest = json.loads(
        (directory / "global_reference_manifest.json").read_text(encoding="utf-8")
    )
    tfs = manifest["tfs"]
    if (
        not tfs
        or len(set(tfs)) != len(tfs)
        or (manifest["subset_dar"], manifest["subset_deg"]) != ("no", "no")
        or manifest["completed_queries"] != 2 * len(tfs)
        or manifest["checked_stages"] != [0, 14]
        or manifest["temporal_scope"] != "global_static"
    ):
        raise ValueError("Incomplete or incompatible SCENIC+ global reference.")
    expected = {"table_layout.json"} | {
        f"global_t{stage:02d}_tf{index:03d}.json"
        for index in range(len(tfs))
        for stage in [0, 14]
    }
    if set(manifest["snapshot_sha256"]) != expected:
        raise ValueError("SCENIC+ global reference has missing endpoint snapshots.")
    for name, sha in manifest["snapshot_sha256"].items():
        if sha256(directory / name) != sha:
            raise ValueError(f"SCENIC+ snapshot changed: {name}")
    path = directory / "global_eregulons.tsv.gz"
    if sha256(path) != manifest["table_sha256"]:
        raise ValueError("SCENIC+ global table changed.")
    table = pd.read_csv(path, sep="\t")
    if (
        not set(COLS[1:]).issubset(table)
        or len(table) != manifest["source_triple_rows"]
        or not table.TF.isin(tfs).all()
        or not np.isfinite(table.R2G_importance).all()
        or (table.R2G_importance < 0).any()
    ):
        raise ValueError("Invalid SCENIC+ global table.")
    return table, {"directory": str(directory), **manifest}


def map_regions(regions, peaks):
    """Prefer exact coordinates, otherwise a unique maximum covering >=50% of source."""
    exact = {region_key(peak): i for i, peak in enumerate(peaks)}
    if len(exact) != len(peaks):
        raise ValueError("Ambiguous evaluation peak identifiers.")

    def coordinates(value):
        match = re.fullmatch(r"([^:]+):(\d+)-(\d+)", region_key(value))
        if match is None or int(match[3]) <= int(match[2]):
            raise ValueError(f"Invalid genomic interval: {value}.")
        return match[1], int(match[2]), int(match[3])

    chroms = {}
    for i, peak in enumerate(peaks):
        chrom, start, end = coordinates(peak)
        chroms.setdefault(chrom, []).append((start, end, i))
    chroms = {
        chrom: np.asarray(sorted(values), dtype=np.int64)
        for chrom, values in chroms.items()
    }
    end_max = {
        chrom: np.maximum.accumulate(values[:, 1]) for chrom, values in chroms.items()
    }
    lookup, rows = {}, []
    for region in sorted(set(regions)):
        chrom, start, end = coordinates(region)
        index = exact.get(region_key(region))
        status, overlap, n_overlap = "exact", end - start, 1
        if index is None:
            status, overlap, n_overlap = "no_overlap", 0, 0
            values = chroms.get(chrom)
            if values is not None:
                lo = np.searchsorted(end_max[chrom], start, side="right")
                hi = np.searchsorted(values[:, 0], end, side="left")
                candidates = values[lo:hi]
                overlaps = np.minimum(candidates[:, 1], end) - np.maximum(
                    candidates[:, 0], start
                )
                valid = overlaps > 0
                candidates, overlaps = candidates[valid], overlaps[valid]
                n_overlap = len(overlaps)
                if n_overlap:
                    overlap = int(overlaps.max())
                    best = np.flatnonzero(overlaps == overlap)
                    if overlap / (end - start) < 0.5:
                        status = "insufficient_overlap"
                    elif len(best) != 1:
                        status = "ambiguous_best_overlap"
                    else:
                        index = int(candidates[best[0], 2])
                        status = "unique_max_overlap"
        if index is not None:
            lookup[region] = index
        rows.append(
            {
                "scenic_region": region,
                "evaluation_RE": peaks[index] if index is not None else "",
                "mapping": status,
                "overlap_bp": overlap,
                "source_overlap_fraction": overlap / (end - start),
                "n_overlapping_evaluation_regions": n_overlap,
            }
        )
    return lookup, pd.DataFrame(rows)


def map_stage_edges(table, peaks, genes, tg, *, return_region_mapping=False):
    """Project reported links onto the predeclared DRAFT/scDoRI evaluation pool."""
    pi, region_mapping = map_regions(table.Region, peaks)
    gi = {g: i for i, g in enumerate(genes)}
    if len(gi) != len(genes):
        raise ValueError("Ambiguous evaluation feature identifiers.")
    for row in tg.itertuples(index=False):
        if row.FBgn_ID in gi:
            symbol = str(row.gene_name).strip()
            if symbol in gi and gi[symbol] != gi[row.FBgn_ID]:
                raise ValueError(f"Ambiguous SCENIC+ target alias: {symbol}.")
            gi[symbol] = gi[row.FBgn_ID]
    lookups, audits = [], []
    for stage in range(15):
        data = table.loc[table.time_idx.eq(stage)]
        mapped, missing_gene, missing_peak = {}, 0, 0
        for row in data.itertuples(index=False):
            g = gi.get(row.Gene)
            p = pi.get(row.Region)
            if g is None:
                missing_gene += 1
            elif p is None:
                missing_peak += 1
            else:
                key = (p, g)
                mapped[key] = max(mapped.get(key, 0.0), float(row.R2G_importance))
        lookups.append(mapped)
        audits.append(
            {
                "time_idx": stage,
                "source_triple_rows": len(data),
                "mapped_unique_re_tg_pairs": len(mapped),
                "missing_gene_rows": missing_gene,
                "missing_peak_rows_after_gene_mapping": missing_peak,
            }
        )
    if return_region_mapping:
        return lookups, audits, region_mapping
    return lookups, audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument(
        "--global-reference",
        action="store_true",
        help="Disable DAR/DEG filtering and retrieve a separate global eRegulon reference.",
    )
    args = parser.parse_args()
    if args.global_reference:
        fetch_global_reference(args.output or GLOBAL_CACHE, workers=args.workers)
    else:
        fetch_reference(args.output or CACHE, workers=args.workers)


if __name__ == "__main__":
    main()
