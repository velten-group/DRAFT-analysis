"""Map the official GSE108222 UMI tables to the existing seven mouse clusters."""

from __future__ import annotations

import argparse
import gzip
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mouse_benchmark"))
from common import CLUSTERS, digest, gene_key, gene_name, new_run, write_json
from prepare import prepare

HERE = Path(__file__).resolve().parent
SOURCE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE108nnn/GSE108222/suppl/GSE108222_RAW.tar"
ARCHIVE_SHA256 = "72392feb2f0da55541db1b6123820e160a2cb4c441c7718597caada7dc1f1764"
SAMPLES = {
    3595429: ("MEF-Day0", 1),
    3595430: ("MEF-Day0", 2),
    3595431: ("FBS-mESCs", 1),
    3595432: ("FBS-mESCs", 2),
    3595435: ("FBS-Day3", 1),
    3595436: ("FBS-Day3", 2),
    3595437: ("FBS-Day6", 1),
    3595438: ("FBS-Day6", 2),
    3595439: ("FBS-Day9", 1),
    3595440: ("FBS-Day9", 2),
    3595441: ("FBS-Day12", 1),
    3595442: ("FBS-Day12", 2),
}


def layout(data_root: Path, clusters: list[int]) -> tuple[dict, dict]:
    result, hashes, all_cells = {}, {}, set()
    for cid in clusters:
        path = data_root / f"cluster{cid}.table"
        cells = pd.read_csv(path, sep="\t", index_col=0, nrows=0).columns.tolist()
        genes = (
            pd.read_csv(path, sep="\t", usecols=[0]).iloc[:, 0].map(gene_name).tolist()
        )
        keys = [gene_key(x) for x in genes]
        if (
            not cells
            or not genes
            or len(set(keys)) != len(keys)
            or len(set(cells)) != len(cells)
        ):
            raise ValueError(f"Empty or ambiguous gene/cell identifiers: {path}")
        if all_cells & set(cells):
            raise ValueError(
                "A cell occurs in more than one existing cluster; do not reassign it silently."
            )
        all_cells.update(cells)
        result[cid] = {"genes": genes, "gene_keys": keys, "cells": cells}
        hashes[str(path.resolve())] = digest(path)
    return result, hashes


def export_counts(
    archive: Path,
    data_root: Path,
    clusters: list[int] | None = None,
    expected_hash: str = ARCHIVE_SHA256,
    validation_only: bool = False,
) -> Path:
    archive, data_root = archive.resolve(), data_root.resolve()
    actual_hash = digest(archive)
    if actual_hash != expected_hash:
        raise ValueError(
            f"GSE108222 archive checksum mismatch: expected {expected_hash}, got {actual_hash}. Check the downloaded source, do not substitute normalized tables."
        )
    if expected_hash != ARCHIVE_SHA256 and not validation_only:
        raise ValueError(
            "Alternative archive hashes are only supported by marked software tests."
        )
    cluster_layout, input_hashes = layout(data_root, clusters or CLUSTERS)
    wanted_cells = {
        cell for entry in cluster_layout.values() for cell in entry["cells"]
    }
    wanted_genes = {
        gene for entry in cluster_layout.values() for gene in entry["gene_keys"]
    }
    frames, reports, seen_cells, seen_samples = [], [], set(), set()
    pattern = re.compile(
        r"GSM(\d+)_(.+)-Rep(\d)_S1\.counts\.umiCounts\.passingKneeFilter\.table\.csv\.gz$"
    )
    # Read tar members as streams; never extract arbitrary archive paths.
    with tarfile.open(archive, "r") as package:
        for member in package.getmembers():
            if not member.isfile():
                continue
            match = pattern.fullmatch(Path(member.name).name)
            if not match or int(match[1]) not in SAMPLES:
                continue
            gsm = int(match[1])
            if gsm in seen_samples or (match[2], int(match[3])) != SAMPLES[gsm]:
                raise ValueError(
                    f"Duplicate or unexpected sample annotation: {member.name}"
                )
            seen_samples.add(gsm)
            with gzip.GzipFile(fileobj=package.extractfile(member)) as handle:
                frame = pd.read_csv(handle)
            if frame.columns[0] != "CellId":
                raise ValueError(
                    f"Expected GEO CellId followed by gene count columns: {member.name}"
                )
            values = frame.iloc[:, 1:].to_numpy()
            if (
                not np.isfinite(values).all()
                or (values < 0).any()
                or not np.array_equal(values, np.rint(values))
            ):
                raise ValueError(
                    f"Source table is not nonnegative integer UMI counts: {member.name}"
                )
            if values.max() > np.iinfo(np.int32).max:
                raise ValueError("Count values exceed int32; investigate the source.")
            cells = [
                f"{match[2]}-{barcode}-rep{match[3]}"
                for barcode in frame.CellId.astype(str)
            ]
            if len(set(cells)) != len(cells) or seen_cells & set(cells):
                raise ValueError(f"Ambiguous source cell names: {member.name}")
            seen_cells.update(cells)
            gene_keys = [gene_key(x) for x in frame.columns[1:]]
            if len(set(gene_keys)) != len(gene_keys):
                raise ValueError(
                    f"Ambiguous source gene names: {member.name}; do not merge distinct genes."
                )
            selected_genes = [
                i for i, gene in enumerate(gene_keys) if gene in wanted_genes
            ]
            selected_cells = [i for i, cell in enumerate(cells) if cell in wanted_cells]
            selected = values[np.ix_(selected_cells, selected_genes)].astype(np.int32)
            frames.append(
                pd.DataFrame(
                    selected.T,
                    index=[gene_keys[i] for i in selected_genes],
                    columns=[cells[i] for i in selected_cells],
                )
            )
            reports.append(
                {
                    "file": member.name,
                    "sample": match[2],
                    "replicate": int(match[3]),
                    "n_source_cells": len(cells),
                    "n_matched_cells": len(selected_cells),
                    "n_source_genes": len(gene_keys),
                    "integer_counts": True,
                }
            )
            print(
                f"GSM{gsm}: {len(selected_cells)}/{len(cells)} cells matched; original UMI counts verified",
                flush=True,
            )
    if seen_samples != set(SAMPLES):
        raise ValueError(
            f"Missing expected source samples: {sorted(set(SAMPLES) - seen_samples)}"
        )
    if wanted_cells - seen_cells:
        raise ValueError(
            f"{len(wanted_cells - seen_cells)} existing cells have no exact GEO match; first missing: {sorted(wanted_cells - seen_cells)[:5]}"
        )
    matrix = pd.concat(frames, axis=1, join="outer")
    if matrix.isna().any().any():
        raise ValueError(
            "Source gene coverage differs across samples; do not fill unknown counts with zero."
        )
    missing_genes = wanted_genes - set(matrix.index)
    if missing_genes:
        raise ValueError(
            f"Local genes absent from GEO counts: {sorted(missing_genes)[:10]}"
        )
    output = new_run(HERE / "runs", "geo_counts_GSE108222")
    counts_root = output / "counts_cluster_tables"
    counts_root.mkdir()
    cluster_reports = []
    for cid, entry in cluster_layout.items():
        counts = matrix.loc[entry["gene_keys"], entry["cells"]].copy()
        counts.index = entry["genes"]
        counts.index.name = "Gene"
        counts.to_csv(counts_root / f"cluster{cid}.table", sep="\t")
        cluster_reports.append(
            {
                "cluster": f"C{cid}",
                "n_genes": len(counts),
                "n_cells": counts.shape[1],
                "n_missing_cells": 0,
                "n_missing_genes": 0,
            }
        )
        print(
            f"C{cid}: exported {counts.shape[0]} genes x {counts.shape[1]} original cells",
            flush=True,
        )
    write_json(
        counts_root / "source_manifest.json",
        {
            "dataset": "GSE108222",
            "source_url": SOURCE_URL,
            "archive_sha256": actual_hash,
            "archive": str(archive),
            "validation_only": validation_only,
            "expression_source_sha256": input_hashes,
            "cell_mapping": "sample + CellId + replicate; exact names, original cluster/order retained",
            "gene_mapping": "case-insensitive exact symbol match; literal mouse symbol dots retained; no aggregation",
            "counts": "original GEO UMI counts, no log, depth normalization, rescaling or invented zeros",
            "n_matched_cells": len(wanted_cells),
            "sample_reports": reports,
            "cluster_reports": cluster_reports,
            "files_sha256": {
                p.name: digest(p) for p in counts_root.glob("cluster*.table")
            },
        },
    )
    print(f"Original mouse counts: {counts_root}", flush=True)
    return counts_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=HERE / "runs/source_inspection_GSE108222/GSE108222_RAW.tar",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Existing normalized cluster tables, priors and ChIP gold; never overwritten.",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Also freeze a new shared-prior run using the verified original counts and existing DRAFT checkpoint.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=HERE.parent.parent / "docs/grn_best.pt"
    )
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    counts_root = export_counts(args.archive, args.data_root)
    if args.prepare:
        prepared = prepare(
            args.data_root,
            args.checkpoint,
            counts_root,
            seed=args.seed,
            gold_header="absent",
            expression_kind="normalized",
        )
        write_json(
            counts_root.parent / "run.json",
            {"counts_root": str(counts_root), "prepared_run": str(prepared)},
        )
        print(
            f"Next: python comparisons/mouse_dictys/workflow.py --prepared-run {prepared} --cpus 8 --device cuda:0",
            flush=True,
        )


if __name__ == "__main__":
    main()
