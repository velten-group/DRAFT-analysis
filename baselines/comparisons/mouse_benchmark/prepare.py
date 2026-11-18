"""Freeze independent mouse baseline inputs and existing DRAFT scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from common import (
    HERE,
    IDENTIFIER_POLICY,
    digest,
    edge_codes,
    gene_index,
    gene_key,
    gene_name,
    load_expression,
    load_network,
    new_run,
    read_gold,
    read_json,
    validate_counts,
    write_json,
)


def shared_qc(counts: pd.DataFrame, min_counts: int, min_cells: int) -> pd.DataFrame:
    """Same input-only gene/cell QC for both external models; no gold access."""
    while True:
        old_shape = counts.shape
        keep_genes = (counts.sum(axis=1) >= min_counts) & (
            (counts > 0).sum(axis=1) >= min_cells
        )
        keep_cells = counts.sum(axis=0) >= 1
        counts = counts.loc[keep_genes, keep_cells]
        if counts.shape[0] < 2 or counts.shape[1] < 3:
            raise ValueError("Too few genes/cells after shared count QC.")
        if counts.shape == old_shape:
            return counts


def prepare(
    data_root: Path,
    checkpoint: Path,
    counts_root: Path | None,
    seed: int = 2025,
    gold_header: str = "auto",
    min_counts: int = 10,
    min_cells: int = 3,
    validation_only: bool = False,
    expression_kind: str = "normalized",
) -> Path:
    import torch  # only the preparation environment needs DRAFT's checkpoint runtime

    data_root, checkpoint = data_root.resolve(), checkpoint.resolve()
    raw_counts_available = counts_root is not None or expression_kind == "counts"
    counts_root = (counts_root or data_root).resolve() if raw_counts_available else None
    # User-owned research checkpoint; load its saved arrays, never instantiate DRAFT.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    clusters = [int(cid) for cid in meta["topo"]]
    tfs, tgs = (
        [gene_name(x) for x in meta["TF_list"]],
        [gene_name(x) for x in meta["TG_list"]],
    )
    tf_idx, tg_idx = gene_index(tfs), gene_index(tgs)
    threshold = float(meta["cfg"]["network_score_threshold"])
    sims = ckpt.get("sims_best")
    if sims is None or len(sims) != len(clusters):
        raise ValueError(
            "Checkpoint must contain one saved sims_best matrix per cluster."
        )
    canonical = {gene_key(name): name for name in tgs + tfs}
    source_hashes = {str(checkpoint): digest(checkpoint)}
    count_provenance = None
    if counts_root and (counts_root / "source_manifest.json").exists():
        provenance_path = counts_root / "source_manifest.json"
        count_provenance = read_json(provenance_path)
        if count_provenance.get("validation_only") and not validation_only:
            raise ValueError(
                "Synthetic count exports cannot be prepared as biological runs."
            )
        for name, expected in count_provenance["files_sha256"].items():
            if digest(counts_root / name) != expected:
                raise ValueError(
                    f"Imported GEO count table changed: {counts_root / name}"
                )
        source_hashes[str(provenance_path)] = digest(provenance_path)
    run = new_run(HERE / "runs", f"shared_prior_s{seed}")
    print(f"Preparing mouse run: {run}", flush=True)
    stats = []
    features = {"TF": tfs, "TG": tgs, "clusters": clusters}
    write_json(run / "features.json", features)
    for tid, cid in enumerate(clusters):
        print(f"Preparing C{cid} ({tid + 1}/{len(clusters)})", flush=True)
        expression_path = data_root / f"cluster{cid}.table"
        counts_path = (
            counts_root / f"cluster{cid}.table" if counts_root else expression_path
        )
        prior_path = data_root / f"cluster{cid}_network.txt"
        for path in {expression_path, counts_path, prior_path}:
            source_hashes[str(path)] = digest(path)
        expression = load_expression(expression_path)
        counts = (
            expression
            if expression_path == counts_path
            else load_expression(counts_path)
        )
        if raw_counts_available:
            validate_counts(counts)
        if set(counts.columns) != set(expression.columns):
            raise ValueError(
                f"Raw counts and DRAFT expression must contain the same cells: C{cid}"
            )
        counts = counts.loc[:, expression.columns]
        known = counts.index.isin(set(tf_idx) | set(tg_idx))
        counts = counts.loc[known]
        before = counts.shape
        counts = shared_qc(counts, min_counts if raw_counts_available else 0, min_cells)
        # Preserve Mouse/MGI symbol case for cisTarget; uppercase only for matching.
        counts.index = [canonical[key] for key in counts.index]
        net = load_network(prior_path, threshold)
        all_codes = edge_codes(net, tfs, tgs)
        present = {gene_key(x) for x in counts.index}
        available = net.TF.isin(present) & net.TG.isin(present)
        codes = edge_codes(net.loc[available], tfs, tgs)
        # Self-links are excluded for all methods: pySCENIC adds TFs to modules.
        keep = (
            np.asarray(list(tf_idx))[codes // len(tgs)]
            != np.asarray(list(tg_idx))[codes % len(tgs)]
        )
        codes = codes[keep]
        if not len(codes):
            raise ValueError(f"No evaluable shared-prior edges after QC: C{cid}")
        score_matrix = np.asarray(sims[tid])
        if score_matrix.shape != (len(tfs), len(tgs)):
            raise ValueError(
                f"Saved DRAFT matrix is not TF x TG: C{cid}, {score_matrix.shape}"
            )
        values = score_matrix[codes // len(tgs), codes % len(tgs)].astype(np.float32)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Saved DRAFT scores must be finite and nonnegative.")
        stage = run / f"C{cid}"
        stage.mkdir()
        np.save(stage / "candidate_codes.npy", codes)
        np.save(stage / "draft_scores.npy", values)
        if raw_counts_available:
            counts.to_csv(stage / "counts.tsv.gz", sep="\t")
            scenic_expression = np.log1p(counts.div(counts.sum(axis=0), axis=1) * 1e4).T
        else:
            scenic_expression = (
                counts.T
            )  # already normalized input; do not log/CPM again
        scenic_expression.to_csv(stage / "scenic_expression.tsv", sep="\t")
        active_tfs = sorted({tfs[i] for i in codes // len(tgs)})
        (stage / "regulators.txt").write_text(
            "\n".join(active_tfs) + "\n", encoding="utf-8"
        )
        prior = pd.DataFrame(
            {
                "TF": [tfs[i] for i in codes // len(tgs)],
                "TG": [tgs[j] for j in codes % len(tgs)],
            }
        )
        prior.to_csv(stage / "prior.tsv.gz", sep="\t", index=False)
        stats.append(
            {
                "cluster": f"C{cid}",
                "n_genes_before_qc": before[0],
                "n_cells_before_qc": before[1],
                "n_genes_after_qc": counts.shape[0],
                "n_cells_after_qc": counts.shape[1],
                "n_prior_before_qc": len(all_codes),
                "n_candidates": len(codes),
                "n_regulators": len(active_tfs),
            }
        )
    # Gold is read after model input/feature/candidate choices have been frozen.
    gold_path = data_root / "chipseq_TF_TG_filtered.tsv"
    source_hashes[str(gold_path)] = digest(gold_path)
    gold = read_gold(gold_path, gold_header)
    gold_codes = edge_codes(gold, tfs, tgs)
    gold.to_csv(run / "gold_edges.tsv.gz", sep="\t", index=False)
    np.save(run / "gold_codes.npy", gold_codes)
    pd.DataFrame(stats).to_csv(run / "input_summary.tsv", sep="\t", index=False)
    manifest = {
        "protocol_version": 2,
        "identifier_policy": IDENTIFIER_POLICY,
        "seed": seed,
        "validation_only": validation_only,
        "clusters": clusters,
        "parents": meta.get("parent_cid", {}),
        "checkpoint": str(checkpoint),
        "source_sha256": source_hashes,
        "gold_header": gold_header,
        "n_gold_edges": len(gold),
        "n_gold_mapped": len(gold_codes),
        "prior_threshold": threshold,
        "self_edges": "excluded for all methods",
        "expression_kind": expression_kind,
        "raw_counts_available": raw_counts_available,
        "counts_root": str(counts_root) if counts_root else None,
        "counts_import_provenance": count_provenance,
        "counts_kind": "user-supplied original counts; integer check passed"
        if raw_counts_available
        else "unavailable; official Dictys must wait for original counts",
        "shared_qc": {
            "min_gene_counts": min_counts if raw_counts_available else 0,
            "min_gene_cells": min_cells,
            "min_cell_expression": 1,
        },
        "scenic_expression": "log1p(CP10K)"
        if raw_counts_available
        else "supplied normalized values without further transformation",
        "draft": "read-only saved checkpoint scores, no retraining or inference; original training cohort",
        "universe": "per-cluster DRAFT prior above checkpoint threshold, endpoints passing shared count QC",
        "incomplete_gold": "unreported ChIP edges are putative negatives, not verified non-binding",
        "files_sha256": {
            str(p.relative_to(run)): digest(p) for p in run.rglob("*") if p.is_file()
        },
    }
    write_json(run / "manifest.json", manifest)
    print(f"Prepared mouse run: {run}", flush=True)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=HERE.parent.parent / "docs/grn_best.pt"
    )
    parser.add_argument(
        "--counts-root",
        type=Path,
        help="Original counts in cluster{cid}.table format, same cells as DRAFT.",
    )
    parser.add_argument(
        "--expression-kind", choices=["counts", "normalized"], default="normalized"
    )
    parser.add_argument(
        "--confirm-raw-counts",
        action="store_true",
        help="Confirm supplied count tables are original counts, not rounded normalized values.",
    )
    parser.add_argument(
        "--gold-header", choices=["auto", "present", "absent"], default="auto"
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-gene-counts", type=int, default=10)
    parser.add_argument("--min-gene-cells", type=int, default=3)
    args = parser.parse_args()
    if (
        args.counts_root or args.expression_kind == "counts"
    ) and not args.confirm_raw_counts:
        parser.error(
            "Confirm original RNA counts with --confirm-raw-counts; use --counts-root if cluster*.table is normalized."
        )
    prepare(
        args.data_root,
        args.checkpoint,
        args.counts_root,
        args.seed,
        args.gold_header,
        args.min_gene_counts,
        args.min_gene_cells,
        expression_kind=args.expression_kind,
    )


if __name__ == "__main__":
    main()
