"""Hi-C-supported RE–TG recovery from the saved global scDoRI gene-peak factors."""

from __future__ import annotations

import uuid
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from hic_common import map_positives, metrics, negative_pairs
from scdori_inputs import load_run
from scdori_native import PIN, check_outputs, digest, read_json, write_json


def benchmark(
    run: Path, data_root: Path | None, thresholds=(0.5, 0.95), seed=42
) -> Path:
    manifest = load_run(run)
    if manifest["upstream_commit"] != PIN:
        raise ValueError("Benchmark expects the recorded official scDoRI release.")
    thresholds = list(dict.fromkeys(float(t) for t in thresholds))
    if not thresholds or not np.isfinite(thresholds).all() or min(thresholds) < 0:
        raise ValueError("Use finite nonnegative weight thresholds.")
    completed = run / "grn_complete.json"
    if not completed.is_file():
        raise RuntimeError("Finish GRN training before benchmarking.")
    record = read_json(completed)
    check_outputs(record, run)
    checkpoint = (run / record["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(run.resolve()):
        raise ValueError("Checkpoint must be inside the selected run.")
    # Native torch.save uses ZIP serialization. Memory mapping avoids loading
    # unused topic/TF-GRN parameters or constructing a second full model.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    learnt = state["gene_peak_factor_learnt"]
    fixed = state["gene_peak_factor_fixed"]
    rna = ad.read_h5ad(run / "data/rna_processed.h5ad", backed="r")
    atac = ad.read_h5ad(run / "data/atac_processed.h5ad", backed="r")
    try:
        indices = np.flatnonzero(rna.var.gene_type.eq("HVG"))
        genes = rna.var_names[indices].tolist()
        peaks = atac.var_names.tolist()
        full_shape = (rna.n_vars, atac.n_vars)
    finally:
        rna.file.close()
        atac.file.close()
    if (
        not genes
        or tuple(learnt.shape) != full_shape
        or tuple(fixed.shape) != full_shape
    ):
        raise ValueError(
            "Checkpoint gene-peak dimensions differ from the prepared inputs."
        )
    prior = np.load(run / "data/gene_peak_distance_exp.npy", mmap_mode="r")
    if prior.shape != full_shape:
        raise ValueError("Prepared distance matrix has invalid dimensions.")
    allowed = []
    for index, model_index in enumerate(indices):
        expected = prior[model_index] > 0
        mask = fixed[model_index].numpy()
        if not np.array_equal(mask, expected):
            raise ValueError(
                "Saved fixed gene-peak mask differs from the prepared prior."
            )
        allowed.append(np.flatnonzero(expected) * len(genes) + index)
    allowed_codes = np.sort(np.concatenate(allowed))
    root = (
        data_root or Path(manifest["input_sources"]["TG.tsv"]["path"]).parent
    ).resolve()
    tg_path = root / "TG.tsv"
    if digest(tg_path) != manifest["input_sources"]["TG.tsv"]["sha256"]:
        raise ValueError("TG identifiers differ from the preparation's source table.")
    tg = pd.read_csv(tg_path, sep="\t", dtype=str).fillna("")
    names = dict(zip(tg.FBgn_ID, tg.gene_name))
    rows, coverage, tables, sources = [], [], {}, {}
    for tag in ["spg", "spc"]:
        source = root / f"{tag}_hic.tsv"
        source_hash = digest(source)
        if source_hash in {v["sha256"] for v in manifest["input_sources"].values()}:
            raise ValueError(
                "Hi-C validation data must not duplicate a preparation input."
            )
        sources[tag] = {"path": str(source), "sha256": source_hash}
        positives, mapping = map_positives(source, peaks, genes, tg)
        p, g = positives.T
        reachable = prior[indices[g], p] > 0
        n_positive, n_reachable = len(positives), int(reachable.sum())
        coverage.append(
            {
                "dataset": tag,
                **mapping,
                "positive_pairs_inside_prior": n_reachable,
                "positive_pairs_outside_prior": n_positive - n_reachable,
                "mapped_positive_fraction_inside_prior": n_reachable / n_positive,
                "max_recall_from_fixed_mask": n_reachable / n_positive,
                "max_f1_from_fixed_mask_if_precision_one": 2
                * n_reachable
                / (n_positive + n_reachable),
            }
        )
        for universe in ["all_features", "within_prior"]:
            selected = positives if universe == "all_features" else positives[reachable]
            if not len(selected):
                continue
            negatives = negative_pairs(
                selected,
                len(peaks),
                len(genes),
                seed,
                allowed_codes if universe == "within_prior" else None,
            )
            pairs = np.vstack([selected, negatives])
            p, g = pairs.T
            model_g = indices[g]
            scores = (learnt[model_g, p] * fixed[model_g, p]).numpy()
            distances = prior[model_g, p]
            if not np.isfinite(scores).all() or not np.isfinite(distances).all():
                raise ValueError("Nonfinite RE–TG weights in evaluated pairs.")
            truth = np.r_[
                np.ones(len(selected), dtype=int), np.zeros(len(negatives), dtype=int)
            ]
            common = {
                "dataset": tag,
                "universe": universe,
                "n_positive": len(selected),
                "n_negative": len(negatives),
            }
            rows.append(
                {
                    **common,
                    "model": "shared_prior_mask",
                    "threshold": 0.0,
                    **metrics(truth, distances > 0, 0),
                }
            )
            for threshold in thresholds:
                for label, weights in [
                    ("scDoRI", scores),
                    ("initial_distance_prior", distances),
                ]:
                    rows.append(
                        {
                            **common,
                            "model": label,
                            "threshold": threshold,
                            **metrics(truth, weights, threshold),
                        }
                    )
            tables[f"pairs_{tag}_{universe}.tsv.gz"] = pd.DataFrame(
                {
                    "RE": np.asarray(peaks)[p],
                    "gene_id": np.asarray(genes)[g],
                    "gene_name": [names.get(genes[i], genes[i]) for i in g],
                    "label": truth,
                    "scdori_score": scores,
                    "initial_distance_score": distances,
                    "prior_supported": distances > 0,
                }
            )
    out = run / f"benchmark_hic_{uuid.uuid4().hex[:10]}"
    out.mkdir(exist_ok=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "hic_f1.tsv", sep="\t", index=False)
    pd.DataFrame(coverage).to_csv(out / "hic_coverage.tsv", sep="\t", index=False)
    for name, table in tables.items():
        table.to_csv(out / name, sep="\t", index=False)
    write_json(
        out / "benchmark_report.json",
        {
            "run": str(run.resolve()),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "manifest_sha256": digest(run / "manifest.json"),
            "upstream_commit": PIN,
            "validation_only": bool(manifest["configuration"].get("validation_only")),
            "sources": sources,
            "n_peaks": len(peaks),
            "n_non_tf_targets": len(genes),
            "n_allowed_pairs": len(allowed_codes),
            "negative_seed": seed,
            "thresholds": thresholds,
            "metric": "global gene_peak_factor_learnt * gene_peak_factor_fixed; F1 from score > threshold; no sigmoid, test-tuned threshold or maximum normalization",
            "negative_sampling": "1:1 uniform unobserved pairs, with replacement, excluding the corresponding assay's positives; unobserved does not imply an experimentally confirmed negative",
            "all_features": "primary recovery within selected non-TF RNA genes and peaks; unreachable Hi-C positives retain label 1",
            "within_prior": "conditional secondary recovery restricted to the fixed support; excluded positives are disclosed in coverage",
            "stages": "global shared RE–TG parameters; one SPG and one SPC assay evaluation per threshold, not fifteen independently inferred networks",
            "comparison": "DRAFT must be rescored on these same feature pools and saved pairs before comparing its F1; numerical score thresholds are method-specific",
        },
    )
    fixture_label = (
        "SYNTHETIC SOFTWARE BENCHMARK: not a biological result.\n\n"
        if manifest["configuration"].get("validation_only")
        else ""
    )
    (out / "README.md").write_text(
        fixture_label
        + "Hi-C-supported RE–TG recovery on the trained model's non-TF target/peak pool. "
        "hic_f1.tsv contains fixed-threshold scDoRI F1, precision/recall and two prior baselines. "
        "hic_coverage.tsv discloses omitted features, positives outside the fixed mask and the recall/F1 ceiling imposed by that mask. "
        "all_features is the primary recovery evaluation; within_prior is conditional on supplied support. "
        "Unobserved sampled pairs are proxy negatives. The shared gene-peak matrix has no separate stage-specific parameter. "
        "RE_TG_100kb.tsv is an input prior, not the validation truth. "
        "Existing DRAFT F1 tables use a different target pool; rescore DRAFT on the exported pairs before a direct comparison.\n",
        encoding="utf-8",
    )
    if fixture_label:
        print(fixture_label.strip(), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(
        pd.DataFrame(coverage)[
            [
                "dataset",
                "mapped_unique_positive_pairs",
                "positive_pairs_inside_prior",
                "max_recall_from_fixed_mask",
                "max_f1_from_fixed_mask_if_precision_one",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"Benchmark: {out}", flush=True)
    return out
