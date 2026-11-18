"""Independent native SCENIC+ RE–TG inference with the shared germline inputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from native import bootstrap, digest, infer_r2g, load_native

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
SCDORI = HERE.parent / "scdori_mondo"
LOG = logging.getLogger(__name__)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def check_files(base, hashes):
    for name, expected in hashes.items():
        if digest(Path(base) / name) != expected:
            raise ValueError(f"Saved input/output changed: {Path(base) / name}")


def resolve_comparison(scdori_run, comparison=Path("latest")):
    if Path(comparison) != Path("latest"):
        path = Path(comparison).expanduser().resolve()
    else:
        run = Path(scdori_run).expanduser().resolve()
        if not run.is_relative_to(SCDORI / "runs"):
            raise ValueError(
                "scDoRI run must be inside its independent runs directory."
            )
        reports = list(
            run.glob("stage_networks_*/comparison_hic_*/comparison_report.json")
        )
        if not reports:
            raise FileNotFoundError(
                "No completed three-method comparison; export/compare the existing networks first."
            )
        path = max(reports, key=lambda p: p.stat().st_mtime_ns).parent
    if not path.is_relative_to(SCDORI / "runs"):
        raise ValueError("Comparison must be inside comparisons/scdori_mondo/runs.")
    return path


def normalize_cell(value):
    return re.sub(
        r"#s?0*([0-9]+)$", r"_sample\1", str(value).strip(), flags=re.IGNORECASE
    )


def region_key(value):
    return re.sub(r"^chr", "", str(value).strip(), flags=re.IGNORECASE)


def prepare(comparison, root=None, *, seed=2025, topics=40, iterations=150):
    comparison = Path(comparison).resolve()
    if not comparison.is_relative_to(SCDORI / "runs"):
        raise ValueError(
            "Comparison must be inside the independent scDoRI runs directory."
        )
    report = read_json(comparison / "comparison_report.json")
    networks = Path(report["stage_networks"]).resolve()
    if not networks.is_relative_to(SCDORI / "runs"):
        raise ValueError(
            "Stage networks must stay inside the independent scDoRI runs directory."
        )
    if digest(networks / "stage_report.json") != report["stage_report_sha256"]:
        raise ValueError("Parent stage report changed.")
    stage = read_json(networks / "stage_report.json")
    if bool(stage["validation_only"]) != bool(report["validation_only"]):
        raise ValueError("Parent comparison/stage validation scopes differ.")
    if seed != stage.get("seed"):
        raise ValueError("New SCENIC+ seed must match the original model comparison.")
    if topics < 2 or iterations < 1:
        raise ValueError("Need at least two topics and one LDA iteration.")
    root = (
        Path(root or Path(stage["input_sources"]["TG.tsv"]["path"]).parent)
        .expanduser()
        .resolve()
    )
    for name, record in stage["input_sources"].items():
        if digest(root / name) != record["sha256"]:
            raise ValueError(f"Shared input source changed: {name}")
    for name in ["stage_arrays.npz", "cells.tsv.gz"]:
        if digest(networks / name) != stage["files_sha256"][name]:
            raise ValueError(f"Shared feature/cell export changed: {name}")
    with np.load(networks / "stage_arrays.npz", allow_pickle=False) as arrays:
        stage_genes, stage_peaks = arrays["genes"].tolist(), arrays["peaks"].tolist()
        pair_gene, pair_peak = arrays["pair_gene"], arrays["pair_peak"]
    if (comparison / "evaluation_features.json").is_file():
        features = read_json(comparison / "evaluation_features.json")
        if report.get("evaluation_features_sha256"):
            check_files(
                comparison,
                {"evaluation_features.json": report["evaluation_features_sha256"]},
            )
        genes, peaks = features["genes"], features["peaks"]
    else:
        tg = pd.read_csv(root / "TG.tsv", sep="\t", dtype=str)
        raw_peaks = {region_key(p) for p in (root / "RE.txt").read_text().splitlines()}
        genes = [g for g in stage_genes if g in set(tg.FBgn_ID)]
        peaks = [p for p in stage_peaks if region_key(p) in raw_peaks]
    if len(genes) != report["n_common_genes"] or len(peaks) != report["n_common_peaks"]:
        raise ValueError(
            "Cannot recover the original unlabeled feature pool; rerun compare_stage_hic.py to save it explicitly."
        )
    cells = pd.read_csv(networks / "cells.tsv.gz", sep="\t")
    if not cells.cell_id.is_unique or set(cells.time_idx) != set(range(15)):
        raise ValueError("Need the original unique paired cells at all fifteen stages.")
    rna, atac = [
        ad.read_h5ad(root / name) for name in ["A_rna_focus.h5ad", "B_atac_focus.h5ad"]
    ]
    ordered = pd.Index(cells.cell_id.map(normalize_cell))
    rna_ids = pd.Index([normalize_cell(c) for c in rna.obs_names])
    atac_ids = pd.Index([normalize_cell(c) for c in atac.obs_names])
    if not ordered.is_unique or not rna_ids.is_unique or not atac_ids.is_unique:
        raise ValueError("Normalized cell identifiers must be unique.")
    ri, ai = rna_ids.get_indexer(ordered), atac_ids.get_indexer(ordered)
    if (ri < 0).any() or (ai < 0).any():
        raise ValueError("A comparison cell is missing in original RNA/ATAC.")
    if "lognorm" not in rna.layers or "counts" not in atac.layers:
        raise ValueError("Require original RNA lognorm and ATAC fragment-count layers.")
    gene_indices = rna.var_names.get_indexer(genes)
    raw_peak_lookup = {region_key(p): i for i, p in enumerate(atac.var_names)}
    if (gene_indices < 0).any() or any(
        region_key(p) not in raw_peak_lookup for p in peaks
    ):
        raise ValueError("Shared feature identifiers differ from original modalities.")
    expression = rna.layers["lognorm"][ri][:, gene_indices]
    expression = np.asarray(
        expression.toarray() if sp.issparse(expression) else expression,
        dtype=np.float32,
    )
    counts = sp.csr_matrix(
        atac.layers["counts"][ai][:, [raw_peak_lookup[region_key(p)] for p in peaks]]
    )
    counts.eliminate_zeros()
    if not np.isfinite(expression).all() or (expression < 0).any():
        raise ValueError("Invalid original normalized RNA values.")
    if (
        not np.isfinite(counts.data).all()
        or (counts.data < 0).any()
        or not np.allclose(counts.data, np.round(counts.data))
    ):
        raise ValueError("ATAC requires nonnegative integer fragment counts.")
    if np.any(np.asarray(counts.sum(axis=1)).ravel() == 0):
        raise ValueError(
            "A paired cell has no ATAC counts; do not silently change the cell cohort."
        )
    gi, pi = set(genes), {region_key(p): p for p in peaks}
    search = pd.DataFrame(
        {
            "Gene": [stage_genes[g] for g in pair_gene],
            "Name": [pi.get(region_key(stage_peaks[p]), "") for p in pair_peak],
        }
    )
    search = (
        search.loc[search.Gene.isin(gi) & search.Name.ne("")]
        .drop_duplicates()
        .sort_values(["Gene", "Name"])
        .reset_index(drop=True)
    )
    if search.empty:
        raise ValueError("Shared fixed RE–TG candidate support is empty.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = HERE / "runs" / f"shared_prior_s{seed}_{stamp}_{uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    np.save(run / "rna_expression.npy", expression)
    sp.save_npz(run / "atac_fragment_counts.npz", counts.astype(np.int32))
    cells.to_csv(run / "cells.tsv.gz", sep="\t", index=False)
    search.to_csv(run / "search_space.tsv.gz", sep="\t", index=False)
    write_json(run / "features.json", {"genes": genes, "peaks": peaks})
    names = [
        "rna_expression.npy",
        "atac_fragment_counts.npz",
        "cells.tsv.gz",
        "search_space.tsv.gz",
        "features.json",
    ]
    write_json(
        run / "manifest.json",
        {
            "comparison": str(comparison),
            "comparison_report_sha256": digest(comparison / "comparison_report.json"),
            "stage_networks": str(networks),
            "stage_report_sha256": digest(networks / "stage_report.json"),
            "source_cell_table_sha256": digest(networks / "cells.tsv.gz"),
            "input_sources": stage["input_sources"],
            "validation_only": bool(stage["validation_only"]),
            "seed": seed,
            "topics": topics,
            "lda_iterations": iterations,
            "gbm_trees": 500,
            "n_cells": len(cells),
            "n_genes": len(genes),
            "n_peaks": len(peaks),
            "n_candidate_pairs": len(search),
            "files_sha256": {n: digest(run / n) for n in names},
            "prepare_code_sha256": digest(Path(__file__)),
            "upstream": read_json(HERE / "upstream_api.json"),
            "design": "native SCENIC+ R2G core with pycisTopic CGS/imputation, shared paired germline cells and complete fixed RE–TG prior support; no motif/eRegulon/DAR/DEG filters; no Hi-C labels read during inference; global fit, not a temporal TF–RE–TG model",
        },
    )
    LOG.info(
        "Prepared %d cells, %d genes, %d peaks and %d candidate pairs: %s",
        len(cells),
        len(genes),
        len(peaks),
        len(search),
        run,
    )
    return run


def resolve_run(value):
    if Path(value) == Path("latest"):
        candidates = [
            p
            for p in (HERE / "runs").glob("shared_prior_s*/manifest.json")
            if not read_json(p).get("validation_only", False)
        ]
        if not candidates:
            raise FileNotFoundError("No prepared biological SCENIC+ R2G run.")
        value = max(candidates, key=lambda p: p.stat().st_mtime_ns).parent
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(HERE / "runs"):
        raise ValueError("SCENIC+ outputs must stay in their own runs directory.")
    return path


def fit(run, cpus=16, api=None):
    run = resolve_run(run)
    manifest = read_json(run / "manifest.json")
    check_files(run, manifest["files_sha256"])
    if manifest["upstream"] != read_json(HERE / "upstream_api.json"):
        raise ValueError(
            "Native source pins differ from preparation; create a new run."
        )
    if (run / "r2g_report.json").exists():
        check_files(run, read_json(run / "r2g_report.json")["files_sha256"])
        LOG.info("Completed R2G output verified; reusing %s", run)
        return run
    if cpus < 1:
        raise ValueError("CPU count must be positive.")
    api = api or load_native()
    features = read_json(run / "features.json")
    cells = pd.read_csv(run / "cells.tsv.gz", sep="\t").cell_id.tolist()
    names = features["peaks"]
    if (run / "topics_complete.json").exists():
        check_files(run, read_json(run / "topics_complete.json")["files_sha256"])
    else:
        counts = sp.load_npz(run / "atac_fragment_counts.npz")
        # Same >=1 accessibility and all-zero-region removal as native create_cistopic_object.
        keep = np.flatnonzero(counts.getnnz(axis=0) > 0)
        binary = counts[:, keep].copy()
        binary.data[:] = 1
        names = [names[i] for i in keep]
        if manifest["topics"] > min(binary.shape):
            raise ValueError("Topic count exceeds the paired cell or nonzero RE count.")
        LOG.info(
            "Native pycisTopic CGS: %d topics, %d iterations, seed %d (one model/CPU)",
            manifest["topics"],
            manifest["lda_iterations"],
            manifest["seed"],
        )
        topic = api.run_cgs_model(
            binary,
            manifest["topics"],
            cells,
            names,
            n_iter=manifest["lda_iterations"],
            random_state=manifest["seed"],
        )
        np.savez_compressed(
            run / "topic_arrays.npz",
            cells=np.asarray(cells),
            peaks=np.asarray(names),
            cell_topic=topic.cell_topic.to_numpy(),
            topic_region=topic.topic_region.to_numpy(),
        )
        topic.parameters.to_csv(run / "native_topic_parameters.tsv", sep="\t")
        write_json(
            run / "topics_complete.json",
            {
                "files_sha256": {
                    n: digest(run / n)
                    for n in ["topic_arrays.npz", "native_topic_parameters.tsv"]
                }
            },
        )
    if (run / "imputation_complete.json").exists():
        check_files(run, read_json(run / "imputation_complete.json")["files_sha256"])
    else:
        with np.load(run / "topic_arrays.npz", allow_pickle=False) as arrays:
            names = arrays["peaks"].tolist()
            topics = [f"Topic{i + 1}" for i in range(arrays["cell_topic"].shape[0])]
            model = SimpleNamespace(
                cell_topic=pd.DataFrame(
                    arrays["cell_topic"], index=topics, columns=cells
                ),
                topic_region=pd.DataFrame(
                    arrays["topic_region"], index=names, columns=topics
                ),
            )
        imputed = api.impute_accessibility(
            SimpleNamespace(selected_model=model, cell_names=cells, region_names=names)
        )
        if imputed.cell_names != cells:
            raise ValueError("Native imputation changed cell order.")
        np.save(run / "imputed_accessibility.npy", imputed.mtx)
        write_json(run / "imputed_peaks.json", imputed.feature_names)
        write_json(
            run / "imputation_complete.json",
            {
                "files_sha256": {
                    n: digest(run / n)
                    for n in ["imputed_accessibility.npy", "imputed_peaks.json"]
                }
            },
        )
    imputed_names = read_json(run / "imputed_peaks.json")
    expression = pd.DataFrame(
        np.load(run / "rna_expression.npy", mmap_mode="r"),
        index=cells,
        columns=features["genes"],
    )
    accessibility = pd.DataFrame(
        np.load(run / "imputed_accessibility.npy", mmap_mode="r").T,
        index=cells,
        columns=imputed_names,
    )
    original_search = pd.read_csv(run / "search_space.tsv.gz", sep="\t")
    search = original_search.loc[original_search.Name.isin(imputed_names)].copy()
    if search.empty:
        raise ValueError("No candidate REs survive native topic imputation.")
    LOG.info(
        "Native SCENIC+ R2G: %d pairs, GBM 500 trees + Spearman, %d CPUs, seed %d",
        len(search),
        cpus,
        manifest["seed"],
    )
    table = infer_r2g(
        api,
        expression,
        accessibility,
        search,
        run / "joblib_tmp",
        cpus,
        manifest["seed"],
    )
    expected = set(zip(search.Name, search.Gene))
    if len(table) != len(expected) or set(zip(table.region, table.target)) != expected:
        raise ValueError(
            "Native R2G output does not score every retained candidate exactly once."
        )
    if not np.isfinite(table.importance).all() or (table.importance < 0).any():
        raise ValueError("Native R2G importance is invalid.")
    table.to_csv(run / "region_to_gene.tsv.gz", sep="\t", index=False)
    discarded = original_search.loc[~original_search.Name.isin(imputed_names)]
    discarded.to_csv(
        run / "candidates_lost_in_preprocessing.tsv.gz", sep="\t", index=False
    )
    names = ["region_to_gene.tsv.gz", "candidates_lost_in_preprocessing.tsv.gz"]
    packages = {
        p: importlib.metadata.version(p)
        for p in [
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "lda",
            "arboreto",
            "joblib",
            "tmtoolkit",
        ]
    }
    write_json(
        run / "r2g_report.json",
        {
            "manifest_sha256": digest(run / "manifest.json"),
            "validation_only": manifest["validation_only"],
            "comparison": manifest["comparison"],
            "comparison_report_sha256": manifest["comparison_report_sha256"],
            "source_cell_table_sha256": manifest["source_cell_table_sha256"],
            "seed": manifest["seed"],
            "n_cells": len(cells),
            "n_candidate_pairs": len(original_search),
            "n_scored_candidate_pairs": len(table),
            "n_preprocessing_lost_pairs": len(discarded),
            "n_positive_importance_pairs": int(table.importance.gt(0).sum()),
            "temporal_scope": "global_static",
            "source_kind": "native_r2g_shared_prior",
            "files_sha256": {n: digest(run / n) for n in names},
            "packages": packages,
            "upstream": manifest["upstream"],
            "code_sha256": digest(Path(__file__)),
            "native_adapter_sha256": digest(HERE / "native.py"),
            "score": "native GBM feature importance; every retained fixed candidate scored, including zero importance; no DAR/DEG/eRegulon/TF-path/significance/Top-K filters; outside-candidate/preprocessing-lost pairs stay unsupported in the saved common evaluation pool",
        },
    )
    print(f"R2G complete: {run}", flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["bootstrap", "prepare", "fit", "all"])
    parser.add_argument("--scdori-run", type=Path)
    parser.add_argument("--comparison", type=Path, default=Path("latest"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run", type=Path, default=Path("latest"))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--topics", type=int, default=40)
    parser.add_argument("--lda-iterations", type=int, default=150)
    parser.add_argument("--cpus", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.command == "bootstrap":
        print(bootstrap())
        return
    if args.command in ["prepare", "all"]:
        if args.comparison == Path("latest") and args.scdori_run is None:
            parser.error("Pass --scdori-run or pin --comparison /EXACT/PATH.")
        comparison = resolve_comparison(args.scdori_run, args.comparison)
        run = prepare(
            comparison,
            args.data_root,
            seed=args.seed,
            topics=args.topics,
            iterations=args.lda_iterations,
        )
        print(f"Prepared R2G run: {run}", flush=True)
    else:
        run = resolve_run(args.run)
    if args.command in ["fit", "all"]:
        fit(run, args.cpus)


if __name__ == "__main__":
    main()
