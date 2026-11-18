"""Run official LIANA+ on the frozen cells and fly LR prior used for CellChat.

Use the isolated .cache/ccc_liana/venv environment. No project model imports.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import anndata as ad
import liana as li
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.io import mmread


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ccc_liana.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root, out = Path(cfg["benchmark_dir"]), Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(root / "cells.tsv", sep="\t").set_index("cell_id")
    genes = (root / "genes.tsv").read_text(encoding="utf-8").splitlines()
    matrix = sparse.csr_matrix(mmread(root / "expression_genes_by_cells.mtx").T)
    data = ad.AnnData(matrix, obs=cells, var=pd.DataFrame(index=genes))
    data.obs["cell_type"] = data.obs.cell_type.astype("category")
    resource = pd.read_csv(root / "cellchat_fly_interactions.tsv", sep="\t")
    resource = resource[["ligand", "receptor"]].drop_duplicates()
    assert data.n_obs == 9221 and len(resource) == 385
    assert data.obs.cell_type.nunique() == 25
    assert {"wnt4", "arr"}.issubset(data.var_names)
    print(
        f"LIANA+ {li.__version__}: {data.n_obs} cells; {len(resource)} fly LR pairs",
        flush=True,
    )
    methods = [method.method_name for method in li.mt.rank_aggregate.methods]
    options = {"default": cfg["expr_prop"], "unfiltered": cfg["sensitivity_expr_prop"]}
    counts = {}
    for label, expr_prop in options.items():
        print(f"Running {label}: expr_prop={expr_prop}", flush=True)
        result = li.mt.rank_aggregate(
            data,
            groupby="cell_type",
            resource=resource,
            expr_prop=expr_prop,
            min_cells=cfg["min_cells"],
            use_raw=False,
            return_all_lrs=True,
            n_perms=cfg["n_perms"],
            seed=cfg["seed"],
            n_jobs=1,
            aggregate_method=cfg["aggregate_method"],
            inplace=False,
            verbose=True,
        )
        assert isinstance(result, pd.DataFrame)
        keys = ["source", "target", "ligand_complex", "receptor_complex"]
        assert not result.duplicated(keys).any()
        ranks = result[["magnitude_rank", "specificity_rank"]].to_numpy()
        assert np.isfinite(ranks).all() and ((ranks >= 0) & (ranks <= 1)).all()
        # Official high-is-stronger transform, not a calibrated activity scale.
        processed = li.mt.process_scores(result, score_key="magnitude_rank")
        result["magnitude_strength"] = np.maximum(
            processed.magnitude_rank.to_numpy(), 0
        )
        result["interaction_name"] = (
            result.ligand_complex + "__" + result.receptor_complex
        )
        result.to_csv(out / f"liana_{label}_all_edges.tsv.gz", sep="\t", index=False)
        case = result[result.interaction_name.eq("wnt4__arr")].copy()
        case.to_csv(out / f"wnt4_arr_{label}_native_edges.tsv", sep="\t", index=False)
        counts[label] = {
            "rows": len(result),
            "wnt4_arr_rows": len(case),
            "expr_prop": expr_prop,
        }
        print(counts[label], flush=True)
    source = Path(li.__file__).resolve().parent
    source_files = [
        source / "method/sc/_rank_aggregate.py",
        source / "method/sc/_liana_pipe.py",
        source / "_core/_pipe_utils/_aggregate.py",
    ]
    input_files = [
        root / f
        for f in [
            "cells.tsv",
            "genes.tsv",
            "expression_genes_by_cells.mtx",
            "cellchat_fly_interactions.tsv",
            "fixed_mapping.tsv",
        ]
    ]
    manifest = {
        "upstream_commit": cfg["upstream_commit"],
        "version": li.__version__,
        "config": cfg,
        "methods": methods,
        "magnitude_specs": li.mt.rank_aggregate.magnitude_specs,
        "cells": data.n_obs,
        "input_genes": data.n_vars,
        "groups": data.obs.cell_type.cat.categories.tolist(),
        "resource_pairs": len(resource),
        "runs": counts,
        "resource": "same custom single-gene fly prior as official R CellChat run",
        "ranking_universe": "all 25 x 25 group combinations and all covered fly LR candidates",
        "curve_score": "official process_scores(magnitude_rank): -log10(rank + float64 eps), clipped at 0 for roundoff",
        "return_all_lrs": "below-expression-threshold rows receive worst method scores per upstream",
        "model_retrained": False,
        "time_mapping_refitted": False,
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in [
                "liana",
                "anndata",
                "scanpy",
                "numpy",
                "scipy",
                "pandas",
                "numba",
            ]
        },
        "inputs": [{"path": str(f), "sha256": digest(f)} for f in input_files],
        "source_files": [{"path": str(f), "sha256": digest(f)} for f in source_files],
    }
    (out / "execution_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
