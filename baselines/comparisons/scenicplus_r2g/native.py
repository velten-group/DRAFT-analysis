"""Execute fingerprinted upstream R2G/topic functions without unused toolchains.

Selected definitions retain their complete native bodies. Only the Ray dispatch
decorator of the single-topic-model runner is omitted for local execution; no
LDA, imputation, correlation or GBM mathematics is rewritten.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
import pathlib
import pickle
import sys
import time
import types
import warnings
from itertools import chain
from urllib.request import Request, urlopen

import cloudpickle
import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from tqdm import tqdm

HERE = pathlib.Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def bootstrap(directory=None):
    directory = pathlib.Path(directory or HERE / "upstream").resolve()
    spec = json.loads((HERE / "upstream_api.json").read_text())
    for repo, record in spec.items():
        for name, expected in record["sha256_lf"].items():
            target = directory / repo / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                url = f"https://raw.githubusercontent.com/aertslab/{repo}/{record['commit']}/{name}"
                request = Request(
                    url, headers={"User-Agent": "DRAFT-independent-R2G-baseline"}
                )
                for attempt in range(3):
                    try:
                        with urlopen(request, timeout=30) as response:
                            raw = response.read()
                        break
                    except OSError:
                        if attempt == 2:
                            raise
                        time.sleep(attempt + 1)
                if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != expected:
                    raise ValueError(
                        f"Downloaded upstream fingerprint differs: {repo}/{name}"
                    )
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(raw)
                temporary.replace(target)
            actual = hashlib.sha256(
                target.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            if actual != expected:
                raise ValueError(f"Upstream source changed: {repo}/{name}")
    return directory


def definitions(path, names, namespace, *, undecorate=()):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            if node.name in undecorate:
                node.decorator_list = []
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            targets = {n.id for n in node.targets if isinstance(n, ast.Name)}
            if targets & set(names):
                selected.append(node)
                found.update(targets & set(names))
    if found != set(names):
        raise ValueError(f"Missing native definitions at {path}: {set(names) - found}")
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *selected], type_ignores=[])
    )
    # load_native() verifies pinned SHA-256 fingerprints before selecting code.
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102


def load_native(directory=None):
    import lda
    import tmtoolkit

    directory = bootstrap(directory)
    module = types.ModuleType("_independent_scenicplus_r2g_native")
    ns = module.__dict__
    ns.update(
        {
            "np": np,
            "pd": pd,
            "scipy": scipy,
            "sparse": sparse,
            "joblib": joblib,
            "logging": logging,
            "sys": sys,
            "os": os,
            "pathlib": pathlib,
            "math": math,
            "time": time,
            "warnings": warnings,
            "pickle": pickle,
            "chain": chain,
            "tqdm": tqdm,
            "lda": lda,
            "tmtoolkit": tmtoolkit,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "RandomForestRegressor": RandomForestRegressor,
            "ExtraTreesRegressor": ExtraTreesRegressor,
            "pearsonr": pearsonr,
            "spearmanr": spearmanr,
        }
    )
    pc = directory / "pycisTopic/src/pycisTopic"
    definitions(pc / "utils.py", ["subset_list", "non_zero_rows", "loglikelihood"], ns)
    definitions(
        pc / "lda_models.py",
        ["CistopicLDAModel", "run_cgs_model"],
        ns,
        undecorate=["run_cgs_model"],
    )
    definitions(
        pc / "diff_features.py", ["CistopicImputedFeatures", "impute_accessibility"], ns
    )
    definitions(
        directory / "scenicplus/src/scenicplus/enhancer_to_gene.py",
        [
            "SKLEARN_REGRESSOR_FACTORY",
            "SCIPY_CORRELATION_FACTORY",
            "GBM_KWARGS",
            "_score_regions_to_single_gene",
            "_get_acc_idx_per_gene",
            "_score_regions_to_genes",
            "calculate_regions_to_genes_relationships",
        ],
        ns,
    )
    # Joblib workers must receive definitions by value, rather than import this
    # deliberately private runtime namespace as an installed SCENIC+ package.
    sys.modules[module.__name__] = module
    cloudpickle.register_pickle_by_value(module)
    return module


def seed_arboreto(seed):
    """Set the native regressor seed parameter in parent and joblib workers."""
    from arboreto import core

    defaults = list(core.fit_model.__defaults__)
    defaults[-1] = int(seed)
    core.fit_model.__defaults__ = tuple(defaults)


def infer_r2g(
    api,
    expression,
    accessibility,
    search_space,
    temp_dir,
    cpus,
    seed,
    *,
    n_estimators=500,
):
    from arboreto import core

    original = core.fit_model.__defaults__
    single_gene = api._score_regions_to_single_gene

    def seeded_single_gene(*args, **kwargs):
        # Native SCENIC+ calls arboreto without exposing its seed argument.
        # Set that native parameter per task, including spawned joblib workers.
        seed_arboreto(seed)
        return single_gene(*args, **kwargs)

    api._score_regions_to_single_gene = seeded_single_gene
    seed_arboreto(seed)
    try:
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            return api.calculate_regions_to_genes_relationships(
                expression,
                accessibility,
                search_space,
                pathlib.Path(temp_dir),
                mask_expr_dropout=False,
                importance_scoring_method="GBM",
                importance_scoring_kwargs={
                    **api.GBM_KWARGS,
                    "n_estimators": n_estimators,
                },
                correlation_scoring_method="SR",
                n_cpu=cpus,
                add_distance=False,
            )
    finally:
        api._score_regions_to_single_gene = single_gene
        core.fit_model.__defaults__ = original
