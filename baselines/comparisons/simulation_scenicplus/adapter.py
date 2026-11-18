"""Unchanged, fingerprinted official SCENIC+ TF2G function bodies."""

from __future__ import annotations

import ast
import functools
import logging
import os
import pathlib
import sys
import types

import numpy as np
import pandas as pd
from common import load_module, log_expression, read_json, source_file, write_json

PIN = "840dab85de3044846234c157e327b6ea0290abfb"
SHA = "e87013543fc1cde0085708fae759c078047aa063d8a1394c7800907623cc86ce"
HERE = pathlib.Path(__file__).resolve().parent


def software(directory):
    runtime = load_module(
        "_simulation_scenicplus_runtime", directory / "check_runtime.py"
    ).check()
    source_file(directory / "upstream/TF_to_gene.py", SHA)
    return {
        "repository": "aertslab/scenicplus",
        "commit": PIN,
        "TF_to_gene_sha256_lf": SHA,
        "runtime": runtime,
    }


def monitored_infer(cache, **kwargs):
    from arboreto import core

    target = kwargs["target_gene_name"]
    path = pathlib.Path(cache) / f"target_{target}.json"
    observed = {}
    failures = []
    target_values = np.asarray(kwargs["target_gene_expression"], dtype=float)

    def capture(kind, options, model, names, gene):
        values = core.to_feature_importances(kind, options, model)
        if (
            len(names) != len(values)
            or not np.isfinite(values).all()
            or (values < 0).any()
        ):
            raise ValueError("Invalid native feature importances")
        links = core.to_links_df(kind, options, model, names, gene)
        observed.update({"weights": dict(zip(names, map(float, values)))})
        return links

    def track_retry(fn, *args, **kwargs):
        def tracked():
            try:
                return fn()
            except Exception as error:
                failures.append(f"{type(error).__name__}: {error}")
                raise

        # Preserve Arboreto's retry count, warning behavior and fallback schema.
        return core.retry(tracked, *args, **kwargs)

    native = core.infer_partial_network
    namespace = dict(native.__globals__, to_links_df=capture, retry=track_retry)
    infer = types.FunctionType(
        native.__code__,
        namespace,
        native.__name__,
        native.__defaults__,
        native.__closure__,
    )
    links = infer(**kwargs)
    status = "native_fit_completed_after_retry" if failures else "native_fit_completed"
    if not observed:
        # Arboreto's official result after exhausting its ten retries is an
        # empty sparse adjacency table. Absent links are zero in our dense score
        # tensor, but the failed-fit status remains explicit in the audit.
        observed["weights"] = {name: 0.0 for name in kwargs["tf_matrix_gene_names"]}
        status = (
            "constant_target_official_fallback"
            if np.ptp(target_values) == 0
            else "native_retry_exhausted_official_fallback"
        )
    write_json(
        path,
        {
            "target": target,
            "status": status,
            "failure_count": len(failures),
            "failure_reasons": list(dict.fromkeys(failures)),
            "target_min": float(target_values.min()),
            "target_max": float(target_values.max()),
            "target_variance": float(target_values.var()),
            **observed,
        },
    )
    return links


def native_api(cache):
    import arboreto.algo
    import arboreto.core
    import cloudpickle
    import joblib
    from joblib.externals import cloudpickle as jcloudpickle
    from tqdm import tqdm

    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    jcloudpickle.register_pickle_by_value(sys.modules[__name__])
    namespace = {
        "np": np,
        "pd": pd,
        "logging": logging,
        "joblib": joblib,
        "pathlib": pathlib,
        "os": os,
        "tqdm": tqdm,
        "log": logging.getLogger("SCENICplus-TF2G"),
        "_prepare_input": arboreto.algo._prepare_input,
        "infer_partial_network": functools.partial(monitored_infer, cache),
    }
    for name in (
        "EARLY_STOP_WINDOW_LENGTH",
        "RF_KWARGS",
        "SGBM_KWARGS",
        "to_tf_matrix",
    ):
        namespace[name] = getattr(arboreto.core, name)
    names = {
        "COLUMN_NAME_TARGET",
        "COLUMN_NAME_WEIGHT",
        "COLUMN_NAME_REGULATION",
        "COLUMN_NAME_CORRELATION",
        "COLUMN_NAME_TF",
        "COLUMN_NAME_SCORE_1",
        "COLUMN_NAME_SCORE_2",
        "RHO_THRESHOLD",
        "_add_correlation",
        "_inject_TF_as_its_own_target",
        "calculate_TFs_to_genes_relationships",
    }
    path = HERE / "upstream/TF_to_gene.py"
    source_file(path, SHA)
    nodes, found = [], set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            assigned = {n.id for n in node.targets if isinstance(n, ast.Name)} & names
            if assigned:
                nodes.append(node)
                found.update(assigned)
    if found != names:
        raise ValueError(f"Missing official definitions: {names - found}")
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102 - pinned source SHA-256 checked above
    return namespace["calculate_TFs_to_genes_relationships"]


def fit(data, settings, run, seed, cpus, device, smoke):
    import joblib

    matrices, status_by_time = [], {}
    for time, counts in zip(data.times, data.counts):
        folder = run / f"time_{time}"
        folder.mkdir(exist_ok=True)
        cache = folder / "target_importances"
        cache.mkdir(exist_ok=True)
        expression = log_expression(counts, settings["target_sum"])
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            adjacency = native_api(cache)(
                expression,
                list(data.tfs),
                folder / "joblib_tmp",
                method=settings["method"],
                n_cpu=cpus,
                seed=seed,
            )
        adjacency.to_csv(folder / "official_adjacencies.tsv.gz", sep="\t", index=False)
        rows, audit = [], []
        for gene in data.targets:
            record = read_json(cache / f"target_{gene}.json")
            if record["target"] != gene or set(record["weights"]) != set(data.tfs):
                raise ValueError(f"Incomplete native regression for {gene}")
            rows.append([record["weights"][tf] for tf in data.tfs])
            audit.append(
                {key: value for key, value in record.items() if key != "weights"}
            )
        pd.DataFrame(audit).to_csv(folder / "target_audit.tsv", sep="\t", index=False)
        counts_by_status = pd.Series(
            [record["status"] for record in audit]
        ).value_counts()
        status_by_time[str(time)] = {
            str(status): int(count) for status, count in counts_by_status.items()
        }
        matrices.append(np.asarray(rows))
    return np.stack(matrices), {
        "label": "SCENIC+ TF2G (official RNA component)",
        "score": "native GBM/RF feature importance, including observed zero importance; no rescaling",
        "prior": "official TF2G does not accept an edge prior; fixed union is used only for within-union evaluation",
        "scope": "independent per-time RNA inference; not the complete TF-RE-TG/eRegulon pipeline",
        "normalization": f"per-cell library normalization to {settings['target_sum']} then log1p",
        "official_fallback": "after Arboreto exhausts its ten native retries it returns an empty sparse adjacency table; absent TF-target links receive zero in the dense benchmark tensor and remain explicitly distinguished from successful zero feature importance",
        "target_status_counts": status_by_time,
    }
