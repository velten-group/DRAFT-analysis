"""TF statistics of the mouse edge clusters and the topic helpers, computed by the cells of the original notebooks."""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from notebook_cells import ENRICH_SHA, cell_source
from notebook_runner import EDGE_HASHES


def source_statistics(source_root, run):
    edge_path = source_root / "edge_cluster.ipynb"
    source = cell_source(edge_path, 3, EDGE_HASHES[3])
    tree = ast.parse(source)
    stop = next(
        i
        for i, n in enumerate(tree.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "TOPK_PER_EC" for t in n.targets)
    )
    # the cell's statistics, without its plotting code
    tree.body = tree.body[:stop]
    namespace = {
        "pd": pd,
        "df_edges": pd.read_csv(run / "edge/edges_selected_from_network_all.csv"),
        "df_edge_clusters": pd.read_csv(
            run / "edge/edge_clusters_12k_from_network_all.csv"
        ),
    }
    exec(compile(tree, str(edge_path), "exec"), namespace)  # noqa: S102
    topic_path = source_root / "topic.ipynb"
    source = cell_source(topic_path, 1, ENRICH_SHA)
    tree = ast.parse(source)
    tree.body = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_topic_cluster_suffixes_from_H", "prep_enrichment_bar_df"}
    ]
    helpers = {"np": np, "pd": pd}
    exec(compile(tree, str(topic_path), "exec"), helpers)  # noqa: S102
    return namespace, helpers
