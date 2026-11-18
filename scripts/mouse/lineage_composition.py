"""Cell counts per sample day in each mouse cell cluster, and the cluster lineage (Fig. 3a)."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

STAGES = ("MEF", "FBS-Day3", "FBS-Day6", "FBS-Day9", "FBS-Day12", "mESC")


def sample_stage(cell: str) -> str:
    if cell.startswith("MEF-Day0-"):
        return "MEF"
    if cell.startswith("FBS-mESCs-"):
        return "mESC"
    match = re.match(r"(FBS-Day(?:3|6|9|12))-", cell)
    if match:
        return match[1]
    raise ValueError(f"Unrecognized sample label: {cell}")


def read_inputs(data_dir: Path):
    order_path = data_dir / "celltype_order.txt"
    tree_path = data_dir / "celltype_tree_ancestor.txt"
    order = [
        line.strip() for line in order_path.read_text().splitlines() if line.strip()
    ]
    parent = {}
    for line in tree_path.read_text().splitlines():
        if line.strip():
            child, ancestor, *_ = line.split("\t")
            if child in parent:
                raise ValueError(f"Multiple parent records: {child}")
            parent[child] = ancestor
    if len(order) != len(set(order)) or set(order) != {"cluster4", *parent}:
        raise ValueError("The cluster order and lineage do not agree.")

    def ancestors(node):
        chain = [node]
        while node in parent:
            node = parent[node]
            if node in chain:
                raise ValueError("Cycle in lineage.")
            chain.append(node)
        return chain

    success_path = set(ancestors("cluster5"))
    unsuccessful_path = set(ancestors("cluster2"))
    shared = success_path & unsuccessful_path
    rows, sources, seen = [], [order_path, tree_path], set()
    for cluster in order:
        path = data_dir / f"{cluster}.table"
        with path.open(encoding="utf-8") as handle:
            cells = next(csv.reader(handle, delimiter="\t"))[1:]
        if not cells or len(cells) != len(set(cells)) or seen.intersection(cells):
            raise ValueError(f"Missing or duplicate cells: {cluster}")
        seen.update(cells)
        counts = Counter(sample_stage(cell) for cell in cells)
        if sum(counts.values()) != len(cells):
            raise ValueError(f"Sample counts do not sum to the cell total: {cluster}")
        branch = (
            "Shared"
            if cluster in shared
            else ("Successful" if cluster in success_path else "Unsuccessful")
        )
        if cluster not in success_path | unsuccessful_path:
            raise ValueError(f"Unexpected lineage node: {cluster}")
        rows.append(
            {
                "cluster": cluster.replace("cluster", "C"),
                "n_cells": len(cells),
                "branch": branch,
                **{stage: counts[stage] for stage in STAGES},
            }
        )
        sources.append(path)
    return rows, order, parent, sources
