"""Lineage tree helpers."""

from __future__ import annotations

import re
from typing import Any


def tail_num(value: str) -> int:
    match = re.search(r"(\d+)$", value)
    return int(match.group(1)) if match else 1 << 30


def parse_stage_branch(name: str) -> tuple[int, str]:
    text = str(name)
    match_num = re.search(r"(\d+)", text)
    stage = int(match_num.group(1)) if match_num else 10**9
    match_branch = re.search(r"(?:[_\-])([A-Za-z]+)$", text)
    branch = match_branch.group(1).lower() if match_branch else ""
    return stage, branch


def select_lineage_with_branching(
    graphs_dict: dict[str, Any],
    prefix_keys: tuple[str, ...] = ("somatic", "soma"),
) -> tuple[list[Any], list[int], list[list[int]], list[str]]:
    items = [
        (cell_type, graphs_dict[cell_type])
        for cell_type in graphs_dict
        if any(cell_type.lower().startswith(prefix) for prefix in prefix_keys)
        or any(prefix in cell_type.lower() for prefix in prefix_keys)
    ]
    if not items:
        return [], [], [], []
    items.sort(key=lambda pair: parse_stage_branch(pair[0]))
    names = [cell_type for cell_type, _ in items]
    graph_list = [graph for _, graph in items]
    stages = [parse_stage_branch(name)[0] for name in names]
    idx_by_stage: dict[int, list[int]] = {}
    for idx, stage in enumerate(stages):
        idx_by_stage.setdefault(stage, []).append(idx)
    topo = list(range(len(graph_list)))
    parents = [[] for _ in graph_list]
    sorted_stages = sorted(idx_by_stage.keys())
    for stage_idx, stage in enumerate(sorted_stages):
        if stage_idx == 0:
            continue
        prev_indices = idx_by_stage[sorted_stages[stage_idx - 1]]
        for idx in idx_by_stage[stage]:
            parents[idx] = list(prev_indices)
    return graph_list, topo, parents, names
