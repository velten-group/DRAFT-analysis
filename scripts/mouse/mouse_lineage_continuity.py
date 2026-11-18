"""Temporal smoothness of the mouse cluster networks along the lineage tree (Fig. 3c)."""

from itertools import pairwise

import numpy as np

ORDER = ["C4", "C8", "C7", "C3", "C2", "C1", "C5"]
BRANCHES = {
    "Successful": ["C4", "C8", "C1", "C5"],
    "Unsuccessful": ["C4", "C8", "C7", "C3", "C2"],
}
LINEAGE_EDGES = list(
    dict.fromkeys(
        edge for path in BRANCHES.values() for edge in pairwise(path)
    )
)


def scope_continuity(raw, nodes, edges):
    """1 - mean distance over the lineage edges / largest distance between the nodes (1 for identical networks);
    all_pairs_smoothness uses the mean over all node pairs instead."""
    distances = raw.loc[nodes, nodes].to_numpy(dtype=float)
    if len(nodes) < 2 or not edges or len(edges) != len(set(edges)):
        raise ValueError("A scope needs at least two nodes and unique adjacent edges")
    if not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("Distances must be finite and nonnegative")
    np.testing.assert_allclose(distances, distances.T, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(np.diag(distances), 0)
    if any(a not in nodes or b not in nodes or a == b for a, b in edges):
        raise ValueError("Adjacent edges must connect different nodes in the scope")
    diameter = float(distances.max())
    mean_adjacent = float(np.mean([raw.loc[a, b] for a, b in edges]))
    upper = distances[np.triu_indices_from(distances, 1)]
    return {
        "mean_continuity": float(1 - mean_adjacent / diameter) if diameter else 1.0,
        "all_pairs_smoothness": float(1 - upper.mean() / diameter)
        if diameter
        else None,
        "mean_adjacent_distance": mean_adjacent,
        "diameter": diameter,
        "n_nodes": len(nodes),
        "n_adjacent_edges": len(edges),
        "adjacent_edges": [list(edge) for edge in edges],
        "constant_networks": diameter == 0,
    }


def lineage_summary(raw):
    result = {"overall": scope_continuity(raw, ORDER, LINEAGE_EDGES)}
    for name, path in BRANCHES.items():
        result[name] = scope_continuity(raw, path, list(pairwise(path)))
    return result
