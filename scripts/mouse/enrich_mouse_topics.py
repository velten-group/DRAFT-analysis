"""GO over-representation of topic genes against the network's target genes (hypergeometric test, BH-adjusted)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


def bh_adjust(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return values.copy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid P-values.")
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum.accumulate(
        (values[order] * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result


def parse_library(path):
    library = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        columns = line.split("\t")
        if len(columns) >= 3:
            library[columns[0]] = {
                name.split(",", 1)[0].strip().upper()
                for name in columns[2:]
                if name.strip()
            }
    if len(library) < 100:
        raise ValueError("Gene-set library is empty or malformed.")
    return library


def background_ora(query, background, library):
    background = {name.upper() for name in background}
    query = {name.upper() for name in query}
    if not query or not query <= background:
        raise ValueError(
            "Query must be a nonempty subset of the network target background."
        )
    rows = []
    for term, members in library.items():
        eligible = members & background
        if not eligible:
            continue
        overlap = eligible & query
        p = (
            hypergeom.sf(len(overlap) - 1, len(background), len(eligible), len(query))
            if overlap
            else 1.0
        )
        rows.append(
            {
                "Term": term,
                "P-value": float(p),
                "overlap": len(overlap),
                "term_background_size": len(eligible),
                "query_size": len(query),
                "background_size": len(background),
                "Genes": ";".join(sorted(overlap)),
            }
        )
    result = pd.DataFrame(rows)
    result["Adjusted P-value"] = bh_adjust(result["P-value"])
    result["n_tested_terms"] = len(result)
    # terms without overlap count in the BH correction and are dropped afterwards
    return (
        result[result.overlap > 0]
        .sort_values(["Adjusted P-value", "P-value", "Term"])
        .reset_index(drop=True)
    )
