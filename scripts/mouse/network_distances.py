"""Full-network sparse Frobenius distances, without candidate/gold filtering."""

from functools import cache

import numpy as np
import pandas as pd
from common import gene_index, gene_key
from scipy import sparse


@cache
def normalized_name(name):
    return gene_key(name)


def consensus_vector(path, tfs, tgs, chunksize=200000):
    """Align native consensus scores to full axes; missing=0, duplicates=max."""
    tf_idx, tg_idx = gene_index(tfs), gene_index(tgs)
    pieces, n_rows, n_mapped = [], 0, 0
    for frame in pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["TF", "TG", "score"],
        dtype={"TF": str, "TG": str, "score": float},
        chunksize=chunksize,
    ):
        scores = frame.score.to_numpy()
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError("Consensus confidences must be finite and in [0, 1]")
        i = frame.TF.map(normalized_name).map(tf_idx)
        j = frame.TG.map(normalized_name).map(tg_idx)
        valid = i.notna() & j.notna()
        codes = i[valid].to_numpy(dtype=np.int64) * len(tgs) + j[valid].to_numpy(
            dtype=np.int64
        )
        part = pd.Series(scores[valid.to_numpy()], index=codes)
        pieces.append(part.groupby(level=0).max())
        n_rows += len(frame)
        n_mapped += int(valid.sum())
    merged = (
        pd.concat(pieces).groupby(level=0).max() if pieces else pd.Series(dtype=float)
    )
    merged = merged.loc[merged > 0]
    vector = sparse.csr_array(
        (
            merged.to_numpy(),
            (np.zeros(len(merged), dtype=int), merged.index.to_numpy(dtype=np.int64)),
        ),
        shape=(1, len(tfs) * len(tgs)),
    )
    return vector, {
        "native_rows": n_rows,
        "mapped_rows": n_mapped,
        "unique_nonzero_edges": vector.nnz,
    }


def frobenius_distances(vectors):
    """Unsquared Frobenius distance: Euclidean norm of aligned matrix entries."""
    if not vectors or any(v.shape != vectors[0].shape for v in vectors):
        raise ValueError("Nonempty vectors must share the same complete TF/TG axes")
    result = np.zeros((len(vectors), len(vectors)), dtype=float)
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            difference = (vectors[i] - vectors[j]).data
            result[i, j] = result[j, i] = np.sqrt(difference @ difference)
    return result
