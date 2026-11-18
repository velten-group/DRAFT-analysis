"""TF-TG scores of DRAFT (from the checkpoint) and of a baseline consensus network, on given regulators and targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import CLUSTERS, digest
from official_aupr_protocol import canonical_edges, clean_symbol

CHECKPOINT_SHA = "0a7599ba08e8d2fa7572212d1927dca81b508fad42a2aa81f91b3a23da6da725"


def load_checkpoint(path, tfs, tgs):
    if digest(path) != CHECKPOINT_SHA:
        raise ValueError("grn_best.pt differs from the deposited checkpoint (SHA-256).")
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    meta = checkpoint["meta"]
    if (meta["topo"] != CLUSTERS
            or [clean_symbol(x) for x in meta["TF_list"]] != tfs
            or [clean_symbol(x) for x in meta["TG_list"]] != tgs):
        raise ValueError("Checkpoint TF/TG lists or cluster order differ from the evaluation features.")
    return checkpoint


def checkpoint_prediction(matrix, tfs, tgs, regulators, targets):
    """Checkpoint scores of every regulator x target pair, without prior mask or top-k cut."""
    values = np.asarray(matrix)
    if values.shape != (len(tfs), len(tgs)) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid full checkpoint score matrix.")
    ii = [i for i, gene in enumerate(tfs) if gene in regulators]
    jj = [j for j, gene in enumerate(tgs) if gene in targets]
    return pd.DataFrame({"TF": np.repeat(np.asarray(tfs)[ii], len(jj)),
                         "TG": np.tile(np.asarray(tgs)[jj], len(ii)),
                         "score": values[np.ix_(ii, jj)].ravel()})


def consensus_prediction(path, tfs, tgs, regulators, targets):
    parts = []
    for part in pd.read_csv(path, sep="\t", header=None, names=["TF", "TG", "score"], chunksize=200000):
        mapped, _ = canonical_edges(part, tfs, tgs)
        parts.append(mapped.loc[mapped.TF.isin(regulators) & mapped.TG.isin(targets)])
    if not parts:
        return pd.DataFrame(columns=["TF", "TG", "score"])
    return pd.concat(parts, ignore_index=True).groupby(["TF", "TG"], sort=False, as_index=False).score.max()
