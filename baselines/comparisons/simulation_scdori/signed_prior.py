"""Observed-data signed binding from the pinned official correlation routine."""

from pathlib import Path

import numpy as np
import pandas as pd
from common import load_module, read_json, source_file, write_json

HERE = Path(__file__).resolve().parent


def signed_binding(data, rna_train, atac_train, settings, run):
    """Fit on training cells only; simulator truth and time labels are not inputs."""
    if settings != {
        "binding": "observed_correlation",
        "percentile": 0.0,
        "n_bg": 500,
        "fit_cells": "training_prefix_only",
    }:
        raise ValueError(
            "Use the declared signed-binding protocol; no implicit parameter search"
        )
    spec = read_json(HERE / "preprocessing_api.json")
    path = HERE / "upstream" / spec["path"]
    source_file(path, spec["sha256_lf"])
    native = load_module("_simulation_scdori_signed_correlation", path)
    rna = np.asarray(rna_train, dtype=np.float32)
    atac = np.asarray(atac_train, dtype=np.float32)
    motif = np.asarray(data.tf_re, dtype=np.float32)
    if (
        rna.shape != (len(atac), len(data.tfs))
        or atac.shape[1] != len(data.regions)
        or not np.isfinite(rna).all()
        or not np.isfinite(atac).all()
    ):
        raise ValueError("Signed-binding cell/feature axes or observed values differ")
    activator, repressor = native.compute_in_silico_chipseq(
        atac,
        rna,
        pd.DataFrame(motif, index=data.regions, columns=data.tfs),
        percentile=settings["percentile"],
        n_bg=settings["n_bg"],
    )
    # The native MinMaxScaler([-1,0]) maps an entirely zero column to -1.
    # With percentile0, a repressor column has evidence iff a supported pair
    # has negative observed correlation. Preserve zero evidence as zero.
    x = (atac - atac.mean(0)) / (atac.std(0) + 1e-8)
    y = (rna - rna.mean(0)) / (rna.std(0) + 1e-8)
    corr = np.nan_to_num(x.T @ y / len(atac))
    no_negative = ~np.any((corr < 0) & (motif > 0), axis=0)
    repressor[:, no_negative] = 0
    # Exact zeros can acquire tiny signed offsets during MinMax scaling. Keep
    # each branch only where its original correlation has the intended sign.
    activator = np.where((motif > 0) & (corr > 0), np.clip(activator, 0, 1), 0).astype(
        np.float32
    )
    repressor = np.where((motif > 0) & (corr < 0), np.clip(repressor, -1, 0), 0).astype(
        np.float32
    )
    if (
        np.any(activator < 0)
        or np.any(repressor > 0)
        or not np.isfinite(activator).all()
        or not np.isfinite(repressor).all()
        or np.any((activator > 0) & (repressor < 0))
    ):
        raise ValueError("Invalid signed evidence after native preprocessing")
    np.savez_compressed(
        run / "signed_binding.npz",
        activator=activator,
        repressor=repressor,
        correlation=corr,
    )
    write_json(
        run / "binding_preprocessing.json",
        {
            "protocol": settings,
            "source": spec,
            "training_cells": len(rna),
            "public_tf_re_links": int(motif.sum()),
            "positive_links": int((activator > 0).sum()),
            "negative_links": int((repressor < 0).sum()),
            "no_evidence_columns_restored_to_zero": np.flatnonzero(
                no_negative
            ).tolist(),
            "scale_guard": "Restore zero evidence/branch sign after MinMax numerical offsets and enforce public support; native source unchanged",
            "scope": "Single-cell observed correlations, no metacell/genomic motif reconstruction; percentile0 avoids an additional hard edge filter on the small simulated panel",
        },
    )
    return activator, repressor
