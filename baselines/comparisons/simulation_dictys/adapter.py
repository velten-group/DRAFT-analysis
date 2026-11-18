"""Official Dictys context-specific OU inference and normalization."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from common import source_file, write_json

PIN = "f409a632eec99c2cbc71f3c69b779964649bfc8b"
HASHES = {
    "network.py": "18a03d4120e5d89c5981795b6c465cdb2b904c85e236611744b0815f928b2439",
    "preproc.py": "865faa6407fdd18b404b9fc700c786a0eb1c0123d9a212d1b399fe898800da54",
    "dynamic.py": "171b725118fde150ec50b4904ef83f17c918c98836c6f035e42d87b9611ba795",
    "traj.py": "de692398abf1cb58ba3a6e6501633467c4bdabf675f9e7db6025c3cf3c721c03",
    "net/stat.py": "0ede3331b276caed5137b6597d5be296c6732b9334ddc44cccd7892214c2217d",
}


def software(directory):
    os.environ.setdefault("MPLCONFIGDIR", str(directory / "runs/.mplconfig"))
    source = directory / "upstream/src"
    if (directory / "upstream/.git").exists():
        checkout = directory / "upstream"
        head = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != PIN:
            raise ValueError("Dictys checkout differs from pinned commit")
        subprocess.run(
            ["git", "-C", str(checkout), "diff", "--exit-code", "HEAD", "--", "src"],
            check=True,
        )
    for name, sha in HASHES.items():
        source_file(source / "dictys" / name, sha)
    sys.path.insert(0, str(source))
    import dictys.network
    import pyro

    root = Path(dictys.network.__file__).parent
    for name, sha in HASHES.items():
        source_file(root / name, sha)
    if pyro.__version__ != "1.9.1":
        raise RuntimeError("Use the isolated environment with pyro-ppl==1.9.1")
    return {
        "repository": "pinellolab/dictys",
        "commit": PIN,
        "api_sha256_lf": HASHES,
    }


def fit(data, settings, run, seed, cpus, device, smoke):
    import torch
    from dictys.network import normalize, reconstruct
    from dictys.preproc import qc_reads

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    matrices, ranks, audits = [], {}, {}
    # Native pandas readers infer an all-numeric index as int but header IDs as
    # str. Reversible nonnumeric aliases keep gene identity equal on both axes.
    aliases = {gene: f"gene_{gene}" for gene in data.genes}
    write_json(run / "native_gene_aliases.json", aliases)
    mask = pd.DataFrame(0, index=data.tfs, columns=data.genes, dtype=np.int8)
    mask.loc[:, list(data.targets)] = np.asarray(data.prior.T, dtype=np.int8)
    mask = mask.rename(index=aliases, columns=aliases)
    steps = settings["nstep"] if not smoke else 3
    qc = settings["smoke_qc"] if smoke else settings["qc"]
    for time, counts in zip(data.times, data.counts):
        folder = run / f"time_{time}"
        folder.mkdir(exist_ok=True)
        counts.rename(index=aliases).to_csv(folder / "expression0.tsv.gz", sep="\t")
        qc_reads(
            str(folder / "expression0.tsv.gz"),
            str(folder / "expression.tsv.gz"),
            *qc,
        )
        expression = pd.read_csv(folder / "expression.tsv.gz", sep="\t", index_col=0)
        expression.index, expression.columns = (
            expression.index.astype(str),
            expression.columns.astype(str),
        )
        regulators = [aliases[tf] for tf in data.tfs if aliases[tf] in expression.index]
        local_mask = mask.loc[regulators, expression.index]
        local_mask = local_mask.loc[local_mask.sum(axis=1) > 0]
        if local_mask.empty:
            raise RuntimeError(
                f"T{time}: no regulator with an allowed target remains after official Dictys QC"
            )
        rank = min(len(local_mask), len(expression) - 1)
        if expression.shape[1] <= rank:
            raise ValueError(
                f"T{time}: Dictys needs >{rank} post-QC cells for native covariance initialization"
            )
        local_mask.to_csv(folder / "binlinking.tsv.gz", sep="\t")
        kept = set(expression.index)
        audit = {
            "time": time,
            "parameters": qc,
            "genes_before": len(counts),
            "genes_after": len(expression),
            "cells_before": counts.shape[1],
            "cells_after": expression.shape[1],
            "removed_genes": [gene for gene in data.genes if aliases[gene] not in kept],
            "removed_cells": [
                cell for cell in counts.columns if cell not in expression.columns
            ],
            "removed_tfs_by_expression_qc": [
                tf for tf in data.tfs if aliases[tf] not in kept
            ],
            "regulators_without_retained_allowed_target": [
                tf
                for tf in data.tfs
                if aliases[tf] in kept and aliases[tf] not in local_mask.index
            ],
            "active_regulators": len(local_mask),
        }
        write_json(folder / "qc.json", audit)
        reconstruct(
            str(folder / "expression.tsv.gz"),
            str(folder / "binlinking.tsv.gz"),
            *[
                str(folder / f"net_{name}.tsv.gz")
                for name in ("weight", "meanvar", "covfactor", "loss", "stats")
            ],
            lr=settings["lr"],
            lrd=settings["lrd"],
            nstep=steps,
            npc=0,
            model="ou",
            nstep_report=1 if smoke else 100,
            rseed=seed,
            device=device,
            dtype="float",
            nth=cpus,
        )
        normalize(
            *[
                str(folder / f"net_{name}.tsv.gz")
                for name in ("weight", "meanvar", "covfactor", "nweight")
            ],
            norm=3,
            nth=cpus,
        )
        weights = pd.read_csv(folder / "net_nweight.tsv.gz", sep="\t", index_col=0)
        weights.index, weights.columns = (
            weights.index.astype(str),
            weights.columns.astype(str),
        )
        # Genes filtered by native QC are unestimated and receive zero score in
        # the fixed full benchmark universe, as in the real-data Dictys export.
        result = np.zeros((len(data.targets), len(data.tfs)))
        for gi, gene in enumerate(data.targets):
            for ti, tf in enumerate(data.tfs):
                if (
                    data.prior[gi, ti]
                    and aliases[tf] in weights.index
                    and aliases[gene] in weights.columns
                ):
                    result[gi, ti] = weights.loc[aliases[tf], aliases[gene]]
        if not np.isfinite(result).all():
            raise ValueError("Official Dictys returned nonfinite edge weights")
        pd.DataFrame(result, index=data.targets, columns=data.tfs).to_csv(
            folder / "signed_edges.tsv", sep="\t"
        )
        matrices.append(abs(result))
        ranks[str(time)] = rank
        audits[str(time)] = audit
    return np.stack(matrices), {
        "label": "Dictys official OU (shared TF-TG prior)",
        "score": "absolute official norm=3 normalized direct edge coefficient; signed coefficients retained",
        "prior": "same fixed union binary mask at every time, projected to genes retained by official per-time RNA QC; replaces footprint/motif prior construction",
        "scope": "official context-specific network per time point; no dynamic windows or interpolation",
        "steps": steps,
        "official_qc": qc,
        "qc_audit": audits,
        "missing_score_policy": "edges incident to a gene removed by official per-time RNA QC receive zero in the fixed evaluation universe",
        "native_expression_covariance_rank": ranks,
    }
