"""Official Linux scMTNI binary; no Python surrogate optimizer."""

from __future__ import annotations

import os
import re
import subprocess
from itertools import pairwise
from pathlib import Path

import numpy as np
from common import digest, log_expression, write_json

PIN = "c4d4850629f3529cc7a8b4dae54d91daf0702809"
HERE = Path(__file__).resolve().parent


def software(directory):
    root = directory / "upstream"
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PIN:
        raise ValueError("scMTNI checkout differs from pinned official commit")
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "Code",
            ":(top,literal,exclude)Code/scMTNI",
        ],
        check=True,
    )
    binary = root / "Code/scMTNI"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError("Build official scMTNI on Linux with setup.sh first")
    binary_sha = digest(binary)
    receipt = directory / "runs/build/verified_binary.sha256"
    if (
        not receipt.is_file()
        or receipt.read_text(encoding="utf-8").strip() != binary_sha
    ):
        raise RuntimeError(
            "scMTNI binary has no matching successful-build receipt. "
            "Rerun bash comparisons/simulation_benchmark/setup.sh scmtni."
        )
    return {
        "repository": "Roy-lab/scMTNI",
        "commit": PIN,
        "binary_sha256": binary_sha,
    }


def lines(path, values):
    path.write_text("".join(f"{v}\n" for v in values), encoding="utf-8")


def prepare(data, settings, folder, cell_indices):
    """Write native six-column configs, suffixed genes, OG maps and four-column tree."""
    folder.mkdir(parents=True, exist_ok=True)
    # Upstream uses shell mkdir and fixed C buffers for paths; reject unsupported
    # paths instead of allowing silent partial output or shell interpretation.
    if not re.fullmatch(r"[A-Za-z0-9_./:\\-]+", str(folder)) or len(str(folder)) > 700:
        raise ValueError(
            "Official scMTNI requires a short path without spaces or shell metacharacters"
        )
    labels = [f"time{time}" for time in data.times]
    lines(folder / "order.txt", labels)
    lines(folder / "regulator_ogs.txt", range(1, len(data.tfs) + 1))
    lines(folder / "target_ogs.txt", range(len(data.tfs) + 1, len(data.genes) + 1))
    with (folder / "orthogroups.txt").open("w", encoding="utf-8") as stream:
        stream.write("Gene_OGID\tNAME\n")
        for i, gene in enumerate(data.genes, 1):
            stream.write(
                f"OG{i}_1\t" + ",".join(f"{gene}_{label}" for label in labels) + "\n"
            )
    tree = [
        f"{child}\t{parent}\t{settings['gain']}\t{settings['loss']}"
        for parent, child in pairwise(labels)
    ]
    lines(folder / "tree.txt", tree)
    configs = []
    selected = {}
    for label, counts, indices in zip(labels, data.counts, cell_indices):
        expr = log_expression(counts.iloc[:, indices], settings["target_sum"]).T
        expr.index = [f"{gene}_{label}" for gene in data.genes]
        expr.index.name = "Gene"
        expr.to_csv(folder / f"{label}.table", sep="\t")
        selected[label] = counts.columns[indices].tolist()
        lines(folder / f"{label}_regulators.txt", [f"{tf}_{label}" for tf in data.tfs])
        lines(folder / f"{label}_targets.txt", [f"{tg}_{label}" for tg in data.targets])
        with (folder / f"{label}_prior.txt").open("w", encoding="utf-8") as stream:
            for gi, ti in np.argwhere(data.prior):
                stream.write(f"{data.tfs[ti]}_{label}\t{data.targets[gi]}_{label}\t1\n")
        output = folder / label
        output.mkdir(exist_ok=True)
        configs.append(
            "\t".join(
                map(
                    str,
                    [
                        label,
                        folder / f"{label}.table",
                        output,
                        folder / f"{label}_regulators.txt",
                        folder / f"{label}_targets.txt",
                        folder / f"{label}_prior.txt",
                    ],
                )
            )
        )
    lines(folder / "config.txt", configs)
    write_json(folder / "sampled_cells.json", selected)


def read_network(data, folder, maximum):
    result = []
    for time in data.times:
        label = f"time{time}"
        output = folder / label / "fold0"
        path = output / f"var_mb_pw_k{maximum}.txt"
        # A present empty edge file may be a valid sparse network, but not a failed process.
        params = output / "modelparams.txt"
        if not path.is_file() or not params.is_file() or params.stat().st_size == 0:
            raise RuntimeError(f"Official scMTNI output is incomplete: {output}")
        matrix = np.zeros((len(data.targets), len(data.tfs)))
        tids = {f"{v}_{label}": i for i, v in enumerate(data.tfs)}
        gids = {f"{v}_{label}": i for i, v in enumerate(data.targets)}
        seen = set()
        for line in path.read_text().splitlines():
            tf, gene, weight = line.split("\t")
            if tf not in tids or gene not in gids or (tf, gene) in seen:
                raise ValueError(f"Unexpected/duplicate native edge: {line}")
            seen.add((tf, gene))
            matrix[gids[gene], tids[tf]] = float(weight)
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite scMTNI coefficients")
        result.append(matrix)
    return np.stack(result)


def fit(data, settings, run, seed, cpus, device, smoke):
    n = 1 if smoke else int(settings["subsamples"])
    fraction = float(settings["subsample_fraction"])
    if n < 1 or not 0 < fraction <= 1:
        raise ValueError("Invalid stability-selection settings")
    maximum = min(int(settings["max_regulators"]), len(data.tfs))
    counts, means = [], []
    for sample in range(n):
        folder = run / f"subsample_{sample:03d}"
        rng = np.random.default_rng(np.random.SeedSequence([seed, sample]))
        indices = [
            np.sort(
                rng.choice(
                    c.shape[1], max(4, int(c.shape[1] * fraction)), replace=False
                )
            )
            for c in data.counts
        ]
        prepare(data, settings, folder, indices)
        command = [
            str(HERE / "upstream/Code/scMTNI"),
            "-f",
            str(folder / "config.txt"),
            "-x",
            str(maximum),
            "-l",
            str(folder / "regulator_ogs.txt"),
            "-n",
            str(folder / "target_ogs.txt"),
            "-d",
            str(folder / "tree.txt"),
            "-m",
            str(folder / "orthogroups.txt"),
            "-s",
            str(folder / "order.txt"),
            "-p",
            str(settings["root_probability"]),
            "-c",
            "yes",
            "-b",
            str(settings["beta0"]),
            "-q",
            str(settings["beta1"]),
        ]
        write_json(folder / "command.json", command)
        print(f"Official scMTNI stability subsample {sample + 1}/{n}", flush=True)
        with (folder / "native.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=folder
            )
        signed = read_network(data, folder, maximum)
        np.save(folder / "signed_coefficients.npy", signed)
        counts.append((signed != 0).astype(float))
        means.append(signed)
    np.save(run / "mean_signed_coefficients.npy", np.mean(means, axis=0))
    # Native unthresholded consensus = frequency of an edge across subsamples.
    score = np.mean(counts, axis=0)
    return score, {
        "label": "scMTNI official + stability selection",
        "subsamples": n,
        "subsample_fraction": fraction,
        "score": "unthresholded edge-selection frequency (0..1)",
        "prior": "fixed union passed as motif score 1 at every time; native soft prior",
        "lineage": "ordered time chain; gain/loss are prespecified constants, not measured GRN turnover",
        "normalization": f"per-cell library size {settings['target_sum']}, log1p; -c yes selects requested target orthogroups",
        "model_seed": "controls adapter cell subsampling; official optimizer is invoked unchanged",
    }
