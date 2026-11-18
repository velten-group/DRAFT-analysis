"""Official pySCENIC GRNBoost2 + cisTarget on independent mouse cluster inputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mouse_benchmark"))
from common import (
    digest,
    mark_complete,
    model_run,
    read_json,
    resolve_prepared,
    verify_complete,
    write_json,
)


def official_command(arguments: list[str], log: Path) -> None:
    env = dict(os.environ)
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        env[key] = "1"  # eight worker processes, no eight-by-eight oversubscription
    print(f"Official pySCENIC command; log: {log}", flush=True)
    command = [sys.executable, "-u", "-m", *arguments]
    write_json(log.with_suffix(".command.json"), {"argv": command})
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        tail = "\n".join(
            log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        )
        raise RuntimeError(
            f"Official SCENIC exited with code {result.returncode}; log {log}\n{tail}"
        )


def regulon_edges(regulons) -> pd.DataFrame:
    rows = []
    for reg in regulons:
        for target, weight in reg.gene2weight.items():
            if target != reg.transcription_factor:
                rows.append(
                    {
                        "TF": reg.transcription_factor,
                        "TG": target,
                        "score": float(weight),
                        "regulon": reg.name,
                    }
                )
    result = pd.DataFrame(rows, columns=["TF", "TG", "score", "regulon"])
    if result.empty:
        return result
    # Same TF/TG may survive multiple motifs/modules. Never sum motif multiplicity.
    return result.sort_values("score", ascending=False).drop_duplicates(["TF", "TG"])


def train(
    prepared: Path,
    databases: list[Path],
    annotations: Path,
    cpus: int,
    clusters: list[int] | None = None,
    aucell: bool = False,
) -> Path:
    if importlib.metadata.version("pyscenic") != "0.12.1":
        raise RuntimeError(
            "Use the isolated mouse_scenic requirements (pySCENIC 0.12.1)."
        )
    from ctxcore.rnkdb import FeatherRankingDatabase
    from pyscenic.cli.utils import load_signatures

    manifest = read_json(prepared / "manifest.json")
    databases = [p.resolve() for p in databases]
    annotations = annotations.resolve()
    if not databases or not annotations.is_file():
        raise FileNotFoundError(
            "SCENIC needs mouse gene-ranking feather databases and motif-to-TF annotations."
        )
    db_objects = [FeatherRankingDatabase(str(p), p.stem) for p in databases]
    db_gene_sets = [set(db.genes) for db in db_objects]
    annotations_frame = pd.read_csv(
        annotations, sep="\t", usecols=["gene_name", "#motif_id"]
    )
    motif_overlap = [
        len(
            set(annotations_frame["#motif_id"])
            & set(db.ct_db.all_motif_or_track_ids.ids)
        )
        for db in db_objects
    ]
    if not all(motif_overlap):
        raise ValueError(
            "A cisTarget database has no motif-ID overlap with annotations; match mc9nr/v9 or mc_v10/v10 resources."
        )
    software = {
        name: importlib.metadata.version(name)
        for name in [
            "pyscenic",
            "arboreto",
            "ctxcore",
            "numpy",
            "pandas",
            "scikit-learn",
        ]
    }
    import pyscenic.cli.arboreto_with_multiprocessing as mp_cli
    import pyscenic.cli.pyscenic as scenic_cli

    software["api_sha256"] = {
        "grn_cli": digest(Path(mp_cli.__file__)),
        "scenic_cli": digest(Path(scenic_cli.__file__)),
    }
    configuration = {
        "software": software,
        "cpus": cpus,
        "seed": manifest["seed"],
        "aucell": aucell,
        "database_sha256": {str(p): digest(p) for p in databases},
        "annotations_sha256": digest(annotations),
        "method": "official multiprocessing GRNBoost2 followed by official cisTarget pruning",
        "modules": "activating and repressing (--all_modules); unmasked dropouts; otherwise native defaults",
        "pruning_backend": "official dask_multiprocessing (avoids legacy multiprocessing_on_dill)",
        "score": "max native regulon target importance across retained motifs/modules; unreported edges zero",
        "training_prior": "TF candidate list only; no DRAFT edge mask imposed on GRNBoost2/cisTarget",
    }
    run = model_run("mouse_scenic", prepared, configuration)
    selected = clusters or manifest["clusters"]
    if not set(selected).issubset(manifest["clusters"]):
        raise ValueError("Requested cluster is not in the prepared inputs.")
    for cid in selected:
        folder = run / f"C{cid}"
        folder.mkdir(exist_ok=True)
        if (folder / "complete.json").exists():
            verify_complete(folder)
            print(f"C{cid}: verified completed SCENIC output", flush=True)
            continue
        inputs = prepared / f"C{cid}"
        expr = pd.read_csv(inputs / "scenic_expression.tsv", sep="\t", index_col=0)
        matching = [len(set(expr.columns) & names) for names in db_gene_sets]
        fractions = [n / len(expr.columns) for n in matching]
        if min(fractions) < 0.5:
            raise ValueError(
                f"C{cid}: less than 50% gene symbol coverage in a cisTarget database: {fractions}. Check MGI symbol case/genome/resources."
            )
        annotation_tfs = annotations_frame["gene_name"]
        regulators = (
            (inputs / "regulators.txt").read_text(encoding="utf-8").splitlines()
        )
        if not set(regulators) & set(annotation_tfs):
            raise ValueError(
                "No candidate TF matches motif annotations; check gene symbols/resources."
            )
        official_command(
            [
                "pyscenic.cli.arboreto_with_multiprocessing",
                str(inputs / "scenic_expression.tsv"),
                str(inputs / "regulators.txt"),
                "--method",
                "grnboost2",
                "--seed",
                str(manifest["seed"]),
                "--num_workers",
                str(cpus),
                "--output",
                str(folder / "adjacencies.tsv"),
            ],
            folder / "grnboost2.log",
        )
        official_command(
            [
                "pyscenic.cli.pyscenic",
                "ctx",
                str(folder / "adjacencies.tsv"),
                *[str(p) for p in databases],
                "--annotations_fname",
                str(annotations),
                "--expression_mtx_fname",
                str(inputs / "scenic_expression.tsv"),
                "--mode",
                "dask_multiprocessing",
                "--num_workers",
                str(cpus),
                "--all_modules",
                "--output",
                str(folder / "regulons.csv"),
            ],
            folder / "cistarget.log",
        )
        regulons = load_signatures(str(folder / "regulons.csv"))
        edges = regulon_edges(regulons)
        edges.to_csv(folder / "edges.tsv.gz", sep="\t", index=False)
        if aucell and regulons:
            official_command(
                [
                    "pyscenic.cli.pyscenic",
                    "aucell",
                    str(inputs / "scenic_expression.tsv"),
                    str(folder / "regulons.csv"),
                    "--seed",
                    str(manifest["seed"]),
                    "--num_workers",
                    str(cpus),
                    "--output",
                    str(folder / "aucell.tsv"),
                ],
                folder / "aucell.log",
            )
        mark_complete(
            folder,
            {
                "model": "SCENIC",
                "cluster": cid,
                "n_regulons": len(regulons),
                "n_edges": len(edges),
                "database_gene_coverage": fractions,
                "database_motif_annotation_overlap": motif_overlap,
                "n_TFs_with_motif_annotations": len(
                    set(regulators) & set(annotation_tfs)
                ),
                "input_manifest_sha256": digest(prepared / "manifest.json"),
                "score": configuration["score"],
                "validation_only": manifest.get("validation_only", False),
            },
        )
        print(
            f"C{cid}: official SCENIC complete, {len(regulons)} regulons / {len(edges)} edges",
            flush=True,
        )
    print(f"SCENIC results: {run}", flush=True)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-run", type=Path, default=Path("latest"))
    parser.add_argument("--databases", type=Path, nargs="+", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--clusters", type=int, nargs="+")
    parser.add_argument(
        "--aucell",
        action="store_true",
        help="Also compute official cell regulon activity; not needed for TF-TG edge evaluation.",
    )
    args = parser.parse_args()
    if args.cpus < 1:
        parser.error("--cpus must be positive")
    train(
        resolve_prepared(args.prepared_run),
        args.databases,
        args.annotations,
        args.cpus,
        args.clusters,
        args.aucell,
    )


if __name__ == "__main__":
    main()
