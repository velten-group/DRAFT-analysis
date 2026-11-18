"""draft-analysis: reproduce the figures of the DRAFT manuscript along two lines.

Quick line (no inputs needed; a few seconds)
    draft-analysis quick                 data panels of Figs. 2-6, all Source Data workbooks and Supp. Tables 2, 4-9,
                                         from source_data/

Checkpoint line (deposited checkpoints, saved simulation predictions and inputs in data/; about 4 hours; Java for
Fig. 3b)
    draft-analysis data [--verify]       list the inputs (data/manifest.tsv), check presence (and SHA-256)
    draft-analysis checkpoint            run all stages, build the panel tables of Figs. 2-6 and Supp. Figs. 1-14
                                         and the recomputed cells of Supp. Tables 2 and 4-9, compare them with the
                                         published values and draw Figs. 2-6
    draft-analysis checkpoint --stages hic_benchmark networks  rerun some stages, then rebuild, compare and draw
    draft-analysis checkpoint --no-run   only rebuild the tables, compare and draw from existing stage outputs

Results: results/quick/ and results/checkpoint/.
"""
from __future__ import annotations

import argparse
import os
import sys

# Every computation runs on one CPU thread (also in the scripts started as subprocesses, which inherit these).
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_name] = "1"

from . import paths

CHECKPOINT_FIGURES = (2, 3, 4, 5, 6)


def quick(figures: list[int] | None) -> int:
    from . import plots, source_data

    for out in plots.draw("quick", figures=figures):
        print(out)
    for kind in source_data.KINDS:
        for out in source_data.build_workbooks(kind, source_data.frozen_dir(kind),
                                               paths.QUICK / "source_data_workbooks" / kind):
            print(out)
    for out in source_data.build_workbooks(source_data.TABLES, source_data.frozen_dir(source_data.TABLES),
                                           paths.QUICK / "supplementary_tables"):
        print(out)
    return 0


def check_own_packages() -> None:
    """Stop when drosophila_gnn or network_evaluation is not the copy in this repository's src/ (an older install in
    the user site-packages, ~/.local, or on PYTHONPATH comes first on sys.path and would run older code)."""
    import importlib.util
    from pathlib import Path

    src = (paths.REPO / "src").resolve()
    for name in ("drosophila_gnn", "network_evaluation"):
        spec = importlib.util.find_spec(name)
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        if origin is None or src not in origin.parents:
            raise SystemExit(
                f"{name} is loaded from {origin}, not from {src}. Another installation (in ~/.local or on PYTHONPATH) "
                "comes first. Set PYTHONNOUSERSITE=1 (conda: conda env config vars set PYTHONNOUSERSITE=1) and unset "
                "PYTHONPATH, reinstall with pip install -r requirements.lock and pip install -e . --no-deps, and run "
                "again.")


def checkpoint(stages: list[str] | None, run: bool) -> int:
    check_own_packages()
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from . import compare, pipeline, plots
    from .panels import fig3, fig4, fig5, fig6, sim, supp, tables

    if run:
        pipeline.run(stages)
    for name, module in (("Figure 2 and Supplementary Figs. 1-4", sim), ("Figure 3", fig3), ("Figure 4", fig4),
                         ("Figure 5", fig5), ("Figure 6", fig6),
                         ("Supplementary", supp), ("Supplementary Tables", tables)):
        print(f"== panel tables: {name}", flush=True)
        module.build()
    results = [r for kind in ("main", "supplementary", "tables") for r in compare.run(kind)]
    compare.write_report(results, paths.CHECKPOINT / "comparison_with_source_data.tsv")
    for out in plots.draw("checkpoint", figures=CHECKPOINT_FIGURES):
        print(out)
    return 0 if all(r.ok for r in results) else 1


def main(argv: list[str] | None = None) -> int:
    from .pipeline import STAGES

    p = argparse.ArgumentParser(prog="draft-analysis", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quick", help="figures, Source Data workbooks and supplementary tables from source_data/")
    q.add_argument("--figures", nargs="+", type=int, help="subset of Figs. 2-6 to draw")
    d = sub.add_parser("data", help="list and check the inputs of the checkpoint line")
    d.add_argument("--verify", action="store_true", help="also check SHA-256 (slow for large files)")
    d.add_argument("--download", action="store_true", help="first fetch the third-party tools that are not deposited")
    c = sub.add_parser("checkpoint", help="reproduce Figs. 2-6, Supp. Figs. 1-14 and Supp. Tables 2, 4-9 from the "
                                         "checkpoints and saved predictions")
    c.add_argument("--stages", nargs="+", choices=list(STAGES), help="run only these stages (in the given order)")
    c.add_argument("--no-run", action="store_true", help="skip the stages; rebuild tables, compare and draw")
    args = p.parse_args(argv)

    if args.cmd == "quick":
        return quick(args.figures)
    if args.cmd == "data":
        from . import manifest
        if args.download:
            manifest.download()
        return manifest.check(verify=args.verify)
    return checkpoint(args.stages, run=not args.no_run)


if __name__ == "__main__":
    sys.exit(main())
