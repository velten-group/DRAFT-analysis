"""The checkpoint line, stage by stage: simulation, Drosophila (germline, soma, replicate and control models), mouse.

Each stage reads only DATA (inputs) and the outputs of earlier stages in results/checkpoint. Run all stages with
`draft-analysis checkpoint`, or a subset with `--stages`.
"""
from __future__ import annotations

import time
from collections.abc import Callable


def _simulation():
    from . import simulation
    simulation.run()


def _reproduce():
    from .fly import model_stages
    model_stages.reproduce("germ")
    model_stages.reproduce("soma")


def _hic():
    from .fly import hic
    hic.run()


def _network():
    from .fly import model_stages
    model_stages.network("germ")
    model_stages.network("soma")


def _numbering():
    from .fly import numbering
    numbering.edge_clusters()
    numbering.topics()
    numbering.install_frozen()


def _ovo():
    from .fly import regulon
    regulon.ovo()


def _mondo():
    from .fly import mondo_regulon
    mondo_regulon.run_stage_lists()
    mondo_regulon.run_r_candidates()


def _mondo_ranks():
    from .fly import mondo_ranks
    mondo_ranks.run()


def _ccc():
    from .fly import ccc
    ccc.full_scores("germ")
    ccc.full_scores("soma")
    ccc.unified()


def _seeds():
    """Germline replicate models (seeds 2024, 2026-2028): network and CCC exports, then calls in all five models."""
    from .fly import ccc, model_stages
    for seed in (2024, 2026, 2027, 2028):
        model_stages.network("germ", seed)
        ccc.full_scores("germ", seed)
    ccc.peak_shifts()


def _controls():
    from .fly import controls
    controls.run()


def _ccc_supp():
    from .fly import ccc
    ccc.supplementary()


def _mouse_notebooks():
    from .mouse import modules
    modules.replay()


def _mouse_benchmark():
    from .mouse import benchmark
    benchmark.run()


STAGES: dict[str, tuple[str, Callable[[], None]]] = {
    "simulation": ("simulation benchmark: every fit scored again from its saved predictions (Fig. 2, Supp. Figs. 1-4)",
                   _simulation),
    "stage_distances": ("stage embeddings and TF-TG distance matrices (Fig. 4c)", _reproduce),
    "hic_benchmark": ("DRAFT and SCENIC+ per-stage scores of the Hi-C within-prior candidates (Fig. 4b)", _hic),
    "networks": ("strict and dense stage networks", _network),
    "edge_clusters": ("edge clusters and topics recomputed, compared with and replaced by the frozen numbering", _numbering),
    "ovo_regulon": ("ovo eRegulon at GSC (Fig. 4d)", _ovo),
    "mondo_regulon": ("Mondo stage lists and edge-cluster candidates (Fig. 5c)", _mondo),
    "mondo_target_ranks": ("kuz and mamo ranks among the early Mondo targets of DRAFT and its ablations (Supp. Table 5)",
                    _mondo_ranks),
    "cell_communication": ("receptor-side scores and LR timing calls (Fig. 6)", _ccc),
    "replicate_models": ("five-model calls and profiles (Figs. 5a, 6d)", _seeds),
    "control_models": ("network export of the ablation and prior-control models (Supp. Figs. 10, 11)", _controls),
    "communication_checks": ("call stability, sequential alignment and alignment benchmark (Supp. Figs. 12-14)", _ccc_supp),
    "mouse_networks": ("mouse: network export, edge clusters, topics and GO from grn_best.pt (Fig. 3c-e, Supp. Figs. 5-6)",
                   _mouse_notebooks),
    "mouse_benchmark": ("mouse: official AUPR and balanced AUROC of the five methods (Fig. 3b; needs Java)",
                        _mouse_benchmark),
}


def run(stages: list[str] | None = None) -> None:
    for name in stages or list(STAGES):
        what, fn = STAGES[name]
        print(f"== {name}: {what}", flush=True)
        t0 = time.time()
        fn()
        print(f"   done in {time.time() - t0:.0f} s", flush=True)
