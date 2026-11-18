"""Ranks of the Mondo candidate targets in the early germline (Supplementary Table 5): DRAFT and its two ablations.

Early window GSC-Sp-4:
- one fixed target universe for every model: genes of TG.tsv expressed (RNA lognorm > 0) in 10-78% of the paired
  germline cells of each of GSC, Gb, Sp-2 and Sp-4; symbols of the form CG<digits> and Mondo itself excluded;
- score of every candidate at each of the four stages: (cosine of the Mondo and target embeddings + 1) / 2, from the
  checkpoint before any edge selection;
- per stage, ranks by descending score (ties averaged); the four ranks combined by their geometric mean and re-ranked
  (ties share the lowest rank).
Models: the manuscript germline model (seed 2025) and its ablations without GRU and with Linear + GRU (seed 2025).
The table reports these ranks for the two ablations; its DRAFT row is the consensus rank of the Fig. 5c eRegulon.
Output: results/checkpoint/fly/mondo_ranks/universe.tsv and ranks.tsv.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import config, paths

STAGES = (0, 1, 2, 3)
STAGE_LABELS = ("GSC", "Gb", "Sp-2", "Sp-4")
MONDO = "FBgn0032940"
BOUNDS = (0.1, 0.78)
MODELS = {"DRAFT": None, "DRAFT w/o GRU": "no_gru", "Linear + GRU": "linear_gru"}


def _cell(value) -> str:
    return re.sub(r"#s?0*([0-9]+)$", r"_sample\1", str(value).strip(), flags=re.IGNORECASE)


def universe() -> pd.DataFrame:
    import anndata as ad

    raw = paths.DATA / "fly" / "raw"
    tg = pd.read_csv(paths.data("fly", "raw", "TG.tsv"), sep="\t", dtype=str, keep_default_na=False)
    anno = pd.read_csv(paths.data("fly", "raw", "cell_annotation.csv"), dtype=str)
    ccol = next(c for c in ("cell_id", "cell", "barcode", "Unnamed: 0") if c in anno)
    scol = next(c for c in ("GEX_latent_time_bins_annotated", "cell_type", "annotation") if c in anno)
    rna = ad.read_h5ad(raw / "A_rna_focus.h5ad", backed="r")
    atac = ad.read_h5ad(raw / "B_atac_focus.h5ad", backed="r")
    try:
        rna_cells = pd.Index([_cell(v) for v in rna.obs_names])
        paired = rna_cells.intersection(pd.Index([_cell(v) for v in atac.obs_names]))
        anno["cell"] = anno[ccol].map(_cell)
        anno = anno[anno.cell.isin(paired)]
        u = pd.DataFrame({"TG_id": tg.FBgn_ID, "TG": tg.gene_name.where(tg.gene_name.ne(""), tg.FBgn_ID)})
        keep = np.ones(len(u), bool)
        for s, label in zip(STAGES, STAGE_LABELS):
            cells = anno.loc[anno[scol].eq(f"Germline_{s}"), "cell"]
            m = rna.layers["lognorm"][rna_cells.get_indexer(cells), :]
            frac = pd.Series(np.asarray((m > 0).mean(axis=0)).ravel(), index=rna.var_names).reindex(u.TG_id)
            u[f"expressing_fraction_{label}"] = frac.to_numpy()
            keep &= frac.between(*BOUNDS).to_numpy()
    finally:
        rna.file.close()
        atac.file.close()
    keep &= ~u.TG.str.fullmatch(r"CG\d+").to_numpy() & u.TG_id.ne(MONDO).to_numpy()
    return u[keep].reset_index(drop=True)


def _embeddings() -> tuple[dict, dict]:
    import torch

    from drosophila_gnn.checkpoint import torch_load_compat
    from drosophila_gnn.infer import reload_germ_from_checkpoint
    from drosophila_gnn.models.gnn import build_model_for_lineage

    cfg = config.load("germ_reproduce", {"reproduce": {"checkpoint_path": str(config.checkpoint("germ"))}})
    base = reload_germ_from_checkpoint(cfg)
    out = {"DRAFT": (base["out_tf"], base["out_tg"])}
    for model, variant in MODELS.items():
        if variant is None:
            continue
        ck = torch_load_compat(paths.data("fly", "checkpoints", "ablation", "seed_2025", variant, "model.pt"),
                               map_location="cpu")
        hp = ck["hp"]
        assert ck.get("seed") == 2025 and bool(hp.get("use_gru", True)) == (variant != "no_gru"), variant
        net = build_model_for_lineage("germ", int(ck["in_channels"]), int(hp["hidden_channels"]),
                                      int(hp["out_channels"]), heads=int(hp["gat_heads"]),
                                      dropout_rate=float(hp["dropout_rate"]), use_gru=bool(hp.get("use_gru", True)),
                                      spatial_encoder=hp.get("spatial_encoder", "gat")).to(base["device"])
        net.load_state_dict(ck["model_state_dict"])
        net.eval()
        with torch.no_grad():
            tf, _, tg = net(base["graphs_germline"], ck["topo_order"], ck["parent"])
        out[model] = (tf, tg)
    return base["maps"], out


def run() -> pd.DataFrame:
    import torch

    u = universe()
    maps, emb = _embeddings()
    tf_row = list(maps["tf"]).index(MONDO)
    tg_row = {g: i for i, g in enumerate(maps["tg"])}
    rows = [tg_row[g] for g in u.TG_id]
    frames = []
    for model, (tf, tg) in emb.items():
        a = torch.nn.functional.normalize(tf[tf_row].detach().cpu().float(), dim=-1)
        b = torch.nn.functional.normalize(tg[rows].detach().cpu().float(), dim=-1)
        score = (((b * a[None]).sum(-1) + 1) / 2).clamp(0, 1)[:, list(STAGES)].numpy()
        ranks = pd.DataFrame(score, columns=[f"rank_{s}" for s in STAGE_LABELS]).rank(ascending=False,
                                                                                       method="average")
        f = u[["TG_id", "TG"]].assign(model=model)
        f[[f"score_{s}" for s in STAGE_LABELS]] = score
        f[ranks.columns] = ranks.to_numpy()
        f["geometric_mean_rank"] = np.exp(np.log(ranks).mean(axis=1)).to_numpy()
        f["aggregate_rank"] = f.geometric_mean_rank.rank(method="min").astype(int)
        frames.append(f)
    out = paths.result("fly", "mondo_ranks", "ranks.tsv")
    u.to_csv(out.with_name("universe.tsv"), sep="\t", index=False)
    ranks = pd.concat(frames, ignore_index=True)
    ranks.to_csv(out, sep="\t", index=False)
    for model in MODELS:
        r = ranks[ranks.model.eq(model)].set_index("TG").aggregate_rank
        print(f"   {model}: kuz {r['kuz']}, mamo {r['mamo']} of {len(u)} targets", flush=True)
    return ranks
