"""Per-stage graphs of TF, RE and TG nodes with their prior edges, for the germline and soma models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .lineage import select_lineage_with_branching, tail_num
from .preprocess import pick_device_by_free_mem, prepare_common_inputs


NODE_TYPES = ("TF", "RE", "TG")


def build_graphs_by_cell_type(common: dict[str, Any]) -> dict[str, HeteroData]:
    graphs: dict[str, HeteroData] = {}
    anno = common["anno"]
    expr_df = common["expr_df"]
    peaks_df = common["peaks_df"]
    tf_list = common["tf_list"]
    tg_list = common["tg_list"]
    re_list = common["re_list"]
    edge_index_tf_re = common["edge_index_tf_re"]
    edge_index_re_tf = common["edge_index_re_tf"]
    edge_index_re_tg = common["edge_index_re_tg"]
    edge_index_tg_re = common["edge_index_tg_re"]

    for cell_type, subanno in anno.groupby("cell_type"):
        cells = subanno["cell_id"].tolist()
        tf_x = torch.tensor(expr_df.loc[tf_list, cells].fillna(0).values.astype(np.float32))
        tg_x = torch.tensor(expr_df.loc[tg_list, cells].fillna(0).values.astype(np.float32))
        re_x = torch.tensor(peaks_df.loc[re_list, cells].fillna(0).values.astype(np.float32))

        data = HeteroData()
        data["TF"].x = tf_x
        data["RE"].x = re_x
        data["TG"].x = tg_x
        data["TF", "combine", "RE"].edge_index = edge_index_tf_re
        data["RE", "combined_by", "TF"].edge_index = edge_index_re_tf
        data["RE", "regulate", "TG"].edge_index = edge_index_re_tg
        data["TG", "regulated_by", "RE"].edge_index = edge_index_tg_re
        graphs[str(cell_type)] = data
    return graphs


def unify_feature_dim(graphs: list[HeteroData], in_channels: int) -> None:
    for graph in graphs:
        for node_type in NODE_TYPES:
            x = graph[node_type].x.float()
            if x.size(1) < in_channels:
                pad = torch.zeros(x.size(0), in_channels - x.size(1), dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
            elif x.size(1) > in_channels:
                x = x[:, :in_channels]
            graph[node_type].x = x


def build_soma_graph_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    common = prepare_common_inputs(config)
    graphs = build_graphs_by_cell_type(common)

    germline_types = [cell_type for cell_type in graphs.keys() if cell_type.lower().startswith("germline_")]
    germline_types.sort(key=tail_num)
    graphs_germline = [graphs[cell_type] for cell_type in germline_types]
    in_channels_germline = graphs_germline[0]["TF"].x.size(1) if graphs_germline else 0
    unify_feature_dim(graphs_germline, in_channels_germline)
    topo_germline = list(range(len(graphs_germline)))
    parents_germline = [[]] + [[idx - 1] for idx in range(1, len(graphs_germline))]
    germline_names = germline_types

    graphs_somatic, topo_somatic, parents_somatic, somatic_names = select_lineage_with_branching(
        graphs,
        prefix_keys=("somatic", "soma"),
    )
    if graphs_somatic:
        in_channels_somatic = graphs_somatic[0]["TF"].x.size(1)
        unify_feature_dim(graphs_somatic, in_channels_somatic)
    else:
        in_channels_somatic = 0

    device = pick_device_by_free_mem(min_required_mb=6000)
    graphs_germline = [graph.to(device) for graph in graphs_germline]
    graphs_somatic = [graph.to(device) for graph in graphs_somatic]

    num_nodes = {"TF": len(common["tf_list"]), "RE": len(common["re_list"]), "TG": len(common["tg_list"])}
    maps = {"tf": common["tf_list"], "re": common["re_list"], "tg": common["tg_list"]}

    summary_lines = [
        "=== PREPROCESS SUMMARY ===",
        f"RNA h5ad (layer) : {common['input_paths'].rna_h5ad.name} ({common['used_rna_layer']})  shape={common['expr_df'].shape}",
        f"ATAC h5ad(layer) : {common['input_paths'].atac_h5ad.name} ({common['used_atac_layer']}) shape={common['peaks_df'].shape}",
        f"Lists            : TF={len(common['tf_list'])}/{common['tf_df'].shape[0]} | RE={len(common['re_list'])} | TG={len(common['tg_list'])}/{len(common['tg_list_all'])}",
        f"Edges TF->RE     : {common['edge_index_tf_re'].size(1)}",
        f"Edges RE->TG     : {common['edge_index_re_tg'].size(1)}",
        f"Device           : {device}",
        "-- Germline --",
        f"  count={len(graphs_germline)} | in_channels={in_channels_germline} | names={germline_names}",
        "-- Somatic --",
        f"  count={len(graphs_somatic)}  | in_channels={in_channels_somatic}  | names={somatic_names}",
    ]

    return {
        "mode": "soma",
        "common": common,
        "graphs": graphs,
        "graphs_germline": graphs_germline,
        "topo_germline": topo_germline,
        "parents_germline": parents_germline,
        "germline_names": germline_names,
        "graphs_somatic": graphs_somatic,
        "topo_somatic": topo_somatic,
        "parents_somatic": parents_somatic,
        "somatic_names": somatic_names,
        "in_channels_germline": in_channels_germline,
        "in_channels_somatic": in_channels_somatic,
        "device": device,
        "num_nodes": num_nodes,
        "maps": maps,
        "summary_lines": summary_lines,
    }


def build_germ_graph_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    common = prepare_common_inputs(config)
    graphs = build_graphs_by_cell_type(common)

    germline_types = [cell_type for cell_type in graphs.keys() if cell_type.startswith("Germline_")]
    germline_types.sort(key=tail_num)
    graphs_germline = [graphs[cell_type] for cell_type in germline_types]

    topo_order_orig = [tail_num(name) for name in germline_types]
    new_idx_map = {timepoint: idx for idx, timepoint in enumerate(topo_order_orig)}
    topo_order = [new_idx_map[timepoint] for timepoint in topo_order_orig]
    parent = [-1] + list(range(0, len(topo_order) - 1)) if len(topo_order) > 0 else []

    for graph in graphs_germline:
        for node_type in NODE_TYPES:
            graph[node_type].x = graph[node_type].x.float()
    all_dims = [graph["TF"].x.size(1) for graph in graphs_germline]
    in_channels = max(all_dims) if all_dims else 0
    unify_feature_dim(graphs_germline, in_channels)

    device = pick_device_by_free_mem(min_required_mb=6000)
    graphs_germline = [graph.to(device) for graph in graphs_germline]

    num_nodes = {"TF": len(common["tf_list"]), "RE": len(common["re_list"]), "TG": len(common["tg_list"])}
    maps = {"tf": common["tf_list"], "re": common["re_list"], "tg": common["tg_list"]}

    summary_lines = [
        "=== PREPROCESS SUMMARY ===",
        f"Base dir               : {common['input_paths'].base_dir}",
        f"RNA h5ad (layer)       : {common['input_paths'].rna_h5ad.name} ({common['used_rna_layer']})  shape={common['expr_df'].shape}",
        f"ATAC h5ad (layer)      : {common['input_paths'].atac_h5ad.name} ({common['used_atac_layer']}) shape={common['peaks_df'].shape}",
        f"Lists                  : TF={len(common['tf_list'])}/{common['tf_df'].shape[0]} | RE={len(common['re_list'])} | TG={len(common['tg_list'])}/{len(common['tg_list_all'])}",
        f"Edges TF->RE           : {common['edge_index_tf_re'].size(1)}",
        f"Edges RE->TG           : {common['edge_index_re_tg'].size(1)}",
        f"Graphs (by cell_type)  : {len(graphs)}",
        f"Germline types (ordered): {', '.join(germline_types) if germline_types else '<empty>'}",
        f"in_channels (unified)   : {in_channels}",
        f"Device                  : {device}",
        "--------------------------",
    ]

    return {
        "mode": "germ",
        "common": common,
        "graphs": graphs,
        "graphs_germline": graphs_germline,
        "topo_order": topo_order,
        "parent": parent,
        "germline_types": germline_types,
        "in_channels": in_channels,
        "device": device,
        "num_nodes": num_nodes,
        "maps": maps,
        "summary_lines": summary_lines,
    }


def build_lineage_graph_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    lineage = str(config.get("lineage", "")).lower()
    if lineage in {"soma", "somatic"}:
        return build_soma_graph_artifacts(config)
    if lineage in {"germ", "germline"}:
        return build_germ_graph_artifacts(config)
    raise ValueError(f"Unsupported lineage '{config.get('lineage')}'. Expected soma or germ.")
