"""DRAFT model and its ablations."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch_geometric.nn import GATConv, HeteroConv


class PairNorm(nn.Module):
    def __init__(self, scale: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        x = x - mean
        rownorm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        return self.scale * x / rownorm


class HeteroDynamicGRN_TF_RE_TG(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 2,
        dropout_rate: float = 0.2,
        use_gru: bool = True,
        spatial_encoder: str = "gat",
    ) -> None:
        super().__init__()
        self.use_gru = use_gru
        self.spatial_encoder = spatial_encoder.strip().lower()
        if self.spatial_encoder not in {"gat", "linear"}:
            raise ValueError(f"Unsupported spatial encoder: {spatial_encoder}")
        h2 = hidden_channels * heads

        self.conv1 = HeteroConv(
            {
                ("TF", "combine", "RE"): GATConv(
                    in_channels, hidden_channels, heads=heads, add_self_loops=False
                ),
                ("RE", "combined_by", "TF"): GATConv(
                    in_channels, hidden_channels, heads=heads, add_self_loops=False
                ),
                ("RE", "regulate", "TG"): GATConv(
                    in_channels, hidden_channels, heads=heads, add_self_loops=False
                ),
                ("TG", "regulated_by", "RE"): GATConv(
                    in_channels, hidden_channels, heads=heads, add_self_loops=False
                ),
            },
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {
                ("TF", "combine", "RE"): GATConv(
                    h2, hidden_channels, heads=heads, add_self_loops=False
                ),
                ("RE", "combined_by", "TF"): GATConv(
                    h2, hidden_channels, heads=heads, add_self_loops=False
                ),
                ("RE", "regulate", "TG"): GATConv(
                    h2, hidden_channels, heads=heads, add_self_loops=False
                ),
                ("TG", "regulated_by", "RE"): GATConv(
                    h2, hidden_channels, heads=heads, add_self_loops=False
                ),
            },
            aggr="sum",
        )

        self.bn_tf1 = nn.BatchNorm1d(h2)
        self.bn_re1 = nn.BatchNorm1d(h2)
        self.bn_tg1 = nn.BatchNorm1d(h2)
        self.bn_tf2 = nn.BatchNorm1d(h2)
        self.bn_re2 = nn.BatchNorm1d(h2)
        self.bn_tg2 = nn.BatchNorm1d(h2)

        self.pn_tf = PairNorm(scale=1.0)
        self.pn_re = PairNorm(scale=1.0)
        self.pn_tg = PairNorm(scale=1.0)

        self.res_tf = nn.Linear(in_channels, h2)
        self.res_re = nn.Linear(in_channels, h2)
        self.res_tg = nn.Linear(in_channels, h2)

        self.dropout = nn.Dropout(dropout_rate)

        self.gru_tf = nn.GRUCell(h2, hidden_channels)
        self.gru_re = nn.GRUCell(h2, hidden_channels)
        self.gru_tg = nn.GRUCell(h2, hidden_channels)

        self.gru_skip_tf = nn.Linear(h2, hidden_channels)
        self.gru_skip_re = nn.Linear(h2, hidden_channels)
        self.gru_skip_tg = nn.Linear(h2, hidden_channels)

        self.fc_tf = nn.Linear(hidden_channels, out_channels)
        self.fc_re = nn.Linear(hidden_channels, out_channels)
        self.fc_tg = nn.Linear(hidden_channels, out_channels)

        # Preserve the reference initialization order for paired-seed comparisons.
        # The ablation contains no GRU modules or trainable GRU parameters.
        if not self.use_gru:
            del self.gru_tf, self.gru_re, self.gru_tg

        if self.spatial_encoder == "linear":
            del self.conv1, self.conv2
            # Match retained GRU/residual/output weights and preserve the
            # random-sampling stream for paired-seed comparisons.
            with torch.random.fork_rng(devices=[]):
                self.linear1 = nn.ModuleDict(
                    {node: nn.Linear(in_channels, h2) for node in ("TF", "RE", "TG")}
                )
                self.linear2 = nn.ModuleDict(
                    {node: nn.Linear(h2, h2) for node in ("TF", "RE", "TG")}
                )

    def _project_without_gru(
        self,
        features: tuple[
            dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]
        ],
        topo_order: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_tf, feat_re, feat_tg = features
        out_tf = torch.stack([self.gru_skip_tf(feat_tf[t]) for t in topo_order], dim=1)
        out_re = torch.stack([self.gru_skip_re(feat_re[t]) for t in topo_order], dim=1)
        out_tg = torch.stack([self.gru_skip_tg(feat_tg[t]) for t in topo_order], dim=1)
        return self.fc_tf(out_tf), self.fc_re(out_re), self.fc_tg(out_tg)

    def _encode_spatial(
        self, graphs: list[Any]
    ) -> tuple[
        dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]
    ]:
        feat_tf: dict[int, torch.Tensor] = {}
        feat_re: dict[int, torch.Tensor] = {}
        feat_tg: dict[int, torch.Tensor] = {}

        for t, data in enumerate(graphs):
            x_dict0 = data.x_dict
            x0_tf, x0_re, x0_tg = x_dict0["TF"], x_dict0["RE"], x_dict0["TG"]

            if self.spatial_encoder == "gat":
                x1 = self.conv1(x_dict0, data.edge_index_dict)
            else:
                x1 = {
                    node: self.linear1[node](x_dict0[node])
                    for node in ("TF", "RE", "TG")
                }
            x1["TF"] = self.dropout(self.bn_tf1(x1["TF"]))
            x1["RE"] = self.dropout(self.bn_re1(x1["RE"]))
            x1["TG"] = self.dropout(self.bn_tg1(x1["TG"]))

            if self.spatial_encoder == "gat":
                x2 = self.conv2(x1, data.edge_index_dict)
            else:
                x2 = {node: self.linear2[node](x1[node]) for node in ("TF", "RE", "TG")}
            x_tf_2 = self.bn_tf2(x2["TF"])
            x_re_2 = self.bn_re2(x2["RE"])
            x_tg_2 = self.bn_tg2(x2["TG"])

            x_tf_2 = self.pn_tf(x_tf_2)
            x_re_2 = self.pn_re(x_re_2)
            x_tg_2 = self.pn_tg(x_tg_2)

            x_tf_2 = x_tf_2 + self.res_tf(x0_tf)
            x_re_2 = x_re_2 + self.res_re(x0_re)
            x_tg_2 = x_tg_2 + self.res_tg(x0_tg)

            x_tf_2 = self.dropout(x_tf_2)
            x_re_2 = self.dropout(x_re_2)
            x_tg_2 = self.dropout(x_tg_2)

            feat_tf[t], feat_re[t], feat_tg[t] = x_tf_2, x_re_2, x_tg_2

        return feat_tf, feat_re, feat_tg

    def forward_multi_parent(
        self, graphs: list[Any], topo_order: list[int], parents: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_tf, feat_re, feat_tg = self._encode_spatial(graphs)
        if not self.use_gru:
            return self._project_without_gru((feat_tf, feat_re, feat_tg), topo_order)

        h_tf: dict[int, torch.Tensor] = {}
        h_re: dict[int, torch.Tensor] = {}
        h_tg: dict[int, torch.Tensor] = {}
        hidden_size = self.gru_tf.hidden_size

        for t in topo_order:
            x_tf, x_re, x_tg = feat_tf[t], feat_re[t], feat_tg[t]
            parents_t = [p for p in (parents[t] or []) if p is not None and p >= 0]

            if len(parents_t) == 0:
                h0_tf = x_tf.new_zeros(x_tf.size(0), hidden_size)
                h0_re = x_re.new_zeros(x_re.size(0), hidden_size)
                h0_tg = x_tg.new_zeros(x_tg.size(0), hidden_size)
            elif len(parents_t) == 1:
                h0_tf, h0_re, h0_tg = (
                    h_tf[parents_t[0]],
                    h_re[parents_t[0]],
                    h_tg[parents_t[0]],
                )
            else:
                h0_tf = torch.stack([h_tf[p] for p in parents_t], dim=0).mean(dim=0)
                h0_re = torch.stack([h_re[p] for p in parents_t], dim=0).mean(dim=0)
                h0_tg = torch.stack([h_tg[p] for p in parents_t], dim=0).mean(dim=0)

            h_tf[t] = self.gru_tf(x_tf, h0_tf) + self.gru_skip_tf(x_tf)
            h_re[t] = self.gru_re(x_re, h0_re) + self.gru_skip_re(x_re)
            h_tg[t] = self.gru_tg(x_tg, h0_tg) + self.gru_skip_tg(x_tg)

        out_tf = torch.stack([h_tf[t] for t in topo_order], dim=1)
        out_re = torch.stack([h_re[t] for t in topo_order], dim=1)
        out_tg = torch.stack([h_tg[t] for t in topo_order], dim=1)
        return self.fc_tf(out_tf), self.fc_re(out_re), self.fc_tg(out_tg)

    def forward_single_parent(
        self, graphs: list[Any], topo_order: list[int], parent: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_tf, feat_re, feat_tg = self._encode_spatial(graphs)
        if not self.use_gru:
            return self._project_without_gru((feat_tf, feat_re, feat_tg), topo_order)

        h_tf: dict[int, torch.Tensor] = {}
        h_re: dict[int, torch.Tensor] = {}
        h_tg: dict[int, torch.Tensor] = {}
        hidden_size = self.gru_tf.hidden_size

        for t in topo_order:
            x_tf, x_re, x_tg = feat_tf[t], feat_re[t], feat_tg[t]
            if parent[t] is None or parent[t] < 0:
                h0_tf = x_tf.new_zeros(x_tf.size(0), hidden_size)
                h0_re = x_re.new_zeros(x_re.size(0), hidden_size)
                h0_tg = x_tg.new_zeros(x_tg.size(0), hidden_size)
            else:
                h0_tf = h_tf[parent[t]]
                h0_re = h_re[parent[t]]
                h0_tg = h_tg[parent[t]]

            h_tf[t] = self.gru_tf(x_tf, h0_tf) + self.gru_skip_tf(x_tf)
            h_re[t] = self.gru_re(x_re, h0_re) + self.gru_skip_re(x_re)
            h_tg[t] = self.gru_tg(x_tg, h0_tg) + self.gru_skip_tg(x_tg)

        out_tf = torch.stack([h_tf[t] for t in topo_order], dim=1)
        out_re = torch.stack([h_re[t] for t in topo_order], dim=1)
        out_tg = torch.stack([h_tg[t] for t in topo_order], dim=1)
        return self.fc_tf(out_tf), self.fc_re(out_re), self.fc_tg(out_tg)


class SomaticLineageGNN(HeteroDynamicGRN_TF_RE_TG):
    def forward(
        self, graphs: list[Any], topo_order: list[int], parents: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_multi_parent(graphs, topo_order, parents)


class GermlineLineageGNN(HeteroDynamicGRN_TF_RE_TG):
    def forward(
        self, graphs: list[Any], topo_order: list[int], parent: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_single_parent(graphs, topo_order, parent)


def build_model_for_lineage(
    lineage: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    heads: int = 2,
    dropout_rate: float = 0.2,
    use_gru: bool = True,
    spatial_encoder: str = "gat",
) -> HeteroDynamicGRN_TF_RE_TG:
    lineage_key = lineage.lower()
    if lineage_key in {"soma", "somatic"}:
        return SomaticLineageGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            dropout_rate=dropout_rate,
            use_gru=use_gru,
            spatial_encoder=spatial_encoder,
        )
    if lineage_key in {"germ", "germline"}:
        return GermlineLineageGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            dropout_rate=dropout_rate,
            use_gru=use_gru,
            spatial_encoder=spatial_encoder,
        )
    raise ValueError(f"Unsupported lineage '{lineage}'. Expected soma or germ.")
