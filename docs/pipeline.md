# Pipeline map

## Figures and lines

| Figures | Lines | Computed from |
|---|---|---|
| 2; Supp. 1-4 (simulation) | quick, checkpoint | simulated datasets and the saved final scores of the 639 benchmark fits |
| 3; Supp. 5-6 (mouse) | quick, checkpoint | mouse model `grn_best.pt`, cluster expression tables, baseline outputs (Dictys, SCENIC, scMTNI+Prior), scMTNI gold standards and evaluator |
| 4, 5a, 5c, 6; Supp. 8, 9, 12-14 | quick, checkpoint | germline models (five seeds) and soma model |
| Supp. 10-11 | quick, checkpoint | as above, plus the ablation and prior-control models |
| Supp. 7 | quick, checkpoint | baseline networks (SCENIC+, scDoRI, random) |
| 5b, 5d-f | quick | experimental measurements |

## Checkpoint line: stages

Outputs in `results/checkpoint/`.

| Stage | Computes | Output |
|---|---|---|
| simulation | candidate AUROC per stage, temporal smoothness, change and branch AUROC, prior SNR, Random control, branch t-SNE | `simulation/` |
| stage_distances | stage embeddings, TF-TG distances between stages | `fly/{germ,soma}/reproduce/` |
| hic_benchmark | DRAFT and SCENIC+ scores of the 120,111 Hi-C candidates | `fly/germ/hic_within_prior/` |
| networks | strict and 50,000-edge dense stage networks | `fly/{germ,soma}/network/`, `network_dense/` |
| edge_clusters | edge clusters and topics, compared with the fixed numbering | `fly/germ/edge_cluster_recomputed/`, `topic_notebook/` |
| ovo_regulon | ovo eRegulon at GSC | `fly/germ/single_tf/` |
| mondo_regulon | Mondo target lists, GSC to Sp-4 | `fly/fig5c_root/germ/single_tf/` |
| mondo_target_ranks | kuz and mamo ranks in DRAFT and its ablations | `fly/mondo_ranks/` |
| cell_communication | receptor scores, early/late calls, literature AUROC | `fly/ccc_full_scores/`, `fly/ccc_unified/` |
| replicate_models | networks and calls of the four other germline models | `fly/germ_seed<seed>/` |
| control_models | networks of the ablation and prior-control models | `fly/controls/` |
| communication_checks | call stability, sequential alignment, alignment benchmark | `fly/ccc_unified/`, `fly/ccc_unified_sequential/` |
| mouse_networks | mouse networks, edge clusters, topics and GO terms | `mouse/notebook_run/` |
| mouse_benchmark | AUPR (official scMTNI evaluator) and balanced AUROC of the five methods | `mouse/benchmark/` |

## Checkpoint line: panels

Tables in `results/checkpoint/panels/`. Each is compared with the published Source Data table; the result, tolerance
and reason per table are in `results/checkpoint/comparison_with_source_data.tsv`.

| Panel | Computed from |
|---|---|
| 2b, 2e, 2f | 120 fits of the main design (3 datasets x 5 model seeds x 8 methods), ground truth, Random (seeds 0-4) |
| 2c, 2d | fits at 2, 3, 6 and 8 cross-module candidates per TG, prior SNR of each dataset, random-network fits; fly prior SNR from the deposited priors, mouse prior SNR given in the code |
| Supp. 1 | distances of the 120 main fits and the ground truth |
| Supp. 2 | 24 random-network fits, Random, ground truth |
| Supp. 3 | prior-size, turnover and inactive-candidate fits |
| Supp. 4 | 45 branching-lineage fits; RNA of dataset 101 for the t-SNE |
| 3a | cluster expression tables |
| 3b | stage mouse_benchmark |
| 3c | DRAFT: mouse model; Dictys, SCENIC: native scores; scMTNI+Prior: published networks |
| 3d, 3e; Supp. 5, 6 | fixed edge clusters and topics of the mouse networks; selected TFs and GO terms |
| 4a | RNA h5ad and cell annotation |
| 4b | DRAFT: germline model; SCENIC+: per-stage fits; scDoRI and Random: deposited; balanced Hi-C pairs |
| 4c | germline and soma models |
| 4d | germline model, fixed edge clusters |
| 4e, 4f, 4g; Supp. 8, 9 | fixed edge clusters and topics; Fig. 4g content (`configs/fly/figure4g_*`); Enrichr results |
| 5a | five germline models; RNA h5ad |
| 5c | germline model; edge clusters of Fig. 5c (Mondo in EC11; Fig. 4 uses another numbering, Mondo in EC9) |
| 6b | RNA/ATAC h5ad, DTW alignment |
| 6c | germline and soma models, CellChat and LIANA+ outputs |
| 6d | as 6c, plus the five germline models |
| Supp. 7 | SCENIC+ dAWA eRegulons, scDoRI stage networks, random networks |
| Supp. 10 | seed-2025 model and its two ablations; SCENIC+ eRegulons |
| Supp. 11 | five germline models, ablation and prior-control models; SCENIC+ and scDoRI peak table |
| Supp. 12-14 | early/late calls under the parallel and sequential alignments; alignment benchmark |

## Checkpoint line: Supplementary Tables

The computed cells of Supplementary Tables 2 and 4-9 are rebuilt (`panels/tables.py`) and compared cell by cell with
the published tables (`source_data/tables/`). Tables 1, 3 and 10-13 (text, literature, settings, reagents) are not
recomputed; the TF lists of Tables 2 and 6 and the literature pairs of Tables 8-9 are inputs.

| Table | Recomputed |
|---|---|
| 2 | rank of each Fig. 4g TF in the edge clusters of its stage |
| 4 | TFs whose network peak is >= 3 stages from their RNA peak and that are hubs of an edge cluster active in their peak window; offsets in the five models |
| 5 | kuz and mamo ranks of DRAFT (Fig. 5c consensus) and its two ablations |
| 6 | network and expression peaks of the selected TFs in the five models |
| 7 | stage pairing of the two lineages |
| 8 | literature pairs listed and evaluated per pathway |
| 9 | literature pairs placed early, with model counts and peak stages |
