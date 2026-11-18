# DRAFT manuscript analyses

## Introduction

This repository reproduces the figures and tables of the DRAFT manuscript
([bioRxiv](https://doi.org/10.64898/2026.11.17.767145)) in two ways:

- **Quick line**: redraws the figures and rebuilds the Source Data workbooks and supplementary tables from the
  published values included in this repository.
- **Checkpoint line**: recomputes the figure panels and supplementary tables from the trained DRAFT models
  (checkpoints), the saved predictions of the simulation benchmark and the input data, and compares every result with
  the published values.

To infer GRNs with DRAFT on your own data, use the DRAFT project: [github.com/velten-group/DRAFT](https://github.com/velten-group/DRAFT).

## One command (Linux, macOS)

```bash
git clone https://github.com/velten-group/DRAFT-analysis.git && cd DRAFT-analysis
bash reproduce.sh      # about 4 hours; ends with REPRODUCED or NOT REPRODUCED (log: reproduce.log)
```

`reproduce.sh` downloads the five data archives from Zenodo (or uses them if they are already in this folder), sets
up the environment (with conda: Python 3.12, R and Java 17 in `.tools/`), checks every input and runs the checkpoint
line. The same steps one by one are below.

## Install

Python 3.11 or newer (3.12 used). The checkpoint line also needs R (`Rscript`) and Java 17 (`java`, or set `DRAFT_JAVA`).

```bash
python -m venv .venv && source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e . --no-deps
```

If packages are also installed in your user site-packages (`~/.local`), set `PYTHONNOUSERSITE=1` before installing and
running (conda: `conda env config vars set PYTHONNOUSERSITE=1`), so that only the new environment is used.

## Quick line (no download, a few seconds)

```bash
draft-analysis quick
```

| Output in `results/quick/` | Figures and tables |
|---|---|
| `figures/` | data panels of Figs. 2-6 |
| `source_data_workbooks/` | Source Data of Figs. 2-6 and Supplementary Figs. 1-14 |
| `supplementary_tables/` | Supplementary Tables 2, 4-9 (computed columns) |

`draft-analysis quick --figures 4 5` draws only the listed figures.

## Checkpoint line (about 4 hours)

Download the five `draft_analysis_data_*.tar.gz` archives from Zenodo
([10.5281/zenodo.23678662](https://doi.org/10.5281/zenodo.23678662)) into the top folder
of this repository (next to this README), then run from that folder:

```bash
for a in draft_analysis_data_*.tar.gz; do tar -xzf "$a" -C data; done   # unpacks into data/
draft-analysis data --download --verify
draft-analysis checkpoint
```

| Output in `results/checkpoint/` | Figures and tables |
|---|---|
| `panels/main/` | Figs. 2, 3, 4, 5a, 5c, 6 |
| `panels/supplementary/` | Supplementary Figs. 1-14 |
| `panels/tables/` | Supplementary Tables 2, 4-9 |
| `figures/` | Figs. 2-6 |
| `comparison_with_source_data.tsv` | every table above against the published values |

### Running part of the checkpoint line

`draft-analysis checkpoint` runs the stages below in this order, writing their outputs to `results/checkpoint/`, then
builds every table from those outputs, compares it with the published values and draws the figures.

| Stage | Computes | Used for |
|---|---|---|
| `simulation` | metrics of every simulation benchmark fit, from its saved predictions | Fig. 2; Supp. Figs. 1-4 |
| `stage_distances` | stage embeddings, TF-TG distances between stages | Fig. 4c |
| `hic_benchmark` | DRAFT and SCENIC+ scores of the Hi-C candidates | Fig. 4b |
| `networks` | stage networks of the germline and soma models | later stages |
| `edge_clusters` | edge clusters and topics, checked against the published numbering | Fig. 4e-g; Supp. Figs. 8, 9; Supp. Table 2 |
| `ovo_regulon` | ovo eRegulon | Fig. 4d |
| `mondo_regulon` | Mondo target lists per stage | Fig. 5c |
| `mondo_target_ranks` | kuz and mamo ranks of DRAFT and its ablations | Supp. Table 5 |
| `cell_communication` | receptor scores and early/late calls | Fig. 6 |
| `replicate_models` | networks and calls of the four other germline models | Figs. 5a, 6d; Supp. Tables 4, 6, 9 |
| `control_models` | networks of the ablation and prior-control models | Supp. Figs. 10, 11 |
| `communication_checks` | call stability across models, alignment benchmark | Supp. Figs. 12-14 |
| `mouse_networks` | mouse networks, edge clusters, topics and GO terms | Fig. 3c-e; Supp. Figs. 5, 6 |
| `mouse_benchmark` | mouse AUPR and AUROC with the official scMTNI evaluator (Java) | Fig. 3b |

`--stages` runs only the listed stages, then still builds and compares every table; it does not run the stages they
read from, so it is for resuming an interrupted run or redoing one part after a full run. `--no-run` runs no stage.

```bash
draft-analysis checkpoint --stages control_models communication_checks mouse_networks mouse_benchmark   # resume after an interruption in control_models
draft-analysis checkpoint --stages mouse_networks mouse_benchmark   # redo the mouse part
draft-analysis checkpoint --no-run                                  # rebuild tables and figures only
```

### Differences from the published values

Values from the checkpoint line can differ from the published values at the order of 1e-4: floating-point rounding
differs between machines (the published values were computed on a GPU cluster), and at a top-k cutoff or a near tie
this can change which edge or rank is chosen. The comparison allows for this; the tolerance of each table is listed in
`comparison_with_source_data.tsv`.

## Baseline methods

`baselines/` holds, for reference, the code and settings that ran the methods DRAFT is compared with. The two lines
do not run it: they use the deposited outputs of those runs. Versions and sources are in
[`baselines/README.md`](baselines/README.md).

### Which baseline is used where

| Baseline | Version | Figures and tables | Deposited output (`data/`) |
|---|---|---|---|
| SCENIC+ region-to-gene, fitted per stage | SCENIC+ 840dab85 | Fig. 4b | `fly/external/scenicplus_native_r2g/`, `fly/external/hic_within_prior/` |
| SCENIC+ eRegulons of the original study (downloaded) | as published | Supp. Figs. 7, 10, 11a; Supp. Table 4 | `fly/external/scenicplus_published_eregulons/`, `fly/external/tf_peak_candidate_table/` |
| scDoRI | v1.0.0 | Fig. 4b; Supp. Figs. 7, 11a | `fly/external/scdori_stage_networks/`, `fly/external/hic_within_prior/`, `fly/external/tf_peak_candidate_table/` |
| CellChat, LIANA+ | CellChat 75253cd0, LIANA+ 1.10.1.dev0 | Fig. 6c; Supp. Figs. 12-14 | `fly/external/ccc_regulatory_benchmark/` |
| Dictys, SCENIC (mouse) | Dictys f409a632, pySCENIC 0.12.1 | Fig. 3b, c | `mouse/native_scores/` |
| scMTNI+Prior (published networks, downloaded) | as published | Fig. 3b, c | `mouse/raw/scMTNI_plus_prior/` |
| SCENIC+, scMTNI, scDoRI, Dictys (simulated data) | SCENIC+ 840dab85, scMTNI c4d48506, scDoRI f02cf47d, Dictys f409a632 | Fig. 2; Supp. Figs. 1-4 | `simulation/fits/` |

## Citation

Zhang Z, Santhi Sekhar P, van Nierop y Sanchez P, Lohmann I, Velten B. DRAFT: graph-recurrent learning of
dynamic gene regulatory networks along single-cell multi-omics trajectories. bioRxiv (2026).
[doi:10.64898/2026.11.17.767145](https://doi.org/10.64898/2026.11.17.767145)

## License

MIT (see `LICENSE`).
