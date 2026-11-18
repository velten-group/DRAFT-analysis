# Input data (checkpoint line)

The five archives of the Zenodo deposit ([10.5281/zenodo.23678662](https://doi.org/10.5281/zenodo.23678662)) unpack
into this folder (`tar -xzf <archive> -C data`); `draft-analysis data --download --verify` then fetches the scMTNI
evaluator and checks every file against `manifest.tsv`. The quick line needs none of these files.

```
data/
  simulation/datasets/     simulated datasets: RNA, ATAC, priors, ground-truth GRNs, gene modules, settings.json;
                           <design>/<case>/seed_<n>: linear (main design and its sensitivity cases), random_network,
                           prior_size, branching (branching/seed_<n>)
  simulation/fits/         final TF-TG scores of each benchmark fit (<design>/<case>/<method>/), indexed by
                           simulation/fits.tsv
  fly/raw/                 snRNA/snATAC h5ad, cell annotation, TF/TG/RE lists, TF-RE and RE-TG priors, LR pairs, Hi-C
  fly/checkpoints/         germ_seed2025.pt (manuscript germline model), germ_seed2024/2026/2027/2028.pt (replicate
                           germline models), soma.pt and soma_graphs.pt (soma model and its input graphs);
                           ablation/ and prior_*/: control models of Supp. Figs. 10-11 and their priors
  fly/frozen_numbering/    fixed edge clusters, topics and TF list
  fly/external/            outputs of other software (scDoRI, SCENIC+, CellChat, LIANA+, Enrichr) and the Hi-C
                           evaluation inputs
  mouse/checkpoint/        grn_best.pt (unpaired mouse model)
  mouse/raw/               cluster expression tables, lineage, GO library, scMTNI+Prior published networks
  mouse/official_aupr/     scMTNI gold standards and node lists; source/ (official evaluator) is downloaded
  mouse/native_scores/     dictys/ and scenic/: native edge scores per cluster; features/: their common feature axes
  mouse/frozen_numbering/  fixed edge clusters, topics and GO tables
  mouse/random_distance_inputs.json   Random control (axes, edge budgets, genes, seed)
```

Public sources: the SCENIC+ stage eRegulons are from the dAWA eRegulon table (dawa.cos.uni-heidelberg.de, DAR = yes,
DEG = yes); the mouse gold standards, node lists and scMTNI+Prior networks are from the scMTNI source data (Zenodo
7879228); the official evaluator (`prepareAUClistfile_intersect.py`, `auc.jar`) is downloaded from
github.com/Roy-lab/scMTNI (commit c4d48506, `Scripts/Evaluation/`).

Columns of `manifest.tsv`: `group`, `path`, `level` (raw, checkpoint, frozen, external; third-party = downloaded by
`draft-analysis data --download`), `used_by` (figures), `origin`, `bytes`, `sha256`.
