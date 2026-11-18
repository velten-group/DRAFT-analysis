# Baseline methods

The code and settings used to run the comparison methods, for reference. The quick and checkpoint lines do not run
this code: they read the deposited outputs of these runs (`data/`). Each method ran in its own environment
(`requirements.txt` and `upstream_api.json` in its folder pin the versions); paths in the code are relative to this
folder.

| Method | Version | Code | Deposited output (`data/`) | Used in |
|---|---|---|---|---|
| scDoRI (germline and soma; stage networks) | scDoRI v1.0.0 (f02cf47d) | `comparisons/scdori_mondo/` | `fly/external/scdori_stage_networks/` | Fig. 4b, Supp. Figs. 7, 11a |
| SCENIC+ region-to-gene, per stage | SCENIC+ 840dab85, pycisTopic 787ce422 | `comparisons/scenicplus_r2g/` | `fly/external/scenicplus_native_r2g/`, `fly/external/hic_within_prior/scenicplus_stage_r2g/` | Fig. 4b |
| Hi-C benchmark: scores of DRAFT, SCENIC+, scDoRI and the random control on the shared prior | - | `comparisons/hic_within_prior/` | `fly/external/hic_within_prior/` | Fig. 4b |
| CellChat, LIANA+ | CellChat 75253cd0 (R 4.6.1), LIANA+ 1.10.1.dev0 (08590dbd) | `scripts/` | `fly/external/ccc_regulatory_benchmark/` | Fig. 6, Supp. Figs. 12-14 |
| Dictys and SCENIC (mouse) | Dictys f409a632, pySCENIC 0.12.1 | `comparisons/mouse_benchmark/`, `comparisons/mouse_dictys/`, `comparisons/mouse_scenic/` | `mouse/native_scores/` | Fig. 3b, c |
| TF network and expression peaks of SCENIC+ and scDoRI | - | `comparisons/tf_peak_offsets/` | `fly/external/tf_peak_candidate_table/` | Supp. Fig. 11a, Supp. Table 4 |
| SCENIC+ (TF-to-gene), scMTNI, scDoRI and Dictys on the simulated datasets | SCENIC+ 840dab85, scMTNI c4d48506, scDoRI f02cf47d, Dictys f409a632 | `comparisons/simulation_benchmark/run.py` with `comparisons/simulation_<method>/`; prior-size controls `comparisons/simulation_main_prior_controls/`; branching lineage `comparisons/simulation_branch/` | `simulation/fits/` | Fig. 2, Supp. Figs. 1-4 |

Downloaded, not run: the SCENIC+ eRegulons of the original study (dAWA), the scMTNI+Prior networks and the gold
standards (scMTNI source data, Zenodo 7879228), and the Enrichr GO results.
