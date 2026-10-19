# Hidden decision states and individuality in mouse perceptual decision-making

An independent re-analysis of public behavioral data from the International Brain Laboratory (IBL) visual decision task. I fit hidden-state models (GLM-HMMs) to 1,154,948 trials from 33 mice, reproduce the main qualitative result of Ashwood et al. (2022), and then ask whether the inferred states are individual to each mouse and whether they are reproducible.

**Full write-up: [`report/report.pdf`](report/report.pdf)** (6 pages: background, methods, results, comparison with the published work, limitations).

## Scope, stated plainly

- The first part of this repository (the report in `report/`) is a **replication with a small extension**, done on public data by one person. It is not peer reviewed and **I do not claim novelty for it**. A second, separate analysis (see `history_baselines/` below) makes a more specific claim and is described as a preprint draft. Ashwood et al. (2022) already reported mouse-specific fits, and Mohammadi, Ashwood & Pillow (2025) analyze these data with a dynamic GLM-HMM.
- The planned analysis of **drift across days was not completed**. I report only indirect observations, not tests.
- The fitting code was validated on synthetic data, and every number in the report is produced by a script in this repository.

## Main results

| Question | Result |
|---|---|
| Do hidden states predict held-out sessions better than a single GLM? | Yes, in 33/33 mice. K=3 gains +0.113 bits/trial over K=1 (all trials) and +0.095 on an Ashwood-style 50/50 subset (published: +0.13). |
| Do the states match the published ones? | Qualitatively yes (one engaged state, plus left/right-biased states); quantitatively weaker (median engaged occupancy 59% vs 69% published). |
| Caveat found along the way | With all trials, the "biased" states track the 20/80 block structure (46/47 states at K=3), so they are not purely disengagement. |
| Is each mouse best predicted by its own model? | Yes: 33/33 (random session split) and 31/33 (early-train / late-test). Not a lab effect. A plain GLM shows it too, so hidden states are not required. Sharper in the random split (own model ranks 1st in 30/33 mice) than in the early/late split (9/33). |
| Are the states reproducible? | Partly. Restarts reaching the same likelihood find the same states. Split-half fits of one mouse resemble each other more than other mice, but the nearest match is the same mouse in only 39-48% of cases (chance 3%). |
| Number of states | Held-out score keeps improving through K=6 with shrinking gains. K=3 was chosen for interpretability, not because the score plateaued. |

## New (Oct 2026): GLM-HMM vs stateless history baselines and a persistence control

Folder `history_baselines/` (code, preprint draft in `history_baselines/paper/`) and `results/history_baselines/` (per-mouse results and a running analysis log).

**Question.** How much of the GLM-HMM's held-out advantage over a plain GLM survives strong stateless baselines (action/stimulus/accuracy history kernels, time in session), and does the rest require temporal persistence of states, or only trial-to-trial mixing of response policies?

| Result (33 mice, 5-fold session-grouped CV, bits/trial) | |
|---|---|
| Stateless history GLM (S2) recovers this share of the 3-state HMM's gain over the 4-input GLM | median 83% (IQR 72-92%) |
| i.i.d. mixture of GLMs (no persistence) recovers this share of the remaining HMM gain | median 92% (IQR 85-97%) |
| HMM minus i.i.d. mixture, full covariates | +0.005 (30/33 mice); +0.0026 to +0.0071 for K=2..5 |
| Same difference on data simulated from a persistent HMM | +0.024 |
| 4-input HMM minus mixture on purely *stateless* simulated data | +0.076 (apparent persistence from unmodelled history) |

Scope and caveats: one dataset and task; all trials after dropping no-go trials (1,149,100 of the 1,154,948 above), including the first 90 trials per session (50/50 prior, 12% of trials) and the later 20/80 blocks; Ashwood et al. analysed only the first-90-trial segment, so absolute numbers are not comparable; one simulated replicate per mouse; K up to 5; hand-designed covariates; a drifting-weights (PsyTrack-style) comparator was not tested; wide-GLM optimiser convergence warnings were not quantified. This does not show that mice lack internal states.

Run (from the repo root, after the pipeline above has produced `data/features.parquet`): `python history_baselines/full_ak.py all full_ak.csv` (and `pers_ctrl.py`, `sim_mix.py`, `ksweep.py`, each taking an output CSV name; the simulation and sweep take hours). `python history_baselines/make_figs.py` rebuilds the figures from `results/history_baselines/`. These scripts were run in the original working folder; only `make_figs.py` was re-run from this repository layout.

## Data

IBL public behavioral data, accessed through the [ONE API](https://github.com/int-brain-lab/ONE) from the public server (`openalyx.internationalbrainlab.org`, public account `intbrainlab`, password `international`, as documented by IBL). Data are downloaded to ONE's cache on your machine and are not stored in this repository.

Cohort: mice with at least 30 biased/ephys-protocol sessions, at most 5 per lab (33 mice, 9 labs, 1,552 sessions). The cohort list is in `results/cohort.csv`.

## Repository layout

```
glmhmm.py        GLM-HMM with Bernoulli emissions, EM with numba forward-backward (written for this project)
scripts/         numbered pipeline, run in order from the repository root
results/         CSV outputs of the analyses (all small)
figures/         figures used in the report
report/          build_report.py and report.pdf
data/            empty; filled by the scripts (git-ignored)
```

| Scripts | Step |
|---|---|
| 01-03 | Select cohort; download trial tables; recover sessions stored in the old per-file format |
| 04-05 | Psychometric curves; baseline GLM with cross-validation (also builds `data/features.parquet`) |
| 06 | Synthetic test: recover a known 2-state model |
| 07-09 | Choose the number of states K (cross-validation, K=1..6) |
| 10-12 | Full-data fits; inspect states; state timelines |
| 13-16 | Individuality test (random split, then early-train / late-test) |
| 17-19 | Reproducibility: random restarts and split-half fits |
| 20-22 | Block-structure control; closer replication on first-90-trial 50/50 data |
| 23 | Extra figures for the report |

## Running it

```
python -m venv .venv && .venv/Scripts/activate      # Python 3.12
pip install -r requirements.txt
python scripts/01_select_cohort.py                   # then 02, 03, ... in order, from the repo root
python report/build_report.py                        # rebuild the PDF (uses Arial from C:/Windows/Fonts)
```

The multi-process scripts (07, 08, 10, 13, 15, 18, 21) use 16 to 20 worker processes; edit `Pool(...)` to match your machine. The longest ones took about 20 to 80 minutes on 20 cores, and the full pipeline takes a few hours. Downloads are the other slow step (about 2 seconds per session with 8 threads).

**Verification status.** Scripts 04, 05, 06, 14, 16 and 23 were re-run from a clean copy of this repository layout and reproduce the numbers in the report. The long-running scripts were run with identical code in the original working folder, before paths were reorganized into `scripts/` and `results/`; I have not re-run them end to end from a fresh clone.

## References

- Ashwood, Z. C., Roy, N. A., Stone, I. R., The International Brain Laboratory, Urai, A. E., Churchland, A. K., Pouget, A., & Pillow, J. W. (2022). Mice alternate between discrete strategies during perceptual decision-making. *Nature Neuroscience* 25, 201-212.
- Mohammadi, Z., Ashwood, Z. C., & Pillow, J. W. (2025). Identifying the factors governing internal state switches during nonstationary sensory decision-making. *Nature Communications*.
- The International Brain Laboratory (2021). Standardized and reproducible measurement of decision-making in mice. *eLife* 10, e63711.

Thanks to the IBL for making these data public.

## License

MIT (see `LICENSE`). The IBL data are subject to IBL's own terms.
