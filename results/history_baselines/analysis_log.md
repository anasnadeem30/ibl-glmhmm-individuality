# Results log

## Milestone 1, step 1: cohort and data (2026-10-16)
- IBL public server (openalyx) needs username `intbrainlab`, password `international` (the brief's snippet omitted the username).
- Cohort: 33 mice, 9 labs, >=30 biased/ephys sessions each, max 5 mice per lab (`cohort.csv`).
- 1,552 sessions, 1,154,948 trials. 193 old (2019) sessions stored in per-file `.npy` format and were recovered via `load_object`.
- No-go trials (choice == 0): 0.5%, dropped. 1,149,100 trials remain (`data/trials_clean.parquet`).
- IBL coding check: choice -1 is the rewarded response to a right-side stimulus, so `right = (choice == -1)`.
- Per-mouse accuracy at non-zero contrast: min 0.68, median 0.85, max 0.93.
- Figure: `figures/psychometric_cohort.png`. Curves are sigmoidal; the 20/80 blocks shift the curve in the expected direction (mice use the block prior).
- ONE downloads cache to `C:\Users\Anas\Downloads\ONE\`.

## Milestone 1, step 2: plain GLM baseline (2026-10-16)
Per-mouse logistic regression, 5-fold CV grouped by session. Features: stimulus (signed contrast), previous choice, win-stay-lose-switch (prev choice x prev reward), optional true block prior. Script `glm_baseline.py`, output `glm_baseline_cv.csv`, `glm_weights.csv`, `figures/glm_baseline.png`.
Mean held-out log-likelihood (bits/trial) and accuracy:
| model | LL | acc |
|---|---|---|
| bias only | -0.987 | |
| stim | -0.714 | 0.783 |
| stim + prev | -0.640 | 0.804 |
| stim + prev + wsls | -0.638 | 0.803 |
| + block prior | -0.619 | 0.811 |
Mean weights (stim/prev/wsls/bias): 5.86 / 0.72 / 0.13 / 0.02. These numbers are the bar a GLM-HMM must beat. The block-prior feature uses the true block label, which the mouse does not see, so it is a reference only.

## Milestone 1, step 3: GLM-HMM implementation and pilot (2026-10-16)
- `glmhmm.py`: own EM implementation (numba forward-backward, weighted logistic M-step, sessions as independent sequences).
- Synthetic check (`test_synth.py`): 2-state ground truth (weights 6/0/0 and 0.5/0/2, stay-prob .97/.96) recovered as 6.18/.5/2.0 and .956/.97; state decoding 92.8%; EM log-lik monotone.
- Pilot, 2 mice, 20% of sessions held out, best of 3 restarts, bits/trial on held-out sessions:
  - KS019: K=1 -0.645, K=2 -0.531, K=3 -0.520, K=4 -0.514
  - CSHL_014: K=1 -0.680, K=2 -0.633, K=3 -0.588, K=4 -0.583
- Cost: about 1-2.5 min per mouse per fold for K=1..4 x 3 restarts. A full 33-mouse x 5-fold sweep needs parallelism.

## Milestone 1, step 4: choosing K (2026-10-17)
Full sweep (`sweep_k.py`, `analyze_k.py`): 33 mice, 5-fold CV grouped by session, K=1..4, best of 3 restarts on training LL. 1580 s on 20 cores.
Mean held-out bits/trial: K=1 -0.6385, K=2 -0.5488, K=3 -0.5258, K=4 -0.5149. Gains: +0.090, +0.023, +0.011.
K=4 is the best K for all 33 mice; every step up in K helped every mouse. No clear elbow at K=3. Within 0.005 bits of best: K=3 for 3 mice, K=4 for 30.
Not yet tried: K>=5 (would show whether the curve plateaus). Figure: `figures/choose_K.png`.
Note: the claim that Ashwood et al. chose 3 states was recalled from memory and must be verified against the paper.

## Milestone 1, step 5: inspecting the states (2026-10-17)
Full-data fits (`fit_full.py`, 5 restarts, best by LL; `inspect_states.py`; `states_full.csv`, `figures/states_K3_K4.png`). States ranked within each mouse by stimulus weight. Medians over 33 mice:
- K=3: stim 21.0 / 3.5 / 1.9; occupancy .58 / .22 / .16 (sum <1 only because medians); dwell 27 / 20 / 10 trials; |bias|>1 in 9 / 32 / 23 mice.
- K=4: stim 25.0 / 14.8 / 3.0 / 1.3; occupancy .38 / .34 / .16 / .10; dwell 11 / 21 / 15 / 3 trials.
- Pattern: one or two high-stimulus-weight "engaged-like" states, plus low-stimulus states with large bias to the left or right depending on the mouse (sign varies by mouse).
- Restart reproducibility (first look, same cohort): best vs 2nd-best restart LL differ by median 0.0000, max 0.0027 bits/trial. This compares likelihoods only; whether the restarts found the same states (weights) has not been checked yet.

## Milestone 1, step 6: state timelines (2026-10-17)
`timelines.py`, `figures/state_timelines.png` (K=3, mice KS019 and CSHL_014, middle session of each). Within a session the posterior switches between states every ~10-30 trials. Across sessions, the engaged-state share falls late in the mouse's history in both mice (KS019 after ~session 50, CSHL_014 with a dip around 40 and a low stretch 45-55). Observation from 2 mice only; not tested. Possible confounds: later sessions may differ in task conditions (e.g. protocol type, recording sessions), and the model is fit on all sessions pooled, so a pooled fit can blur a changing strategy. This is the starting point for the Milestone 3 drift analysis, not a finding.

## Milestone 1, step 7: K = 5 and 6 (2026-10-17)
`sweep_k56.py`, `analyze_k56.py`, `sweep_k_summary.csv`, `figures/choose_K.png`. Same CV protocol as K=1..4.
Mean held-out bits/trial: K=1 -0.6385, 2 -0.5488, 3 -0.5258, 4 -0.5149, 5 -0.5105, 6 -0.5072. Gains per step: .090, .023, .011, .0044, .0033.
K=6 is best for 32 of 33 mice. K=5 beats K=4 in 31/33 mice (median +0.0043); K=6 beats K=5 in 32/33 (median +0.0025).
No plateau: the score keeps rising, with shrinking gains. Choice: K=3 as primary model (interpretable, matches the published 3-state framing to be verified), K=4 as robustness check. Not claiming 3 is the "true" number of states.

## Milestone 2a: individuality test (2026-10-17)
`individuality.py` (165 fold models, K=1/3/4, best of 3 restarts), `analyze_ind.py`; `individuality_matrix.csv`, `individuality_summary.csv`, `figures/individuality_K3.png`. 5-fold CV grouped by session (GroupKFold, not temporal). Model of mouse i scored on mouse j's held-out sessions of the same fold; own = diagonal, compared against the mean of the other 32 models on the same test mouse.
| K | own beats others | mean advantage (bits/trial) | mean own rank (chance 17) | rank 1 | top 5 | Wilcoxon p |
|---|---|---|---|---|---|---|
| 1 | 33/33 | +0.072 | 1.9 | 21 | 32 | 2.3e-10 |
| 3 | 33/33 | +0.056 | 1.6 | 30 | 32 | 2.3e-10 |
| 4 | 33/33 | +0.055 | 1.6 | 28 | 31 | 2.3e-10 |
Lab control: same-lab models are not better than other-lab models (same-lab minus other-lab = -0.0015 bits at K=3), so the effect is not a lab/rig effect.
Reading: each mouse is best predicted by its own model. The plain GLM shows it too (even a larger average gap), so individuality does not require hidden states; the GLM-HMM makes own-model the top rank more consistently (30 vs 21 of 33) and shrinks the gap, consistent with the states capturing some shared structure.
Caveats: (1) folds are random across sessions, so own-mouse test sessions are interleaved in time with its training sessions; part of the advantage could be time-local state/drift rather than a stable trait. A train-early/test-late version is needed. (2) Only 33 mice from 9 labs; one session-split scheme. (3) Not yet separated into which parameters (bias, stimulus sensitivity, history, transitions) carry the individuality.

## Milestone 2b: temporal individuality and restart reproducibility (2026-10-17)
**Temporal individuality** (`individuality_temporal.py`, `analyze_ind_t.py`, `individuality_temporal_matrix.csv`): each mouse's model fit on its first 60% of sessions (by date, best of 5 restarts), scored on every mouse's last 40%.
| K | own beats others | mean advantage (bits) | own rank mean (chance 17) | rank 1 | top 5 |
|---|---|---|---|---|---|
| 1 | 31/33 | +0.084 | 7.6 | 3 | 18 |
| 3 | 31/33 | +0.061 | 5.3 | 9 | 24 |
| 4 | 31/33 | +0.057 | 5.2 | 10 | 23 |
Compared with the random-split test (30/33 at rank 1 for K=3), the average advantage is similar but "own model is the single best" drops a lot. Individuality persists across time but is less sharp. Own-model score falls from -0.510 (early train) to -0.556 (late test) at K=3; this mixes ordinary generalization gap with drift and cannot be separated here. Same-lab vs other-lab difference still ~0 (-0.003).
Mice where own model is not best at K=3 include several CSHL animals (ranks 2-5); not investigated.
**Restart reproducibility** (`repro_restarts.py`, `repro_restarts.csv`, full-data fits, 5 restarts): states matched by Hungarian assignment on weights, Euclidean distance. Restarts with LL within 0.001 bits of the best recover the same states (median distance 0.008 at K=3, 0.049 at K=4). Restarts with worse LL land on different states (median ~4.3-4.6), comparable to between-mouse distance (median 5.7 at K=3, 6.0 at K=4). Share of restarts within 0.05 of best weights: 46% (K=3), 26% (K=4). Caveat: distance is dominated by the large stimulus weights; standardized comparison and a split-half-sessions test not yet run.

## Milestone 2c: split-half reproducibility of states (2026-10-17)
`split_half_fit.py` (33 mice x {interleaved, early/late} x 2 halves x K=3,4, best of 5 restarts), `analyze_split_half.py`, `split_half_summary.csv`. States matched by Hungarian assignment on weights standardized by cross-mouse SD.
| split | K | median dist same mouse | median dist different mice | nearest other-half is same mouse (chance 3%) | mean rank of same mouse (chance 17) | dwell-time r |
|---|---|---|---|---|---|---|
| interleaved | 3 | 0.66 | 1.54 | 39% / 48% | 6.5 | 0.59 |
| interleaved | 4 | 1.08 | 1.59 | 33% / 48% | 8.3 | 0.58 |
| early/late | 3 | 1.06 | 1.40 | 24% / 15% | 10.8 | 0.43 |
| early/late | 4 | 0.95 | 1.37 | 24% / 30% | 8.7 | 0.35 |
Same-mouse halves are closer than other-mouse halves in 31/33 mice (interleaved K=3) and 27/33 (early/late K=3). States are partly reproducible and partly mouse-specific, but noisy: even for the best case, the same mouse is the nearest match in under half of mice, and the early/late split is clearly worse than the interleaved one.
Caveats: each half has half the data, so some of the mismatch is estimation noise rather than real change; the dwell-time correlation pools states that have different typical dwell times, so it is inflated by state rank and should not be read as mouse-specific stability.

## Literature check and two controls (2026-10-17)
Papers read (via PMC): Ashwood et al. 2022 (Nat Neurosci; PMC8890994) and Mohammadi, Ashwood & Pillow 2025 (Nat Commun; PMC12749821). Key facts from them:
- Ashwood: 37 IBL mice, animals with >=3000 trials, ONLY the first 90 trials of each session (50/50 block); inputs = stimulus, bias, previous choice, previous stimulus side; 5-fold CV; 3-state model chosen "for reasons of simplicity and interpretability" with improvement "approximately leveling off at 3"; test LL gain 0.13 bits/trial of 3-state over 1-state; median mouse 69% engaged, median dwell 24 (engaged) vs 13 and 12 (biased) trials; states "remarkably consistent across mice" (engaged / biased-left / biased-right) but individual fits per mouse; analysed data after learning.
- Mohammadi et al.: IBL data with biased blocks, 37 mice with >=30 sessions; dynamic GLM-HMM (weights and transitions evolve across sessions); four states (Engaged-L, Engaged-R, Biased-L, Biased-R). I did not verify what drift findings they report.
- IBL 2021 eLife: after training, no significant differences between labs (consistent with our null lab effect).

**Block confound control** (`block_control.py`, `block_control_states.csv`): our fits use all trials including 20/80 blocks without block information as input. Low-stimulus side-biased states have higher occupancy in the block favouring their side in 46/47 states (K=3; mean +0.15 on mean occupancy 0.21) and 54/58 (K=4). So part of the "biased" states reflects block-tracking, not only disengagement. TV distance between state mix in right- vs left-favouring blocks: median 0.185 (K=3).

**Closer replication** (`replication_5050.py`, `analyze_replication.py`): first 90 trials of each session at 50/50 block only (138,559 trials, median 4,137 per mouse), inputs stimulus/bias/prev choice/prev stimulus side, 5-fold CV by session.
Held-out bits/trial: K=1 -0.7045, K=2 -0.6296, K=3 -0.6093, K=4 -0.6010. Gain K=3 vs K=1: 0.095 (Ashwood: 0.13). Gains per step 0.075, 0.020, 0.008. K=3 beats K=1 in 33/33 mice; K=4 beats K=3 in 26/33.
K=3 full-data medians: engaged occupancy 0.59 (Ashwood 0.69), dwell engaged 22.8 / others 17.8 and 6.9 (Ashwood 24 / 13 and 12); 24/33 mice have two non-engaged states with opposite-sign bias.
Reading: qualitatively consistent with Ashwood et al. (engaged state plus side-biased states, elbow near K=3), quantitatively somewhat weaker (smaller gain, lower engaged occupancy). Differences in cohort, session selection, fitting procedure (they initialised from a global fit) and features are not controlled.

## Next-project pilot: action-kernel baseline vs GLM-HMM (2026-10-18)
`pilot_ak.py` -> `pilot_ak.csv`; design in `Next_Project_Design.md`. 5 mice (KS016, CSHL051, DY_006, NYU-09, ZM_1746), 5-fold CV grouped by session, all trials, K=3, best of 3 restarts. Action kernel = exp-weighted mean of past choices (decay tau chosen on training LL from 1.5..40 trials) added as a GLM regressor. PILOT ONLY, 5 mice, not a result.
Mean held-out bits/trial: M0 GLM (stim/prev/wsls/bias) -0.669; +action kernel -0.584; +kernel+stimulus kernel -0.578; +true block prior -0.653; +kernel+block -0.584; GLM-HMM K=3 -0.522; GLM-HMM K=3 + kernel -0.495.
Kernel captured 31-79% (per mouse) of the GLM-HMM's gain over M0 (0.79, 0.78, 0.61, 0.31, 0.40). GLM-HMM still beat the best stateless model in all 5 mice (+0.020 to +0.114). GLM-HMM + kernel beat GLM-HMM alone in all 5.
Caveats: kernel is additive only; one timescale (DY_006 chose 40/20, near grid edge); stateless models are not tuned as hard as the HMM; 5 mice.

## Next-project main run: stateless baselines vs GLM-HMM, 33 mice (2026-10-18)
`feats_ak.py`, `full_ak.py` -> `full_ak.csv`, `full_ak_bits.csv`. 5-fold CV grouped by session, all trials, K=3, best of 3 restarts, 3786 s on 24 cores. Some sklearn lbfgs convergence warnings (max_iter=500) in the wider GLMs; not yet quantified.
Feature sets: X0 = stim, prev, wsls, bias. S1 = X0 + 5 action kernels (EWMA of past choices, tau 2/5/10/20/40). S2 = S1 + stimulus kernels (tau 5/20) + EWMA of past correctness (tau 5/20) + stim x accuracy gain terms + within-session trial fraction.
Mean held-out bits/trial: GLM X0 -0.6385, S1 -0.5696, S2 -0.5504; GLM X0 + true block prior -0.6198; HMM3 X0 -0.5258, S1 -0.4963, S2 -0.4914.
- Total HMM3(X0) gain over GLM(X0): +0.113 (33/33).
- Stateless S1 captures median 64% of it (IQR 47-79%); stateless S2 captures median 83% (IQR 72-92%).
- HMM3(X0) still beats GLM(S2) in 32/33 mice, mean +0.0245 (median +0.016); close in many mice (<0.01 in 10).
- HMM3(S2) beats GLM(S2) in 33/33, mean +0.059, median +0.054; HMM3(S2) beats HMM3(X0) in 33/33 (+0.034): history/engagement covariates and hidden states are complementary.
- Caveat not yet controlled: HMM3(S2) has 3x the weights of GLM(S2), so the +0.059 may reflect extra capacity (a mixture) and not temporal persistence. Control queued: iid mixture of 3 GLMs (no Markov persistence).

## Mimicry / calibration simulation (2026-10-18)
`sim_mimic.py` -> `sim_mimic.csv`. 8 mice (CSHL051, DY_006, KS016, NYU-09, ZM_1746, CSHL_014, SWC_021, ibl_witten_15) x 2 replicates x 2 generators; closed-loop replay of the real stimulus sequence and true correct side; same 5-fold CV pipeline as the real data. Generators: (a) stateless GLM(S2) fitted to the mouse; (b) 3-state GLM-HMM on X0 fitted to the mouse. Mean bits/trial:
- Stateless-generated data: HMM3(S2) minus GLM(S2) = -0.001 (all 8 mice within +-0.002); HMM3(X0) minus GLM(S2) = -0.045. So the pipeline does NOT produce a hidden-state advantage from purely stateless data (false-positive control passes).
- Real data, same 8 mice: HMM3(S2) minus GLM(S2) = +0.053; HMM3(X0) minus GLM(S2) = +0.017.
- HMM-generated data: stateless S2 captured 52% of the HMM's gain (36-76% by mouse), real data 88%. HMM3(X0) over GLM(S2) = +0.063 in HMM-generated data vs +0.017 in real. Real data look like a mix: more stateless-history structure than a pure 3-state HMM, plus structure beyond it.
Limits: 8 mice, 2 replicates, one stateless generator (S2 itself), generators tuned on the same model families being compared.

## Persistence control: iid mixture of GLMs vs GLM-HMM, 33 mice (2026-10-18)
`pers_ctrl.py` (MixGLM: K=3 GLM mixture, i.i.d. state assignment each trial, rows of A = marginal state distribution) -> `pers_ctrl.csv`. Same CV, 3 restarts, 2688 s.
Mean bits/trial: MIX3(X0) -0.5688, MIX3(S2) -0.4965 (vs HMM3(X0) -0.5258, HMM3(S2) -0.4914, GLM(S2) -0.5504).
- X0 features: HMM3 beats MIX3 by +0.043 (33/33): temporal persistence matters when the covariates are poor.
- S2 features: MIX3(S2) beats GLM(S2) by +0.054 (33/33); HMM3(S2) beats MIX3(S2) by only +0.005 (median +0.004; 30/33 mice, p=1e-7). The iid mixture reaches median 92% (IQR 85-97%) of HMM3(S2)'s gain over GLM(S2).
- Reading: once observable history/engagement covariates are in the model, most remaining gain is trial-by-trial heterogeneity (multiple response policies), not temporal persistence of hidden states. Persistence adds a small but consistent amount.
Caveats: S2 history covariates themselves carry temporal structure, so "persistence" here means beyond what S2 captures; K=3 only; one mixture family; EM local optima (3 restarts); mice treated as independent units.

## Calibration of the persistence test (sim_mix.py, 33 mice x 3 generators, 1 replicate, 2026-10-19)
Closed-loop replay of real stimuli; generators fit to each mouse. Bits/trial, 5-fold CV by session, K=3.
| generator | HMM3(S2)-MIX3(S2) | HMM3(X0)-MIX3(X0) | MIX share of HMM3(S2) gain over GLM(S2) |
|---|---|---|---|
| stateless S2 GLM | -0.0003 | +0.076 | n/a (gains ~0) |
| iid mixture (no persistence) | -0.001 (HMM>MIX in 4/33) | -0.0006 | 101% |
| 3-state HMM (persistent) | +0.024 (33/33) | +0.064 | 62% |
| REAL data | +0.005 (30/33) | +0.043 | 92% |
Reading: (1) with rich covariates the test shows no false persistence on stateless or iid-mixture data; (2) with only X0, HMM-vs-mixture "persistence" appears even in stateless data (+0.076) - unmodelled history masquerades as persistence, so X0-only comparisons overstate it; (3) a truly persistent HMM would give +0.024 and 62%; real mice sit at +0.005 and 92%, far nearer the iid-mixture end. Caveat: 1 replicate; generators are X0-only HMM/mixture, so real states with history-dependent emissions are not covered.

## K sweep (ksweep.py, 33 mice x 5 folds, S2 inputs; bits/trial)
K | HMM-MIX mean (median) | HMM>MIX | mixture share of HMM gain (median)
2 | +0.0026 (0.0023) | 29/33 | 96%
3 | +0.0051 (0.0039) | 30/33 | 92%
4 | +0.0067 (0.0060) | 31/33 | 90%
5 | +0.0071 (0.0075) | 31/33 | 89%
Persistence gap grows slowly with K but stays well below the +0.024 produced by simulated persistent-HMM data (K=3). HMM-gain over S2 GLM: 0.055/0.059/0.061/0.061. No K selection beyond 5.
