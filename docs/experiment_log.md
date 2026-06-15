# Kaggle Stellar Class Experiment Log

Current best public LB: **0.97044**

Best submission file:

`output/submission_external_5model_blend.csv`

## Active Files

These files are kept in `output/` because they are useful for final submission or future blending.

| Role | File | Notes |
| --- | --- | --- |
| Best submission | `submission_external_5model_blend.csv` | LB 0.97044 |
| Best external OOF/test proba | `oof_external_5model_blend.csv`, `test_proba_external_5model_blend.csv` | Current strongest base |
| Best LGB proba | `oof_lgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_10fold_average.csv` | LGB 10-fold OOF |
| Best LGB test proba | `test_proba_lgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_10fold_average.csv` | Used in blends |
| XGB v1 proba | `oof_xgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_5fold_average.csv` | raw OOF 0.966950 |
| XGB v2 seed63 proba | `oof_xgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_5fold_average_depth5_child3_lambda3_seed63.csv` | raw OOF 0.967585 |
| XGB v2 seed71 proba | `oof_xgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_5fold_average_depth5_child3_lambda3_seed71.csv` | raw OOF 0.967579 |
| XGB v3 proba | `oof_xgb_original_w0p25_spatial_all_density_notebook_features_target_encoding_5fold_average_depth5_child5_lambda4.csv` | raw OOF 0.967584, LB blend lower |
| CatBoost proba | `oof_catboost_original_w0p25_te_5fold.csv` | raw OOF 0.965535, blend weight became 0 |

## Main Results

| Experiment | OOF | LB | Status |
| --- | ---: | ---: | --- |
| Categorical baseline | - | 0.5658 | Old baseline |
| Add redshift | - | 0.62512 | Old baseline |
| Redshift binning | - | 0.88665 | Old baseline |
| Add color features | - | 0.90095 / 0.90371 | Old baseline |
| LightGBM | - | 0.93085 | First strong model |
| Add raw magnitudes | - | 0.95110 | Useful |
| Add spatial features | - | 0.96430 | Major gain |
| Add original data + notebook features + TE | - | 0.96730-0.96764 | Strong LGB baseline |
| Best LGB calibrated | 0.967064 | 0.96764 | Kept as base |
| LGB + XGB v1 | 0.967817 | 0.96814 | Improved |
| LGB + XGB v2 | 0.968112 | 0.96883 | **Current best** |
| LGB + XGB v2 + XGB v3 | 0.968218 | 0.96862 | Rejected: OOF up, LB down |
| LGB + XGB v2 seed63 + XGB v2 seed71 | 0.968212 | TBD | Next submission candidate |
| LGB + XGB v2 + Green Valley XGB | 0.968311 | TBD | New candidate, needs LB check |
| LGB + XGB v2 + XGB v4 shallow | 0.968155 | TBD | Conservative candidate, 184 rows changed |
| External 5-model blend | 0.969853 | 0.97044 | New best from Kaggle GPU/notebook |
| External 5-model + local LGB blend | 0.969940 | 0.97035 | Rejected: LB lower |
| External 5-model calibrated | 0.969856 | TBD | Candidate, 176 rows changed vs external |
| External 5-model + local LR stacker | 0.969770 | TBD | Rejected locally: OOF lower than external |
| Kaggle CatBoost GPU/CPU diverse | 0.964423 | - | Rejected: external blend chose CatBoost weight 0 |
| Probability stacker: LGB + XGB v2 | 0.967965 | TBD | Lower than manual blend |
| Probability stacker: all local proba models | 0.968085 | TBD | Lower than manual blend |
| Submission hard vote softmax | not locally measurable | TBD | Final-polish candidate, 206 rows changed |
| Submission hard vote shifted | not locally measurable | TBD | More aggressive final-polish candidate, 251 rows changed |
| CatBoost blend | 0.967817 | - | Rejected: CatBoost weight 0 |
| RealMLP local smoke test | 0.956646 on 1 fold | - | Rejected locally |

## Current Best Blend

File:

`output/submission_lgb_xgb_xgb2_blend_tiny.csv`

Blend:

- LightGBM: 0.51464
- XGB v2: 0.48536
- QSO multiplier: 1.22
- STAR multiplier: 1.58

## Next Candidate

Generated one more XGB v2-style model with a different seed:

- `max_depth=5`
- `min_child_weight=3`
- `reg_lambda=3`
- `random_state=71`

Blend candidate:

- LightGBM
- XGB v2 seed 63
- XGB v2 seed 71

Candidate file:

`output/submission_lgb_xgb2_seed63_seed71_blend_tiny.csv`

Blend:

- LightGBM: 0.45
- XGB v2 seed 63: 0.25
- XGB v2 seed 71: 0.30
- QSO multiplier: 1.22
- STAR multiplier: 1.58

OOF: **0.968212**

This is slightly above the current best candidate's OOF (`0.968112`) but has not been validated on LB yet.

## Green Valley Candidate

Added features based on the galaxy-population boundary at `u-r = 2.2`:

- `_green_valley_margin = (u-r) - 2.2`
- `_green_valley_abs_margin = abs((u-r) - 2.2)`
- `_green_valley_margin_x_mag_mean`

Candidate file:

`output/submission_lgb_xgb2_green_valley_blend_grid.csv`

Blend:

- LightGBM: 0.35
- XGB v2 seed63: 0.25
- XGB Green Valley: 0.40
- QSO multiplier: 1.24
- STAR multiplier: 1.50

OOF: **0.968311**

This is the highest OOF candidate so far, but prior experiments showed that OOF gains can fail on LB. Treat it as a submit-and-check candidate, not a new best until LB confirms it.

## Hard Vote Blender Candidates

Local script:

`src/blend_submissions.py`

Input folder:

`output/submissions_for_blend/`

The script parses the leading number in each CSV filename as its LB score, then uses weighted hard voting on rows where submissions disagree.

Current inputs:

- `0.96883_lgb_xgb2.csv`
- `0.96863_seed_blend.csv`
- `0.96863_green_valley.csv`
- `0.96862_xgb3.csv`
- `0.96814_lgb_xgb1.csv`
- `0.96764_lgb.csv`

Candidates:

- `output/submission_hard_vote_blend_softmax.csv`
  - changed 206 rows vs current best
  - default softmax weighting, decorrelation on
- `output/submission_hard_vote_blend_shifted.csv`
  - changed 251 rows vs current best
  - shifted-score weighting, decorrelation on

These cannot be honestly validated locally because they use only test-set submission labels. Treat them as final-polish LB probes.

## Probability Core / Logistic Stacker

Local script:

`src/stack_probability_core.py`

This mirrors the notebook's Layer A idea, using local OOF/test probability CSVs instead of `oof_*.npy` and `test_*.npy`.

Results:

- `output/submission_probability_stacker_lgb_xgb2.csv`
  - models: LightGBM + XGB v2 seed63
  - OOF: **0.967965**
  - lower than manual LGB + XGB v2 blend (`0.968112`)
- `output/submission_probability_stacker_all.csv`
  - models: LightGBM + XGB v2 seed63 + XGB v2 seed71 + Green Valley XGB + XGB v3
  - OOF: **0.968085**
  - still lower than manual LGB + XGB v2 blend

Conclusion: logistic probability stacking is useful as a locally measurable core, but with current local models it does not beat the hand-tuned probability blend. Keep it as a diagnostic, not a primary submission candidate.

## XGB v4 Candidate

XGB v4 parameters:

- `max_depth=4`
- `min_child_weight=2`
- `reg_lambda=1`
- `n_estimators=700`
- `learning_rate=0.035`

Raw OOF: **0.967385**

Blend candidate:

`output/submission_lgb_xgb2_xgb4_blend_grid.csv`

Blend:

- LightGBM: 0.45464
- XGB v2 seed63: 0.48536
- XGB v4: 0.06
- QSO multiplier: 1.20
- STAR multiplier: 1.54

OOF: **0.968155**

Changed 184 rows vs current best. This is a conservative candidate, but OOF gain is smaller than prior candidates that failed on LB.

## External 5-Model Blend

Files imported from Kaggle notebook:

- `output/oof_external_5model_blend.csv`
- `output/test_proba_external_5model_blend.csv`
- `output/submission_external_5model_blend.csv`

Single submission LB: **0.97044**

OOF: **0.969853**

Local blend candidate:

`output/submission_external_lgb_xgb2_blend_tiny.csv`

Blend:

- Local LightGBM: 0.20
- Local XGB v2: 0.00
- External 5-model blend: 0.80
- QSO multiplier: 1.10
- STAR multiplier: 1.20

OOF: **0.969940**

Changed 689 rows vs `submission_external_5model_blend.csv`. This is the next best submission candidate after the external 5-model single submission.
