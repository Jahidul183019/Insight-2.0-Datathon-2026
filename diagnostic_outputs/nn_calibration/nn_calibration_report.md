# Neural-network OOF calibration audit

## Decision

**Submission D does not pass the submission gate.** The raw NN has a localized calibration deviation, but leakage-safe isotonic calibration does not improve the downstream 80/20 tree/NN candidate.

No submission or saved test-probability artifact was created or modified.

The competition's support-weighted F1 is the primary F1 measure in this report. Binary Dead-class F1 is retained only in explicit `legacy_dead_class_*` fields and deprecated compatibility aliases (`metric_schema_version=2_weighted_f1_primary`).

## Direct findings

- The NN's 10-quantile ECE is `0.010364`, RMS calibration error is `0.015562`, and maximum bin gap is `0.042590`.
- The corresponding tree OOF values are ECE `0.003452`, RMS error `0.003965`, and maximum gap `0.006353`.
- The NN's largest deviation is the second probability decile: mean prediction `0.555327` versus observed Dead rate `0.597917` (gap `+0.042590`, 2,400 rows).
- Five-fold cross-fit isotonic calibration reduces the NN ECE to `0.001925` for the primary seed, but its Brier loss changes from `0.088704` to `0.088764` and log loss from `0.290800` to `0.293971`.
- The calibrated 80/20 blend improves official support-weighted fixed-rate F1 in only `0/21` cross-fitting seeds.
- In 50 repeated 40/60 calibration/holdout checks, the calibrated blend improves official support-weighted fixed-rate F1 in only `9/50` splits; mean delta is `-0.000252`.

## Primary-seed blend comparison

| Metric | Raw 80/20 blend | Isotonic 80/20 blend |
|---|---:|---:|
| Brier loss | 0.083565 | 0.083593 |
| Log loss | 0.273668 | 0.273781 |
| ROC AUC | 0.901494 | 0.901356 |
| ECE | 0.003381 | 0.004216 |
| Official support-weighted F1 at fixed 84.5% | 0.878447 | 0.878023 |
| Legacy Dead-class F1 at fixed 84.5% | 0.928756 | 0.928507 |

## Method

- Reliability uses 10 equal-frequency bins, matching the existing v6 reliability method.
- For every five-fold audit, the isotonic mapping for a row is fitted without that row or its label.
- The repeated 40/60 audit fits the mapping on 40% of labelled OOF rows and evaluates it only on the untouched 60%.
- Fixed-rate F1 predicts exactly `round(0.845 * n)` positive rows using stable rank ordering.
- The 84.5% rate is a historically selected policy, not an independently estimated value in this audit; it must not be refined through leaderboard probing.
- The existing tree and NN vectors came from different CV schemes. This is a screening audit, not a guarantee about Kaggle's hidden labels.
- Cross-fit isotonic uses fold-specific monotonic mappings; this can alter global ordering and exposes the downstream instability that an in-sample calibration curve would conceal.

## Input provenance

```csv
file,bytes,sha256
train.csv,11299297,28189b2140be2fdfff65c5dae8cf5d2ffeaffd009ee962047396096cba4ab3c2
y_step1.npy,192128,00b24b69a78146ce19d7a057d7f7e21404c42efda06033e9d8e2e4296457d9db
oof_step1.npy,192128,14fbad5a4584b61ca8ca6b8beeeeb3c6f8ec7ba167f023b9676be33653f641df
archive/oof_nn.npy,192128,5620400c786fa180abec67f919920015ac8796d69a5ea5dbeb5c05e714232f8d
```

## Output guide

- `nn_oof_reliability_curve.png`: raw NN, raw tree, and held-out isotonic reliability curves.
- `nn_oof_reliability_bins.csv`: exact bin counts, means, gaps, and Wilson intervals.
- `nn_calibration_metrics.csv`: primary-seed metrics.
- `nn_isotonic_crossfit_seed_metrics.csv`: every metric and seed.
- `nn_isotonic_crossfit_summary.csv`: aggregate cross-fit evidence.
- `nn_isotonic_40_60_holdout_metrics.csv` and `nn_isotonic_40_60_holdout_summary.csv`: repeated held-out robustness checks.
