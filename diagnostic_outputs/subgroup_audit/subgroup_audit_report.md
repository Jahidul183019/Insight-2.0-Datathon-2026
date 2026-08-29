# Subgroup F1 Audit

## Decision

**NO-SUBMIT GATE:** on official weighted F1, the prospective pseudo90 + 20% NN blend is not clearly outside uncertainty and/or has a material large-subgroup regression.

This is a saved-OOF diagnostic, not a reconstruction of Kaggle's hidden split. The NN vector is ordinary five-fold OOF rather than a jointly nested rebuild with pseudo90, so the combined estimate remains a screening result.

## Metric policy

Predictions select exactly `round(0.845 * n)` Dead rows by descending probability with stable tie-breaking. **Support-weighted F1 is the official competition metric and drives every gate below.** Binary Dead-class F1 is reported only as a legacy diagnostic.

| candidate | rows | fixed_positive_rate | weighted_f1 | dead_class_f1 | roc_auc |
|---|---|---|---|---|---|
| frozen_tree | 24000 | 0.845000 | 0.877513 | 0.928209 | 0.901623 |
| nested_pseudo90 | 24000 | 0.845000 | 0.874797 | 0.926617 | 0.896890 |
| nested_pseudo90_nn20 | 24000 | 0.845000 | 0.876155 | 0.927413 | 0.898453 |

## Composite-stage definition

This is an explicit disease-extent grouping, not a claim of formal AJCC I-IV stage:

- `M1 distant/metastatic`: M starts with M1, regardless of T or N.
- `M0 N1-3 node-positive`: M0 and N is N1, N2, or N3.
- `M0 N0 T0-2 localized`: M0, N0, and T is T0, T1*, or T2*.
- `M0 N0 T3-4 locally advanced`: M0, N0, and T is T3 or T4*.
- `M0 indeterminate T/N`: M0 with T or N not classifiable above.
- `EOD unavailable (all blank)`: T, N, and M are all Blank(s).
- `EOD not applicable (88)`: T, N, and M are all code 88.
- `Other/discordant EOD`: any remaining T/N/M combination.

## Weak, intervention-sized frozen-tree slices on official weighted F1

All slices with at least 200 rows are in `subgroup_slice_metrics.csv`. The table below requires at least 500 rows and an official weighted-F1 gap of at least 0.020.

| dimension | slice | rows | actual_dead_rate | global_policy_predicted_dead_rate | global_policy_weighted_f1 | global_policy_weighted_f1_gap | within_slice_weighted_f1 | fold_global_policy_weighted_f1_mean | fold_global_policy_weighted_f1_std | global_policy_dead_class_f1 | global_policy_dead_f1_gap | within_slice_dead_class_f1 | fold_global_policy_dead_class_f1_mean | fold_global_policy_dead_class_f1_std | fold_global_policy_dead_class_f1_min | fold_global_policy_dead_class_f1_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| derived_eod_composite_stage | M0 N0 T0-2 localized | 2566 | 0.4201 | 0.3164 | 0.7449 | -0.1326 | 0.5116 | 0.7432 | 0.0185 | 0.6635 | -0.2647 | 0.6556 | 0.6595 | 0.0379 | 0.6072 | 0.6976 |
| histologic_type_icdo3 | 8551 | 514 | 0.2918 | 0.1440 | 0.7685 | -0.1090 | 0.3967 | 0.7637 | 0.0543 | 0.5268 | -0.4014 | 0.5068 | 0.5195 | 0.0779 | 0.4091 | 0.6222 |
| derived_eod_composite_stage | M0 N0 T3-4 locally advanced | 611 | 0.6612 | 0.6367 | 0.7808 | -0.0967 | 0.7276 | 0.7805 | 0.0273 | 0.8298 | -0.0984 | 0.8391 | 0.8278 | 0.0246 | 0.8065 | 0.8591 |
| derived_eod_composite_stage | M0 N1-3 node-positive | 1829 | 0.7233 | 0.7266 | 0.7875 | -0.0900 | 0.7872 | 0.7876 | 0.0071 | 0.8537 | -0.0745 | 0.8770 | 0.8542 | 0.0065 | 0.8438 | 0.8602 |
| derived_eod_composite_stage | EOD not applicable (88) | 754 | 0.7653 | 0.7891 | 0.8135 | -0.0640 | 0.8171 | 0.8137 | 0.0408 | 0.8823 | -0.0460 | 0.8946 | 0.8822 | 0.0209 | 0.8509 | 0.9091 |
| age_recode | 55-59 years | 1821 | 0.7924 | 0.8204 | 0.8392 | -0.0383 | 0.8317 | 0.8383 | 0.0242 | 0.9030 | -0.0252 | 0.9027 | 0.9027 | 0.0143 | 0.8857 | 0.9239 |
| age_recode | 65-69 years | 3851 | 0.7988 | 0.8065 | 0.8509 | -0.0266 | 0.8505 | 0.8505 | 0.0084 | 0.9078 | -0.0204 | 0.9134 | 0.9074 | 0.0059 | 0.8987 | 0.9144 |
| age_recode | 50-54 years | 789 | 0.7693 | 0.7985 | 0.8533 | -0.0242 | 0.8637 | 0.8557 | 0.0420 | 0.9086 | -0.0196 | 0.9215 | 0.9103 | 0.0275 | 0.8646 | 0.9380 |
| age_recode | 60-64 years | 2925 | 0.8027 | 0.8113 | 0.8562 | -0.0213 | 0.8546 | 0.8574 | 0.0203 | 0.9117 | -0.0165 | 0.9158 | 0.9124 | 0.0126 | 0.8940 | 0.9241 |

F1 is prevalence-sensitive. In particular, forcing 84.5% Dead inside a low-death slice can reduce F1; a weak slice is not by itself evidence for a subgroup threshold.

## Pseudo90 + NN regressions on official weighted F1

| dimension | slice | rows | combined_minus_pseudo90_weighted_f1 | combined_minus_pseudo90_dead_f1 | changed_predictions | corrected_predictions | harmed_predictions | fold_weighted_delta_mean | fold_weighted_delta_min | fold_weighted_delta_max | fold_weighted_wins | fold_weighted_ties | fold_weighted_losses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| age_recode | 55-59 years | 1821 | -0.0023 | -0.0014 | 16 | 6 | 10 | -0.0022 | -0.0089 | 0.0048 | 1 | 1 | 3 |

## Legacy Dead-class diagnostic-only regressions

| dimension | slice | rows | combined_minus_pseudo90_dead_f1 | combined_minus_pseudo90_weighted_f1 | changed_predictions | fold_dead_delta_mean | fold_dead_delta_min | fold_dead_delta_max | fold_dead_wins | fold_dead_ties | fold_dead_losses |
|---|---|---|---|---|---|---|---|---|---|---|---|
| derived_eod_composite_stage | M0 N0 T0-2 localized | 2566 | -0.0017 | -0.0007 | 47 | -0.0022 | -0.0116 | 0.0081 | 1 | 1 | 3 |
| histologic_type_icdo3 | 8551 | 514 | -0.0042 | 0.0003 | 7 | -0.0087 | -0.0389 | 0.0139 | 1 | 2 | 2 |

## Paired bootstrap: combined minus pseudo90

- **Official weighted F1:** observed full-OOF delta `+0.001358`; bootstrap mean `+0.001195`, 95% interval `[-0.000255, +0.002631]`.
- Legacy Dead-class F1: mean `+0.000701`, 95% interval `[-0.000149, +0.001542]`.

## Reproducibility outputs

- `global_candidate_metrics.csv`: full-OOF metrics.
- `subgroup_slice_metrics.csv`: both policies and both F1 definitions for every slice.
- `subgroup_fold_metrics.csv`: raw five-fold measurements using StratifiedKFold seed 42.
- `subgroup_stability_summary.csv`: mean, standard deviation, minimum, and maximum.
- `pseudo90_nn20_regression_audit.csv`: hard-label changes and fold deltas.
- `subgroup_audit_summary.json`: definitions, artifact hashes, bootstrap, and gate status.

The stability table contains 90 candidate/slice summaries.
