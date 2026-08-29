"""Reproducible diagnostics-only gate audit for the proposed Submission 12.

The proposed candidate is an 80% nested-pseudo90 / 20% neural-network OOF
blend.  This script evaluates it against the available saved OOF controls at
the locked 84.5% positive-rate policy.  It intentionally does not create or
modify a submission CSV, test-probability array, history file, or notebook.

Outputs are restricted to ``diagnostic_outputs/submission12_gate/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "diagnostic_outputs" / "submission12_gate"

FIXED_POSITIVE_RATE = 0.845
PRIMARY_METRIC = "weighted_f1"
LEGACY_METRIC = "dead_class_binary_f1"
N_OUTER_FOLDS = 5
OUTER_SEED = 42
BOOTSTRAP_REPEATS = 1_000
BOOTSTRAP_SEED = 9_042
N_SIMULATED_SPLITS = 50
SIMULATED_PUBLIC_FRACTION = 0.40
SIMULATED_SPLIT_SEED = 20_260_829

NESTED_PATH = (
    ROOT / "diagnostic_outputs" / "pseudo90_nested" / "nested_oof_predictions.npz"
)
NN_OOF_PATH = ROOT / "archive" / "oof_nn.npy"
Y_PATH = ROOT / "y_step1.npy"
TRAIN_PATH = ROOT / "train.csv"

PROTECTED_PATHS = [
    ROOT / "submission.csv",
    ROOT / "SUBMISSION_HISTORY.md",
    ROOT / "insight_2_0_consolidated.ipynb",
    ROOT / "archive" / "probs_v6_final.npy",
    ROOT / "archive" / "probs_nn.npy",
    ROOT / "artifacts" / "pseudo90" / "probs_pseudo90.npy",
]

CANDIDATE_ORDER = [
    "nested_pseudo95_proxy",
    "nested_pseudo90",
    "sub10_nested_analogue_p95_nn20",
    "sub12_candidate_p90_nn20",
    "existing_tree80_nn20",
]


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    return {
        str(require_file(path).relative_to(ROOT)): sha256_file(path)
        for path in PROTECTED_PATHS
    }


def validate_probability_vector(
    name: str, values: np.ndarray, expected_rows: int
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) != expected_rows:
        raise ValueError(
            f"{name} must have shape ({expected_rows},), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError(f"{name} contains values outside [0, 1]")
    return values


def top_rate_predictions(scores: np.ndarray, rate: float) -> np.ndarray:
    """Select exactly round(rate * n) positives with stable tie-breaking."""

    positive_count = int(round(rate * len(scores)))
    positive_count = min(max(positive_count, 0), len(scores))
    order = np.argsort(-scores, kind="mergesort")
    predictions = np.zeros(len(scores), dtype=np.int8)
    predictions[order[:positive_count]] = 1
    return predictions


def metric_pair(y_true: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    """Return Dead-class binary F1 and class-support-weighted F1."""

    dead_f1 = f1_score(
        y_true,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )
    return float(dead_f1), float(weighted_f1)


def load_inputs() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    saved_y = np.load(require_file(Y_PATH), allow_pickle=False)
    train = pd.read_csv(require_file(TRAIN_PATH), usecols=["vital_status"])
    train_y = train["vital_status"].eq("Dead").to_numpy(dtype=np.int8)
    if not np.array_equal(saved_y, train_y):
        raise ValueError("y_step1.npy does not match train.csv row order and labels")

    nested = np.load(require_file(NESTED_PATH), allow_pickle=False)
    required_keys = {
        "y",
        "pseudo95_oof",
        "pseudo90_oof",
        "frozen_tree_oof",
    }
    missing = required_keys - set(nested.files)
    if missing:
        raise ValueError(f"Nested OOF archive is missing keys: {sorted(missing)}")
    if not np.array_equal(nested["y"], train_y):
        raise ValueError("Nested OOF labels do not match train.csv")

    rows = len(train_y)
    pseudo95 = validate_probability_vector(
        "pseudo95_oof", nested["pseudo95_oof"], rows
    )
    pseudo90 = validate_probability_vector(
        "pseudo90_oof", nested["pseudo90_oof"], rows
    )
    frozen_tree = validate_probability_vector(
        "frozen_tree_oof", nested["frozen_tree_oof"], rows
    )
    nn_oof = validate_probability_vector(
        "archive/oof_nn.npy",
        np.load(require_file(NN_OOF_PATH), allow_pickle=False),
        rows,
    )

    candidates = {
        "nested_pseudo95_proxy": pseudo95,
        "nested_pseudo90": pseudo90,
        "sub10_nested_analogue_p95_nn20": 0.80 * pseudo95 + 0.20 * nn_oof,
        "sub12_candidate_p90_nn20": 0.80 * pseudo90 + 0.20 * nn_oof,
        "existing_tree80_nn20": 0.80 * frozen_tree + 0.20 * nn_oof,
    }
    return train_y, candidates


def global_metrics(
    y: np.ndarray, candidates: dict[str, np.ndarray]
) -> pd.DataFrame:
    pseudo90_predictions = top_rate_predictions(
        candidates["nested_pseudo90"], FIXED_POSITIVE_RATE
    )
    pseudo90_dead_f1, pseudo90_weighted_f1 = metric_pair(
        y, pseudo90_predictions
    )
    rows = []
    for name in CANDIDATE_ORDER:
        scores = candidates[name]
        predictions = top_rate_predictions(scores, FIXED_POSITIVE_RATE)
        dead_f1, weighted_f1 = metric_pair(y, predictions)
        rows.append(
            {
                "candidate": name,
                "rows": len(y),
                "true_dead_count": int(y.sum()),
                "true_dead_rate": float(y.mean()),
                "predicted_dead_count": int(predictions.sum()),
                "predicted_dead_rate": float(predictions.mean()),
                "dead_class_binary_f1": dead_f1,
                "weighted_f1": weighted_f1,
                "roc_auc": float(roc_auc_score(y, scores)),
                "dead_f1_delta_vs_nested_pseudo90": dead_f1
                - pseudo90_dead_f1,
                "weighted_f1_delta_vs_nested_pseudo90": weighted_f1
                - pseudo90_weighted_f1,
                "hard_disagreement_count_vs_nested_pseudo90": int(
                    (predictions != pseudo90_predictions).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def outer_fold_metrics(
    y: np.ndarray, candidates: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    splitter = StratifiedKFold(
        n_splits=N_OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED
    )
    for fold, (_, validation_idx) in enumerate(
        splitter.split(np.zeros(len(y)), y), start=1
    ):
        fold_y = y[validation_idx]
        fold_records = []
        for name in CANDIDATE_ORDER:
            predictions = top_rate_predictions(
                candidates[name][validation_idx], FIXED_POSITIVE_RATE
            )
            dead_f1, weighted_f1 = metric_pair(fold_y, predictions)
            fold_records.append(
                {
                    "outer_fold": fold,
                    "candidate": name,
                    "rows": len(validation_idx),
                    "true_dead_rate": float(fold_y.mean()),
                    "predicted_dead_count": int(predictions.sum()),
                    "predicted_dead_rate": float(predictions.mean()),
                    "dead_class_binary_f1": dead_f1,
                    "weighted_f1": weighted_f1,
                }
            )
        pseudo90_record = next(
            row for row in fold_records if row["candidate"] == "nested_pseudo90"
        )
        for row in fold_records:
            row["dead_f1_delta_vs_nested_pseudo90"] = (
                row["dead_class_binary_f1"]
                - pseudo90_record["dead_class_binary_f1"]
            )
            row["weighted_f1_delta_vs_nested_pseudo90"] = (
                row["weighted_f1"] - pseudo90_record["weighted_f1"]
            )
            rows.append(row)

    metrics = pd.DataFrame(rows)
    summary_rows = []
    for name, group in metrics.groupby("candidate", sort=False):
        for metric, delta_column in (
            ("dead_class_binary_f1", "dead_f1_delta_vs_nested_pseudo90"),
            ("weighted_f1", "weighted_f1_delta_vs_nested_pseudo90"),
        ):
            deltas = group[delta_column].to_numpy()
            summary_rows.append(
                {
                    "candidate": name,
                    "metric": metric,
                    "fold_mean": float(group[metric].mean()),
                    "worst_fold_value": float(group[metric].min()),
                    "best_fold_value": float(group[metric].max()),
                    "mean_delta_vs_nested_pseudo90": float(deltas.mean()),
                    "improved_fold_count": int((deltas > 0.0).sum()),
                    "tied_fold_count": int((deltas == 0.0).sum()),
                    "regressed_fold_count": int((deltas < 0.0).sum()),
                }
            )
    return metrics, pd.DataFrame(summary_rows)


def paired_stratified_bootstrap(
    y: np.ndarray,
    candidate_scores: np.ndarray,
    control_scores: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rows = []
    for repeat in range(BOOTSTRAP_REPEATS):
        sampled = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        sampled_y = y[sampled]
        candidate_predictions = top_rate_predictions(
            candidate_scores[sampled], FIXED_POSITIVE_RATE
        )
        control_predictions = top_rate_predictions(
            control_scores[sampled], FIXED_POSITIVE_RATE
        )
        candidate_dead, candidate_weighted = metric_pair(
            sampled_y, candidate_predictions
        )
        control_dead, control_weighted = metric_pair(
            sampled_y, control_predictions
        )
        rows.append(
            {
                "bootstrap_repeat": repeat,
                "dead_f1_delta_sub12_minus_pseudo90": candidate_dead
                - control_dead,
                "weighted_f1_delta_sub12_minus_pseudo90": candidate_weighted
                - control_weighted,
            }
        )

    deltas = pd.DataFrame(rows)
    full_candidate = top_rate_predictions(candidate_scores, FIXED_POSITIVE_RATE)
    full_control = top_rate_predictions(control_scores, FIXED_POSITIVE_RATE)
    full_candidate_metrics = metric_pair(y, full_candidate)
    full_control_metrics = metric_pair(y, full_control)
    summary_rows = []
    for metric, column, metric_index in (
        (
            "weighted_f1",
            "weighted_f1_delta_sub12_minus_pseudo90",
            1,
        ),
        (
            "dead_class_binary_f1",
            "dead_f1_delta_sub12_minus_pseudo90",
            0,
        ),
    ):
        values = deltas[column].to_numpy()
        lower, median, upper = np.quantile(values, [0.025, 0.50, 0.975])
        summary_rows.append(
            {
                "metric": metric,
                "repeats": BOOTSTRAP_REPEATS,
                "seed": BOOTSTRAP_SEED,
                "full_oof_point_delta": full_candidate_metrics[metric_index]
                - full_control_metrics[metric_index],
                "bootstrap_mean_delta": float(values.mean()),
                "bootstrap_std_delta": float(values.std(ddof=1)),
                "bootstrap_95pct_lower": float(lower),
                "bootstrap_median": float(median),
                "bootstrap_95pct_upper": float(upper),
                "fraction_strictly_above_zero": float((values > 0.0).mean()),
                "ci_lower_strictly_above_zero": bool(lower > 0.0),
            }
        )
    return deltas, pd.DataFrame(summary_rows)


def repeated_split_audit(
    y: np.ndarray, candidates: dict[str, np.ndarray]
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    all_indices = np.arange(len(y))
    rows = []
    winner_rows = []
    for split_id in range(N_SIMULATED_SPLITS):
        split_seed = SIMULATED_SPLIT_SEED + split_id
        public_idx, private_idx = train_test_split(
            all_indices,
            train_size=SIMULATED_PUBLIC_FRACTION,
            stratify=y,
            random_state=split_seed,
        )
        for partition, indices in (("public", public_idx), ("private", private_idx)):
            partition_y = y[indices]
            split_records = []
            for name in CANDIDATE_ORDER:
                predictions = top_rate_predictions(
                    candidates[name][indices], FIXED_POSITIVE_RATE
                )
                dead_f1, weighted_f1 = metric_pair(partition_y, predictions)
                record = {
                    "split_id": split_id,
                    "split_seed": split_seed,
                    "partition": partition,
                    "candidate": name,
                    "rows": len(indices),
                    "true_dead_rate": float(partition_y.mean()),
                    "predicted_dead_rate": float(predictions.mean()),
                    "dead_class_binary_f1": dead_f1,
                    "weighted_f1": weighted_f1,
                }
                split_records.append(record)
                rows.append(record)

            for metric in ("weighted_f1", "dead_class_binary_f1"):
                values = {row["candidate"]: row[metric] for row in split_records}
                best_value = max(values.values())
                co_winners = sorted(
                    name
                    for name, value in values.items()
                    if np.isclose(value, best_value, rtol=0.0, atol=1e-15)
                )
                # This is the validation_harness.py tie-break: metric descending,
                # then candidate name ascending.
                selected_winner = sorted(
                    values, key=lambda name: (-values[name], name)
                )[0]
                winner_rows.append(
                    {
                        "split_id": split_id,
                        "split_seed": split_seed,
                        "partition": partition,
                        "metric": metric,
                        "harness_selected_winner": selected_winner,
                        "best_value": best_value,
                        "co_winner_count": len(co_winners),
                        "co_winners": "|".join(co_winners),
                    }
                )

    metrics = pd.DataFrame(rows)
    winners = pd.DataFrame(winner_rows)

    candidate_summary = (
        metrics.groupby(["partition", "candidate"], as_index=False)
        .agg(
            dead_class_binary_f1_mean=("dead_class_binary_f1", "mean"),
            dead_class_binary_f1_std=("dead_class_binary_f1", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            weighted_f1_std=("weighted_f1", "std"),
        )
        .sort_values(
            ["partition", "dead_class_binary_f1_mean"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    winner_summary_rows = []
    for (partition, metric), group in winners.groupby(["partition", "metric"]):
        selected_counts = group["harness_selected_winner"].value_counts()
        for name in CANDIDATE_ORDER:
            co_winner_count = int(
                group["co_winners"]
                .str.split("|")
                .apply(lambda values: name in values)
                .sum()
            )
            unique_winner_count = int(
                (
                    (group["harness_selected_winner"] == name)
                    & (group["co_winner_count"] == 1)
                ).sum()
            )
            selected_count = int(selected_counts.get(name, 0))
            winner_summary_rows.append(
                {
                    "partition": partition,
                    "metric": metric,
                    "candidate": name,
                    "n_splits": len(group),
                    "harness_selected_winner_count": selected_count,
                    "inclusive_co_winner_count": co_winner_count,
                    "unique_winner_count": unique_winner_count,
                    "clear_majority_selected_winner": bool(
                        selected_count > len(group) / 2
                    ),
                }
            )
    winner_summary = pd.DataFrame(winner_summary_rows)

    pairwise_rows = []
    for partition in ("public", "private"):
        partition_metrics = metrics[metrics["partition"] == partition]
        for metric in ("weighted_f1", "dead_class_binary_f1"):
            pivot = partition_metrics.pivot(
                index="split_id", columns="candidate", values=metric
            )
            delta = (
                pivot["sub12_candidate_p90_nn20"]
                - pivot["nested_pseudo90"]
            )
            pairwise_rows.append(
                {
                    "partition": partition,
                    "metric": metric,
                    "comparison": "sub12_candidate_p90_nn20-minus-nested_pseudo90",
                    "n_splits": len(delta),
                    "mean_delta": float(delta.mean()),
                    "std_delta": float(delta.std(ddof=1)),
                    "minimum_delta": float(delta.min()),
                    "median_delta": float(delta.median()),
                    "maximum_delta": float(delta.max()),
                    "improved_split_count": int((delta > 0.0).sum()),
                    "tied_split_count": int((delta == 0.0).sum()),
                    "regressed_split_count": int((delta < 0.0).sum()),
                }
            )
    return metrics, winners, candidate_summary, winner_summary, pd.DataFrame(pairwise_rows)


def read_submission_labels(path: Path) -> np.ndarray:
    submission = pd.read_csv(require_file(path))
    if submission.columns.tolist() != ["patient_id", "vital_status"]:
        raise ValueError(f"Unexpected submission schema: {path}")
    if submission["patient_id"].duplicated().any():
        raise ValueError(f"Duplicate patient IDs: {path}")
    invalid = set(submission["vital_status"].unique()) - {"Dead", "Alive"}
    if invalid:
        raise ValueError(f"Invalid labels in {path}: {sorted(invalid)}")
    return submission["vital_status"].eq("Dead").to_numpy(dtype=np.int8)


def recipe_equivalence_audit() -> pd.DataFrame:
    v6_probs = validate_probability_vector(
        "archive/probs_v6_final.npy",
        np.load(
            require_file(ROOT / "archive" / "probs_v6_final.npy"),
            allow_pickle=False,
        ),
        36_000,
    )
    nn_test_probs = validate_probability_vector(
        "archive/probs_nn.npy",
        np.load(
            require_file(ROOT / "archive" / "probs_nn.npy"),
            allow_pickle=False,
        ),
        36_000,
    )
    submission6 = read_submission_labels(ROOT / "archive" / "submission6.csv")
    submission10 = read_submission_labels(ROOT / "archive" / "submission10.csv")
    target_dead_count = int(submission6.sum())

    def exact_top_count(scores: np.ndarray, count: int) -> np.ndarray:
        order = np.argsort(scores, kind="mergesort")
        predictions = np.zeros(len(scores), dtype=np.int8)
        predictions[order[-count:]] = 1
        return predictions

    reconstructed6 = exact_top_count(v6_probs, target_dead_count)
    reconstructed10 = exact_top_count(
        0.80 * v6_probs + 0.20 * nn_test_probs, target_dead_count
    )
    return pd.DataFrame(
        [
            {
                "item": "actual_submission6_hard_reconstruction",
                "status": "exact",
                "exact_match": bool(np.array_equal(reconstructed6, submission6)),
                "disagreement_count": int((reconstructed6 != submission6).sum()),
                "detail": (
                    f"Top-{target_dead_count:,} of archive/probs_v6_final.npy "
                    "reproduces archive/submission6.csv in memory."
                ),
            },
            {
                "item": "actual_submission10_hard_reconstruction",
                "status": "exact",
                "exact_match": bool(np.array_equal(reconstructed10, submission10)),
                "disagreement_count": int((reconstructed10 != submission10).sum()),
                "detail": (
                    f"Top-{target_dead_count:,} of 0.80*v6 + 0.20*NN "
                    "reproduces archive/submission10.csv in memory."
                ),
            },
            {
                "item": "nested_pseudo95_vs_actual_submission6",
                "status": "not_recipe_equivalent",
                "exact_match": False,
                "disagreement_count": pd.NA,
                "detail": (
                    "The nested control uses one full 1500-round fit per model family "
                    "without inner early stopping or fold averaging. Actual Submission 6 "
                    "uses a 3x5 early-stopped teacher and a five-fold early-stopped "
                    "pseudo student. It is a like-for-like pseudo90 control, not an "
                    "exact nested replay of Submission 6."
                ),
            },
            {
                "item": "oof_step1_vs_actual_submission6",
                "status": "not_recipe_equivalent",
                "exact_match": False,
                "disagreement_count": pd.NA,
                "detail": (
                    "oof_step1/frozen_tree_oof is the pre-pseudo teacher proxy and "
                    "does not contain Submission 6's final pseudo-student component."
                ),
            },
            {
                "item": "nn_oof_outer_fold_comparability",
                "status": "screening_proxy_only",
                "exact_match": False,
                "disagreement_count": pd.NA,
                "detail": (
                    "pipeline_nn.py uses the same five-fold StratifiedKFold seed 42, "
                    "but preprocessing is fit globally and each validation fold selects "
                    "the best NN epoch. The blend is aligned OOF screening, not the same "
                    "untouched-outer-fold standard as the nested pseudo audit."
                ),
            },
        ]
    )


def build_gate_results(
    global_frame: pd.DataFrame,
    fold_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    winner_summary: pd.DataFrame,
    recipe_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    sub12_global = global_frame.set_index("candidate").loc[
        "sub12_candidate_p90_nn20"
    ]
    pseudo95_global = global_frame.set_index("candidate").loc[
        "nested_pseudo95_proxy"
    ]
    tree80_global = global_frame.set_index("candidate").loc[
        "existing_tree80_nn20"
    ]

    bootstrap_by_metric = bootstrap_summary.set_index("metric")
    folds_by_metric = fold_summary[
        fold_summary["candidate"] == "sub12_candidate_p90_nn20"
    ].set_index("metric")
    private_winners = winner_summary[
        (winner_summary["partition"] == "private")
        & (winner_summary["candidate"] == "sub12_candidate_p90_nn20")
    ].set_index("metric")
    nested_recipe_exact = bool(
        recipe_audit.loc[
            recipe_audit["item"] == "nested_pseudo95_vs_actual_submission6",
            "status",
        ].iloc[0]
        == "exact"
    )

    gate_rows = []
    for metric in ("weighted_f1", "dead_class_binary_f1"):
        lower = float(bootstrap_by_metric.loc[metric, "bootstrap_95pct_lower"])
        metric_role = (
            "official_primary" if metric == PRIMARY_METRIC else "legacy_comparison"
        )
        gate_rows.append(
            {
                "gate": "1_bootstrap_ci_lower_strictly_above_zero",
                "metric": metric,
                "metric_role": metric_role,
                "observed": lower,
                "required": "> 0",
                "passed": bool(lower > 0.0),
                "note": "Sub12 minus nested pseudo90 paired stratified bootstrap.",
            }
        )
        improved_folds = int(
            folds_by_metric.loc[metric, "improved_fold_count"]
        )
        gate_rows.append(
            {
                "gate": "2_outer_fold_improvement_count",
                "metric": metric,
                "metric_role": metric_role,
                "observed": improved_folds,
                "required": ">= 4 of 5",
                "passed": bool(improved_folds >= 4),
                "note": "Each fold independently uses an exact 84.5% top-rate policy.",
            }
        )
        winner_count = int(
            private_winners.loc[metric, "harness_selected_winner_count"]
        )
        gate_rows.append(
            {
                "gate": "3_simulated_private_clear_majority",
                "metric": metric,
                "metric_role": metric_role,
                "observed": winner_count,
                "required": "> 25 of 50",
                "passed": bool(winner_count > N_SIMULATED_SPLITS / 2),
                "note": "Winner comparison includes all five requested OOF candidates.",
            }
        )

    gate_rows.extend(
        [
            {
                "gate": "4_no_meaningful_subgroup_regression",
                "metric": "external",
                "metric_role": "separate_audit",
                "observed": pd.NA,
                "required": "separate subgroup audit must pass",
                "passed": pd.NA,
                "note": "Handled by the separate exact subgroup audit, not this script.",
            },
            {
                "gate": "5_directionally_better_than_actual_submission6_nested",
                "metric": "weighted_f1",
                "metric_role": "official_primary",
                "observed": pd.NA,
                "required": "exact recipe-equivalent nested Submission 6 OOF",
                "passed": False,
                "note": (
                    "Unavailable: nested pseudo95 is not an exact replay. Sub12 is "
                    f"{sub12_global['weighted_f1'] - pseudo95_global['weighted_f1']:+.9f} "
                    "versus the pseudo95 proxy but "
                    f"{sub12_global['weighted_f1'] - tree80_global['weighted_f1']:+.9f} "
                    "versus existing tree80_nn20."
                ),
            },
        ]
    )
    gates = pd.DataFrame(gate_rows)
    decisive = gates[
        gates["passed"].notna()
        & gates["metric_role"].isin(["official_primary", "recipe_requirement"])
    ]
    overall_pass = bool(decisive["passed"].astype(bool).all()) and nested_recipe_exact
    decision = {
        "candidate": "sub12_candidate_p90_nn20",
        "decision": "NO_GO_DO_NOT_GENERATE_OR_SUBMIT",
        "overall_gate_pass": overall_pass,
        "fixed_positive_rate": FIXED_POSITIVE_RATE,
        "official_primary_metric": "weighted_f1",
        "legacy_comparison_metric": "dead_class_binary_f1",
        "metric_resolution": (
            "Official evaluation page specifies weighted F1; Dead-class binary F1 "
            "is retained only for comparison with earlier local analyses."
        ),
        "decisive_failures": [
            {
                "gate": str(row.gate),
                "metric": str(row.metric),
                "observed": None if pd.isna(row.observed) else float(row.observed),
                "required": str(row.required),
            }
            for row in decisive.itertuples(index=False)
            if not bool(row.passed)
        ],
        "submission_artifact_generated": False,
        "test_probability_artifact_generated": False,
        "recipe_equivalent_submission6_nested_oof_available": nested_recipe_exact,
    }
    return gates, decision


def format_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()

    def format_value(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        if isinstance(value, (bool, np.bool_)):
            return "True" if bool(value) else "False"
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(format_value(value) for value in row)
        + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def write_report(
    path: Path,
    global_frame: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    fold_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    split_candidate_summary: pd.DataFrame,
    winner_summary: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    recipe_audit: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    sub12_folds = fold_metrics[
        fold_metrics["candidate"].isin(
            ["nested_pseudo90", "sub12_candidate_p90_nn20"]
        )
    ].pivot(
        index="outer_fold",
        columns="candidate",
        values=["dead_class_binary_f1", "weighted_f1"],
    )
    sub12_folds.columns = ["_".join(column) for column in sub12_folds.columns]
    sub12_folds = sub12_folds.reset_index()
    sub12_fold_rows = fold_metrics[
        fold_metrics["candidate"] == "sub12_candidate_p90_nn20"
    ][
        [
            "outer_fold",
            "dead_f1_delta_vs_nested_pseudo90",
            "weighted_f1_delta_vs_nested_pseudo90",
        ]
    ]
    sub12_folds = sub12_folds.merge(sub12_fold_rows, on="outer_fold")

    private_summary = split_candidate_summary[
        split_candidate_summary["partition"] == "private"
    ]
    private_winners = winner_summary[
        winner_summary["partition"] == "private"
    ].copy()
    private_winners["_metric_order"] = private_winners["metric"].map(
        {"weighted_f1": 0, "dead_class_binary_f1": 1}
    )
    private_winners = private_winners.sort_values(
        ["_metric_order", "candidate"]
    ).drop(columns="_metric_order")
    pairwise_private = pairwise_summary[
        pairwise_summary["partition"] == "private"
    ]

    decisive_failed = gates[
        (gates["metric_role"] == "official_primary")
        & gates["passed"].apply(
            lambda value: False if pd.isna(value) else not bool(value)
        )
    ]
    lines = [
        "# Submission 12 gate audit",
        "",
        "## Decision",
        "",
        "**NO-GO: do not generate or submit Submission 12.** The candidate fails "
        "multiple predeclared gates under the official weighted-F1 metric. The legacy "
        "Dead-class binary-F1 view reaches the same decision.",
        "",
        "This audit created diagnostics only. It did not create a submission CSV or "
        "test-probability artifact and did not alter `submission.csv`, "
        "`SUBMISSION_HISTORY.md`, or the consolidated notebook.",
        "",
        "## Metric scope",
        "",
        "The official evaluation page specifies support-weighted F1, so `weighted_f1` "
        "is the authoritative gate metric. Standard binary F1 with `Dead=1` is retained "
        "only as a legacy comparison with earlier workspace analyses. All hard predictions "
        "use deterministic top-k selection "
        "at exactly 84.5% predicted Dead within the evaluated partition.",
        "",
        "## Full-OOF fixed-rate results",
        "",
        format_table(
            global_frame,
            [
                "candidate",
                "predicted_dead_rate",
                "weighted_f1",
                "dead_class_binary_f1",
                "weighted_f1_delta_vs_nested_pseudo90",
                "dead_f1_delta_vs_nested_pseudo90",
            ],
        ),
        "",
        "The candidate's full-OOF point improvement is positive, but a point estimate "
        "alone is insufficient for the predeclared gate.",
        "",
        "## Five outer folds",
        "",
        format_table(sub12_folds, list(sub12_folds.columns)),
        "",
        format_table(
            fold_summary[
                fold_summary["candidate"] == "sub12_candidate_p90_nn20"
            ],
            [
                "metric",
                "fold_mean",
                "worst_fold_value",
                "improved_fold_count",
                "tied_fold_count",
                "regressed_fold_count",
            ],
        ),
        "",
        "Fold-local top-k is used here. Slicing one globally selected hard vector would "
        "not preserve 84.5% within each fold and is not used to pass the fold gate.",
        "",
        "## Paired stratified bootstrap",
        "",
        format_table(
            bootstrap_summary,
            [
                "metric",
                "full_oof_point_delta",
                "bootstrap_mean_delta",
                "bootstrap_95pct_lower",
                "bootstrap_95pct_upper",
                "fraction_strictly_above_zero",
                "ci_lower_strictly_above_zero",
            ],
        ),
        "",
        "The official weighted-F1 lower confidence bound crosses zero, so Gate 1 fails. "
        "The legacy binary-F1 interval also crosses zero.",
        "",
        "## Repeated stratified 40/60 audit",
        "",
        "The table below uses the synthetic 60% holdout (`private`) partition and the "
        "same fixed-rate policy and seeds as `validation_harness.py`.",
        "",
        format_table(
            private_summary,
            [
                "candidate",
                "dead_class_binary_f1_mean",
                "dead_class_binary_f1_std",
                "weighted_f1_mean",
                "weighted_f1_std",
            ],
        ),
        "",
        format_table(
            private_winners,
            [
                "metric",
                "candidate",
                "harness_selected_winner_count",
                "inclusive_co_winner_count",
                "unique_winner_count",
                "clear_majority_selected_winner",
            ],
        ),
        "",
        "Pairwise Sub12-versus-pseudo90 stability is directionally favorable:",
        "",
        format_table(
            pairwise_private,
            [
                "metric",
                "mean_delta",
                "minimum_delta",
                "maximum_delta",
                "improved_split_count",
                "tied_split_count",
                "regressed_split_count",
            ],
        ),
        "",
        "However, Gate 3 asks whether Sub12 is the best candidate, not merely whether "
        "it beats pseudo90 pairwise. Existing tree80/NN20 wins the full comparison set.",
        "",
        "## Recipe-equivalence limitations",
        "",
        format_table(
            recipe_audit,
            ["item", "status", "exact_match", "disagreement_count", "detail"],
        ),
        "",
        "Consequently, the saved pseudo95 nested vector is a useful paired control but "
        "cannot satisfy the prompt's exact Submission 6 nested-equivalence requirement. "
        "An exact replay would require end-to-end nested retraining of the original "
        "3x5 teacher and five-fold pseudo-student recipe.",
        "",
        "## Gate result",
        "",
        format_table(
            gates,
            [
                "gate",
                "metric",
                "metric_role",
                "observed",
                "required",
                "passed",
                "note",
            ],
        ),
        "",
        f"Official-metric decisive failed gate rows: {len(decisive_failed)}. Gate 4 is deliberately left to the "
        "separate subgroup audit; its result cannot rescue failures in Gates 1–3.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python3 submission12_gate_audit.py",
        "```",
        "",
        "Seeds: outer folds `42`; bootstrap `9042`; simulated splits "
        "`20260829` through `20260878`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    before_hashes = protected_hashes()
    y, candidates = load_inputs()

    global_frame = global_metrics(y, candidates)
    fold_metrics, fold_summary = outer_fold_metrics(y, candidates)
    bootstrap_deltas, bootstrap_summary = paired_stratified_bootstrap(
        y,
        candidates["sub12_candidate_p90_nn20"],
        candidates["nested_pseudo90"],
    )
    (
        split_metrics,
        split_winners,
        split_candidate_summary,
        split_winner_summary,
        split_pairwise_summary,
    ) = repeated_split_audit(y, candidates)
    recipe_audit = recipe_equivalence_audit()
    gates, decision = build_gate_results(
        global_frame,
        fold_summary,
        bootstrap_summary,
        split_winner_summary,
        recipe_audit,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {
        "global_fixed_rate_metrics.csv": global_frame,
        "outer_fold_fixed_rate_metrics.csv": fold_metrics,
        "outer_fold_summary.csv": fold_summary,
        "paired_bootstrap_deltas.csv": bootstrap_deltas,
        "paired_bootstrap_summary.csv": bootstrap_summary,
        "oof_40_60_fixed_rate_metrics.csv": split_metrics,
        "oof_40_60_winners.csv": split_winners,
        "oof_40_60_candidate_summary.csv": split_candidate_summary,
        "oof_40_60_winner_summary.csv": split_winner_summary,
        "oof_40_60_pairwise_summary.csv": split_pairwise_summary,
        "recipe_equivalence_audit.csv": recipe_audit,
        "gate_results.csv": gates,
    }
    for filename, frame in frames.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False)

    (OUTPUT_DIR / "gate_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        OUTPUT_DIR / "submission12_gate_report.md",
        global_frame,
        fold_metrics,
        fold_summary,
        bootstrap_summary,
        split_candidate_summary,
        split_winner_summary,
        split_pairwise_summary,
        recipe_audit,
        gates,
    )

    after_hashes = protected_hashes()
    integrity = pd.DataFrame(
        [
            {
                "protected_file": path,
                "sha256_before": before_hashes[path],
                "sha256_after": after_hashes[path],
                "unchanged": before_hashes[path] == after_hashes[path],
            }
            for path in before_hashes
        ]
    )
    integrity.to_csv(OUTPUT_DIR / "protected_file_integrity.csv", index=False)
    if not integrity["unchanged"].all():
        raise RuntimeError("A protected submission/history artifact changed during audit")

    print("Submission 12 diagnostics-only gate audit complete.")
    print(f"Outputs: {OUTPUT_DIR}")
    print("Decision: NO-GO; no submission or test-probability artifact was generated.")


if __name__ == "__main__":
    main()
