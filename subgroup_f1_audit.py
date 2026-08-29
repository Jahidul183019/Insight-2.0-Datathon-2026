"""Reproducible subgroup audit for the fixed 84.5% prediction-rate policy.

This is a read-only modelling audit: it loads saved OOF vectors and writes
diagnostic tables under ``diagnostic_outputs/subgroup_audit``.  It does not
train models or create/modify a Kaggle submission.

The competition Evaluation page confirms support-weighted F1 as the official
metric.  Binary Dead-class F1 is retained only as a legacy diagnostic so older
local analyses can still be reconciled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "diagnostic_outputs" / "subgroup_audit"
FIXED_POSITIVE_RATE = 0.845
MIN_REPORT_ROWS = 200
MIN_INTERVENTION_ROWS = 500
MEANINGFUL_F1_GAP = -0.020
MATERIAL_REGRESSION = -0.001
REQUIRED_POINT_DELTA = 0.0004
N_OUTER_FOLDS = 5
OUTER_SEED = 42
BOOTSTRAP_SEED = 9_042

T_COLUMN = "derived_eod2018t_recode2018"
N_COLUMN = "derived_eod2018n_recode2018"
M_COLUMN = "derived_eod2018m_recode2018"

COMPOSITE_STAGE_DEFINITION = {
    "M1 distant/metastatic": "M starts with M1, regardless of T or N",
    "M0 N1-3 node-positive": "M0 and N is N1, N2, or N3",
    "M0 N0 T0-2 localized": "M0, N0, and T is T0, T1*, or T2*",
    "M0 N0 T3-4 locally advanced": "M0, N0, and T is T3 or T4*",
    "M0 indeterminate T/N": "M0 with T or N not classifiable above",
    "EOD unavailable (all blank)": "T, N, and M are all Blank(s)",
    "EOD not applicable (88)": "T, N, and M are all code 88",
    "Other/discordant EOD": "any remaining T/N/M combination",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated audit tables and report.",
    )
    parser.add_argument(
        "--fixed-positive-rate",
        type=float,
        default=FIXED_POSITIVE_RATE,
        help="Fraction selected as Dead by deterministic score rank.",
    )
    parser.add_argument(
        "--min-report-rows",
        type=int,
        default=MIN_REPORT_ROWS,
        help="Minimum subgroup size included in the report.",
    )
    parser.add_argument(
        "--min-intervention-rows",
        type=int,
        default=MIN_INTERVENTION_ROWS,
        help="Minimum subgroup size considered for an intervention.",
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=1_000,
        help="Stratified paired bootstrap repeats for pseudo90+NN vs pseudo90.",
    )
    args = parser.parse_args()
    if not 0.0 < args.fixed_positive_rate < 1.0:
        parser.error("--fixed-positive-rate must be between 0 and 1")
    if args.min_report_rows < 1:
        parser.error("--min-report-rows must be positive")
    if args.min_intervention_rows < args.min_report_rows:
        parser.error("--min-intervention-rows must be >= --min-report-rows")
    if args.bootstrap_repeats < 1:
        parser.error("--bootstrap-repeats must be positive")
    return args


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_probability_vector(name: str, values: np.ndarray, n_rows: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError(f"{name} must have shape ({n_rows},), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError(f"{name} contains values outside [0, 1]")
    return values


def top_rate_predictions(scores: np.ndarray, positive_rate: float) -> np.ndarray:
    """Select exactly round(rate * n) positives with stable tie-breaking."""

    n_positive = int(round(positive_rate * len(scores)))
    n_positive = min(max(n_positive, 0), len(scores))
    order = np.argsort(-scores, kind="mergesort")
    predictions = np.zeros(len(scores), dtype=np.int8)
    predictions[order[:n_positive]] = 1
    return predictions


def metric_pair(y_true: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    return (
        float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
    )


def derive_composite_stage(train: pd.DataFrame) -> pd.Series:
    """Build a transparent disease-extent grouping from the three EOD fields.

    This is deliberately not labelled AJCC stage I/II/III/IV because a formal
    AJCC mapping depends on cancer-specific rules and edition details that are
    not encoded here.
    """

    missing = [column for column in (T_COLUMN, N_COLUMN, M_COLUMN) if column not in train]
    if missing:
        raise ValueError(f"train.csv is missing EOD columns: {missing}")

    def classify(row: pd.Series) -> str:
        t_value, n_value, m_value = (str(row[column]) for column in (T_COLUMN, N_COLUMN, M_COLUMN))
        if m_value.startswith("M1"):
            return "M1 distant/metastatic"
        if m_value == "M0":
            if n_value in {"N1", "N2", "N3"}:
                return "M0 N1-3 node-positive"
            if n_value == "N0":
                if t_value == "T0" or t_value.startswith("T1") or t_value.startswith("T2"):
                    return "M0 N0 T0-2 localized"
                if t_value == "T3" or t_value.startswith("T4"):
                    return "M0 N0 T3-4 locally advanced"
            return "M0 indeterminate T/N"
        if (t_value, n_value, m_value) == ("Blank(s)", "Blank(s)", "Blank(s)"):
            return "EOD unavailable (all blank)"
        if (t_value, n_value, m_value) == ("88", "88", "88"):
            return "EOD not applicable (88)"
        return "Other/discordant EOD"

    return train[[T_COLUMN, N_COLUMN, M_COLUMN]].apply(classify, axis=1)


def make_group_masks(train: pd.DataFrame, min_rows: int) -> list[tuple[str, str, np.ndarray]]:
    dimensions = {
        "histologic_type_icdo3": train["histologic_type_icdo3"].astype(str),
        "derived_eod_composite_stage": train["derived_eod_composite_stage"].astype(str),
        "age_recode": train["age_recode"].fillna("<missing>").astype(str),
    }
    groups: list[tuple[str, str, np.ndarray]] = []
    for dimension, values in dimensions.items():
        for value in sorted(values.unique()):
            indices = np.flatnonzero(values.to_numpy() == value)
            if len(indices) >= min_rows:
                groups.append((dimension, value, indices))
    return groups


def global_candidate_metrics(
    y: np.ndarray,
    candidates: dict[str, np.ndarray],
    positive_rate: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    predictions: dict[str, np.ndarray] = {}
    for candidate, scores in candidates.items():
        hard_predictions = top_rate_predictions(scores, positive_rate)
        predictions[candidate] = hard_predictions
        dead_f1, weighted_f1 = metric_pair(y, hard_predictions)
        rows.append(
            {
                "candidate": candidate,
                "rows": len(y),
                "actual_dead_rate": float(y.mean()),
                "fixed_positive_rate": float(hard_predictions.mean()),
                "dead_class_f1": dead_f1,
                "weighted_f1": weighted_f1,
                "roc_auc": float(roc_auc_score(y, scores)),
            }
        )
    return pd.DataFrame(rows), predictions


def subgroup_metrics(
    y: np.ndarray,
    groups: list[tuple[str, str, np.ndarray]],
    candidates: dict[str, np.ndarray],
    global_predictions: dict[str, np.ndarray],
    global_metrics: pd.DataFrame,
    positive_rate: float,
    min_intervention_rows: int,
) -> pd.DataFrame:
    global_lookup = global_metrics.set_index("candidate").to_dict("index")
    rows = []
    for dimension, value, indices in groups:
        subgroup_y = y[indices]
        for candidate, scores in candidates.items():
            sliced_global_predictions = global_predictions[candidate][indices]
            within_predictions = top_rate_predictions(scores[indices], positive_rate)
            global_dead_f1, global_weighted_f1 = metric_pair(
                subgroup_y, sliced_global_predictions
            )
            within_dead_f1, within_weighted_f1 = metric_pair(subgroup_y, within_predictions)
            dead_gap = global_dead_f1 - global_lookup[candidate]["dead_class_f1"]
            weighted_gap = global_weighted_f1 - global_lookup[candidate]["weighted_f1"]
            rows.append(
                {
                    "dimension": dimension,
                    "slice": value,
                    "candidate": candidate,
                    "rows": len(indices),
                    "dead_count": int(subgroup_y.sum()),
                    "alive_count": int((subgroup_y == 0).sum()),
                    "actual_dead_rate": float(subgroup_y.mean()),
                    "global_policy_predicted_dead_rate": float(sliced_global_predictions.mean()),
                    "global_policy_dead_class_f1": global_dead_f1,
                    "global_policy_weighted_f1": global_weighted_f1,
                    "global_policy_dead_f1_gap": dead_gap,
                    "global_policy_weighted_f1_gap": weighted_gap,
                    "within_slice_predicted_dead_rate": float(within_predictions.mean()),
                    "within_slice_dead_class_f1": within_dead_f1,
                    "within_slice_weighted_f1": within_weighted_f1,
                    "within_minus_global_dead_f1": within_dead_f1 - global_dead_f1,
                    "within_minus_global_weighted_f1": within_weighted_f1 - global_weighted_f1,
                    "large_enough_for_intervention": len(indices) >= min_intervention_rows,
                    "weak_by_dead_f1": dead_gap <= MEANINGFUL_F1_GAP,
                    "weak_by_weighted_f1": weighted_gap <= MEANINGFUL_F1_GAP,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["dimension", "candidate", "global_policy_dead_class_f1", "slice"]
    ).reset_index(drop=True)


def fold_metrics(
    y: np.ndarray,
    groups: list[tuple[str, str, np.ndarray]],
    candidates: dict[str, np.ndarray],
    positive_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_ids = np.empty(len(y), dtype=np.int8)
    splitter = StratifiedKFold(N_OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED)
    for fold, (_, validation_indices) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        fold_ids[validation_indices] = fold

    rows = []
    fold_groups = [("GLOBAL", "all rows", np.arange(len(y)))] + groups
    for fold in range(1, N_OUTER_FOLDS + 1):
        fold_indices = np.flatnonzero(fold_ids == fold)
        position = np.full(len(y), -1, dtype=np.int64)
        position[fold_indices] = np.arange(len(fold_indices))
        fold_predictions = {
            candidate: top_rate_predictions(scores[fold_indices], positive_rate)
            for candidate, scores in candidates.items()
        }
        for dimension, value, group_indices in fold_groups:
            subgroup_indices = group_indices[fold_ids[group_indices] == fold]
            subgroup_y = y[subgroup_indices]
            local_positions = position[subgroup_indices]
            for candidate, scores in candidates.items():
                sliced_fold_predictions = fold_predictions[candidate][local_positions]
                within_predictions = top_rate_predictions(scores[subgroup_indices], positive_rate)
                global_dead_f1, global_weighted_f1 = metric_pair(
                    subgroup_y, sliced_fold_predictions
                )
                within_dead_f1, within_weighted_f1 = metric_pair(
                    subgroup_y, within_predictions
                )
                rows.append(
                    {
                        "dimension": dimension,
                        "slice": value,
                        "candidate": candidate,
                        "outer_fold": fold,
                        "rows": len(subgroup_indices),
                        "actual_dead_rate": float(subgroup_y.mean()),
                        "fold_global_policy_predicted_dead_rate": float(
                            sliced_fold_predictions.mean()
                        ),
                        "fold_global_policy_dead_class_f1": global_dead_f1,
                        "fold_global_policy_weighted_f1": global_weighted_f1,
                        "fold_within_slice_predicted_dead_rate": float(within_predictions.mean()),
                        "fold_within_slice_dead_class_f1": within_dead_f1,
                        "fold_within_slice_weighted_f1": within_weighted_f1,
                    }
                )

    raw = pd.DataFrame(rows).sort_values(
        ["dimension", "slice", "candidate", "outer_fold"]
    ).reset_index(drop=True)
    metric_columns = [
        "fold_global_policy_dead_class_f1",
        "fold_global_policy_weighted_f1",
        "fold_within_slice_dead_class_f1",
        "fold_within_slice_weighted_f1",
    ]
    summary = (
        raw.groupby(["dimension", "slice", "candidate"], as_index=False)[metric_columns]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return raw, summary


def comparison_metrics(
    y: np.ndarray,
    groups: list[tuple[str, str, np.ndarray]],
    global_predictions: dict[str, np.ndarray],
    fold_raw: pd.DataFrame,
    min_intervention_rows: int,
) -> pd.DataFrame:
    pseudo = global_predictions["nested_pseudo90"]
    combined = global_predictions["nested_pseudo90_nn20"]
    rows = []
    comparison_groups = [("GLOBAL", "all rows", np.arange(len(y)))] + groups
    for dimension, value, indices in comparison_groups:
        subgroup_y = y[indices]
        pseudo_predictions = pseudo[indices]
        combined_predictions = combined[indices]
        pseudo_dead_f1, pseudo_weighted_f1 = metric_pair(subgroup_y, pseudo_predictions)
        combined_dead_f1, combined_weighted_f1 = metric_pair(
            subgroup_y, combined_predictions
        )
        changed = pseudo_predictions != combined_predictions
        corrected = (pseudo_predictions != subgroup_y) & (combined_predictions == subgroup_y)
        harmed = (pseudo_predictions == subgroup_y) & (combined_predictions != subgroup_y)

        selection = fold_raw[
            (fold_raw["dimension"] == dimension)
            & (fold_raw["slice"] == value)
            & fold_raw["candidate"].isin(["nested_pseudo90", "nested_pseudo90_nn20"])
        ]
        pivot = selection.pivot(
            index="outer_fold",
            columns="candidate",
            values=[
                "fold_global_policy_dead_class_f1",
                "fold_global_policy_weighted_f1",
            ],
        )
        fold_deltas = pd.DataFrame(
            {
                "dead": pivot["fold_global_policy_dead_class_f1"][
                    "nested_pseudo90_nn20"
                ]
                - pivot["fold_global_policy_dead_class_f1"]["nested_pseudo90"],
                "weighted": pivot["fold_global_policy_weighted_f1"][
                    "nested_pseudo90_nn20"
                ]
                - pivot["fold_global_policy_weighted_f1"]["nested_pseudo90"],
            }
        )

        row = {
            "dimension": dimension,
            "slice": value,
            "rows": len(indices),
            "pseudo90_dead_class_f1": pseudo_dead_f1,
            "combined_dead_class_f1": combined_dead_f1,
            "combined_minus_pseudo90_dead_f1": combined_dead_f1 - pseudo_dead_f1,
            "pseudo90_weighted_f1": pseudo_weighted_f1,
            "combined_weighted_f1": combined_weighted_f1,
            "combined_minus_pseudo90_weighted_f1": combined_weighted_f1
            - pseudo_weighted_f1,
            "changed_predictions": int(changed.sum()),
            "pseudo90_dead_to_combined_alive": int(
                ((pseudo_predictions == 1) & (combined_predictions == 0)).sum()
            ),
            "pseudo90_alive_to_combined_dead": int(
                ((pseudo_predictions == 0) & (combined_predictions == 1)).sum()
            ),
            "corrected_predictions": int(corrected.sum()),
            "harmed_predictions": int(harmed.sum()),
            "net_corrected": int(corrected.sum() - harmed.sum()),
            "large_enough_for_intervention": len(indices) >= min_intervention_rows,
            "material_dead_f1_regression": combined_dead_f1 - pseudo_dead_f1
            <= MATERIAL_REGRESSION,
            "material_weighted_f1_regression": combined_weighted_f1 - pseudo_weighted_f1
            <= MATERIAL_REGRESSION,
        }
        for metric in ("dead", "weighted"):
            if fold_deltas.empty:
                values = np.array([], dtype=float)
            else:
                values = fold_deltas[metric].to_numpy(dtype=float)
            row.update(
                {
                    f"fold_{metric}_delta_mean": float(values.mean()) if len(values) else np.nan,
                    f"fold_{metric}_delta_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    f"fold_{metric}_delta_min": float(values.min()) if len(values) else np.nan,
                    f"fold_{metric}_delta_max": float(values.max()) if len(values) else np.nan,
                    f"fold_{metric}_wins": int((values > 0).sum()) if len(values) else 0,
                    f"fold_{metric}_ties": int((values == 0).sum()) if len(values) else 0,
                    f"fold_{metric}_losses": int((values < 0).sum()) if len(values) else 0,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["dimension", "combined_minus_pseudo90_dead_f1", "slice"]
    ).reset_index(drop=True)


def paired_stratified_bootstrap(
    y: np.ndarray,
    candidate_scores: np.ndarray,
    control_scores: np.ndarray,
    positive_rate: float,
    repeats: int,
) -> dict[str, object]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    dead_deltas = np.empty(repeats)
    weighted_deltas = np.empty(repeats)
    for iteration in range(repeats):
        sampled = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        candidate_predictions = top_rate_predictions(candidate_scores[sampled], positive_rate)
        control_predictions = top_rate_predictions(control_scores[sampled], positive_rate)
        candidate_dead, candidate_weighted = metric_pair(y[sampled], candidate_predictions)
        control_dead, control_weighted = metric_pair(y[sampled], control_predictions)
        dead_deltas[iteration] = candidate_dead - control_dead
        weighted_deltas[iteration] = candidate_weighted - control_weighted

    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "lower_95pct": float(np.quantile(values, 0.025)),
            "upper_95pct": float(np.quantile(values, 0.975)),
            "fraction_le_zero": float((values <= 0).mean()),
        }

    return {
        "method": "paired stratified bootstrap; fixed 84.5% rank policy re-applied per sample",
        "seed": BOOTSTRAP_SEED,
        "repeats": repeats,
        "dead_class_f1_delta": summarize(dead_deltas),
        "weighted_f1_delta": summarize(weighted_deltas),
    }


def markdown_table(frame: pd.DataFrame, columns: list[str], decimals: int = 4) -> str:
    if frame.empty:
        return "_None._"
    display = frame[columns].copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.{decimals}f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    global_metrics: pd.DataFrame,
    subgroup: pd.DataFrame,
    stability: pd.DataFrame,
    comparisons: pd.DataFrame,
    bootstrap: dict[str, object],
) -> None:
    tree = subgroup[subgroup["candidate"] == "frozen_tree"]
    weak = tree[
        tree["large_enough_for_intervention"]
        & tree["weak_by_weighted_f1"]
    ].sort_values("global_policy_weighted_f1")
    tree_stability = stability[stability["candidate"] == "frozen_tree"][
        [
            "dimension",
            "slice",
            "fold_global_policy_dead_class_f1_mean",
            "fold_global_policy_dead_class_f1_std",
            "fold_global_policy_dead_class_f1_min",
            "fold_global_policy_dead_class_f1_max",
            "fold_global_policy_weighted_f1_mean",
            "fold_global_policy_weighted_f1_std",
        ]
    ]
    weak = weak.merge(tree_stability, on=["dimension", "slice"], how="left")
    primary_regressions = comparisons[
        comparisons["large_enough_for_intervention"]
        & comparisons["material_weighted_f1_regression"]
        & (comparisons["dimension"] != "GLOBAL")
    ]
    legacy_only_regressions = comparisons[
        comparisons["large_enough_for_intervention"]
        & comparisons["material_dead_f1_regression"]
        & ~comparisons["material_weighted_f1_regression"]
        & (comparisons["dimension"] != "GLOBAL")
    ]
    dead_bootstrap = bootstrap["dead_class_f1_delta"]
    weighted_bootstrap = bootstrap["weighted_f1_delta"]
    global_lookup = global_metrics.set_index("candidate")
    weighted_point_delta = float(
        global_lookup.loc["nested_pseudo90_nn20", "weighted_f1"]
        - global_lookup.loc["nested_pseudo90", "weighted_f1"]
    )
    conservative_gate = (
        weighted_point_delta > REQUIRED_POINT_DELTA
        and weighted_bootstrap["lower_95pct"] > 0
        and primary_regressions.empty
    )

    stage_lines = [
        f"- `{name}`: {definition}." for name, definition in COMPOSITE_STAGE_DEFINITION.items()
    ]
    report = [
        "# Subgroup F1 Audit",
        "",
        "## Decision",
        "",
        (
            "**PASS:** the conservative evidence gate supports further consideration."
            if conservative_gate
            else "**NO-SUBMIT GATE:** on official weighted F1, the prospective pseudo90 + "
            "20% NN blend is not clearly outside uncertainty and/or has a material "
            "large-subgroup regression."
        ),
        "",
        "This is a saved-OOF diagnostic, not a reconstruction of Kaggle's hidden split. "
        "The NN vector is ordinary five-fold OOF rather than a jointly nested rebuild with "
        "pseudo90, so the combined estimate remains a screening result.",
        "",
        "## Metric policy",
        "",
        f"Predictions select exactly `round({args.fixed_positive_rate} * n)` Dead rows by "
        "descending probability with stable tie-breaking. **Support-weighted F1 is the "
        "official competition metric and drives every gate below.** Binary Dead-class F1 "
        "is reported only as a legacy diagnostic.",
        "",
        markdown_table(
            global_metrics,
            [
                "candidate",
                "rows",
                "fixed_positive_rate",
                "weighted_f1",
                "dead_class_f1",
                "roc_auc",
            ],
            decimals=6,
        ),
        "",
        "## Composite-stage definition",
        "",
        "This is an explicit disease-extent grouping, not a claim of formal AJCC I-IV stage:",
        "",
        *stage_lines,
        "",
        "## Weak, intervention-sized frozen-tree slices on official weighted F1",
        "",
        f"All slices with at least {args.min_report_rows} rows are in `subgroup_slice_metrics.csv`. "
        f"The table below requires at least {args.min_intervention_rows} rows and an official "
        f"weighted-F1 gap of at least {abs(MEANINGFUL_F1_GAP):.3f}.",
        "",
        markdown_table(
            weak,
            [
                "dimension",
                "slice",
                "rows",
                "actual_dead_rate",
                "global_policy_predicted_dead_rate",
                "global_policy_weighted_f1",
                "global_policy_weighted_f1_gap",
                "within_slice_weighted_f1",
                "fold_global_policy_weighted_f1_mean",
                "fold_global_policy_weighted_f1_std",
                "global_policy_dead_class_f1",
                "global_policy_dead_f1_gap",
                "within_slice_dead_class_f1",
                "fold_global_policy_dead_class_f1_mean",
                "fold_global_policy_dead_class_f1_std",
                "fold_global_policy_dead_class_f1_min",
                "fold_global_policy_dead_class_f1_max",
            ],
        ),
        "",
        "F1 is prevalence-sensitive. In particular, forcing 84.5% Dead inside a low-death "
        "slice can reduce F1; a weak slice is not by itself evidence for a subgroup threshold.",
        "",
        "## Pseudo90 + NN regressions on official weighted F1",
        "",
        markdown_table(
            primary_regressions,
            [
                "dimension",
                "slice",
                "rows",
                "combined_minus_pseudo90_weighted_f1",
                "combined_minus_pseudo90_dead_f1",
                "changed_predictions",
                "corrected_predictions",
                "harmed_predictions",
                "fold_weighted_delta_mean",
                "fold_weighted_delta_min",
                "fold_weighted_delta_max",
                "fold_weighted_wins",
                "fold_weighted_ties",
                "fold_weighted_losses",
            ],
        ),
        "",
        "## Legacy Dead-class diagnostic-only regressions",
        "",
        markdown_table(
            legacy_only_regressions,
            [
                "dimension",
                "slice",
                "rows",
                "combined_minus_pseudo90_dead_f1",
                "combined_minus_pseudo90_weighted_f1",
                "changed_predictions",
                "fold_dead_delta_mean",
                "fold_dead_delta_min",
                "fold_dead_delta_max",
                "fold_dead_wins",
                "fold_dead_ties",
                "fold_dead_losses",
            ],
        ),
        "",
        "## Paired bootstrap: combined minus pseudo90",
        "",
        f"- **Official weighted F1:** observed full-OOF delta `{weighted_point_delta:+.6f}`; "
        f"bootstrap mean `{weighted_bootstrap['mean']:+.6f}`, 95% interval "
        f"`[{weighted_bootstrap['lower_95pct']:+.6f}, {weighted_bootstrap['upper_95pct']:+.6f}]`.",
        f"- Legacy Dead-class F1: mean `{dead_bootstrap['mean']:+.6f}`, 95% interval "
        f"`[{dead_bootstrap['lower_95pct']:+.6f}, {dead_bootstrap['upper_95pct']:+.6f}]`.",
        "",
        "## Reproducibility outputs",
        "",
        "- `global_candidate_metrics.csv`: full-OOF metrics.",
        "- `subgroup_slice_metrics.csv`: both policies and both F1 definitions for every slice.",
        "- `subgroup_fold_metrics.csv`: raw five-fold measurements using StratifiedKFold seed 42.",
        "- `subgroup_stability_summary.csv`: mean, standard deviation, minimum, and maximum.",
        "- `pseudo90_nn20_regression_audit.csv`: hard-label changes and fold deltas.",
        "- `subgroup_audit_summary.json`: definitions, artifact hashes, bootstrap, and gate status.",
        "",
        f"The stability table contains {len(stability):,} candidate/slice summaries.",
    ]
    output_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train.csv": require_file(ROOT / "train.csv"),
        "y_step1.npy": require_file(ROOT / "y_step1.npy"),
        "oof_step1.npy": require_file(ROOT / "oof_step1.npy"),
        "archive/oof_nn.npy": require_file(ROOT / "archive" / "oof_nn.npy"),
        "nested_oof_predictions.npz": require_file(
            ROOT / "diagnostic_outputs" / "pseudo90_nested" / "nested_oof_predictions.npz"
        ),
    }
    train = pd.read_csv(paths["train.csv"])
    if "vital_status" not in train:
        raise ValueError("train.csv does not contain vital_status")
    y = train["vital_status"].eq("Dead").to_numpy(dtype=np.int8)
    saved_y = np.load(paths["y_step1.npy"], allow_pickle=False).astype(np.int8)
    if not np.array_equal(y, saved_y):
        raise ValueError("y_step1.npy does not exactly match train.csv labels/order")

    tree = validate_probability_vector(
        "oof_step1.npy", np.load(paths["oof_step1.npy"], allow_pickle=False), len(y)
    )
    nn = validate_probability_vector(
        "archive/oof_nn.npy", np.load(paths["archive/oof_nn.npy"], allow_pickle=False), len(y)
    )
    nested = np.load(paths["nested_oof_predictions.npz"], allow_pickle=False)
    required_keys = {"y", "pseudo90_oof"}
    missing_keys = required_keys.difference(nested.files)
    if missing_keys:
        raise ValueError(f"nested OOF archive is missing keys: {sorted(missing_keys)}")
    if not np.array_equal(nested["y"].astype(np.int8), y):
        raise ValueError("nested OOF labels do not match train.csv labels/order")
    pseudo90 = validate_probability_vector("pseudo90_oof", nested["pseudo90_oof"], len(y))

    candidates = {
        "frozen_tree": tree,
        "nested_pseudo90": pseudo90,
        "nested_pseudo90_nn20": 0.80 * pseudo90 + 0.20 * nn,
    }
    train = train.copy()
    train["derived_eod_composite_stage"] = derive_composite_stage(train)
    groups = make_group_masks(train, args.min_report_rows)

    global_metrics, global_predictions = global_candidate_metrics(
        y, candidates, args.fixed_positive_rate
    )
    subgroup = subgroup_metrics(
        y,
        groups,
        candidates,
        global_predictions,
        global_metrics,
        args.fixed_positive_rate,
        args.min_intervention_rows,
    )
    fold_raw, stability = fold_metrics(y, groups, candidates, args.fixed_positive_rate)
    comparisons = comparison_metrics(
        y, groups, global_predictions, fold_raw, args.min_intervention_rows
    )
    bootstrap = paired_stratified_bootstrap(
        y,
        candidates["nested_pseudo90_nn20"],
        candidates["nested_pseudo90"],
        args.fixed_positive_rate,
        args.bootstrap_repeats,
    )

    global_metrics.to_csv(output_dir / "global_candidate_metrics.csv", index=False)
    subgroup.to_csv(output_dir / "subgroup_slice_metrics.csv", index=False)
    fold_raw.to_csv(output_dir / "subgroup_fold_metrics.csv", index=False)
    stability.to_csv(output_dir / "subgroup_stability_summary.csv", index=False)
    comparisons.to_csv(output_dir / "pseudo90_nn20_regression_audit.csv", index=False)

    primary_regressions = comparisons[
        comparisons["large_enough_for_intervention"]
        & comparisons["material_weighted_f1_regression"]
        & (comparisons["dimension"] != "GLOBAL")
    ]
    legacy_dead_regressions = comparisons[
        comparisons["large_enough_for_intervention"]
        & comparisons["material_dead_f1_regression"]
        & (comparisons["dimension"] != "GLOBAL")
    ]
    global_lookup = global_metrics.set_index("candidate")
    weighted_point_delta = float(
        global_lookup.loc["nested_pseudo90_nn20", "weighted_f1"]
        - global_lookup.loc["nested_pseudo90", "weighted_f1"]
    )
    conservative_gate = (
        weighted_point_delta > REQUIRED_POINT_DELTA
        and bootstrap["weighted_f1_delta"]["lower_95pct"] > 0
        and primary_regressions.empty
    )
    summary = {
        "purpose": "saved-OOF subgroup screening; not hidden-test reconstruction",
        "official_metric": "support-weighted F1",
        "official_metric_source": "competition Evaluation page",
        "dead_class_f1_role": "legacy diagnostic only",
        "fixed_positive_rate": args.fixed_positive_rate,
        "min_report_rows": args.min_report_rows,
        "min_intervention_rows": args.min_intervention_rows,
        "meaningful_f1_gap": MEANINGFUL_F1_GAP,
        "material_regression": MATERIAL_REGRESSION,
        "outer_folds": N_OUTER_FOLDS,
        "outer_seed": OUTER_SEED,
        "composite_stage_is_formal_ajcc": False,
        "composite_stage_definition": COMPOSITE_STAGE_DEFINITION,
        "input_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "reported_slice_count": len(groups),
        "bootstrap": bootstrap,
        "observed_combined_minus_pseudo90_weighted_f1": weighted_point_delta,
        "required_observed_weighted_f1_delta": REQUIRED_POINT_DELTA,
        "material_large_subgroup_regressions": int(len(primary_regressions)),
        "legacy_dead_class_material_large_subgroup_regressions": int(
            len(legacy_dead_regressions)
        ),
        "conservative_submission_gate_pass": bool(conservative_gate),
        "gate_verdict": "consider" if conservative_gate else "do_not_submit",
        "caveats": [
            "F1 is prevalence-sensitive across subgroups.",
            "Binary Dead-class F1 is retained only to reconcile legacy local analyses.",
            "archive/oof_nn.npy is ordinary OOF, not jointly nested with pseudo90.",
            "A subgroup feature or weight still requires full leakage-safe nested retraining.",
        ],
    }
    (output_dir / "subgroup_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        output_dir / "subgroup_audit_report.md",
        args,
        global_metrics,
        subgroup,
        stability,
        comparisons,
        bootstrap,
    )

    print("Subgroup audit complete.")
    print(f"Outputs: {output_dir}")
    print(global_metrics.to_string(index=False))
    print(json.dumps({"gate_verdict": summary["gate_verdict"], "bootstrap": bootstrap}, indent=2))


if __name__ == "__main__":
    main()
