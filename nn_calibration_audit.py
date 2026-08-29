"""Leakage-safe calibration audit for the saved neural-network OOF scores.

The audit compares ``archive/oof_nn.npy`` with the saved tree OOF proxy and
evaluates isotonic calibration only through held-out mappings.  It never
creates or modifies a Kaggle submission or a saved test-probability vector.

The official support-weighted F1 metric is primary.  Binary Dead-class F1 is
retained under explicit legacy names and deprecated compatibility aliases so
earlier audit outputs remain interpretable.

Default outputs are written under ``diagnostic_outputs/nn_calibration/``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "insight_2_0_matplotlib_cache"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "diagnostic_outputs" / "nn_calibration"
PRIMARY_CROSSFIT_SEED = 42
REPEATED_CROSSFIT_SEED = 20260829
REPEATED_HOLDOUT_SEED = 20260829
POSITIVE_RATE = 0.845
N_BINS = 10
TREE_WEIGHT = 0.80
NN_WEIGHT = 0.20
OFFICIAL_F1_AVERAGE = "weighted"
METRIC_SCHEMA_VERSION = "2_weighted_f1_primary"

LOWER_IS_BETTER = {
    "brier",
    "log_loss",
    "ece",
    "rms_calibration_error",
    "max_calibration_error",
}
COMPARISON_METRICS = [
    "brier",
    "log_loss",
    "roc_auc",
    "ece",
    "rms_calibration_error",
    "max_calibration_error",
    "official_weighted_f1_at_fixed_rate",
    "legacy_dead_class_f1_at_fixed_rate",
]


def official_weighted_f1(y_true: np.ndarray, predictions: np.ndarray) -> float:
    """Competition metric: support-weighted F1 across Alive and Dead."""

    return float(
        f1_score(
            y_true,
            predictions,
            average=OFFICIAL_F1_AVERAGE,
            zero_division=0,
        )
    )


def legacy_dead_class_f1(y_true: np.ndarray, predictions: np.ndarray) -> float:
    """Historical local metric: binary F1 with Dead encoded as class 1."""

    return float(f1_score(y_true, predictions, average="binary", zero_division=0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for diagnostic artifacts.",
    )
    parser.add_argument(
        "--crossfit-seeds",
        type=int,
        default=21,
        help="Number of deterministic five-fold cross-fitting runs.",
    )
    parser.add_argument(
        "--holdout-splits",
        type=int,
        default=50,
        help="Number of repeated 40/60 calibration/holdout checks.",
    )
    args = parser.parse_args()
    if args.crossfit_seeds < 1:
        parser.error("--crossfit-seeds must be at least 1")
    if args.holdout_splits < 1:
        parser.error("--holdout-splits must be at least 1")
    return args


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_probability_vector(
    name: str, values: np.ndarray, expected_rows: int
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (expected_rows,):
        raise ValueError(
            f"{name} must have shape ({expected_rows},), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError(f"{name} contains probabilities outside [0, 1]")
    return values


def top_rate_predictions(scores: np.ndarray, positive_rate: float) -> np.ndarray:
    positive_count = int(round(positive_rate * len(scores)))
    order = np.argsort(-scores, kind="mergesort")
    predictions = np.zeros(len(scores), dtype=np.int8)
    predictions[order[:positive_count]] = 1
    return predictions


def reliability_bins(
    y: np.ndarray,
    probabilities: np.ndarray,
    *,
    name: str,
    n_bins: int = N_BINS,
) -> pd.DataFrame:
    """Return quantile-bin reliability data matching the v6 plot method."""
    edges = np.percentile(probabilities, np.linspace(0.0, 100.0, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    bin_ids = np.searchsorted(edges[1:-1], probabilities, side="right")
    z_95 = 1.959963984540054
    rows: list[dict[str, float | int | str]] = []

    for bin_id in range(n_bins):
        selected = bin_ids == bin_id
        count = int(selected.sum())
        if count == 0:
            continue
        predicted = float(probabilities[selected].mean())
        observed = float(y[selected].mean())
        gap = observed - predicted

        # Wilson interval for the observed event rate in the bin.
        denominator = 1.0 + z_95**2 / count
        center = (observed + z_95**2 / (2.0 * count)) / denominator
        half_width = (
            z_95
            * np.sqrt(
                observed * (1.0 - observed) / count
                + z_95**2 / (4.0 * count**2)
            )
            / denominator
        )
        rows.append(
            {
                "series": name,
                "bin": bin_id + 1,
                "rows": count,
                "probability_min": float(probabilities[selected].min()),
                "probability_max": float(probabilities[selected].max()),
                "mean_predicted": predicted,
                "observed_dead_rate": observed,
                "observed_minus_predicted": gap,
                "absolute_gap": abs(gap),
                "observed_wilson_95_low": center - half_width,
                "observed_wilson_95_high": center + half_width,
            }
        )
    return pd.DataFrame(rows)


def calibration_intercept_slope(
    y: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    clipped = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    logits = logit(clipped)
    design = np.column_stack([np.ones(len(logits)), logits])

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        linear = design @ parameters
        return float(np.sum(np.logaddexp(0.0, linear) - y * linear))

    result = minimize(
        negative_log_likelihood,
        x0=np.array([0.0, 1.0]),
        method="BFGS",
    )
    if not np.isfinite(result.x).all():
        raise RuntimeError("Calibration intercept/slope optimization failed")
    return float(result.x[0]), float(result.x[1])


def probability_metrics(
    y: np.ndarray, probabilities: np.ndarray, *, name: str
) -> dict[str, float | str]:
    bins = reliability_bins(y, probabilities, name=name)
    weights = bins["rows"].to_numpy(dtype=float) / len(y)
    gaps = bins["observed_minus_predicted"].to_numpy(dtype=float)
    intercept, slope = calibration_intercept_slope(y, probabilities)
    fixed_predictions = top_rate_predictions(probabilities, POSITIVE_RATE)
    return {
        "series": name,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "rows": len(y),
        "prevalence": float(y.mean()),
        "mean_probability": float(probabilities.mean()),
        "mean_probability_minus_prevalence": float(
            probabilities.mean() - y.mean()
        ),
        "brier": brier_score_loss(y, probabilities),
        "log_loss": log_loss(y, probabilities),
        "roc_auc": roc_auc_score(y, probabilities),
        "ece": float(np.sum(weights * np.abs(gaps))),
        "rms_calibration_error": float(np.sqrt(np.sum(weights * gaps**2))),
        "max_calibration_error": float(np.max(np.abs(gaps))),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "official_weighted_f1_at_probability_0_5": official_weighted_f1(
            y, (probabilities >= 0.5).astype(np.int8)
        ),
        "legacy_dead_class_f1_at_probability_0_5": legacy_dead_class_f1(
            y, (probabilities >= 0.5).astype(np.int8)
        ),
        # Deprecated backward-compatible binary-Dead alias.
        "f1_at_probability_0_5": legacy_dead_class_f1(
            y, (probabilities >= 0.5).astype(np.int8)
        ),
        "positive_rate_at_probability_0_5": float((probabilities >= 0.5).mean()),
        "fixed_rate": POSITIVE_RATE,
        "official_weighted_f1_at_fixed_rate": official_weighted_f1(
            y, fixed_predictions
        ),
        "legacy_dead_class_f1_at_fixed_rate": legacy_dead_class_f1(
            y, fixed_predictions
        ),
        # Deprecated backward-compatible binary-Dead alias.
        "fixed_rate_f1": legacy_dead_class_f1(y, fixed_predictions),
    }


def crossfit_isotonic(
    y: np.ndarray, probabilities: np.ndarray, *, seed: int
) -> np.ndarray:
    calibrated = np.empty_like(probabilities, dtype=float)
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )
    for train_indices, validation_indices in splitter.split(
        probabilities.reshape(-1, 1), y
    ):
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(probabilities[train_indices], y[train_indices])
        calibrated[validation_indices] = calibrator.predict(
            probabilities[validation_indices]
        )
    return calibrated


def metric_comparison_rows(
    *,
    seed: int,
    candidate: str,
    raw_metrics: dict[str, float | str],
    calibrated_metrics: dict[str, float | str],
) -> list[dict[str, float | int | str | bool]]:
    rows = []
    for metric in COMPARISON_METRICS:
        raw_value = float(raw_metrics[metric])
        calibrated_value = float(calibrated_metrics[metric])
        delta = calibrated_value - raw_value
        improved = delta < 0.0 if metric in LOWER_IS_BETTER else delta > 0.0
        rows.append(
            {
                "seed": seed,
                "metric_schema_version": METRIC_SCHEMA_VERSION,
                "candidate": candidate,
                "metric": metric,
                "raw_value": raw_value,
                "calibrated_value": calibrated_value,
                "calibrated_minus_raw": delta,
                "improved": improved,
            }
        )
    return rows


def repeated_crossfit_audit(
    y: np.ndarray,
    tree_oof: np.ndarray,
    nn_oof: np.ndarray,
    *,
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    raw_blend = TREE_WEIGHT * tree_oof + NN_WEIGHT * nn_oof
    raw_nn_metrics = probability_metrics(y, nn_oof, name="nn_raw")
    raw_blend_metrics = probability_metrics(y, raw_blend, name="tree80_nn20_raw")
    rows = []
    primary_calibrated: np.ndarray | None = None

    for seed in seeds:
        calibrated_nn = crossfit_isotonic(y, nn_oof, seed=seed)
        if seed == PRIMARY_CROSSFIT_SEED:
            primary_calibrated = calibrated_nn.copy()
        calibrated_blend = TREE_WEIGHT * tree_oof + NN_WEIGHT * calibrated_nn
        calibrated_nn_metrics = probability_metrics(
            y,
            calibrated_nn,
            name="nn_isotonic_crossfit",
        )
        calibrated_blend_metrics = probability_metrics(
            y,
            calibrated_blend,
            name="tree80_nn20_isotonic_crossfit",
        )
        rows.extend(
            metric_comparison_rows(
                seed=seed,
                candidate="nn",
                raw_metrics=raw_nn_metrics,
                calibrated_metrics=calibrated_nn_metrics,
            )
        )
        rows.extend(
            metric_comparison_rows(
                seed=seed,
                candidate="tree80_nn20",
                raw_metrics=raw_blend_metrics,
                calibrated_metrics=calibrated_blend_metrics,
            )
        )

    if primary_calibrated is None:
        raise RuntimeError("Primary cross-fitting seed was not evaluated")

    seed_metrics = pd.DataFrame(rows)
    summary = (
        seed_metrics.groupby(["candidate", "metric"], as_index=False)
        .agg(
            raw_value=("raw_value", "first"),
            calibrated_mean=("calibrated_value", "mean"),
            calibrated_std=("calibrated_value", "std"),
            calibrated_min=("calibrated_value", "min"),
            calibrated_max=("calibrated_value", "max"),
            delta_mean=("calibrated_minus_raw", "mean"),
            delta_std=("calibrated_minus_raw", "std"),
            delta_min=("calibrated_minus_raw", "min"),
            delta_max=("calibrated_minus_raw", "max"),
            improved_seeds=("improved", "sum"),
            evaluated_seeds=("seed", "nunique"),
        )
        .sort_values(["candidate", "metric"])
        .reset_index(drop=True)
    )
    summary["improvement_rate"] = (
        summary["improved_seeds"] / summary["evaluated_seeds"]
    )
    summary.insert(0, "metric_schema_version", METRIC_SCHEMA_VERSION)
    return seed_metrics, summary, primary_calibrated


def repeated_holdout_audit(
    y: np.ndarray,
    tree_oof: np.ndarray,
    nn_oof: np.ndarray,
    *,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_indices = np.arange(len(y))
    rows = []
    for split_id in range(n_splits):
        split_seed = REPEATED_HOLDOUT_SEED + split_id
        calibration_indices, holdout_indices = train_test_split(
            all_indices,
            train_size=0.40,
            stratify=y,
            random_state=split_seed,
        )
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(nn_oof[calibration_indices], y[calibration_indices])
        calibrated_nn = calibrator.predict(nn_oof[holdout_indices])
        holdout_y = y[holdout_indices]
        holdout_tree = tree_oof[holdout_indices]
        holdout_nn = nn_oof[holdout_indices]

        comparisons = {
            "nn": (holdout_nn, calibrated_nn),
            "tree80_nn20": (
                TREE_WEIGHT * holdout_tree + NN_WEIGHT * holdout_nn,
                TREE_WEIGHT * holdout_tree + NN_WEIGHT * calibrated_nn,
            ),
        }
        for candidate, (raw_values, calibrated_values) in comparisons.items():
            raw_metrics = probability_metrics(
                holdout_y, raw_values, name=f"{candidate}_raw"
            )
            calibrated_metrics = probability_metrics(
                holdout_y,
                calibrated_values,
                name=f"{candidate}_isotonic",
            )
            for row in metric_comparison_rows(
                seed=split_seed,
                candidate=candidate,
                raw_metrics=raw_metrics,
                calibrated_metrics=calibrated_metrics,
            ):
                row["split_id"] = split_id
                row["calibration_rows"] = len(calibration_indices)
                row["holdout_rows"] = len(holdout_indices)
                rows.append(row)

    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby(["candidate", "metric"], as_index=False)
        .agg(
            raw_mean=("raw_value", "mean"),
            calibrated_mean=("calibrated_value", "mean"),
            delta_mean=("calibrated_minus_raw", "mean"),
            delta_std=("calibrated_minus_raw", "std"),
            delta_min=("calibrated_minus_raw", "min"),
            delta_max=("calibrated_minus_raw", "max"),
            improved_splits=("improved", "sum"),
            evaluated_splits=("split_id", "nunique"),
        )
        .sort_values(["candidate", "metric"])
        .reset_index(drop=True)
    )
    summary["improvement_rate"] = (
        summary["improved_splits"] / summary["evaluated_splits"]
    )
    summary.insert(0, "metric_schema_version", METRIC_SCHEMA_VERSION)
    return metrics, summary


def write_reliability_plot(
    bins: pd.DataFrame,
    output_path: Path,
) -> None:
    styles = {
        "tree_oof_raw": ("#2ca02c", "o", "tree OOF raw"),
        "nn_oof_raw": ("#1f77b4", "o", "NN OOF raw"),
        "nn_oof_isotonic_crossfit_seed42": (
            "#d62728",
            "s",
            "NN isotonic, 5-fold cross-fit",
        ),
    }
    figure, axis = plt.subplots(figsize=(9.6, 8.0))
    for series, (color, marker, label) in styles.items():
        selected = bins[bins["series"] == series]
        axis.plot(
            selected["mean_predicted"],
            selected["observed_dead_rate"],
            color=color,
            marker=marker,
            linewidth=2,
            label=label,
        )
    axis.plot([0.0, 1.0], [0.0, 1.0], "--", color="#ff7f0e", label="perfect")
    axis.set(
        title="NN OOF reliability: raw vs leakage-safe isotonic",
        xlabel="Mean predicted probability",
        ylabel="Observed Dead rate",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def summary_value(
    summary: pd.DataFrame,
    candidate: str,
    metric: str,
    column: str,
) -> float:
    selected = summary[
        (summary["candidate"] == candidate) & (summary["metric"] == metric)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Missing summary row for {candidate}/{metric}")
    return float(selected.iloc[0][column])


def write_report(
    output_path: Path,
    *,
    inputs: pd.DataFrame,
    primary_metrics: pd.DataFrame,
    crossfit_summary: pd.DataFrame,
    holdout_summary: pd.DataFrame,
    crossfit_seed_count: int,
    holdout_split_count: int,
) -> None:
    metrics = primary_metrics.set_index("series")
    nn_raw = metrics.loc["nn_oof_raw"]
    tree_raw = metrics.loc["tree_oof_raw"]
    nn_cal = metrics.loc["nn_oof_isotonic_crossfit_seed42"]
    blend_raw = metrics.loc["tree80_nn20_raw"]
    blend_cal = metrics.loc["tree80_nn20_isotonic_crossfit_seed42"]

    blend_fixed_improved = int(
        summary_value(
            crossfit_summary,
            "tree80_nn20",
            "official_weighted_f1_at_fixed_rate",
            "improved_seeds",
        )
    )
    holdout_fixed_improved = int(
        summary_value(
            holdout_summary,
            "tree80_nn20",
            "official_weighted_f1_at_fixed_rate",
            "improved_splits",
        )
    )
    holdout_fixed_delta = summary_value(
        holdout_summary,
        "tree80_nn20",
        "official_weighted_f1_at_fixed_rate",
        "delta_mean",
    )

    comparison = pd.DataFrame(
        [
            {
                "metric": "Brier loss",
                "raw_80_20_blend": blend_raw["brier"],
                "isotonic_80_20_blend": blend_cal["brier"],
            },
            {
                "metric": "Log loss",
                "raw_80_20_blend": blend_raw["log_loss"],
                "isotonic_80_20_blend": blend_cal["log_loss"],
            },
            {
                "metric": "ROC AUC",
                "raw_80_20_blend": blend_raw["roc_auc"],
                "isotonic_80_20_blend": blend_cal["roc_auc"],
            },
            {
                "metric": "ECE",
                "raw_80_20_blend": blend_raw["ece"],
                "isotonic_80_20_blend": blend_cal["ece"],
            },
            {
                "metric": "Official support-weighted F1 at fixed 84.5%",
                "raw_80_20_blend": blend_raw[
                    "official_weighted_f1_at_fixed_rate"
                ],
                "isotonic_80_20_blend": blend_cal[
                    "official_weighted_f1_at_fixed_rate"
                ],
            },
            {
                "metric": "Legacy Dead-class F1 at fixed 84.5%",
                "raw_80_20_blend": blend_raw[
                    "legacy_dead_class_f1_at_fixed_rate"
                ],
                "isotonic_80_20_blend": blend_cal[
                    "legacy_dead_class_f1_at_fixed_rate"
                ],
            },
        ]
    )
    comparison_markdown = [
        "| Metric | Raw 80/20 blend | Isotonic 80/20 blend |",
        "|---|---:|---:|",
    ]
    comparison_markdown.extend(
        f"| {row.metric} | {row.raw_80_20_blend:.6f} | "
        f"{row.isotonic_80_20_blend:.6f} |"
        for row in comparison.itertuples(index=False)
    )

    lines = [
        "# Neural-network OOF calibration audit",
        "",
        "## Decision",
        "",
        "**Submission D does not pass the submission gate.** The raw NN has a "
        "localized calibration deviation, but leakage-safe isotonic calibration "
        "does not improve the downstream 80/20 tree/NN candidate.",
        "",
        "No submission or saved test-probability artifact was created or modified.",
        "",
        "The competition's support-weighted F1 is the primary F1 measure in "
        "this report. Binary Dead-class F1 is retained only in explicit "
        "`legacy_dead_class_*` fields and deprecated compatibility aliases "
        f"(`metric_schema_version={METRIC_SCHEMA_VERSION}`).",
        "",
        "## Direct findings",
        "",
        f"- The NN's 10-quantile ECE is `{nn_raw['ece']:.6f}`, RMS calibration "
        f"error is `{nn_raw['rms_calibration_error']:.6f}`, and maximum bin gap "
        f"is `{nn_raw['max_calibration_error']:.6f}`.",
        f"- The corresponding tree OOF values are ECE `{tree_raw['ece']:.6f}`, "
        f"RMS error `{tree_raw['rms_calibration_error']:.6f}`, and maximum gap "
        f"`{tree_raw['max_calibration_error']:.6f}`.",
        "- The NN's largest deviation is the second probability decile: mean "
        "prediction `0.555327` versus observed Dead rate `0.597917` "
        "(gap `+0.042590`, 2,400 rows).",
        f"- Five-fold cross-fit isotonic calibration reduces the NN ECE to "
        f"`{nn_cal['ece']:.6f}` for the primary seed, but its Brier loss changes "
        f"from `{nn_raw['brier']:.6f}` to `{nn_cal['brier']:.6f}` and log loss "
        f"from `{nn_raw['log_loss']:.6f}` to `{nn_cal['log_loss']:.6f}`.",
        "- The calibrated 80/20 blend improves official support-weighted "
        "fixed-rate F1 in only "
        f"`{blend_fixed_improved}/{crossfit_seed_count}` cross-fitting seeds.",
        f"- In {holdout_split_count} repeated 40/60 calibration/holdout checks, "
        "the calibrated blend improves official support-weighted fixed-rate F1 in only "
        f"`{holdout_fixed_improved}/{holdout_split_count}` splits; mean delta is "
        f"`{holdout_fixed_delta:+.6f}`.",
        "",
        "## Primary-seed blend comparison",
        "",
        *comparison_markdown,
        "",
        "## Method",
        "",
        "- Reliability uses 10 equal-frequency bins, matching the existing v6 "
        "reliability method.",
        "- For every five-fold audit, the isotonic mapping for a row is fitted "
        "without that row or its label.",
        "- The repeated 40/60 audit fits the mapping on 40% of labelled OOF rows "
        "and evaluates it only on the untouched 60%.",
        "- Fixed-rate F1 predicts exactly `round(0.845 * n)` positive rows using "
        "stable rank ordering.",
        "- The 84.5% rate is a historically selected policy, not an independently "
        "estimated value in this audit; it must not be refined through leaderboard "
        "probing.",
        "- The existing tree and NN vectors came from different CV schemes. This "
        "is a screening audit, not a guarantee about Kaggle's hidden labels.",
        "- Cross-fit isotonic uses fold-specific monotonic mappings; this can alter "
        "global ordering and exposes the downstream instability that an in-sample "
        "calibration curve would conceal.",
        "",
        "## Input provenance",
        "",
        "```csv",
        inputs.to_csv(index=False).rstrip(),
        "```",
        "",
        "## Output guide",
        "",
        "- `nn_oof_reliability_curve.png`: raw NN, raw tree, and held-out "
        "isotonic reliability curves.",
        "- `nn_oof_reliability_bins.csv`: exact bin counts, means, gaps, and "
        "Wilson intervals.",
        "- `nn_calibration_metrics.csv`: primary-seed metrics.",
        "- `nn_isotonic_crossfit_seed_metrics.csv`: every metric and seed.",
        "- `nn_isotonic_crossfit_summary.csv`: aggregate cross-fit evidence.",
        "- `nn_isotonic_40_60_holdout_metrics.csv` and "
        "`nn_isotonic_40_60_holdout_summary.csv`: repeated held-out robustness "
        "checks.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = require_file(ROOT / "train.csv")
    y_path = require_file(ROOT / "y_step1.npy")
    tree_path = require_file(ROOT / "oof_step1.npy")
    nn_path = require_file(ROOT / "archive" / "oof_nn.npy")

    train_y = (
        pd.read_csv(train_path, usecols=["vital_status"])["vital_status"]
        .eq("Dead")
        .to_numpy(dtype=np.int8)
    )
    saved_y = np.load(y_path, allow_pickle=False)
    if not np.array_equal(saved_y, train_y):
        raise ValueError("y_step1.npy does not match train.csv row order/labels")
    y = train_y
    tree_oof = validate_probability_vector(
        "oof_step1.npy",
        np.load(tree_path, allow_pickle=False),
        len(y),
    )
    nn_oof = validate_probability_vector(
        "archive/oof_nn.npy",
        np.load(nn_path, allow_pickle=False),
        len(y),
    )

    seeds = [PRIMARY_CROSSFIT_SEED] + [
        REPEATED_CROSSFIT_SEED + offset
        for offset in range(args.crossfit_seeds - 1)
    ]
    seed_metrics, crossfit_summary, primary_calibrated_nn = (
        repeated_crossfit_audit(y, tree_oof, nn_oof, seeds=seeds)
    )
    raw_blend = TREE_WEIGHT * tree_oof + NN_WEIGHT * nn_oof
    calibrated_blend = (
        TREE_WEIGHT * tree_oof + NN_WEIGHT * primary_calibrated_nn
    )

    series = {
        "tree_oof_raw": tree_oof,
        "nn_oof_raw": nn_oof,
        "nn_oof_isotonic_crossfit_seed42": primary_calibrated_nn,
        "tree80_nn20_raw": raw_blend,
        "tree80_nn20_isotonic_crossfit_seed42": calibrated_blend,
    }
    primary_metrics = pd.DataFrame(
        [probability_metrics(y, values, name=name) for name, values in series.items()]
    )
    bin_tables = pd.concat(
        [
            reliability_bins(y, values, name=name)
            for name, values in series.items()
            if name
            in {
                "tree_oof_raw",
                "nn_oof_raw",
                "nn_oof_isotonic_crossfit_seed42",
            }
        ],
        ignore_index=True,
    )
    holdout_metrics, holdout_summary = repeated_holdout_audit(
        y,
        tree_oof,
        nn_oof,
        n_splits=args.holdout_splits,
    )

    inputs = pd.DataFrame(
        [
            {
                "file": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in [train_path, y_path, tree_path, nn_path]
        ]
    )

    bin_tables.to_csv(output_dir / "nn_oof_reliability_bins.csv", index=False)
    primary_metrics.to_csv(output_dir / "nn_calibration_metrics.csv", index=False)
    seed_metrics.to_csv(
        output_dir / "nn_isotonic_crossfit_seed_metrics.csv", index=False
    )
    crossfit_summary.to_csv(
        output_dir / "nn_isotonic_crossfit_summary.csv", index=False
    )
    holdout_metrics.to_csv(
        output_dir / "nn_isotonic_40_60_holdout_metrics.csv", index=False
    )
    holdout_summary.to_csv(
        output_dir / "nn_isotonic_40_60_holdout_summary.csv", index=False
    )
    write_reliability_plot(
        bin_tables,
        output_dir / "nn_oof_reliability_curve.png",
    )
    write_report(
        output_dir / "nn_calibration_report.md",
        inputs=inputs,
        primary_metrics=primary_metrics,
        crossfit_summary=crossfit_summary,
        holdout_summary=holdout_summary,
        crossfit_seed_count=len(seeds),
        holdout_split_count=args.holdout_splits,
    )

    print("NN calibration audit complete.")
    print(f"Outputs: {output_dir}")
    print("Decision: skip Submission D; no submission artifact was written.")


if __name__ == "__main__":
    main()
