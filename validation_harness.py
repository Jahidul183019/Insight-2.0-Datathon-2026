"""Prediction-level validation harness for saved OOF and submission artifacts.

This script is deliberately lightweight: it does not fit a model, modify an
existing prediction artifact, contact Kaggle, or assume access to hidden test
labels.  It uses saved out-of-fold (OOF) predictions to simulate repeated
40/60 "selection/holdout" evaluations and audits the hard predictions from
Submissions 6--9 separately.

The 40/60 exercise is a proxy for checking whether a decision made on a
smaller labelled partition remains sensible on a larger labelled partition.
It is *not* a reconstruction of Kaggle's public/private split.

Default outputs are written only under:
    diagnostic_outputs/validation_harness/
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "diagnostic_outputs" / "validation_harness"

# These scores are metadata reported in SUBMISSION_HISTORY.md.  They are never
# treated as row-level labels or recomputed by this script.
REPORTED_LB_SCORES = {
    "submission_6": 0.877258,
    "submission_7": 0.876170,
    "submission_8": 0.876305,
    "submission_9": 0.875022,
}

SUBMISSION_FILES = {
    "submission_6": ROOT / "submission6.csv",
    "submission_7": ROOT / "archive" / "submission7.csv",
    "submission_8": ROOT / "archive" / "submission8.csv",
    "submission_9": ROOT / "archive" / "submission9.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for clearly named diagnostic outputs.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=50,
        help="Number of repeated stratified 40/60 prediction-level audits.",
    )
    parser.add_argument(
        "--public-fraction",
        type=float,
        default=0.40,
        help="Fraction used as the synthetic selection partition.",
    )
    parser.add_argument(
        "--fixed-positive-rate",
        type=float,
        default=0.845,
        help="Positive prediction rate tested by the fixed-rate policy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260829,
        help="First split seed; subsequent splits use consecutive seeds.",
    )
    args = parser.parse_args()

    if args.n_splits < 1:
        parser.error("--n-splits must be at least 1")
    if not 0.0 < args.public_fraction < 1.0:
        parser.error("--public-fraction must be between 0 and 1")
    if not 0.0 < args.fixed_positive_rate < 1.0:
        parser.error("--fixed-positive-rate must be between 0 and 1")
    return args


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return path


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
    """Return exactly round(rate * n) positives with deterministic tie-breaking."""
    n_positive = int(round(positive_rate * len(scores)))
    n_positive = min(max(n_positive, 0), len(scores))
    order = np.argsort(-scores, kind="mergesort")
    predictions = np.zeros(len(scores), dtype=np.int8)
    predictions[order[:n_positive]] = 1
    return predictions


def best_probability_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose an exact F1-optimal threshold on one labelled partition."""
    order = np.argsort(-scores, kind="mergesort")
    sorted_y = y_true[order]
    cumulative_tp = np.cumsum(sorted_y)
    predicted_positive = np.arange(1, len(scores) + 1)
    denominator = predicted_positive + int(y_true.sum())
    f1_values = np.divide(
        2.0 * cumulative_tp,
        denominator,
        out=np.zeros_like(cumulative_tp, dtype=float),
        where=denominator > 0,
    )
    best_k = int(np.flatnonzero(f1_values == f1_values.max())[0]) + 1

    sorted_scores = scores[order]
    if best_k == len(scores):
        return float(np.nextafter(sorted_scores[-1], -np.inf))
    upper = float(sorted_scores[best_k - 1])
    lower = float(sorted_scores[best_k])
    if upper == lower:
        # Exact top-k selection is impossible across a tied boundary.  The
        # threshold remains deterministic; the recorded positive rate exposes
        # any difference from best_k / n.
        return upper
    return (upper + lower) / 2.0


def best_rank_rate(
    y_true: np.ndarray,
    scores: np.ndarray,
    rate_grid: np.ndarray,
    tie_reference: float,
) -> float:
    scored_rates = []
    for rate in rate_grid:
        predictions = top_rate_predictions(scores, float(rate))
        score = f1_score(y_true, predictions, zero_division=0)
        scored_rates.append((float(score), -abs(float(rate) - tie_reference), float(rate)))
    # Prefer the rate nearest the fixed-rate reference when F1 ties exactly.
    return max(scored_rates)[2]


def make_oof_candidates(tree_oof: np.ndarray, nn_oof: np.ndarray) -> dict[str, dict[str, object]]:
    weights = [
        ("tree_oof", 1.00, 0.00),
        ("tree95_nn05", 0.95, 0.05),
        ("tree90_nn10", 0.90, 0.10),
        ("tree80_nn20", 0.80, 0.20),
        ("tree70_nn30_previous_candidate", 0.70, 0.30),
        ("nn_oof", 0.00, 1.00),
    ]
    return {
        name: {
            "scores": tree_weight * tree_oof + nn_weight * nn_oof,
            "tree_weight": tree_weight,
            "nn_weight": nn_weight,
        }
        for name, tree_weight, nn_weight in weights
    }


def global_oof_metrics(
    y: np.ndarray,
    candidates: dict[str, dict[str, object]],
    fixed_rate: float,
    rate_grid: np.ndarray,
) -> pd.DataFrame:
    rows = []
    tree_scores = np.asarray(candidates["tree_oof"]["scores"])
    tree_fixed = top_rate_predictions(tree_scores, fixed_rate)

    for name, candidate in candidates.items():
        scores = np.asarray(candidate["scores"])
        threshold = best_probability_threshold(y, scores)
        threshold_predictions = (scores >= threshold).astype(np.int8)
        fixed_predictions = top_rate_predictions(scores, fixed_rate)
        rank_rate = best_rank_rate(y, scores, rate_grid, fixed_rate)
        rank_predictions = top_rate_predictions(scores, rank_rate)
        rows.append(
            {
                "candidate": name,
                "tree_weight": candidate["tree_weight"],
                "nn_weight": candidate["nn_weight"],
                "roc_auc": roc_auc_score(y, scores),
                "pearson_correlation_with_tree": np.corrcoef(tree_scores, scores)[0, 1],
                "f1_at_probability_0_5": f1_score(
                    y, scores >= 0.5, zero_division=0
                ),
                "positive_rate_at_probability_0_5": float((scores >= 0.5).mean()),
                "descriptive_oracle_threshold": threshold,
                "descriptive_oracle_threshold_f1": f1_score(
                    y, threshold_predictions, zero_division=0
                ),
                "descriptive_oracle_threshold_positive_rate": float(
                    threshold_predictions.mean()
                ),
                "fixed_rate": fixed_rate,
                "fixed_rate_f1": f1_score(y, fixed_predictions, zero_division=0),
                "fixed_rate_disagreement_with_tree": float(
                    (fixed_predictions != tree_fixed).mean()
                ),
                "descriptive_oracle_rank_rate": rank_rate,
                "descriptive_oracle_rank_rate_f1": f1_score(
                    y, rank_predictions, zero_division=0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("candidate").reset_index(drop=True)


def repeated_split_audit(
    y: np.ndarray,
    candidates: dict[str, dict[str, object]],
    public_fraction: float,
    fixed_rate: float,
    rate_grid: np.ndarray,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    all_indices = np.arange(len(y))

    for split_id in range(n_splits):
        split_seed = seed + split_id
        public_idx, private_idx = train_test_split(
            all_indices,
            train_size=public_fraction,
            stratify=y,
            random_state=split_seed,
        )

        for name, candidate in candidates.items():
            scores = np.asarray(candidate["scores"])
            public_scores = scores[public_idx]
            private_scores = scores[private_idx]
            public_y = y[public_idx]
            private_y = y[private_idx]

            threshold = best_probability_threshold(public_y, public_scores)
            public_threshold_predictions = (public_scores >= threshold).astype(np.int8)
            private_threshold_predictions = (private_scores >= threshold).astype(np.int8)

            selected_rate = best_rank_rate(
                public_y, public_scores, rate_grid, fixed_rate
            )
            public_rank_predictions = top_rate_predictions(public_scores, selected_rate)
            private_rank_predictions = top_rate_predictions(private_scores, selected_rate)

            policies = [
                (
                    "public_selected_probability_threshold",
                    threshold,
                    public_threshold_predictions,
                    private_threshold_predictions,
                ),
                (
                    "public_selected_rank_rate_0.80_to_0.92",
                    selected_rate,
                    public_rank_predictions,
                    private_rank_predictions,
                ),
                (
                    f"fixed_rank_rate_{fixed_rate:.3f}",
                    fixed_rate,
                    top_rate_predictions(public_scores, fixed_rate),
                    top_rate_predictions(private_scores, fixed_rate),
                ),
            ]

            for policy, selection_value, public_predictions, private_predictions in policies:
                rows.append(
                    {
                        "split_id": split_id,
                        "split_seed": split_seed,
                        "policy": policy,
                        "candidate": name,
                        "tree_weight": candidate["tree_weight"],
                        "nn_weight": candidate["nn_weight"],
                        "selection_value": selection_value,
                        "public_rows": len(public_idx),
                        "private_rows": len(private_idx),
                        "public_prevalence": float(public_y.mean()),
                        "private_prevalence": float(private_y.mean()),
                        "public_positive_rate": float(public_predictions.mean()),
                        "private_positive_rate": float(private_predictions.mean()),
                        "public_f1": f1_score(
                            public_y, public_predictions, zero_division=0
                        ),
                        "private_f1": f1_score(
                            private_y, private_predictions, zero_division=0
                        ),
                    }
                )

    return pd.DataFrame(rows)


def summarize_split_candidates(split_results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        split_results.groupby(["policy", "candidate"], as_index=False)
        .agg(
            public_f1_mean=("public_f1", "mean"),
            public_f1_std=("public_f1", "std"),
            private_f1_mean=("private_f1", "mean"),
            private_f1_std=("private_f1", "std"),
            public_positive_rate_mean=("public_positive_rate", "mean"),
            private_positive_rate_mean=("private_positive_rate", "mean"),
            selection_value_mean=("selection_value", "mean"),
            selection_value_std=("selection_value", "std"),
        )
        .sort_values(["policy", "private_f1_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return summary


def split_selection_stability(split_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_id, policy), group in split_results.groupby(["split_id", "policy"]):
        group = group.sort_values("candidate").copy()
        group["public_rank"] = group["public_f1"].rank(
            method="average", ascending=False
        )
        group["private_rank"] = group["private_f1"].rank(
            method="average", ascending=False
        )
        if group["public_rank"].nunique() == 1 or group["private_rank"].nunique() == 1:
            rank_correlation = np.nan
        else:
            rank_correlation = float(
                np.corrcoef(group["public_rank"], group["private_rank"])[0, 1]
            )

        public_winner_row = group.sort_values(
            ["public_f1", "candidate"], ascending=[False, True]
        ).iloc[0]
        private_winner_row = group.sort_values(
            ["private_f1", "candidate"], ascending=[False, True]
        ).iloc[0]
        selected_private_f1 = float(
            group.loc[
                group["candidate"] == public_winner_row["candidate"], "private_f1"
            ].iloc[0]
        )
        rows.append(
            {
                "split_id": split_id,
                "policy": policy,
                "public_selected_candidate": public_winner_row["candidate"],
                "private_best_candidate": private_winner_row["candidate"],
                "winner_exact_match": bool(
                    public_winner_row["candidate"] == private_winner_row["candidate"]
                ),
                "rank_correlation": rank_correlation,
                "selected_candidate_private_f1": selected_private_f1,
                "best_private_f1": float(private_winner_row["private_f1"]),
                "private_f1_regret": float(private_winner_row["private_f1"])
                - selected_private_f1,
            }
        )
    return pd.DataFrame(rows)


def summarize_selection_stability(stability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in stability.groupby("policy"):
        public_win_counts = group["public_selected_candidate"].value_counts()
        private_win_counts = group["private_best_candidate"].value_counts()
        rows.append(
            {
                "policy": policy,
                "n_splits": len(group),
                "winner_exact_match_rate": float(group["winner_exact_match"].mean()),
                "rank_correlation_mean": float(group["rank_correlation"].mean()),
                "rank_correlation_std": float(group["rank_correlation"].std()),
                "private_f1_regret_mean": float(group["private_f1_regret"].mean()),
                "private_f1_regret_max": float(group["private_f1_regret"].max()),
                "most_frequent_public_winner": public_win_counts.index[0],
                "most_frequent_public_winner_count": int(public_win_counts.iloc[0]),
                "most_frequent_private_winner": private_win_counts.index[0],
                "most_frequent_private_winner_count": int(private_win_counts.iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("policy").reset_index(drop=True)


def load_submission_audit(test_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    required_columns = {"patient_id", "vital_status"}
    test_ids = pd.Index(test_df["patient_id"])
    if not test_ids.is_unique:
        raise ValueError("test.csv contains duplicate patient_id values")

    rows = []
    predictions = {}
    reported_order = {
        name: rank
        for rank, (name, _) in enumerate(
            sorted(REPORTED_LB_SCORES.items(), key=lambda item: item[1], reverse=True),
            start=1,
        )
    }

    for name, path in SUBMISSION_FILES.items():
        require_file(path)
        submission = pd.read_csv(path)
        if set(submission.columns) != required_columns or len(submission.columns) != 2:
            raise ValueError(
                f"{path} must contain exactly patient_id and vital_status"
            )
        if submission["patient_id"].duplicated().any():
            raise ValueError(f"{path} contains duplicate patient_id values")
        if set(submission["patient_id"]) != set(test_ids):
            raise ValueError(f"{path} patient IDs do not exactly match test.csv")
        invalid = set(submission["vital_status"].dropna().unique()) - {"Dead", "Alive"}
        if invalid or submission["vital_status"].isna().any():
            raise ValueError(f"{path} has invalid labels: {sorted(invalid)}")

        aligned = submission.set_index("patient_id").loc[test_ids, "vital_status"]
        dead = aligned.eq("Dead").to_numpy(dtype=bool)
        predictions[name] = dead
        rows.append(
            {
                "candidate": name,
                "file": str(path.relative_to(ROOT)),
                "rows": len(submission),
                "dead_count": int(dead.sum()),
                "dead_rate": float(dead.mean()),
                "reported_public_lb_f1": REPORTED_LB_SCORES[name],
                "reported_public_lb_rank": reported_order[name],
                "score_source": "SUBMISSION_HISTORY.md metadata; not recomputed",
                "like_for_like_oof_available": False,
                "locally_scoreable": False,
                "local_validation_note": (
                    "No recipe-equivalent OOF vector and no hidden test labels"
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("candidate"), predictions


def pairwise_submission_disagreement(
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for left, right in itertools.combinations(sorted(predictions), 2):
        left_dead = predictions[left]
        right_dead = predictions[right]
        disagreement = left_dead != right_dead
        union_dead = left_dead | right_dead
        rows.append(
            {
                "left_candidate": left,
                "right_candidate": right,
                "rows": len(left_dead),
                "disagreement_count": int(disagreement.sum()),
                "disagreement_rate": float(disagreement.mean()),
                "left_dead_right_alive": int((left_dead & ~right_dead).sum()),
                "left_alive_right_dead": int((~left_dead & right_dead).sum()),
                "both_dead": int((left_dead & right_dead).sum()),
                "both_alive": int((~left_dead & ~right_dead).sum()),
                "dead_set_jaccard": float(
                    (left_dead & right_dead).sum() / max(int(union_dead.sum()), 1)
                ),
                "reported_lb_score_left": REPORTED_LB_SCORES[left],
                "reported_lb_score_right": REPORTED_LB_SCORES[right],
                "reported_lb_score_left_minus_right": (
                    REPORTED_LB_SCORES[left] - REPORTED_LB_SCORES[right]
                ),
                "interpretation": "prediction-structure proxy, not a local F1 comparison",
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output_path: Path,
    y: np.ndarray,
    global_metrics: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    submission_audit: pd.DataFrame,
    pairwise: pd.DataFrame,
    n_splits: int,
    public_fraction: float,
    fixed_rate: float,
) -> None:
    fixed_policy = f"fixed_rank_rate_{fixed_rate:.3f}"
    fixed_summary = candidate_summary[candidate_summary["policy"] == fixed_policy]
    fixed_best = fixed_summary.sort_values("private_f1_mean", ascending=False).iloc[0]
    current_candidate = fixed_summary[
        fixed_summary["candidate"] == "tree80_nn20"
    ].iloc[0]
    tree = fixed_summary[fixed_summary["candidate"] == "tree_oof"].iloc[0]
    candidate_delta = float(
        current_candidate["private_f1_mean"] - tree["private_f1_mean"]
    )

    global_best_fixed = global_metrics.sort_values("fixed_rate_f1", ascending=False).iloc[0]
    closest_pair = pairwise.sort_values("disagreement_count").iloc[0]
    reported_order = " > ".join(
        submission_audit.sort_values("reported_public_lb_rank")["candidate"]
    )

    lines = [
        "# Validation-harness report",
        "",
        "## Scope",
        "",
        (
            f"This is a prediction-level audit using {len(y):,} labelled training rows, "
            f"{n_splits} repeated stratified {public_fraction:.0%}/"
            f"{1-public_fraction:.0%} splits, and already-saved OOF probabilities. "
            "It performs no model training and never reads Kaggle public/private labels."
        ),
        "",
        "The words `public` and `private` in CSV column names mean only the synthetic "
        "40/60 selection and holdout partitions. They are not Kaggle partitions.",
        "",
        "## Direct findings",
        "",
        (
            f"- At a fixed {fixed_rate:.1%} positive rate on full OOF data, the best "
            f"tested tree/NN candidate was `{global_best_fixed['candidate']}` "
            f"(F1 {global_best_fixed['fixed_rate_f1']:.6f})."
        ),
        (
            f"- Across repeated 40/60 splits under that fixed-rate policy, "
            f"`{fixed_best['candidate']}` had the highest mean holdout F1 "
            f"({fixed_best['private_f1_mean']:.6f})."
        ),
        (
            f"- The current 80% tree / 20% NN candidate changed mean holdout F1 by "
            f"{candidate_delta:+.6f} relative to the saved tree OOF proxy. This comparison "
            "does not include pseudo-label retraining."
        ),
        (
            f"- The reported leaderboard order is `{reported_order}`, but this harness "
            "cannot honestly reproduce or refute that order from the available files."
        ),
        (
            f"- The closest hard-submission pair is `{closest_pair['left_candidate']}` "
            f"vs `{closest_pair['right_candidate']}` with "
            f"{int(closest_pair['disagreement_count']):,} changed rows "
            f"({closest_pair['disagreement_rate']:.2%}). Disagreement is a diversity/risk "
            "proxy, not evidence that either side is correct."
        ),
        "",
        "## Why Submissions 6--9 cannot be locally ranked",
        "",
        "Only their test probabilities or hard labels are available. There are no "
        "recipe-equivalent OOF vectors for the 95%-pseudo-label model (Sub 6), stack "
        "(Sub 7), super-blend (Sub 8), and 98%-pseudo-label model (Sub 9). Test labels "
        "are hidden. Reported leaderboard scores are included only as historical metadata; "
        "they are not validation targets.",
        "",
        "A valid replay requires regenerating all four candidates inside the same outer "
        "folds, including pseudo-label creation using only information allowed within each "
        "fold. Comparing their current hard test files against training labels would be "
        "invalid because the rows do not correspond.",
        "",
        "## Important limitations",
        "",
        "- The split is applied to locked OOF predictions, not to end-to-end model training. "
        "It tests selection and threshold stability, not full training-set shift.",
        "- `oof_step1.npy` already contains a globally selected tree blend, so its estimate "
        "is not fully nested and can be mildly optimistic.",
        "- The NN and tree OOF vectors were produced by different CV schemes. Their blend "
        "comparison is useful as a screening proxy, not a submission guarantee.",
        "- Repeated splits overlap. Their standard deviations describe observed split "
        "dispersion and are not independent-sample confidence intervals.",
        "- A fixed positive-rate policy depends only on ranking; it does not evaluate "
        "probability calibration.",
        "",
        "## Output guide",
        "",
        "- `oof_global_candidate_metrics.csv`: full-OOF screening metrics.",
        "- `oof_40_60_split_results.csv`: every candidate/policy/split result.",
        "- `oof_40_60_candidate_summary.csv`: mean and dispersion by candidate.",
        "- `oof_40_60_selection_stability.csv`: winner/rank transfer per split.",
        "- `oof_40_60_selection_summary.csv`: aggregate selection stability.",
        "- `submission_6_to_9_artifact_audit.csv`: format, rate, and score provenance.",
        "- `submission_6_to_9_pairwise_disagreement.csv`: hard-prediction diversity.",
        "",
        "The detailed selection-policy summary follows:",
        "",
        "```csv",
        stability_summary.to_csv(index=False, float_format="%.6f").rstrip(),
        "```",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(require_file(ROOT / "train.csv"))
    test_df = pd.read_csv(require_file(ROOT / "test.csv"))
    if "vital_status" not in train_df:
        raise ValueError("train.csv does not contain vital_status")
    if "patient_id" not in test_df:
        raise ValueError("test.csv does not contain patient_id")

    y_from_train = train_df["vital_status"].eq("Dead").to_numpy(dtype=np.int8)
    y_saved = np.load(require_file(ROOT / "y_step1.npy"), allow_pickle=False)
    if not np.array_equal(y_from_train, y_saved):
        raise ValueError("y_step1.npy does not exactly match train.csv row order/labels")
    y = y_from_train

    tree_oof = validate_probability_vector(
        "oof_step1.npy",
        np.load(require_file(ROOT / "oof_step1.npy"), allow_pickle=False),
        len(y),
    )
    nn_oof = validate_probability_vector(
        "archive/oof_nn.npy",
        np.load(require_file(ROOT / "archive" / "oof_nn.npy"), allow_pickle=False),
        len(y),
    )
    candidates = make_oof_candidates(tree_oof, nn_oof)

    # The bounded grid includes the locally interesting region while preventing
    # arbitrary extreme positive-rate searches on the synthetic public split.
    rate_grid = np.round(np.arange(0.80, 0.92001, 0.0025), 4)

    global_metrics = global_oof_metrics(
        y, candidates, args.fixed_positive_rate, rate_grid
    )
    split_results = repeated_split_audit(
        y=y,
        candidates=candidates,
        public_fraction=args.public_fraction,
        fixed_rate=args.fixed_positive_rate,
        rate_grid=rate_grid,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    candidate_summary = summarize_split_candidates(split_results)
    stability = split_selection_stability(split_results)
    stability_summary = summarize_selection_stability(stability)
    submission_audit, submission_predictions = load_submission_audit(test_df)
    pairwise = pairwise_submission_disagreement(submission_predictions)

    outputs = {
        "oof_global_candidate_metrics.csv": global_metrics,
        "oof_40_60_split_results.csv": split_results,
        "oof_40_60_candidate_summary.csv": candidate_summary,
        "oof_40_60_selection_stability.csv": stability,
        "oof_40_60_selection_summary.csv": stability_summary,
        "submission_6_to_9_artifact_audit.csv": submission_audit,
        "submission_6_to_9_pairwise_disagreement.csv": pairwise,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    report_path = output_dir / "validation_report.md"
    write_report(
        output_path=report_path,
        y=y,
        global_metrics=global_metrics,
        candidate_summary=candidate_summary,
        stability_summary=stability_summary,
        submission_audit=submission_audit,
        pairwise=pairwise,
        n_splits=args.n_splits,
        public_fraction=args.public_fraction,
        fixed_rate=args.fixed_positive_rate,
    )

    print("Validation harness complete.")
    print(f"Outputs: {output_dir}")
    print(
        "Important: this is a saved-OOF selection proxy, not a reconstruction "
        "of Kaggle public/private labels or a local ranking of Submissions 6--9."
    )


if __name__ == "__main__":
    main()
