"""Build the next NN-diversity submission from saved OOF/test predictions.

The experiment deliberately keeps the exact positive count from Submission 6.
That makes the leaderboard comparison test only whether the NN improves the
ranking near the decision boundary, rather than mixing in a threshold change.
"""

from pathlib import Path

import numpy as np
import pandas as pd


TREE_WEIGHT = 0.80
NN_WEIGHT = 0.20
TREE_PROBS_PATH = Path("archive/probs_v6_final.npy")
NN_PROBS_PATH = Path("archive/probs_nn.npy")
BASELINE_SUBMISSION_PATH = Path("submission6.csv")
PROBS_OUTPUT_PATH = Path("probs_nn20_blend.npy")
SUBMISSION_OUTPUT_PATH = Path("submission.csv")


def top_k_predictions(probabilities: np.ndarray, positive_count: int) -> np.ndarray:
    """Return exactly ``positive_count`` positives using deterministic ranking."""
    if not 0 < positive_count < len(probabilities):
        raise ValueError("positive_count must be between 1 and len(probabilities) - 1")

    # Stable sorting makes the result reproducible even if boundary scores tie.
    order = np.argsort(probabilities, kind="mergesort")
    predictions = np.zeros(len(probabilities), dtype=np.int8)
    predictions[order[-positive_count:]] = 1
    return predictions


print("=" * 80)
print("  NEXT SUBMISSION: 80% SUBMISSION 6 + 20% NEURAL NETWORK")
print("=" * 80)

test_df = pd.read_csv("test.csv")
baseline = pd.read_csv(BASELINE_SUBMISSION_PATH)
tree_probs = np.load(TREE_PROBS_PATH)
nn_probs = np.load(NN_PROBS_PATH)

expected_columns = ["patient_id", "vital_status"]
if baseline.columns.tolist() != expected_columns:
    raise ValueError(f"{BASELINE_SUBMISSION_PATH} must have columns {expected_columns}")
if not baseline["patient_id"].equals(test_df["patient_id"]):
    raise ValueError("Submission 6 patient IDs/order do not match test.csv")
if not baseline["vital_status"].isin(["Dead", "Alive"]).all():
    raise ValueError("Submission 6 contains an invalid vital_status value")
if len(tree_probs) != len(test_df) or len(nn_probs) != len(test_df):
    raise ValueError("Saved probability arrays do not match the test row count")
if not np.isfinite(tree_probs).all() or not np.isfinite(nn_probs).all():
    raise ValueError("Saved probability arrays contain NaN or infinite values")
if not np.isclose(TREE_WEIGHT + NN_WEIGHT, 1.0):
    raise ValueError("Blend weights must sum to one")

baseline_predictions = (baseline["vital_status"] == "Dead").to_numpy()
dead_count = int(baseline_predictions.sum())
blend_probs = TREE_WEIGHT * tree_probs + NN_WEIGHT * nn_probs
final_predictions = top_k_predictions(blend_probs, dead_count)

np.save(PROBS_OUTPUT_PATH, blend_probs)
submission = pd.DataFrame(
    {
        "patient_id": test_df["patient_id"],
        "vital_status": np.where(final_predictions == 1, "Dead", "Alive"),
    }
)
submission.to_csv(SUBMISSION_OUTPUT_PATH, index=False)

changed = final_predictions.astype(bool) != baseline_predictions
alive_to_dead = int((final_predictions.astype(bool) & ~baseline_predictions).sum())
dead_to_alive = int((~final_predictions.astype(bool) & baseline_predictions).sum())
sorted_probs = np.sort(blend_probs)
boundary_gap = sorted_probs[-dead_count] - sorted_probs[-dead_count - 1]

print(f"  Tree probabilities: {TREE_PROBS_PATH}")
print(f"  NN probabilities:   {NN_PROBS_PATH}")
print(f"  Blend:              {TREE_WEIGHT:.0%} tree + {NN_WEIGHT:.0%} NN")
print(f"  Dead:               {dead_count:,}/{len(submission):,} ({dead_count / len(submission):.3%})")
print(f"  Changed vs Sub 6:   {int(changed.sum()):,} rows")
print(f"    Alive -> Dead:    {alive_to_dead:,}")
print(f"    Dead -> Alive:    {dead_to_alive:,}")
print(f"  Boundary gap:       {boundary_gap:.8f}")
print(f"  Probabilities:      {PROBS_OUTPUT_PATH}")
print(f"  Submission:         {SUBMISSION_OUTPUT_PATH}")
print("=" * 80)
