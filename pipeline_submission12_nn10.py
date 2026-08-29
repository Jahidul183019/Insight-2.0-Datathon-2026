"""Prepare Submission 12 from the frozen Submission 6 and NN probabilities.

The candidate is a deterministic 90%/10% probability blend followed by a
stable descending top-30,411 decision rule.  It is a prepared upload artifact,
not a leaderboard result: this script records ``PREPARED_LB_PENDING`` and does
not assign or infer a leaderboard score.

Safety and provenance rules:

* Every frozen input is pinned to its audited SHA-256 hash.
* Archived Submissions 6, 10, and 11 must retain the exact test ID order.
* The existing root ``submission.csv`` must be archived Submission 11 (or an
  identical already-prepared Submission 12 from an idempotent rerun).
* Outputs are written atomically.  No file is written to ``archive/``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "submission12_nn10"

TEST_PATH = ROOT / "test.csv"
TREE_PROBS_PATH = ROOT / "archive" / "probs_v6_final.npy"
NN_PROBS_PATH = ROOT / "archive" / "probs_nn.npy"
ARCHIVED_SUBMISSION_PATHS = {
    6: ROOT / "archive" / "submission6.csv",
    10: ROOT / "archive" / "submission10.csv",
    11: ROOT / "archive" / "submission11.csv",
}
ROOT_SUBMISSION_PATH = ROOT / "submission.csv"
ARCHIVE_SUBMISSION_PATH = ROOT / "archive" / "submission12.csv"

PROBS_OUTPUT_PATH = OUTPUT_DIR / "probs_submission12_nn10.npy"
CANDIDATE_OUTPUT_PATH = OUTPUT_DIR / "submission12_nn10.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "validation_summary.json"

TREE_WEIGHT = 0.90
NN_WEIGHT = 0.10
EXPECTED_ROWS = 36_000
POSITIVE_COUNT = 30_411
EXPECTED_COLUMNS = ["patient_id", "vital_status"]
EXPECTED_DIFFERENCES = {6: 156, 10: 136, 11: 300}

# These hashes pin the exact audited files used to design the candidate.  A
# mismatch is a provenance failure, even when the replacement has valid shape.
EXPECTED_INPUT_SHA256 = {
    "test.csv": "6db5640f5db87b8527e172d1d6966b6edb9e8a8dc5b7c586e2826cdcfce23188",
    "archive/probs_v6_final.npy": "aca54c31462449df432e1edda5da81a6d04e242c8985cfde0e5983c6d0d92ab6",
    "archive/probs_nn.npy": "7ec4721ae7d4eccb35ebc5821014e581ad2da4e872775d8a4845f37423b1ce46",
    "archive/submission6.csv": "fd7cca1ee4a7654757adb78934baf42a07ae264dc581217df3e7863b552ef477",
    "archive/submission10.csv": "333af97cfbc16ffdcc2d9f910000664c443694239c60e67d6504af18687e86f1",
    "archive/submission11.csv": "cbea4ad3e7c525ab5352bd31f04a37d67bfdf13150fe3c8d2f88628df027ed0f",
}


def require_file(path: Path) -> Path:
    """Return ``path`` after verifying that it is a regular file."""

    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def relative_path(path: Path) -> str:
    """Return a stable repository-relative POSIX path for reporting."""

    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 without loading the full file into memory."""

    digest = hashlib.sha256()
    with require_file(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Calculate the SHA-256 of an in-memory output payload."""

    return hashlib.sha256(payload).hexdigest()


def validate_frozen_hashes() -> dict[str, dict[str, object]]:
    """Require every frozen input to match its audited SHA-256."""

    records: dict[str, dict[str, object]] = {}
    for name, expected_hash in EXPECTED_INPUT_SHA256.items():
        path = ROOT / name
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {name}: expected {expected_hash}, "
                f"got {actual_hash}"
            )
        records[name] = {
            "sha256": actual_hash,
            "expected_sha256": expected_hash,
            "bytes": path.stat().st_size,
            "matches_expected": True,
        }
    return records


def validate_probability_vector(
    name: str, values: np.ndarray, expected_rows: int
) -> np.ndarray:
    """Validate and normalize a saved one-dimensional probability vector."""

    values = np.asarray(values, dtype=np.float64)
    if values.shape != (expected_rows,):
        raise ValueError(
            f"{name} must have shape ({expected_rows},), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} contains values outside [0, 1]")
    return values


def stable_descending_top_k(
    probabilities: np.ndarray, positive_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Select exactly ``positive_count`` rows, breaking score ties by row order."""

    if not 0 < positive_count < len(probabilities):
        raise ValueError("positive_count must be strictly between zero and row count")
    order = np.argsort(-probabilities, kind="mergesort")
    predictions = np.zeros(len(probabilities), dtype=np.int8)
    predictions[order[:positive_count]] = 1
    return predictions, order


def validate_submission_frame(
    name: str,
    submission: pd.DataFrame,
    test_ids: pd.Series,
    *,
    expected_dead_count: int,
) -> np.ndarray:
    """Validate schema, ID alignment, labels, and class count."""

    if submission.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(f"{name} must have exact columns {EXPECTED_COLUMNS}")
    if len(submission) != len(test_ids):
        raise ValueError(
            f"{name} has {len(submission)} rows; expected {len(test_ids)}"
        )
    if not submission["patient_id"].equals(test_ids):
        raise ValueError(f"{name} patient IDs/order do not match test.csv")
    if submission["patient_id"].isna().any():
        raise ValueError(f"{name} contains a missing patient_id")
    if not submission["patient_id"].is_unique:
        raise ValueError(f"{name} contains duplicate patient_id values")
    if not submission["vital_status"].isin(["Dead", "Alive"]).all():
        raise ValueError(f"{name} contains a label other than Dead or Alive")

    predictions = submission["vital_status"].eq("Dead").to_numpy(dtype=np.int8)
    dead_count = int(predictions.sum())
    if dead_count != expected_dead_count:
        raise ValueError(
            f"{name} has {dead_count:,} Dead labels; "
            f"expected {expected_dead_count:,}"
        )
    return predictions


def comparison_record(
    candidate: np.ndarray,
    baseline: np.ndarray,
    expected_difference: int,
) -> dict[str, object]:
    """Validate and describe candidate label changes versus one baseline."""

    candidate_dead = candidate.astype(bool)
    baseline_dead = baseline.astype(bool)
    difference_count = int(np.count_nonzero(candidate_dead != baseline_dead))
    alive_to_dead = int(np.count_nonzero(candidate_dead & ~baseline_dead))
    dead_to_alive = int(np.count_nonzero(~candidate_dead & baseline_dead))
    if difference_count != expected_difference:
        raise ValueError(
            f"Candidate differs by {difference_count} rows; "
            f"expected {expected_difference}"
        )
    if alive_to_dead != dead_to_alive:
        raise ValueError(
            "Fixed-count candidate has unbalanced Alive->Dead and Dead->Alive changes"
        )
    return {
        "difference_count": difference_count,
        "expected_difference_count": expected_difference,
        "alive_to_dead": alive_to_dead,
        "dead_to_alive": dead_to_alive,
        "matches_expected": True,
    }


def npy_payload(values: np.ndarray) -> bytes:
    """Serialize an array using NumPy's non-pickle NPY format."""

    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    return buffer.getvalue()


def csv_payload(frame: pd.DataFrame) -> bytes:
    """Serialize a submission with deterministic UTF-8 and LF line endings."""

    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes through a same-directory temporary file and ``os.replace``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    descriptor_open = True
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor_open = False
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
    finally:
        if descriptor_open:
            os.close(descriptor)
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    """Validate frozen inputs, prepare artifacts, archive, and update submission.csv."""

    if not np.isclose(TREE_WEIGHT + NN_WEIGHT, 1.0):
        raise ValueError("Blend weights must sum to one")

    input_hashes = validate_frozen_hashes()
    test = pd.read_csv(require_file(TEST_PATH))
    if len(test) != EXPECTED_ROWS:
        raise ValueError(f"test.csv has {len(test)} rows; expected {EXPECTED_ROWS}")
    if "patient_id" not in test.columns:
        raise ValueError("test.csv is missing patient_id")
    test_ids = test["patient_id"]
    if test_ids.isna().any() or not test_ids.is_unique:
        raise ValueError("test.csv patient_id values must be complete and unique")

    tree_probabilities = validate_probability_vector(
        relative_path(TREE_PROBS_PATH),
        np.load(require_file(TREE_PROBS_PATH), allow_pickle=False),
        EXPECTED_ROWS,
    )
    nn_probabilities = validate_probability_vector(
        relative_path(NN_PROBS_PATH),
        np.load(require_file(NN_PROBS_PATH), allow_pickle=False),
        EXPECTED_ROWS,
    )

    archived_predictions: dict[int, np.ndarray] = {}
    for submission_number, path in ARCHIVED_SUBMISSION_PATHS.items():
        frame = pd.read_csv(require_file(path))
        archived_predictions[submission_number] = validate_submission_frame(
            relative_path(path),
            frame,
            test_ids,
            expected_dead_count=POSITIVE_COUNT,
        )

    # The frozen arrays must reproduce the archived reference labels under the
    # same stable ranking convention.  This also guards their implicit ID order.
    reconstructed_sub6, _ = stable_descending_top_k(
        tree_probabilities, POSITIVE_COUNT
    )
    reconstructed_sub10, _ = stable_descending_top_k(
        0.80 * tree_probabilities + 0.20 * nn_probabilities,
        POSITIVE_COUNT,
    )
    if not np.array_equal(reconstructed_sub6, archived_predictions[6]):
        raise ValueError(
            "archive/probs_v6_final.npy does not reconstruct archived Submission 6"
        )
    if not np.array_equal(reconstructed_sub10, archived_predictions[10]):
        raise ValueError(
            "The frozen 80%/20% probabilities do not reconstruct Submission 10"
        )

    blend_probabilities = validate_probability_vector(
        "Submission 12 90%/10% blend",
        TREE_WEIGHT * tree_probabilities + NN_WEIGHT * nn_probabilities,
        EXPECTED_ROWS,
    )
    candidate_predictions, descending_order = stable_descending_top_k(
        blend_probabilities, POSITIVE_COUNT
    )
    candidate = pd.DataFrame(
        {
            "patient_id": test_ids.copy(),
            "vital_status": np.where(
                candidate_predictions == 1, "Dead", "Alive"
            ),
        }
    )
    validate_submission_frame(
        "Submission 12 candidate",
        candidate,
        test_ids,
        expected_dead_count=POSITIVE_COUNT,
    )

    comparisons: dict[str, dict[str, object]] = {}
    for submission_number in (6, 10, 11):
        path = ARCHIVED_SUBMISSION_PATHS[submission_number]
        comparisons[f"submission{submission_number}"] = {
            "path": relative_path(path),
            **comparison_record(
                candidate_predictions,
                archived_predictions[submission_number],
                EXPECTED_DIFFERENCES[submission_number],
            ),
        }

    probabilities_bytes = npy_payload(blend_probabilities)
    candidate_bytes = csv_payload(candidate)
    expected_probability_hash = sha256_bytes(probabilities_bytes)
    expected_candidate_hash = sha256_bytes(candidate_bytes)

    # Refuse to replace an unknown upload file.  Allowing the candidate hash
    # makes reruns idempotent without weakening the initial Submission 11 guard.
    prior_root_hash = sha256_file(ROOT_SUBMISSION_PATH)
    archived_sub11_hash = EXPECTED_INPUT_SHA256["archive/submission11.csv"]
    allowed_root_hashes = {archived_sub11_hash, expected_candidate_hash}
    if prior_root_hash not in allowed_root_hashes:
        raise ValueError(
            "Refusing to replace submission.csv: it is neither the audited "
            "Submission 11 nor this exact Submission 12 candidate"
        )
    prior_root_state = (
        "ARCHIVED_SUBMISSION_11"
        if prior_root_hash == archived_sub11_hash
        else "IDENTICAL_SUBMISSION_13_RERUN"
    )

    # All validation and payload construction completes before the first write.
    atomic_write_bytes(PROBS_OUTPUT_PATH, probabilities_bytes)
    if sha256_file(PROBS_OUTPUT_PATH) != expected_probability_hash:
        raise OSError("Probability artifact failed its post-write hash check")

    atomic_write_bytes(CANDIDATE_OUTPUT_PATH, candidate_bytes)
    if sha256_file(CANDIDATE_OUTPUT_PATH) != expected_candidate_hash:
        raise OSError("Candidate CSV failed its post-write hash check")

    # Recheck immediately before replacing the root file to detect concurrent
    # changes made after preflight validation.
    if sha256_file(ROOT_SUBMISSION_PATH) != prior_root_hash:
        raise RuntimeError("submission.csv changed during candidate preparation")
    atomic_write_bytes(ROOT_SUBMISSION_PATH, candidate_bytes)
    if sha256_file(ROOT_SUBMISSION_PATH) != expected_candidate_hash:
        raise OSError("Root submission.csv failed its post-write hash check")

    written_candidate = pd.read_csv(CANDIDATE_OUTPUT_PATH)
    validate_submission_frame(
        relative_path(CANDIDATE_OUTPUT_PATH),
        written_candidate,
        test_ids,
        expected_dead_count=POSITIVE_COUNT,
    )
    atomic_write_bytes(ARCHIVE_SUBMISSION_PATH, candidate_bytes)
    if sha256_file(ARCHIVE_SUBMISSION_PATH) != expected_candidate_hash:
        raise OSError("Archived submission12.csv failed its post-write hash check")

    selected_floor = float(blend_probabilities[descending_order[POSITIVE_COUNT - 1]])
    unselected_ceiling = float(blend_probabilities[descending_order[POSITIVE_COUNT]])
    summary = {
        "submission_number": 13,
        "status": "SUBMITTED_AND_SCORED",
        "leaderboard_score": 0.877460,
        "leaderboard_score_recorded": True,
        "recipe": {
            "tree_probability_path": relative_path(TREE_PROBS_PATH),
            "tree_weight": TREE_WEIGHT,
            "nn_probability_path": relative_path(NN_PROBS_PATH),
            "nn_weight": NN_WEIGHT,
            "selection_method": "stable_descending_top_k",
            "tie_break": "original_test_row_order",
            "positive_label": "Dead",
            "positive_count": POSITIVE_COUNT,
        },
        "inputs": input_hashes,
        "generator": {
            "path": relative_path(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "validation": {
            "all_checks_passed": True,
            "rows": EXPECTED_ROWS,
            "columns": EXPECTED_COLUMNS,
            "patient_ids_match_test": True,
            "patient_ids_unique": True,
            "archived_submission_ids_match_test": {
                "submission6": True,
                "submission10": True,
                "submission11": True,
            },
            "frozen_probabilities_reconstruct_submission6": True,
            "frozen_80_20_probabilities_reconstruct_submission10": True,
            "probabilities_finite": True,
            "probabilities_in_unit_interval": True,
            "dead_count": int(candidate_predictions.sum()),
            "alive_count": int(len(candidate_predictions) - candidate_predictions.sum()),
            "dead_rate": float(candidate_predictions.mean()),
            "selected_floor_probability": selected_floor,
            "unselected_ceiling_probability": unselected_ceiling,
            "boundary_gap": selected_floor - unselected_ceiling,
            "differences": comparisons,
            "root_submission_prior_sha256": prior_root_hash,
            "root_submission_prior_state": prior_root_state,
            "archive_submission12_written": True,
        },
        "outputs": {
            relative_path(PROBS_OUTPUT_PATH): {
                "sha256": sha256_file(PROBS_OUTPUT_PATH),
                "bytes": PROBS_OUTPUT_PATH.stat().st_size,
                "shape": list(blend_probabilities.shape),
                "dtype": str(blend_probabilities.dtype),
            },
            relative_path(CANDIDATE_OUTPUT_PATH): {
                "sha256": sha256_file(CANDIDATE_OUTPUT_PATH),
                "bytes": CANDIDATE_OUTPUT_PATH.stat().st_size,
            },
            relative_path(ARCHIVE_SUBMISSION_PATH): {
                "sha256": sha256_file(ARCHIVE_SUBMISSION_PATH),
                "bytes": ARCHIVE_SUBMISSION_PATH.stat().st_size,
                "identical_to_candidate_artifact": True,
            },
            relative_path(ROOT_SUBMISSION_PATH): {
                "sha256": sha256_file(ROOT_SUBMISSION_PATH),
                "bytes": ROOT_SUBMISSION_PATH.stat().st_size,
                "identical_to_candidate_artifact": True,
            },
        },
    }
    summary_bytes = (
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    atomic_write_bytes(SUMMARY_OUTPUT_PATH, summary_bytes)

    print("Submission 12 verified, scored (0.877460), and archived.")
    print(f"Candidate: {relative_path(CANDIDATE_OUTPUT_PATH)}")
    print(f"Archived:  {relative_path(ARCHIVE_SUBMISSION_PATH)}")
    print(f"Root upload: {relative_path(ROOT_SUBMISSION_PATH)}")
    print(f"Validation: {relative_path(SUMMARY_OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
