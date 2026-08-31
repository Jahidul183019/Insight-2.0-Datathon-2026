"""Build the self-contained, reproducible final competition notebook.

The generated notebook embeds the exact Submission 6 training implementation
instead of loading frozen prediction vectors.  Historical artifacts are used
only for an optional post-training equality check.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "insight_2_0_consolidated.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def embedded_tree_pipeline() -> str:
    source = (ROOT / "pipeline_v6.py").read_text(encoding="utf-8")
    old = 'OUTPUT_DIR = Path("artifacts/v6_rerun")'
    new = 'OUTPUT_DIR = Path("artifacts/final_notebook")'
    if source.count(old) != 1:
        raise RuntimeError("Could not locate the canonical v6 output declaration")
    source = source.replace(old, new)
    return (
        "# Exact final-model training implementation, embedded for portability.\n"
        "# It reads only train.csv/test.csv and writes under artifacts/final_notebook/.\n"
        + source
    )


nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12"},
}

nb.cells = [
    markdown(
        """
# Insight 2.0 — Reproducible Final Model

This notebook trains the final Submission 6 workflow for cancer-patient vital
status prediction. The competition describes the metric as weighted F1 and
identifies `Dead` as the positive class; local results below label the F1
variant used.

The model combines LightGBM, XGBoost, and CatBoost across two feature views,
followed by one pass of model-generated pseudo-label training. Starting from
`train.csv` and `test.csv`, the notebook writes a validated `submission.csv`
with 36,000 rows and 30,411 `Dead` predictions.

Verified Public Leaderboard score: `0.877258`.
"""
    ),
    markdown(
        """
## 1. Environment and data contract

The workflow uses only the organizer-provided training and test files. It
checks the schema, patient identifiers, target labels, and package versions
before modelling. For the audited reproduction, the environment used NumPy
2.4.3, pandas 3.0.3, scikit-learn 1.8.0, LightGBM 4.7.0, XGBoost 3.2.0, and
CatBoost 1.2.10.
"""
    ),
    code(
        """
from pathlib import Path
import hashlib
import platform
import warnings

import numpy as np
import pandas as pd
import sklearn
import lightgbm
import xgboost
import catboost
from sklearn.metrics import f1_score

ROOT = Path.cwd().resolve()
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test.csv"
for required in (TRAIN_PATH, TEST_PATH):
    if not required.is_file():
        raise FileNotFoundError(f"Missing required competition file: {required}")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
assert train.columns.tolist() == ["patient_id", "vital_status", *test.columns.drop("patient_id").tolist()] or set(test.columns) == set(train.columns) - {"vital_status"}
assert "vital_status" not in test.columns
assert train["patient_id"].is_unique and test["patient_id"].is_unique
assert not train["patient_id"].isna().any() and not test["patient_id"].isna().any()
assert set(train["vital_status"].unique()) == {"Dead", "Alive"}

y_binary = train["vital_status"].eq("Dead").astype("int8")
AUDITED_VERSIONS = {
    "numpy": "2.4.3",
    "pandas": "3.0.3",
    "scikit-learn": "1.8.0",
    "lightgbm": "4.7.0",
    "xgboost": "3.2.0",
    "catboost": "1.2.10",
}
RUNTIME_VERSIONS = {
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit-learn": sklearn.__version__,
    "lightgbm": lightgbm.__version__,
    "xgboost": xgboost.__version__,
    "catboost": catboost.__version__,
}
VERSION_MATCHES = {
    name: RUNTIME_VERSIONS[name] == expected
    for name, expected in AUDITED_VERSIONS.items()
}
print("Python:", platform.python_version())
display(pd.DataFrame({
    "audited": AUDITED_VERSIONS,
    "runtime": RUNTIME_VERSIONS,
    "exact_match": VERSION_MATCHES,
}))
print("train/test shapes:", train.shape, test.shape)
print("Dead prevalence:", f"{y_binary.mean():.3%}")
"""
    ),
    markdown(
        """
## 2. Exploratory data analysis

The checks below examine class balance, missingness, temporal coverage, stage,
age, and train/test categorical coverage. These variables are clinically and
statistically relevant to mortality, while patient identifiers are excluded
from modelling.
"""
    ),
    code(
        """
import matplotlib.pyplot as plt

summary = pd.DataFrame({
    "dtype": train.dtypes.astype(str),
    "missing_count": train.isna().sum(),
    "missing_rate": train.isna().mean(),
    "unique_values": train.nunique(dropna=False),
}).sort_values("missing_rate", ascending=False)
display(summary.head(15))

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
train["vital_status"].value_counts().reindex(["Alive", "Dead"]).plot.bar(ax=axes[0], color=["#4C78A8", "#E45756"])
axes[0].set_title("Target distribution")
axes[0].set_ylabel("Patients")

summary.head(12).sort_values("missing_rate")["missing_rate"].plot.barh(ax=axes[1], color="#72B7B2")
axes[1].set_title("Highest missingness rates")
axes[1].set_xlabel("Missing fraction")

stage_rate = train.groupby("summary_stage", dropna=False)["vital_status"].apply(lambda s: s.eq("Dead").mean()).sort_values()
stage_rate.plot.barh(ax=axes[2], color="#F58518")
axes[2].set_title("Dead rate by summary stage")
axes[2].set_xlabel("Dead fraction")
plt.tight_layout()
plt.show()

age_audit = train.groupby("age_recode", dropna=False).agg(
    patients=("patient_id", "size"),
    dead_rate=("vital_status", lambda s: s.eq("Dead").mean()),
).sort_index()
display(age_audit)

categorical_columns = train.drop(columns=["patient_id", "vital_status"]).select_dtypes(include=["object", "string"]).columns
coverage = []
for column in categorical_columns:
    train_values = set(train[column].fillna("__NA__").astype(str))
    test_values = set(test[column].fillna("__NA__").astype(str))
    coverage.append({"feature": column, "test_only_categories": len(test_values - train_values)})
display(pd.DataFrame(coverage).sort_values("test_only_categories", ascending=False).head(15))
"""
    ),
    markdown(
        """
## 3. Preprocessing and feature engineering

- Categorical variables are ordinal encoded with an explicit unknown-category
  sentinel.
- Numeric missing values are retained for tree models until the final matrix
  boundary.
- Target encoding is learned inside training folds; validation targets are
  never used to encode their own rows.
- Engineered variables include age midpoint, diagnosis year, TNM order,
  positive-node ratio, metastasis count, tumour-size reconciliation, treatment
  indicators, missingness flags, and target-free category frequencies.
- `patient_id` is used only to align the final output.

The implementation is embedded in the training cell below, so the notebook
does not depend on cached probability arrays or project-local Python modules.
"""
    ),
    markdown(
        """
## 4. Model development and fixed hyperparameters

Optuna was used during development to search numeric model parameters under
cross-validation. The selected values are fixed in the training cell so the
final notebook does not repeat the search.

The ensemble contains two feature views:

1. v5 without target encoding.
2. v4 with target encoding calculated inside each training fold.

LightGBM, XGBoost, and CatBoost are trained over three repeated five-fold
splits. High-confidence test predictions (`>=0.95` or `<=0.05`) are then used
as model-generated pseudo labels for one additional five-fold training stage.
The original and pseudo-trained probabilities are averaged 50/50. No hidden
labels or manual test annotations are used.
"""
    ),
    markdown(
        """
## 5. End-to-end training

This is the complete final training implementation. It starts from the two CSV
files and writes fresh outputs under `artifacts/final_notebook/`. Depending on
hardware, the 120 boosted-tree fits can take several minutes.
"""
    ),
    code(embedded_tree_pipeline()),
    markdown(
        """
## 6. Local evaluation and exact output validation

The full pseudo-labelled system has no ordinary OOF vector because test-derived
pseudo labels are part of its student training population. To avoid presenting
an optimistic estimate, the local score below is explicitly the pre-pseudo
teacher OOF score at the locked class-count policy. The final cell then verifies
the fresh probability vector and submission structure. If the historical
reference files are present, they are used only after training to measure exact
byte and label-level reproduction; they never contribute to prediction
generation.
"""
    ),
    code(
        """
def stable_top_k(probabilities, count):
    order = np.argsort(-np.asarray(probabilities), kind="mergesort")
    labels = np.zeros(len(order), dtype=np.int8)
    labels[order[:count]] = 1
    return labels

target_dead_count = 30_411
teacher_oof_labels = stable_top_k(oof_blend, round(0.845 * len(oof_blend)))
teacher_weighted_f1 = f1_score(y, teacher_oof_labels, average="weighted", zero_division=0)
print(f"Pre-pseudo teacher OOF support-weighted F1: {teacher_weighted_f1:.6f}")

generated_dir = ROOT / "artifacts" / "final_notebook"
generated_submission_path = generated_dir / "submission.csv"
generated_probability_path = generated_dir / "probs_v6_final.npy"
generated_probabilities = np.load(generated_probability_path, allow_pickle=False)

assert generated_probabilities.shape == (len(test),)
assert np.isfinite(generated_probabilities).all()
assert ((generated_probabilities >= 0.0) & (generated_probabilities <= 1.0)).all()

# Apply the documented decision policy deterministically even if small
# cross-platform floating-point differences change the historical threshold.
generated_labels = stable_top_k(generated_probabilities, target_dead_count)
generated = pd.DataFrame({
    "patient_id": test["patient_id"].copy(),
    "vital_status": np.where(generated_labels == 1, "Dead", "Alive"),
})
generated.to_csv(generated_submission_path, index=False, lineterminator="\\n")

assert generated.columns.tolist() == ["patient_id", "vital_status"]
assert len(generated) == len(test) == 36_000
assert generated["patient_id"].equals(test["patient_id"])
assert generated["patient_id"].is_unique
assert generated["vital_status"].isin(["Dead", "Alive"]).all()
assert not generated.isna().any().any()
assert int(generated["vital_status"].eq("Dead").sum()) == target_dead_count

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

EXPECTED_SUBMISSION6_SHA256 = "fd7cca1ee4a7654757adb78934baf42a07ae264dc581217df3e7863b552ef477"
EXPECTED_PROBABILITY_SHA256 = "aca54c31462449df432e1edda5da81a6d04e242c8985cfde0e5983c6d0d92ab6"
actual_submission_sha256 = sha256(generated_submission_path)
actual_probability_sha256 = sha256(generated_probability_path)
exact_submission_bytes = actual_submission_sha256 == EXPECTED_SUBMISSION6_SHA256
exact_probability_bytes = actual_probability_sha256 == EXPECTED_PROBABILITY_SHA256

if not exact_probability_bytes:
    warnings.warn(
        "Fresh probabilities are not byte-identical to the audited reference. "
        "This can occur across OS, hardware, thread counts, or library builds. "
        "Structural validity and prediction-level agreement are reported below."
    )
if not exact_submission_bytes:
    warnings.warn(
        "Fresh CSV bytes differ from the audited reference; the notebook will "
        "continue because schema, IDs, labels, and class count are valid."
    )
if not all(VERSION_MATCHES.values()):
    warnings.warn(
        "Runtime package versions differ from the audited environment; exact "
        "floating-point reproduction is therefore not expected."
    )

historical_path = ROOT / "archive" / "submission6.csv"
historical_label_disagreements = None
historical_bytes_equal = None
if historical_path.is_file():
    historical = pd.read_csv(historical_path)
    if (
        historical.columns.tolist() == ["patient_id", "vital_status"]
        and len(historical) == len(test)
        and historical["patient_id"].equals(test["patient_id"])
        and historical["vital_status"].isin(["Dead", "Alive"]).all()
    ):
        historical_label_disagreements = int(
            (generated["vital_status"] != historical["vital_status"]).sum()
        )
        historical_bytes_equal = (
            generated_submission_path.read_bytes() == historical_path.read_bytes()
        )
        if historical_label_disagreements:
            warnings.warn(
                f"Fresh predictions differ from the audited Submission 6 on "
                f"{historical_label_disagreements:,} rows."
            )
    else:
        warnings.warn(
            "The optional historical reference is malformed or misaligned; "
            "it was ignored and did not affect prediction generation."
        )

# Promote only after all machine-independent structural checks pass. Exact
# reference hashes remain diagnostics, not cross-platform execution gates.
# Submission 12 remains safely preserved at archive/submission12.csv.
canonical_submission_path = ROOT / "submission.csv"
canonical_submission_path.write_bytes(generated_submission_path.read_bytes())
assert sha256(canonical_submission_path) == actual_submission_sha256

print("Structural validation: PASS")
print("Audited environment exact match:", all(VERSION_MATCHES.values()))
print("Exact probability-byte match:", exact_probability_bytes)
print("Exact CSV-byte match:", exact_submission_bytes)
print("Historical label disagreements:", historical_label_disagreements)
print("Historical CSV bytes equal:", historical_bytes_equal)
print("Final submission:", canonical_submission_path)
print("Rows / Dead / Alive:", len(generated), int(generated.vital_status.eq("Dead").sum()), int(generated.vital_status.eq("Alive").sum()))
print("Actual SHA-256:", actual_submission_sha256)
print("Audited reference SHA-256:", EXPECTED_SUBMISSION6_SHA256)
"""
    ),
    markdown(
        """
## 7. Compact interpretation appendix

### Representative component importance

The chart below reports feature importance from the **final-fold,
pseudo-trained v4 CatBoost component** that remains in memory after the
unchanged training workflow finishes. It is a useful model-inspection view,
but it is not an aggregate importance for all 120 fitted components and must
not be interpreted causally. The final prediction also includes LightGBM,
XGBoost, non-target-encoded models, repeated folds, and the teacher/student
blend.
"""
    ),
    code(
        """
representative_feature_names = list(X_train_base.columns) + [
    f"{column}_te" for column in te_cols
]

if hasattr(m, "get_feature_importance"):
    representative_importance = np.asarray(m.get_feature_importance(), dtype=float)
    if len(representative_importance) == len(representative_feature_names):
        importance_table = (
            pd.DataFrame({
                "feature": representative_feature_names,
                "importance": representative_importance,
            })
            .sort_values("importance", ascending=False)
            .head(15)
            .sort_values("importance")
        )
        ax = importance_table.plot.barh(
            x="feature",
            y="importance",
            figsize=(9, 6),
            legend=False,
            color="#4C78A8",
        )
        ax.set_title("Representative final-fold v4 CatBoost feature importance")
        ax.set_xlabel("CatBoost importance")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.show()
        display(importance_table.sort_values("importance", ascending=False).reset_index(drop=True))
    else:
        warnings.warn(
            "Representative importance length does not match the documented "
            "feature names; the optional interpretation plot was skipped."
        )
else:
    warnings.warn(
        "The final retained component does not expose CatBoost feature "
        "importance; the optional interpretation plot was skipped."
    )
"""
    ),
    markdown(
        """
### Leakage-safe age-band audit

The table below is an extract from
`diagnostic_outputs/age55_investigation/age_band_metrics.csv` using the
`submission6_nested_equivalent` predictions. Each patient was predicted by an
outer fold that excluded that patient during fitting.

| Age band | Patients | Weighted F1 | ROC AUC |
| --- | ---: | ---: | ---: |
| 50–54 | 789 | 0.8514 | 0.8904 |
| 55–59 | 1,821 | 0.8381 | 0.8625 |
| 60–64 | 2,925 | 0.8584 | 0.8860 |
| 65–69 | 3,851 | 0.8484 | 0.8801 |
| 70–74 | 4,233 | 0.8723 | 0.8988 |

Age 55–59 had the lowest weighted F1 among these bands. A dedicated follow-up
found no stable error cluster that justified a subgroup-specific rule.

### Decision rate

Development runs used a fixed rate of approximately 84.5% `Dead`. On the final
probability vector, the documented threshold scan reaches that policy at
`0.684`, producing 30,411 `Dead` predictions (`84.475%`). A separate 85.0%
rate check improved only two of five validation folds, so the original rate was
retained. The rate is a ranking policy, not a probability-calibration claim.

### Local validation

A five-fold nested-equivalent reconstruction excluded each validation patient
from target encoding, teacher fitting, pseudo-label selection, and student
fitting. It predicted all 24,000 labelled rows once and achieved
support-weighted F1 `0.877004`. This is a local recipe check; it is not an
estimate of Kaggle's hidden split.
"""
    ),
    markdown(
        """
## 8. Results and limitations

The notebook regenerates `submission.csv` from the organizer-provided data. In
the audited environment it produced 36,000 rows, 30,411 `Dead` predictions,
and SHA-256
`fd7cca1ee4a7654757adb78934baf42a07ae264dc581217df3e7863b552ef477`.
The verified Public Leaderboard score is `0.877258`.

A clean-kernel audit completed all five code cells without errors in `341.48`
seconds. Runtime varies with CPU, thread scheduling, and library builds. Exact
hashes are therefore reported as diagnostic references rather than
cross-platform execution requirements; schema, patient IDs, labels, missing
values, row count, and class count remain strict checks.

The nested-equivalent local reconstruction achieved support-weighted F1
`0.877004`. The age-band audit found lower performance for ages 55–59, while
the displayed importance chart represents one CatBoost component rather than
the complete ensemble. These associations should not be interpreted as causal
or clinical conclusions.

Pseudo-label training can reinforce confident teacher errors, and a fixed
class-count policy may transfer poorly if the hidden class prevalence differs.
The Public Leaderboard covers only part of the test set, so the Private result
remains uncertain.

The workflow uses manually specified preprocessing, features, models, folds,
and ensemble rules. It uses no AutoML pipeline generator, external data,
manual test labels, or row-level leaderboard probing.
"""
    ),
]

nbf.write(nb, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH} with {len(nb.cells)} cells")
