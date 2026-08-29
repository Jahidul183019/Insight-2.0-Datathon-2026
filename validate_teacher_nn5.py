import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from validation_harness import load_approved_submission6_nested_reference, top_rate_predictions
from subgroup_f1_audit import derive_composite_stage

ROOT = Path("/Users/md.jahidulislam/Desktop/insight-2-0")

def bootstrap_ci(y, pred_a, pred_b, n_repeats=1000, seed=9042):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_repeats):
        indices = rng.choice(n, size=n, replace=True)
        f1_a = f1_score(y[indices], pred_a[indices], average="weighted", zero_division=0)
        f1_b = f1_score(y[indices], pred_b[indices], average="weighted", zero_division=0)
        diffs.append(f1_a - f1_b)
    
    diffs = np.array(diffs)
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), np.mean(diffs)

def main():
    train_df = pd.read_csv(ROOT / "train.csv")
    y = train_df["vital_status"].eq("Dead").to_numpy(dtype=np.int8)

    nested = np.load(ROOT / "diagnostic_outputs" / "submission6_nested" / "nested_oof_predictions.npz")
    teacher_oof = nested["teacher_oof"]

    nn_oof = np.load(ROOT / "archive" / "oof_nn.npy")
    teacher_nn5_oof = 0.95 * teacher_oof + 0.05 * nn_oof

    submission6_nested_reference = load_approved_submission6_nested_reference(y)
    sub10_oof = 0.80 * submission6_nested_reference + 0.20 * nn_oof

    # Step 2.1: 5-Fold OOF Weighted F1
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_f1s_teacher = []
    fold_f1s_sub10 = []
    
    pred_teacher = top_rate_predictions(teacher_nn5_oof, 0.845)
    pred_sub10 = top_rate_predictions(sub10_oof, 0.845)
    
    for train_idx, val_idx in skf.split(y, y):
        f1_t = f1_score(y[val_idx], pred_teacher[val_idx], average="weighted", zero_division=0)
        f1_10 = f1_score(y[val_idx], pred_sub10[val_idx], average="weighted", zero_division=0)
        fold_f1s_teacher.append(f1_t)
        fold_f1s_sub10.append(f1_10)
    
    print("=== 5-Fold OOF Validation ===")
    print(f"Teacher+5%NN: {fold_f1s_teacher}")
    print(f"Teacher+5%NN Worst Fold: {min(fold_f1s_teacher):.6f}")
    print(f"Sub10: {fold_f1s_sub10}")
    print(f"Sub10 Worst Fold: {min(fold_f1s_sub10):.6f}")
    
    wins = sum(1 for t, s in zip(fold_f1s_teacher, fold_f1s_sub10) if t > s)
    print(f"Teacher+5%NN won {wins} out of 5 folds vs Sub10.\n")

    # Step 2.2: Paired Bootstrap vs Submission 10
    print("=== Paired Bootstrap (Teacher+5%NN vs Sub10) ===")
    ci_lower, ci_upper, mean_diff = bootstrap_ci(y, pred_teacher, pred_sub10)
    print(f"Mean Difference: {mean_diff:.6f}")
    print(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    if ci_lower > 0:
        print("CI excludes zero (Significant Improvement)\n")
    else:
        print("CI includes zero (NOT Significant)\n")
        
    # Step 2.4: Subgroup Safety Checks
    print("=== Subgroup Safety Checks ===")
    train_df["composite_stage"] = derive_composite_stage(train_df)
    
    # Age 55-59
    age_idx = train_df["age_recode"].str.contains("55").fillna(False)
    
    f1_t_age = f1_score(y[age_idx], pred_teacher[age_idx], average="weighted", zero_division=0)
    f1_10_age = f1_score(y[age_idx], pred_sub10[age_idx], average="weighted", zero_division=0)
    
    # Localized stage
    loc_idx = train_df["composite_stage"] == "M0 N0 T0-2 localized"
    f1_t_loc = f1_score(y[loc_idx], pred_teacher[loc_idx], average="weighted", zero_division=0)
    f1_10_loc = f1_score(y[loc_idx], pred_sub10[loc_idx], average="weighted", zero_division=0)
    
    print(f"Age 55-59 - Teacher+NN5: {f1_t_age:.6f}, Sub10: {f1_10_age:.6f}, Diff: {f1_t_age - f1_10_age:.6f}")
    print(f"Localized - Teacher+NN5: {f1_t_loc:.6f}, Sub10: {f1_10_loc:.6f}, Diff: {f1_t_loc - f1_10_loc:.6f}")

if __name__ == "__main__":
    main()
