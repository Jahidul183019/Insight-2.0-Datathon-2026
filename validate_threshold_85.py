import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from validation_harness import load_approved_submission6_nested_reference, top_rate_predictions, repeated_split_audit
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

    nn_oof = np.load(ROOT / "archive" / "oof_nn.npy")
    submission6_nested_reference = load_approved_submission6_nested_reference(y)
    sub10_oof = 0.80 * submission6_nested_reference + 0.20 * nn_oof

    pred_845 = top_rate_predictions(sub10_oof, 0.845)
    pred_850 = top_rate_predictions(sub10_oof, 0.850)

    # 1. 5-Fold OOF Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_f1s_850 = []
    fold_f1s_845 = []
    
    for train_idx, val_idx in skf.split(y, y):
        f1_850 = f1_score(y[val_idx], pred_850[val_idx], average="weighted", zero_division=0)
        f1_845 = f1_score(y[val_idx], pred_845[val_idx], average="weighted", zero_division=0)
        fold_f1s_850.append(f1_850)
        fold_f1s_845.append(f1_845)
        
    print("=== 5-Fold OOF Validation (Sub10 85% vs Sub10 84.5%) ===")
    print(f"85%: {fold_f1s_850}")
    print(f"85% Worst Fold: {min(fold_f1s_850):.6f}")
    print(f"84.5%: {fold_f1s_845}")
    print(f"84.5% Worst Fold: {min(fold_f1s_845):.6f}")
    
    wins = sum(1 for a, b in zip(fold_f1s_850, fold_f1s_845) if a > b)
    print(f"85% won {wins} out of 5 folds vs 84.5%.\n")

    # 2. Paired Bootstrap
    print("=== Paired Bootstrap (85% vs 84.5%) ===")
    ci_lower, ci_upper, mean_diff = bootstrap_ci(y, pred_850, pred_845)
    print(f"Mean Difference: {mean_diff:.6f}")
    print(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    if ci_lower > 0:
        print("CI excludes zero (Significant Improvement)\n")
    else:
        print("CI includes zero (NOT Significant)\n")

    # 3. 50-Split Harness
    candidates = {
        "Sub10_845": {
            "scores": sub10_oof,
            "tree_weight": 0.80,
            "nn_weight": 0.20,
        }
    }
    # Wait, repeated_split_audit tests multiple policies at once if we provide a rate grid!
    # But it also creates fixed_rank_rate_X policies explicitly if we provide it.
    # Actually, we can just run the loop manually for 50 splits with top_rate_predictions on public/private
    print("=== 50-Split Harness (85% vs 84.5%) ===")
    split_wins_850 = 0
    split_losses_850 = 0
    split_ties = 0
    
    from sklearn.model_selection import train_test_split
    all_indices = np.arange(len(y))
    for split_idx in range(50):
        split_seed = 42 + split_idx
        public_idx, private_idx = train_test_split(all_indices, train_size=0.40, stratify=y, random_state=split_seed)
        
        # In a real harness, you select the threshold on the public split.
        # But we are testing FIXED thresholds on the private split.
        private_y = y[private_idx]
        private_scores = sub10_oof[private_idx]
        
        pred_pvt_845 = top_rate_predictions(private_scores, 0.845)
        pred_pvt_850 = top_rate_predictions(private_scores, 0.850)
        
        f1_pvt_845 = f1_score(private_y, pred_pvt_845, average="weighted", zero_division=0)
        f1_pvt_850 = f1_score(private_y, pred_pvt_850, average="weighted", zero_division=0)
        
        if f1_pvt_850 > f1_pvt_845:
            split_wins_850 += 1
        elif f1_pvt_850 < f1_pvt_845:
            split_losses_850 += 1
        else:
            split_ties += 1
            
    print(f"85% vs 84.5% - Wins: {split_wins_850}, Losses: {split_losses_850}, Ties: {split_ties}\n")
    
    # 4. Subgroup Safety Checks
    print("=== Subgroup Safety Checks ===")
    train_df["composite_stage"] = derive_composite_stage(train_df)
    
    age_idx = train_df["age_recode"].str.contains("55").fillna(False)
    loc_idx = train_df["composite_stage"] == "M0 N0 T0-2 localized"
    
    f1_850_age = f1_score(y[age_idx], pred_850[age_idx], average="weighted", zero_division=0)
    f1_845_age = f1_score(y[age_idx], pred_845[age_idx], average="weighted", zero_division=0)
    
    f1_850_loc = f1_score(y[loc_idx], pred_850[loc_idx], average="weighted", zero_division=0)
    f1_845_loc = f1_score(y[loc_idx], pred_845[loc_idx], average="weighted", zero_division=0)
    
    print(f"Age 55-59 - 85%: {f1_850_age:.6f}, 84.5%: {f1_845_age:.6f}, Diff: {f1_850_age - f1_845_age:.6f}")
    print(f"Localized - 85%: {f1_850_loc:.6f}, 84.5%: {f1_845_loc:.6f}, Diff: {f1_850_loc - f1_845_loc:.6f}")

if __name__ == "__main__":
    main()
