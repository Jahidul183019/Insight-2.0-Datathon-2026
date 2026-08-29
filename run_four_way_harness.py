import pandas as pd
import numpy as np
from pathlib import Path
from validation_harness import (
    repeated_split_audit, 
    load_approved_submission6_nested_reference,
    validate_probability_vector
)

ROOT = Path("/Users/md.jahidulislam/Desktop/insight-2-0")

def main():
    train_df = pd.read_csv(ROOT / "train.csv")
    y = train_df["vital_status"].eq("Dead").to_numpy(dtype=np.int8)

    submission6_nested_reference = load_approved_submission6_nested_reference(y)
    if submission6_nested_reference is None:
        raise ValueError("Missing canonical Submission 6 nested reference")

    nn_oof = validate_probability_vector(
        "archive/oof_nn.npy",
        np.load(ROOT / "archive" / "oof_nn.npy", allow_pickle=False),
        len(y),
    )
    
    nested = np.load(ROOT / "diagnostic_outputs" / "submission6_nested" / "nested_oof_predictions.npz")
    teacher_oof = nested["teacher_oof"]
    teacher_nn5_oof = 0.95 * teacher_oof + 0.05 * nn_oof

    sub10_oof = 0.80 * submission6_nested_reference + 0.20 * nn_oof
    sub12_oof = 0.90 * submission6_nested_reference + 0.10 * nn_oof

    candidates = {
        "Submission 6": {
            "scores": submission6_nested_reference,
            "tree_weight": 1.0,
            "nn_weight": 0.0,
        },
        "Submission 10": {
            "scores": sub10_oof,
            "tree_weight": 0.80,
            "nn_weight": 0.20,
        },
        "Submission 12": {
            "scores": sub12_oof,
            "tree_weight": 0.90,
            "nn_weight": 0.10,
        },
        "Teacher+NN5": {
            "scores": teacher_nn5_oof,
            "tree_weight": 0.95,
            "nn_weight": 0.05,
        }
    }

    rate_grid = np.round(np.arange(0.80, 0.92001, 0.0025), 4)

    split_results = repeated_split_audit(
        y=y,
        candidates=candidates,
        public_fraction=0.40,
        fixed_rate=0.845,
        rate_grid=rate_grid,
        n_splits=50,
        seed=42,
    )
    
    fixed_results = split_results[split_results["policy"] == "fixed_rank_rate_0.845"]

    splits = []
    
    for split_idx in range(50):
        split_data = fixed_results[fixed_results["split_id"] == split_idx]
        
        f1_10 = split_data.loc[split_data["candidate"] == "Submission 10", "private_weighted_f1"].values[0]
        f1_12 = split_data.loc[split_data["candidate"] == "Submission 12", "private_weighted_f1"].values[0]
        f1_6 = split_data.loc[split_data["candidate"] == "Submission 6", "private_weighted_f1"].values[0]
        f1_tnn = split_data.loc[split_data["candidate"] == "Teacher+NN5", "private_weighted_f1"].values[0]
        
        max_f1 = max(f1_10, f1_12, f1_6, f1_tnn)
        winners = []
        if f1_10 == max_f1: winners.append("Submission 10")
        if f1_12 == max_f1: winners.append("Submission 12")
        if f1_6 == max_f1: winners.append("Submission 6")
        if f1_tnn == max_f1: winners.append("Teacher+NN5")
        best_candidate = "Tie" if len(winners) > 1 else winners[0]
        
        splits.append({
            "split_id": split_idx,
            "winner": best_candidate,
            "F1_Sub10": f1_10,
            "F1_Sub12": f1_12,
            "F1_Sub6": f1_6,
            "F1_TeacherNN5": f1_tnn,
        })
        
    df = pd.DataFrame(splits)
    
    out_dir = ROOT / "diagnostic_outputs" / "validation_harness"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(out_dir / "oof_40_60_four_way_results.csv", index=False)
    
    win_counts = df["winner"].value_counts().to_dict()
    
    def count_pw(f1_a, f1_b):
        wins = (f1_a > f1_b).sum()
        losses = (f1_a < f1_b).sum()
        ties = (f1_a == f1_b).sum()
        return wins, losses, ties
        
    pw_t_10 = count_pw(df["F1_TeacherNN5"], df["F1_Sub10"])
    
    print("WIN COUNTS:")
    print(win_counts)
    print("PAIRWISE Teacher+NN5 vs Sub10:")
    print(f"Wins: {pw_t_10[0]}, Losses: {pw_t_10[1]}, Ties: {pw_t_10[2]}")

if __name__ == "__main__":
    main()
