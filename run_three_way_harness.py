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
        
        max_f1 = max(f1_10, f1_12, f1_6)
        winners = []
        if f1_10 == max_f1: winners.append("Submission 10")
        if f1_12 == max_f1: winners.append("Submission 12")
        if f1_6 == max_f1: winners.append("Submission 6")
        best_candidate = "Tie" if len(winners) > 1 else winners[0]
        
        splits.append({
            "split_id": split_idx,
            "winner": best_candidate,
            "F1_Sub10": f1_10,
            "F1_Sub12": f1_12,
            "F1_Sub6": f1_6,
        })
        
    df = pd.DataFrame(splits)
    
    out_dir = ROOT / "diagnostic_outputs" / "validation_harness"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(out_dir / "oof_40_60_three_way_results.csv", index=False)
    
    win_counts = df["winner"].value_counts().to_dict()
    
    def count_pw(f1_a, f1_b):
        wins = (f1_a > f1_b).sum()
        losses = (f1_a < f1_b).sum()
        ties = (f1_a == f1_b).sum()
        return wins, losses, ties
        
    pw_10_12 = count_pw(df["F1_Sub10"], df["F1_Sub12"])
    pw_10_6 = count_pw(df["F1_Sub10"], df["F1_Sub6"])
    pw_12_6 = count_pw(df["F1_Sub12"], df["F1_Sub6"])
    
    df["top_1"] = df[["F1_Sub10", "F1_Sub12", "F1_Sub6"]].max(axis=1)
    df["top_2"] = df[["F1_Sub10", "F1_Sub12", "F1_Sub6"]].apply(lambda x: sorted(x)[-2], axis=1)
    df["margin"] = df["top_1"] - df["top_2"]
    
    mean_margin = df["margin"].mean()
    std_margin = df["margin"].std()
    
    summary = f"""# Three-Way Validation Harness Summary (Sub 6 vs 10 vs 12)

**Policy**: Fixed 84.5% positive rate on the private (60%) split.
**Splits**: 50 repeated stratified 40/60 splits.

## Win Counts (Out of 50)
- Submission 10: {win_counts.get('Submission 10', 0)}
- Submission 12: {win_counts.get('Submission 12', 0)}
- Submission 6: {win_counts.get('Submission 6', 0)}
- Ties: {win_counts.get('Tie', 0)}

## Pairwise Win Rates (Out of 50)
- Submission 10 vs 12: {pw_10_12[0]} Wins, {pw_10_12[1]} Losses, {pw_10_12[2]} Ties
- Submission 10 vs 6: {pw_10_6[0]} Wins, {pw_10_6[1]} Losses, {pw_10_6[2]} Ties
- Submission 12 vs 6: {pw_12_6[0]} Wins, {pw_12_6[1]} Losses, {pw_12_6[2]} Ties

## Margin of Victory
- Mean F1 gap between 1st and 2nd place candidate: {mean_margin:.6f}
- Standard deviation of gap: {std_margin:.6f}
"""

    (out_dir / "three_way_summary.md").write_text(summary)
    
    print("WIN COUNTS:")
    print(win_counts)
    print("PAIRWISE:")
    print(f"10 vs 12: {pw_10_12}")
    print(f"10 vs 6: {pw_10_6}")
    print(f"12 vs 6: {pw_12_6}")
    print(f"MARGIN: {mean_margin:.6f} +/- {std_margin:.6f}")

if __name__ == "__main__":
    main()
