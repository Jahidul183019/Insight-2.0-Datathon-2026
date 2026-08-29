import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, brier_score_loss
from sklearn.model_selection import cross_val_predict

print("=" * 80)
print("  STEP 2: Calibration Check")
print("=" * 80)

# Load data from Step 1
try:
    oof_probs = np.load("oof_step1.npy")
    y_true = np.load("y_step1.npy")
    print(f"Loaded {len(oof_probs)} OOF predictions and labels.")
except Exception as e:
    print("Error loading Step 1 files. Run step1_cv_diagnostic.py first.")
    exit()

# 1. Base Model Calibration
prob_true, prob_pred = calibration_curve(y_true, oof_probs, n_bins=10, strategy='quantile')
brier_base = brier_score_loss(y_true, oof_probs)

best_f1_base = 0
best_th_base = 0.5
for th in np.arange(0.3, 0.7, 0.005):
    f1 = f1_score(y_true, (oof_probs >= th).astype(int))
    if f1 > best_f1_base:
        best_f1_base = f1
        best_th_base = th

print(f"\n[Base Model]")
print(f"  Brier Score: {brier_base:.5f}")
print(f"  Optimal F1:  {best_f1_base:.5f} (at threshold {best_th_base:.3f})")

# 2. Isotonic Recalibration (using CV to avoid overfitting)
print("\nApplying Isotonic Recalibration...")
iso = IsotonicRegression(out_of_bounds='clip')
oof_recalibrated = cross_val_predict(iso, oof_probs.reshape(-1, 1), y_true, cv=5)

prob_true_cal, prob_pred_cal = calibration_curve(y_true, oof_recalibrated, n_bins=10, strategy='quantile')
brier_cal = brier_score_loss(y_true, oof_recalibrated)

best_f1_cal = 0
best_th_cal = 0.5
for th in np.arange(0.3, 0.9, 0.005):
    f1 = f1_score(y_true, (oof_recalibrated >= th).astype(int))
    if f1 > best_f1_cal:
        best_f1_cal = f1
        best_th_cal = th

print(f"\n[Recalibrated Model]")
print(f"  Brier Score: {brier_cal:.5f}")
print(f"  Optimal F1:  {best_f1_cal:.5f} (at threshold {best_th_cal:.3f})")

# 3. Plotting
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.plot(prob_pred, prob_true, "s-", label=f"Base Model (Brier={brier_base:.4f})")
plt.plot(prob_pred_cal, prob_true_cal, "o-", label=f"Isotonic Calibrated (Brier={brier_cal:.4f})")

plt.ylabel("Fraction of positives (Dead)")
plt.xlabel("Mean predicted probability")
plt.title("Calibration Curve (Reliability Diagram)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig("calibration_curve.png", dpi=150, bbox_inches='tight')
print("\nSaved 'calibration_curve.png'")
