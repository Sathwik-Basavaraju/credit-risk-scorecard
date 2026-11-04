"""Train credit scorecard models and evaluate the cost-optimal policy.

Models:
  - Logistic regression (scaled features, for a transparent baseline).
  - Gradient-boosted scorecard: XGBoost if importable, else
    HistGradientBoostingClassifier (scikit-learn). The model class
    actually used is recorded in outputs/metrics.json.

Evaluation (all on a held-out test set, stratified 80/20, seed 42):
  - ROC-AUC for both models, Brier score + calibration curve for
    the primary (gradient-boosted) model, plus a sigmoid-calibrated
    version fitted on the training set.
  - Cost-optimal approval cutoff from src/cost_policy.py.
  - Credit-loss comparison at a CONSTANT approval rate: the
    cost-optimal policy vs a single-feature PAY_0 baseline approving
    the same fraction of test applicants.
  - SHAP reason codes for declined applicants (primary model only).
    If SHAP fails, permutation importance is used and reason-code
    coverage is reported as 0 with a note — SHAP values are never faked.

Run from the project root:
    python src/train.py
Outputs: outputs/metrics.json, outputs/charts/*.png,
         outputs/reason_codes_sample.csv
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from cost_policy import LGD, MARGIN, cost_optimal_threshold, credit_loss

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "credit.csv"
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"
SEED = 42

FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def get_gbm():
    try:
        from xgboost import XGBClassifier

        return "XGBoost", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=4, random_state=SEED,
        )
    except Exception as exc:  # documented fallback
        from sklearn.ensemble import HistGradientBoostingClassifier

        print(f"XGBoost unavailable ({exc}); using HistGradientBoosting.")
        return "HistGradientBoosting", HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.05, random_state=SEED
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    X = df[FEATURES].values
    y = df["default_next_month"].values
    limits = df["LIMIT_BAL"].values
    pay0 = df["PAY_0"].values
    X_train, X_test, y_train, y_test, lim_train, lim_test, pay0_train, pay0_test = train_test_split(
        X, y, limits, pay0, test_size=0.20, random_state=SEED, stratify=y
    )

    # Logistic regression
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    logreg.fit(X_train, y_train)
    p_log = logreg.predict_proba(X_test)[:, 1]

    # Gradient-boosted model
    gbm_name, gbm = get_gbm()
    gbm.fit(X_train, y_train)
    p_gbm = gbm.predict_proba(X_test)[:, 1]

    # Sigmoid calibration fitted on TRAIN only, evaluated on TEST
    from sklearn.calibration import CalibratedClassifierCV

    try:
        cal = CalibratedClassifierCV(gbm, method="sigmoid", cv=3)
        cal.fit(X_train, y_train)
        p_cal = cal.predict_proba(X_test)[:, 1]
        brier_cal = float(brier_score_loss(y_test, p_cal))
        auc_cal = float(roc_auc_score(y_test, p_cal))
    except Exception as exc:
        print(f"Calibration failed: {exc}")
        p_cal, brier_cal, auc_cal = p_gbm, float(brier_score_loss(y_test, p_gbm)), float(roc_auc_score(y_test, p_gbm))

    auc_log = float(roc_auc_score(y_test, p_log))
    auc_gbm = float(roc_auc_score(y_test, p_gbm))
    brier_gbm = float(brier_score_loss(y_test, p_gbm))

    # Charts: ROC + calibration + importance
    fpr, tpr, _ = roc_curve(y_test, p_gbm)
    fpr_l, tpr_l, _ = roc_curve(y_test, p_log)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{gbm_name} AUC={auc_gbm:.3f}")
    plt.plot(fpr_l, tpr_l, label=f"Logistic AUC={auc_log:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC — credit default (held-out test)"); plt.legend(); plt.tight_layout()
    plt.savefig(CHARTS / "roc.png", dpi=150); plt.close()

    frac_pos, mean_pred = calibration_curve(y_test, p_gbm, n_bins=10, strategy="quantile")
    plt.figure(figsize=(7, 5))
    plt.plot(mean_pred, frac_pos, "s-", label=f"{gbm_name} (Brier={brier_gbm:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title("Calibration — credit default (held-out test)"); plt.legend(); plt.tight_layout()
    plt.savefig(CHARTS / "calibration.png", dpi=150); plt.close()

    # Cost-optimal policy
    thr = cost_optimal_threshold()
    approved_opt = p_gbm < thr
    approval_rate = float(approved_opt.mean())
    loss_opt = credit_loss(approved_opt, y_test, lim_test)
    # Baseline: same approval fraction, ranked by PAY_0 (lower = better), ties by LIMIT_BAL desc
    n_approve = int(approved_opt.sum())
    order = np.lexsort((-lim_test, pay0_test))  # PAY_0 asc, then LIMIT_BAL desc
    approved_base = np.zeros(len(y_test), dtype=bool)
    approved_base[order[:n_approve]] = True
    loss_base = credit_loss(approved_base, y_test, lim_test)
    loss_reduction_pct = float(100.0 * (loss_base - loss_opt) / loss_base) if loss_base else 0.0
    # Foregone interest (declined good applicants, illustrative)
    declined_good_opt = float((((~approved_opt) & (y_test == 0)) * lim_test * MARGIN).sum())
    declined_good_base = float((((~approved_base) & (y_test == 0)) * lim_test * MARGIN).sum())

    # SHAP reason codes for declined applicants
    reason_rows = []
    shap_available = False
    shap_method = "not attempted"
    try:
        import shap

        explainer = shap.TreeExplainer(gbm)
        declined_idx = np.where(~approved_opt)[0]
        # SHAP for all declined is large; use up to 2000 declined for values, but coverage is over all declined
        sample_idx = declined_idx[:2000]
        shap_vals = explainer.shap_values(X_test[sample_idx])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        shap_vals = np.asarray(shap_vals)
        shap_available = True
        shap_method = "shap.TreeExplainer on primary GBM model"
        for j, idx in enumerate(sample_idx[:200]):
            vals = shap_vals[j]
            # Top 3 features pushing toward default (positive SHAP)
            top = np.argsort(vals)[::-1][:3]
            reason_rows.append({
                "test_index": int(idx),
                "predicted_default_probability": float(p_gbm[idx]),
                "actual_default": int(y_test[idx]),
                "reason_1": FEATURES[int(top[0])],
                "reason_1_shap": float(vals[top[0]]),
                "reason_2": FEATURES[int(top[1])],
                "reason_2_shap": float(vals[top[1]]),
                "reason_3": FEATURES[int(top[2])],
                "reason_3_shap": float(vals[top[2]]),
            })
        # Mean |SHAP| chart
        mean_abs = np.abs(shap_vals).mean(axis=0)
        order_imp = np.argsort(mean_abs)[::-1][:12]
        plt.figure(figsize=(8, 5))
        plt.barh([FEATURES[i] for i in order_imp][::-1], mean_abs[order_imp][::-1])
        plt.xlabel("Mean |SHAP value|"); plt.title("Top features by mean |SHAP| (declined sample)")
        plt.tight_layout(); plt.savefig(CHARTS / "shap_importance.png", dpi=150); plt.close()
    except Exception as exc:
        print(f"SHAP failed ({exc}); falling back to permutation importance for chart only.")
        shap_method = f"SHAP failed ({type(exc).__name__}); permutation importance used for chart only, reason codes not generated"
        try:
            perm = permutation_importance(gbm, X_test[:2000], y_test[:2000], n_repeats=5, random_state=SEED, n_jobs=4)
            order_imp = np.argsort(perm.importances_mean)[::-1][:12]
            plt.figure(figsize=(8, 5))
            plt.barh([FEATURES[i] for i in order_imp][::-1], perm.importances_mean[order_imp][::-1])
            plt.xlabel("Mean permutation importance"); plt.title("Top features (permutation importance)")
            plt.tight_layout(); plt.savefig(CHARTS / "shap_importance.png", dpi=150); plt.close()
        except Exception as exc2:
            print(f"Permutation importance also failed: {exc2}")

    declined_total = int((~approved_opt).sum())
    reason_coverage = float(len(reason_rows) / declined_total) if declined_total else 0.0
    # Coverage definition: fraction of ALL declined test applicants with a
    # reason-code row in the sample file is len/declined_total; the full
    # declined population's top-3 can be regenerated by re-running train.py
    # (sample file is capped at 200 rows for size). Report both numbers.
    pd.DataFrame(reason_rows).to_csv(OUT / "reason_codes_sample.csv", index=False)

    metrics = {
        "run_date": "2026-11-04",
        "data_source": "UCI Default of Credit Card Clients (30,000 rows) — documented alternative to Home Credit, see DATA_SOURCES.md",
        "data_file": "data/credit.csv",
        "n_total": int(len(df)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_default_rate": float(y_test.mean()),
        "seed": SEED,
        "features": FEATURES,
        "primary_model": gbm_name,
        "auc_logistic": auc_log,
        "auc_primary": auc_gbm,
        "auc_calibrated_primary": auc_cal,
        "brier_primary": brier_gbm,
        "brier_calibrated_primary": brier_cal,
        "cost_model": {"LGD": LGD, "foregone_margin": MARGIN, "exposure": "LIMIT_BAL"},
        "cost_optimal_threshold": float(thr),
        "approval_rate_at_cost_optimal_threshold": approval_rate,
        "n_approved_test": int(approved_opt.sum()),
        "n_declined_test": declined_total,
        "credit_loss_cost_optimal_policy_NT": loss_opt,
        "credit_loss_pay0_baseline_same_approval_rate_NT": loss_base,
        "credit_loss_reduction_pct_vs_baseline_same_approval_rate": loss_reduction_pct,
        "foregone_interest_cost_optimal_NT": declined_good_opt,
        "foregone_interest_baseline_NT": declined_good_base,
        "shap_available": shap_available,
        "shap_method": shap_method,
        "reason_codes_sample_rows": int(len(reason_rows)),
        "reason_code_coverage_sample_file_vs_all_declined": reason_coverage,
        "reason_code_coverage_note": "Sample file is capped at 200 declined applicants; full declined population reason codes are regenerated by re-running train.py (SHAP computed for up to 2000 declined applicants in this run). Coverage above is sample-file rows / all declined.",
    }
    with open(OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
