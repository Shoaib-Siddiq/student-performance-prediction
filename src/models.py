"""Training and evaluation pipeline for Student Performance Prediction.

Runs the full experiment and writes reproducible artifacts:
- models/   : serialized best models (joblib)
- reports/  : metrics.json + classification report text
- images/   : plots for the README / interview walkthrough

Two tasks:
  1. REGRESSION   : predict final grade G3 (Linear Regression vs Random Forest)
  2. CLASSIFICATION: predict pass/fail (G3 >= 10) (Logistic Regression vs Random Forest)

Evaluation: 80/20 stratified holdout (random_state=42) + 5-fold CV.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.data import load_dataset

RANDOM_STATE = 42
PASS_THRESHOLD = 10

sns.set_theme(style="whitegrid")


def build_preprocessor(df: pd.DataFrame, target_cols: list[str]):
    """One-hot encode categorical features; keep numeric as-is."""
    feature_cols = [c for c in df.columns if c not in target_cols + ["passed"]]
    cat_cols = df[feature_cols].select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return pre, feature_cols


def run_regression(X, y, feature_cols) -> dict:
    """Predict G3 with Linear Regression vs Random Forest."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    pre, _ = build_preprocessor(pd.concat([X, y.to_frame("G3")], axis=1), ["G3"])

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    results = {}
    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results[name] = {
            "R2": round(float(r2_score(y_test, y_pred)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            "MAE": round(float(mean_absolute_error(y_test, y_pred)), 4),
        }
        print(f"  [regression] {name}: R2={results[name]['R2']} "
              f"RMSE={results[name]['RMSE']} MAE={results[name]['MAE']}")

        if name == "Random Forest":
            # --- save actual-vs-predicted plot ---
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.scatter(y_test, y_pred, alpha=0.5, s=28)
            lims = [min(y_test.min(), y_pred.min()) - 1, max(y_test.max(), y_pred.max()) + 1]
            ax.plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
            ax.set_xlabel("Actual final grade (G3)")
            ax.set_ylabel("Predicted final grade")
            ax.set_title("Random Forest — Actual vs Predicted")
            ax.legend()
            fig.tight_layout()
            fig.savefig("images/regression_actual_vs_predicted.png", dpi=150)
            plt.close(fig)

            # --- save feature importance ---
            best_rf = pipe.named_steps["model"]
            feature_names = pipe.named_steps["pre"].get_feature_names_out()
            imp = pd.Series(best_rf.feature_importances_, index=feature_names)
            top = imp.sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            top.iloc[::-1].plot(kind="barh", color="#0f6b8c", ax=ax)
            ax.set_title("Top 10 predictors of final grade (RF importance)")
            ax.set_xlabel("Importance")
            fig.tight_layout()
            fig.savefig("images/feature_importance.png", dpi=150)
            plt.close(fig)

            joblib.dump(pipe, "models/regression_rf.joblib")

    return results


def run_classification(X, y, feature_cols) -> dict:
    """Predict pass/fail with Logistic Regression vs Random Forest."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pre, _ = build_preprocessor(pd.concat([X, y.to_frame("passed")], axis=1), ["G3"])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    results = {}
    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "f1": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_pred if hasattr(y_prob, "shape") else y_pred)), 4),
        }
        # use probability for AUC properly
        results[name]["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
        print(f"  [classification] {name}: acc={results[name]['accuracy']} "
              f"f1={results[name]['f1']} auc={results[name]['roc_auc']}")

        if name == "Random Forest":
            joblib.dump(pipe, "models/classification_rf.joblib")

            # --- confusion matrix ---
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4.6, 3.8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
            ax.set_title("Random Forest — Confusion Matrix (pass/fail)")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            fig.tight_layout()
            fig.savefig("images/confusion_matrix.png", dpi=150)
            plt.close(fig)

            report = classification_report(y_test, y_pred, target_names=["Fail", "Pass"])
            (Path("reports") / "classification_report.txt").write_text(report)
            print(report)

    return results


def main() -> None:
    df = load_dataset("data", PASS_THRESHOLD)
    print(f"Integrated dataset: {len(df)} students | pass rate: "
          f"{df['passed'].mean():.1%} (G3 >= {PASS_THRESHOLD})")

    # ---- EDA plots ----
    fig, ax = plt.subplots(figsize=(6, 3.6))
    sns.histplot(df["G3"], bins=20, color="#0f6b8c", ax=ax)
    ax.set_title("Distribution of final grades (G3)")
    fig.tight_layout()
    fig.savefig("images/grade_distribution.png", dpi=150)
    plt.close(fig)

    num_cols = ["age", "studytime", "failures", "absences", "G1", "G2", "G3"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation heatmap (numeric features)")
    fig.tight_layout()
    fig.savefig("images/correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # ---- Model runs ----
    X = df.drop(columns=["G1", "G2", "G3", "passed"])
    y_reg = df["G3"]
    y_clf = df["passed"]

    print("\n=== REGRESSION: predict final grade G3 ===")
    reg_metrics = run_regression(X, y_reg, list(X.columns))

    print("\n=== CLASSIFICATION: predict pass/fail ===")
    clf_metrics = run_classification(X, y_clf, list(X.columns))

    # ---- CV for stability ----
    pre, _ = build_preprocessor(pd.concat([X, y_clf.to_frame("passed")], axis=1), ["G3"])
    rf_pipe = Pipeline([("pre", pre),
                        ("model", RandomForestClassifier(n_estimators=300, max_depth=12,
                                                         random_state=RANDOM_STATE, n_jobs=-1))])
    cv_scores = cross_val_score(rf_pipe, X, y_clf, cv=5, scoring="roc_auc")
    print(f"\n5-fold CV ROC-AUC (RF classifier): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    summary = {
        "dataset": {"students": int(len(df)), "pass_rate": float(df["passed"].mean())},
        "regression": reg_metrics,
        "classification": clf_metrics,
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
    }
    (Path("reports") / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved reports/metrics.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
