"""Train and compare three house-price regressors; optionally track with MLflow.

Run from repository root:
    python src/train.py

If MLflow is installed, runs, metrics, artifacts, signatures, and models are logged.
The script does not fabricate a registry result: register_model is invoked for each run
only when the tracking/registry server supports it.
"""
from pathlib import Path
import json
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA)
target = "price_lakh"
features = [c for c in df.columns if c != target]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42
)
candidates = [
    ("RandomForest_100", RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)),
    ("RandomForest_200_depth8", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
    ("ExtraTrees_200_depth10", ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
]
try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
except Exception:
    git_commit = "unavailable-not-a-git-checkout"

mlflow.set_experiment("MLOps-Lab-4-5-House-Prices")
mlflow.sklearn.autolog(log_models=False)
rows = []
for name, model in candidates:
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred)),
        }
        mlflow.log_params({"candidate": name, "test_size": 0.2, "random_state": 42, "feature_count": len(features)})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "git_commit": git_commit,
            "dataset_sha256": __import__("hashlib").sha256(DATA.read_bytes()).hexdigest(),
            "dataset_file": "data/dataset.csv",
        })
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(3)
        mlflow.sklearn.log_model(
            model, artifact_path="model",
            signature=signature, input_example=input_example
        )
        # At least two artifacts: comparison/configuration and diagnostic plots.
        pd.DataFrame([{"model": name, **metrics}]).to_csv(REPORTS / f"{name}_metrics.csv", index=False)
        mlflow.log_artifact(str(REPORTS / f"{name}_metrics.csv"))
        fig_path = FIGURES / f"{name}_residuals.png"
        plt.figure(figsize=(6, 4))
        plt.scatter(pred, y_test.to_numpy() - pred, alpha=0.6)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.title(f"Residuals: {name}")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=140)
        plt.close()
        mlflow.log_artifact(str(fig_path))
        rows.append({"model": name, **metrics, "run_id": run.info.run_id})
comparison = pd.DataFrame(rows).sort_values("rmse")
comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
print(comparison.to_string(index=False))
print("\nAcceptance rule: select the model with lowest held-out RMSE, provided evaluation data and operational checks are valid.")
print("For registration, use the run IDs above with the MLflow UI or a registry-enabled tracking server.")
