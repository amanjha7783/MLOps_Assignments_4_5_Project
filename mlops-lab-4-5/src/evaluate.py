"""Evaluate the selected model with reproducible held-out metrics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "dataset.csv")
features = [c for c in df.columns if c != "price_lakh"]
X_train, X_test, y_train, y_test = train_test_split(df[features], df["price_lakh"], test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
result = {
    "model": "RandomForest_200_depth8",
    "test_rows": int(len(X_test)),
    "features": features,
    "mae": float(mean_absolute_error(y_test, pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    "r2": float(r2_score(y_test, pred)),
}
out = ROOT / "reports" / "evaluation.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
