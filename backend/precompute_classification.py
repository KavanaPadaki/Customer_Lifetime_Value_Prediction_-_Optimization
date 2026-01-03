import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
import json

df = pd.read_parquet("backend/data/final/clv_prepared.parquet")

# Define high-value customer
threshold = df["future_clv"].quantile(0.75)
df["high_value"] = (df["future_clv"] >= threshold).astype(int)

features = [c for c in df.columns if c not in {"Customer ID", "future_clv", "future_clv_log", "high_value"}]

X = df[features]
y = df["high_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "roc_auc": roc_auc_score(y_test, proba),
    "positive_class": "High-Value Customer",
    "threshold": 0.5
}

with open("backend/data/classification_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(metrics)
