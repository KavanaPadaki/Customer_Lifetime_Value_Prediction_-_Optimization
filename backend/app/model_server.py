import os
import json
import joblib
import pandas as pd
import numpy as np


# ============================================================
# PATHS (ROBUST: LOCAL + DOCKER + CLOUD)
# ============================================================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)

DATA_PATH = os.path.join(BASE_DIR,  "clv_prepared.parquet")
MODEL_PATH = os.path.join(BASE_DIR,  "model.joblib")
METRICS_PATH = os.path.join(BASE_DIR,  "metrics.json")
CLASSIFICATION_METRICS_PATH = os.path.join(
    BASE_DIR,  "classification_metrics.json"
)


class ModelServer:
    def __init__(self):
        # =====================================================
        # LOAD DATA + MODEL
        # =====================================================
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

        self.data = pd.read_parquet(DATA_PATH)
        self.model = joblib.load(MODEL_PATH)

        # =====================================================
        # LOAD METRICS (OPTIONAL BUT EXPECTED)
        # =====================================================
        self.metrics = {}
        self.classification_metrics = {}

        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                self.metrics = json.load(f)

        if os.path.exists(CLASSIFICATION_METRICS_PATH):
            with open(CLASSIFICATION_METRICS_PATH, "r") as f:
                self.classification_metrics = json.load(f)

        # =====================================================
        # FEATURE COLUMNS
        # =====================================================
        excluded_cols = {
            "Customer ID",
            "future_clv",
            "future_clv_log",
            "pred",
            "cohort_month",
            "first_purchase_date",
            "last_purchase_date",
        }

        self.feature_cols = [
            c for c in self.data.columns if c not in excluded_cols
        ]

        # =====================================================
        # CLEAN DATA (JSON SAFE)
        # =====================================================
        self.data = self.data.replace([np.inf, -np.inf], np.nan).fillna(0)

        # =====================================================
        # PRECOMPUTE PREDICTIONS (ONCE)
        # =====================================================
        self.data["pred"] = self.model.predict(
            self.data[self.feature_cols]
        )

        # =====================================================
        # CUSTOMER INDEX (FAST LOOKUP)
        # =====================================================
        self.customer_index = {
            str(cid): idx
            for idx, cid in enumerate(
                self.data["Customer ID"].astype(str)
            )
        }

        print(f"[INIT] Loaded {len(self.data)} customers")

    # =========================================================
    # SINGLE CUSTOMER PREDICTION
    # =========================================================
    def predict(self, customer_id: str) -> float:
        customer_id = str(customer_id)

        if customer_id not in self.customer_index:
            raise KeyError("Customer ID not found")

        return float(
            self.data.iloc[self.customer_index[customer_id]]["pred"]
        )

    # =========================================================
    # REGRESSION METRICS
    # =========================================================
    def get_metrics(self):
        if not self.metrics:
            raise RuntimeError("Regression metrics not available")
        return self.metrics

    # =========================================================
    # CLASSIFICATION METRICS
    # =========================================================
    def get_classification_metrics(self):
        if not self.classification_metrics:
            raise RuntimeError("Classification metrics not available")
        return self.classification_metrics

    # =========================================================
    # DECILE LIFT
    # =========================================================
    def get_decile_stats(self):
        df = self.data.copy()

        df["decile"] = pd.qcut(
            df["pred"].rank(method="first"),
            10,
            labels=False
        )

        out = (
            df.groupby("decile")
            .agg(
                avg_pred=("pred", "mean"),
                avg_actual_clv=("future_clv", "mean"),
                count=("Customer ID", "count"),
            )
            .reset_index()
        )

        out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
        return out.to_dict(orient="records")

    # =========================================================
    # GAIN CURVE
    # =========================================================
    def get_gain_curve(self):
        df = self.data.copy()
        df = df.sort_values("pred", ascending=False).reset_index(drop=True)

        df["frac"] = (df.index + 1) / len(df)
        total = df["future_clv"].sum() + 1e-12
        df["cum_actual"] = df["future_clv"].cumsum() / total

        return df["frac"].tolist(), df["cum_actual"].tolist()

    # =========================================================
    # ELITE CUSTOMERS
    # =========================================================
    def get_elite(self, top_pct: int):
        if top_pct < 1 or top_pct > 50:
            top_pct = 5

        cutoff = self.data["pred"].quantile(1 - top_pct / 100)
        elite = self.data[self.data["pred"] >= cutoff]

        elite = elite.sort_values("pred", ascending=False)
        elite = elite.replace([np.inf, -np.inf], np.nan).fillna(0)

        return elite.to_dict(orient="records")

    # =========================================================
    # TABLE
    # =========================================================
    def get_table(self, limit: int):
        df = self.data.head(limit).copy()
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return df.to_dict(orient="records")

    # =========================================================
    # META
    # =========================================================
    def meta(self):
        return {
            "rows": len(self.data),
            "features": self.feature_cols,
        }
