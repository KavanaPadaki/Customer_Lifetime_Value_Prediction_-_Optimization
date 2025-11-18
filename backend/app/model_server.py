import os
import joblib
import pandas as pd
import numpy as np

BASE = "/app/data" if os.path.exists("/.dockerenv") else "/data"
DATA_PATH = f"{BASE}/clv_prepared.parquet"
MODEL_PATH = f"{BASE}/model.joblib"


class ModelServer:
    def __init__(self):
        # ---------------------------------------------------------------------
        # LOAD DATA + MODEL
        # ---------------------------------------------------------------------
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

        self.data = pd.read_parquet(DATA_PATH)
        self.model = joblib.load(MODEL_PATH)

        # ---------------------------------------------------------------------
        # FEATURE COLS
        # ---------------------------------------------------------------------
        bad_cols = {
            "Customer ID",
            "future_clv",
            "future_clv_log",
            "pred",  # we will recompute
            "cohort_month",
            "first_purchase_date",
            "last_purchase_date",
        }

        self.feature_cols = [c for c in self.data.columns if c not in bad_cols]

        # ---------------------------------------------------------------------
        # CLEAN ALL NUMERIC DATA ONCE – JSON SAFE
        # ---------------------------------------------------------------------
        self.data = self.data.replace([np.inf, -np.inf], np.nan).fillna(0)

        # ---------------------------------------------------------------------
        # PRECOMPUTE PREDICTIONS (faster endpoints)
        # ---------------------------------------------------------------------
        self.data["pred"] = self.model.predict(self.data[self.feature_cols])

        # ---------------------------------------------------------------------
        # PREPARE FAST CUSTOMER INDEX LOOKUP
        # ---------------------------------------------------------------------
        self.customer_index = {
            str(cid): i for i, cid in enumerate(self.data["Customer ID"].astype(str))
        }

        print(f"[INIT] Loaded {len(self.data)} rows")

    # =========================================================================
    # PREDICT SINGLE CUSTOMER
    # =========================================================================
    def predict(self, customer_id: str) -> float:
        customer_id = str(customer_id)

        if customer_id not in self.customer_index:
            raise KeyError("Customer ID not found")

        row = self.data.iloc[self.customer_index[customer_id]]
        X = row[self.feature_cols].values.reshape(1, -1)
        return float(self.model.predict(X)[0])

    # =========================================================================
    # DECILE STATS
    # =========================================================================
    def get_decile_stats(self):
        df = self.data.copy()

        # Clean BEFORE ranking
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        df["decile"] = pd.qcut(df["pred"].rank(method="first"), 10, labels=False)

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

    # =========================================================================
    # GAIN CURVE
    # =========================================================================
    def get_gain_curve(self):
        df = self.data.copy()
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        df = df.sort_values("pred", ascending=False).reset_index(drop=True)

        df["frac"] = (df.index + 1) / len(df)
        total = df["future_clv"].sum() + 1e-12
        df["cum_actual"] = df["future_clv"].cumsum() / total

        return df["frac"].tolist(), df["cum_actual"].tolist()

    # =========================================================================
    # ELITE CUSTOMERS
    # =========================================================================
    def get_elite(self, top_pct: int):
        if top_pct < 1 or top_pct > 50:
            top_pct = 5

        df = self.data.copy()
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        cutoff = df["pred"].quantile(1 - top_pct / 100)

        elite = df[df["pred"] >= cutoff].copy()

        elite = elite.sort_values("pred", ascending=False)
        elite = elite.replace([np.inf, -np.inf], np.nan).fillna(0)

        return elite.to_dict(orient="records")

    # =========================================================================
    # TABLE
    # =========================================================================
    def get_table(self, limit: int):
        df = self.data.head(limit).copy()
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return df.to_dict(orient="records")

    # =========================================================================
    # META
    # =========================================================================
    def meta(self):
        return {
            "rows": len(self.data),
            "features": self.feature_cols,
        }

