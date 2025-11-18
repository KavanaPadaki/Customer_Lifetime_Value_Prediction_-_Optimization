import os
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Detect environment (container or host)


BASE = "data" 

RAW_PATH = f"{BASE}/processed/cleaned_dataset.csv"
OUT_PARQUET = f"{BASE}/final/clv_prepared.parquet"
OUT_MODEL = f"{BASE}/final/model.joblib"


# ----------------------------
# LOAD CLEANED TRANSACTIONS
# ----------------------------
def load_base():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"{RAW_PATH} not found")

    df = pd.read_csv(RAW_PATH, parse_dates=["InvoiceDate"])

    # Hard requirement check
    required = {"Customer ID", "Invoice", "InvoiceDate", "Quantity", "Price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Add revenue column (your pipeline needs this)
    df["revenue"] = df["Quantity"] * df["Price"]

    return df


# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def build_features_from_past(df_past):
    ref_date = df_past["InvoiceDate"].max()

    # --- Recency ---
    last_purchase = (
        df_past.groupby("Customer ID")["InvoiceDate"]
        .max()
        .reset_index()
    )
    last_purchase["recency"] = (ref_date - last_purchase["InvoiceDate"]).dt.days

    # --- Frequency ---
    freq = (
        df_past.groupby("Customer ID")["Invoice"]
        .nunique()
        .reset_index()
        .rename(columns={"Invoice": "frequency"})
    )

    # --- Monetary ---
    monetary = (
        df_past.groupby("Customer ID")["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"revenue": "monetary"})
    )

    # --- Tenure ---
    first_purchase = (
        df_past.groupby("Customer ID")["InvoiceDate"]
        .min()
        .reset_index()
    )
    first_purchase["tenure_days"] = (
        (ref_date - first_purchase["InvoiceDate"]).dt.days.clip(lower=1)
    )

    # --- Avg Order Value ---
    orders = (
        df_past.groupby("Customer ID")
        .agg(
            total_spend=("revenue", "sum"),
            orders=("Invoice", "nunique")
        )
        .reset_index()
    )

    orders["avg_order_value"] = (
        orders["total_spend"] / orders["orders"].replace(0, np.nan)
    )
    orders["avg_order_value"] = orders["avg_order_value"].fillna(0)

    # --- Avg days between orders ---
    avg_days_between = (
        df_past.groupby("Customer ID")["InvoiceDate"]
        .apply(lambda x: x.sort_values().diff().dt.days.mean())
        .reset_index()
    )
    avg_days_between.columns = ["Customer ID", "avg_days_between_orders"]

    median_days = avg_days_between["avg_days_between_orders"].median()
    if np.isnan(median_days):
        median_days = 0

    avg_days_between["avg_days_between_orders"] = (
        avg_days_between["avg_days_between_orders"].fillna(median_days)
    )

    # --- Purchase velocity ---
    active_months = first_purchase["tenure_days"] / 30
    active_months = active_months.replace(0, 1)

    velocity = freq.copy()
    velocity["purchase_velocity"] = (
        velocity["frequency"] / active_months
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- Merge all ---
    df = (
        last_purchase[["Customer ID", "recency"]]
        .merge(freq, on="Customer ID")
        .merge(monetary, on="Customer ID")
        .merge(first_purchase[["Customer ID", "tenure_days"]], on="Customer ID")
        .merge(orders[["Customer ID", "avg_order_value"]], on="Customer ID")
        .merge(avg_days_between, on="Customer ID")
        .merge(velocity[["Customer ID", "purchase_velocity"]], on="Customer ID")
    )

    # Final NaN/inf cleanup
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df

# ----------------------------
# COMPUTE FUTURE CLV (6 MONTHS)
# ----------------------------
def compute_future_clv(transactions, past_cutoff, horizon_months=6):
    future_end = past_cutoff + relativedelta(months=horizon_months)

    df_future = transactions[
        (transactions["InvoiceDate"] >= past_cutoff)
        & (transactions["InvoiceDate"] < future_end)
    ]

    future = (
        df_future.groupby("Customer ID")["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"revenue": "future_clv"})
    )

    return future


# ----------------------------
# TRAIN MODEL
# ----------------------------
def train_model(df):
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    features = [c for c in df.columns if c not in {"Customer ID", "future_clv", "future_clv_log","cohort_month"}]
    X = df[features]
    y = df["future_clv"].values

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)
    joblib.dump(model, OUT_MODEL)
    print(f"Model saved → {OUT_MODEL}")



# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():
    print("Loading cleaned dataset...")
    df = load_base()

    # last 6 months excluded for target
    cutoff = df["InvoiceDate"].max() - relativedelta(months=6)

    print("Building past features...")
    df_past = df[df["InvoiceDate"] < cutoff]
    features = build_features_from_past(df_past)

    print("Computing future CLV...")
    target = compute_future_clv(df, cutoff)

    print("Merging...")
    full = features.merge(target, on="Customer ID", how="left")
    full["future_clv"] = full["future_clv"].fillna(0)
    full["future_clv_log"] = np.log1p(full["future_clv"])

    print(f"Saving parquet → {OUT_PARQUET}")
    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    full.to_parquet(OUT_PARQUET)

    print("Training model...")
    train_model(full)

    print("DONE.")


if __name__ == "__main__":
    main()
