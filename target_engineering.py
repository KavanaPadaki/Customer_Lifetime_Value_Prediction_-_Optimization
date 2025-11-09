import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

def attach_future_clv(
    transactions_df: pd.DataFrame,
    features_df: pd.DataFrame,
    cutoff_date: str,
    horizon_months: int = 6,
    customer_col: str = "Customer ID",
    date_col: str = "InvoiceDate",
    amount_col: str = "Price",
):
    """
    Merges future CLV target onto an existing feature-engineered dataset.
    """

    # ensure date format
    transactions_df[date_col] = pd.to_datetime(transactions_df[date_col])
    cutoff = pd.to_datetime(cutoff_date)
    future_end = cutoff + relativedelta(months=horizon_months)

    # future CLV
    df_future = transactions_df[
        (transactions_df[date_col] >= cutoff)
        & (transactions_df[date_col] < future_end)
    ]
    clv_target = (
        df_future.groupby(customer_col)[amount_col]
        .sum()
        .reset_index()
        .rename(columns={amount_col: "future_clv"})
    )

    # merge
    df_final = features_df.merge(clv_target, on=customer_col, how="left")
    df_final["future_clv"] = df_final["future_clv"].fillna(0)
    df_final["future_clv_log"] = np.log1p(df_final["future_clv"])

    # diagnostics
    print(f"Cutoff: {cutoff.date()} | Horizon: {horizon_months} months | Future end: {future_end.date()}")
    print(f"Customers: {df_final.shape[0]} | Nonzero CLV: {(df_final['future_clv']>0).mean():.2%}")
    print(df_final["future_clv"].describe())

    return df_final

# Load both datasets
transactions_df = pd.read_csv("data/processed/cleaned_dataset.csv")
features_df = pd.read_csv("data/processed/feature_engineered_clv_dataset.csv")

final_df = attach_future_clv(
    transactions_df,
    features_df,
    cutoff_date="2010-06-30",  # halfway point of your 2-year data
    horizon_months=6
)

final_df.to_parquet("data/final/customers_features_target.parquet", index=False)
