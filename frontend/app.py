import os
import requests
import streamlit as st
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
if not BACKEND_URL:
    st.stop()
    raise RuntimeError("BACKEND_URL environment variable not set")

TIMEOUT = 10

# ============================================================
# API HELPERS
# ============================================================
def api_get(path: str, params=None):
    try:
        r = requests.get(
            f"{BACKEND_URL}{path}",
            params=params,
            timeout=TIMEOUT
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"GET {path} failed: {e}")
        return {}

def api_post(path: str, payload: dict):
    try:
        r = requests.post(
            f"{BACKEND_URL}{path}",
            json=payload,
            timeout=TIMEOUT
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"POST {path} failed: {e}")
        return {}

# ============================================================
# CACHED LOADERS
# ============================================================
@st.cache_data(show_spinner=False)
def load_customer_ids():
    data = api_get("/table", params={"limit": 50000})
    rows = data.get("rows", [])
    if not rows:
        return []

    df = pd.DataFrame(rows)
    if "Customer ID" not in df.columns:
        return []

    return sorted(df["Customer ID"].astype(str).unique())

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Customer Lifetime Value Dashboard",
    layout="wide"
)

st.title("📈 Customer Lifetime Value Dashboard")

# ============================================================
# MODEL SUMMARY
# ============================================================
st.subheader("🧠 Model Overview")
st.markdown("""
- **Problem:** Customer Lifetime Value Prediction  
- **Type:** Regression (RandomForestRegressor)  
- **Target:** 6-month future CLV  
- **Evaluation:** Time-based holdout  
""")

# ============================================================
# REGRESSION METRICS
# ============================================================
st.subheader("📊 Regression Metrics (CLV Estimation)")

reg_metrics = api_get("/metrics")

if reg_metrics:
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{reg_metrics['r2']:.3f}")
    c2.metric("MAE", f"{reg_metrics['mae']:.2f}")
    c3.metric("RMSE", f"{reg_metrics['rmse']:.2f}")

    st.caption(
        f"Train size: {reg_metrics['n_train']} | "
        f"Test size: {reg_metrics['n_test']}"
    )
else:
    st.warning("Regression metrics not available.")

# ============================================================
# CLASSIFICATION METRICS
# ============================================================
st.subheader("🎯 Classification Metrics (High-Value Customer Identification)")

clf_metrics = api_get("/metrics/classification")

if clf_metrics:
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{clf_metrics['accuracy']:.2f}")
    c2.metric("Precision", f"{clf_metrics['precision']:.2f}")
    c3.metric("Recall", f"{clf_metrics['recall']:.2f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("F1 Score", f"{clf_metrics['f1']:.2f}")
    c5.metric("ROC-AUC", f"{clf_metrics['roc_auc']:.2f}")
    c6.metric("Threshold", clf_metrics["threshold"])

    with st.expander("ℹ️ About this classification model"):
        st.markdown("""
        - Predicts whether a customer is **high-value**
        - Used for **targeting & segmentation**
        - Evaluated offline, metrics shown for comparison
        """)
else:
    st.warning("Classification metrics not available.")

# ============================================================
# DECILE LIFT
# ============================================================
st.subheader("🔥 Decile Lift")

deciles = api_get("/decile_lift").get("decile_stats", [])
if deciles:
    st.dataframe(pd.DataFrame(deciles))
else:
    st.warning("Decile lift data not available.")

# ============================================================
# GAIN CURVE
# ============================================================
st.subheader("📈 Gain Curve (Cumulative CLV Capture)")

gc = api_get("/gain_curve")
frac = gc.get("frac", [])
cum = gc.get("cum_actual", [])

if frac and cum:
    chart_df = pd.DataFrame({
        "Population Fraction": frac,
        "Cumulative CLV": cum
    })
    st.line_chart(chart_df, x="Population Fraction", y="Cumulative CLV")
else:
    st.warning("Gain curve not available.")

# ============================================================
# ELITE CUSTOMERS
# ============================================================
st.subheader("👑 Elite Customers")

top_pct = st.slider("Top % of customers", 1, 50, 5)
elite = api_get("/elite", params={"top_pct": top_pct})

if elite and elite.get("elite_rows"):
    st.write(
        f"Top {elite['top_pct']}% customers "
        f"(showing up to 500 rows)"
    )
    st.dataframe(pd.DataFrame(elite["elite_rows"]))
else:
    st.warning("Elite segment not available.")

# ============================================================
# SINGLE CUSTOMER PREDICTOR
# ============================================================
st.subheader("🔍 Predict CLV for a Single Customer")

customer_ids = load_customer_ids()

if not customer_ids:
    st.error("Customer IDs could not be loaded from backend.")
else:
    cid = st.selectbox("Select Customer ID", customer_ids)

    if st.button("Predict CLV"):
        result = api_post("/predict", {"customer_id": cid})
        if "predicted_clv" in result:
            st.success(
                f"Predicted 6-month CLV: "
                f"${result['predicted_clv']:.2f}"
            )
        else:
            st.error("Prediction failed.")

# ============================================================
# TABLE VIEW
# ============================================================
st.subheader("📄 Customer Table")

limit = st.slider("Rows to display", 10, 1000, 100)
table = api_get("/table", params={"limit": limit})

if table.get("rows"):
    st.dataframe(pd.DataFrame(table["rows"]))
else:
    st.warning("No table data available.")
