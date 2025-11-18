import os
import requests
import streamlit as st
import pandas as pd

# ============================================================
# BACKEND URL
# ============================================================
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")


# ============================================================
# API HELPERS
# ============================================================
def api_get(path: str, params=None):
    url = f"{BACKEND}{path}"
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"GET {path} failed: {e}")
        return {}


def api_post(path: str, payload: dict):
    url = f"{BACKEND}{path}"
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"POST {path} failed: {e}")
        return {}


# ============================================================
# CACHED LOADER FOR CUSTOMER IDS
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
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="CLV Dashboard", layout="wide")
st.title("📈 Customer Lifetime Value Dashboard")


# ============================================================
# META
# ============================================================
st.header("📊 Metadata")
meta = api_get("/meta")

if meta:
    st.write(f"**Rows:** {meta.get('rows', 0)}")
    st.write(f"**Features:** {len(meta.get('features', []))}")


# ============================================================
# DECILE LIFT
# ============================================================
st.header("🔥 Decile Lift")

dl = api_get("/decile_lift").get("decile_stats", [])

if dl:
    st.dataframe(pd.DataFrame(dl))
else:
    st.warning("Decile lift data not available.")


# ============================================================
# GAIN CURVE
# ============================================================
st.header("📈 Gain Curve")

gc = api_get("/gain_curve")
frac = gc.get("frac", [])
cum = gc.get("cum_gain", [])

if frac and cum:
    chart_df = pd.DataFrame({"Population": frac, "Gain": cum})
    st.line_chart(chart_df, x="Population", y="Gain")
else:
    st.warning("Gain curve not available.")


# ============================================================
# ELITE SEGMENT
# ============================================================
st.header("👑 Elite Customers")

top_pct = st.slider("Top %", 1, 50, 5)
elite_data = api_get("/elite", params={"top_pct": top_pct})

if elite_data and elite_data.get("elite_rows"):
    st.write(f"Top {elite_data['top_pct']}% customers (showing max 500 rows)")
    st.dataframe(pd.DataFrame(elite_data["elite_rows"]))
else:
    st.warning("Elite segment not available.")


# ============================================================
# SINGLE CUSTOMER PREDICTOR
# ============================================================
st.header("🔍 Predict CLV for a Single Customer")

customer_ids = load_customer_ids()

if not customer_ids:
    st.error("Could not load Customer IDs from backend.")
else:
    cid = st.selectbox("Select Customer ID", customer_ids)

    if st.button("Predict Customer"):
        resp = api_post("/predict", {"customer_id": cid})
        if "predicted_clv" in resp:
            st.success(f"Predicted CLV: ${resp['predicted_clv']:.2f}")
        else:
            st.error("Prediction failed.")


# ============================================================
# TABLE VIEW
# ============================================================
st.header("📄 Customer Table")

limit = st.slider("Rows", 10, 1000, 100)
table_data = api_get("/table", params={"limit": limit})

if "rows" in table_data and table_data["rows"]:
    st.dataframe(pd.DataFrame(table_data["rows"]))
else:
    st.warning("No table data available.")
