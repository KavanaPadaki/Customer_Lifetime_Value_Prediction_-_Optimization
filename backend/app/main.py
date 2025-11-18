import numpy as np
from fastapi import FastAPI, HTTPException
from .model_server import ModelServer

app = FastAPI()
ms = None

@app.on_event("startup")
def init_model():
    global ms
    ms = ModelServer()
    print(f"[INIT] Loaded {ms.data.shape[0]} rows")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/meta")
def meta():
    return ms.meta()

@app.get("/decile_lift")
def decile_lift():
    return {"decile_stats": ms.get_decile_stats()}

@app.get("/gain_curve")
def gain_curve():
    frac, gain = ms.get_gain_curve()
    return {"frac": frac, "cum_gain": gain}

@app.get("/elite")
def elite(top_pct: int = 5):
    try:
        elite_rows = ms.get_elite(top_pct)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"top_pct": top_pct, "count": len(elite_rows), "elite_rows": elite_rows[:500]}

@app.get("/table")
def table(limit: int = 100):
    return {"rows": ms.get_table(limit)}

@app.post("/predict")
def predict_customer(payload: dict):
    cust_id = str(payload.get("customer_id"))
    if not cust_id:
        raise HTTPException(status_code=400, detail="customer_id required")
    try:
        pred = ms.predict(cust_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Customer not found")
    return {"customer_id": cust_id, "predicted_clv": pred}
