from pydantic import BaseModel
from typing import List


# ----------------------------------------------------------
# PREDICT ENDPOINT
# ----------------------------------------------------------
class PredictRequest(BaseModel):
    customer_id: str


class PredictResponse(BaseModel):
    customer_id: str
    predicted_clv: float


# ----------------------------------------------------------
# META
# ----------------------------------------------------------
class MetaResponse(BaseModel):
    rows: int
    features: List[str]


# ----------------------------------------------------------
# DECILE LIFT
# ----------------------------------------------------------
class DecileStat(BaseModel):
    decile: int
    avg_pred: float
    avg_actual_clv: float
    count: int


class DecileLiftResponse(BaseModel):
    decile_stats: List[DecileStat]


# ----------------------------------------------------------
# GAIN CURVE
# ----------------------------------------------------------
class GainCurveResponse(BaseModel):
    frac: List[float]
    cum_gain: List[float]


# ----------------------------------------------------------
# ELITE CUSTOMERS
# ----------------------------------------------------------
class EliteRow(BaseModel):
    Customer_ID: str | int
    pred: float
    future_clv: float


class EliteResponse(BaseModel):
    top_pct: int
    count: int
    elite_rows: List[dict]   # using dict because row fields vary


# ----------------------------------------------------------
# TABLE
# ----------------------------------------------------------
class TableResponse(BaseModel):
    rows: List[dict]   # flexible structure, DF rows 


# ----------------------------------------------------------
# HEALTH
# ----------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
