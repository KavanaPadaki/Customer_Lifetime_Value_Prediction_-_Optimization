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
# MODEL METRICS (REGRESSION)
# ----------------------------------------------------------
class RegressionMetricsResponse(BaseModel):
    r2: float
    mae: float
    rmse: float
    n_train: int
    n_test: int


class ClassificationMetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    threshold: float




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
    cum_actual: List[float]


# ----------------------------------------------------------
# ELITE CUSTOMERS
# ----------------------------------------------------------
class EliteResponse(BaseModel):
    top_pct: int
    count: int
    elite_rows: List[dict]


# ----------------------------------------------------------
# TABLE
# ----------------------------------------------------------
class TableResponse(BaseModel):
    rows: List[dict]


# ----------------------------------------------------------
# HEALTH
# ----------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
