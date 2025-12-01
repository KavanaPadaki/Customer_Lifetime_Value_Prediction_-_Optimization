
# Customer Lifetime Value (CLV) Prediction System

End-to-end machine learning system for predicting 6-month Customer Lifetime Value (CLV).
Backend served via FastAPI, frontend built with Streamlit, deployed using Docker and Railway.

## Live Demo

Frontend (Streamlit):[link](https://clv-app-final-production.up.railway.app/)
Backend (FastAPI):[link](https://customerlifetimevalueprediction-optimizatio-production.up.railway.app/health)

---

## Features

### Analytics

* Gain curve
* Decile lift
* Customer segmentation (Elite / Top-X%)
* Customer data table
* Single-customer real-time prediction

### Backend (FastAPI)

Endpoints:

* `POST /predict`
* `GET /decile_lift`
* `GET /gain_curve`
* `GET /elite`
* `GET /table`
* `GET /meta`
* `GET /health`

### Frontend (Streamlit)

* Real-time charts and segmentation analytics
* Customer browsing interface
* Online CLV prediction interface

---

## Project Structure

```
clv-app/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── model_server.py
│   │   ├── metrics.py
│   │   ├── schemas.py
│   │   └── utils.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── app.py
│   └── requirements.txt
│
├── data/
│   └── final/
│       ├── clv_prepared.parquet
│       └── model.joblib
│
├── docker-compose.yml
└── README.md
```

---

## Local Setup

### Run with Docker

```bash
docker-compose up --build
```

## Deployment

### Deploy to Railway (Backend)

1. Push repository to GitHub
2. Create Railway service → Deploy from GitHub (root: `backend`)
3. Upload ML artifacts to:

```
/app/data/final/
```

4. Add environment variables:

```
DATA_DIR=/app/data
PYTHONUNBUFFERED=1
```

5. Deploy

### Deploy to Streamlit Cloud (Frontend)

Add secret:

```
CLV_API_URL="https://YOUR-RAILWAY-URL"
```

---

## Example Requests

### Predict Endpoint

```json
{
  "customer_id": "12345"
}
```

### Table

```
GET /table?limit=100
```

### Elite segment

```
GET /elite?top_pct=5
```

---

## Future Enhancements

* Database persistence
* Automated model retraining pipeline
* XGBoost version
* CI/CD automation
* SHAP explainability

---

## Purpose

Demonstrates practical ML engineering through model development, API deployment, and production-grade architecture.


Now stop wasting time on formatting and move back to execution.

