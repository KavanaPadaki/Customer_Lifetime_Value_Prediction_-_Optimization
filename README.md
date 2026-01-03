# 📈 Customer Lifetime Value (CLV) Prediction System

An end-to-end **Customer Lifetime Value prediction platform** built using Machine Learning, FastAPI, Docker, and Streamlit, deployed on cloud infrastructure.

This project demonstrates **data preprocessing, feature engineering, model training, evaluation, API serving, and frontend visualization** in a production-style setup.

---

## 🚀 Live Demo

- **Frontend :**  
  [Streamlit Dashboard](https://customerlifetimevalueprediction-optimization-dn5rgub64tqsqzf25.streamlit.app/)

- **Backend :**  
  [FastAPI](https://customer-lifetime-value-prediction.onrender.com)

---

## 🧠 Problem Statement

Customer Lifetime Value (CLV) is a critical metric for:
- Customer segmentation
- Marketing spend optimization
- Retention strategies

The goal of this project is to:
- Predict **future customer value**
- Rank customers by expected value
- Evaluate model performance using **regression and classification metrics**
- Expose predictions and analytics via a REST API
- Visualize results in an interactive dashboard

---


## 📊 Models Implemented

### 1️⃣ Regression Model (CLV Prediction)
- **Model:** RandomForestRegressor
- **Target:** Future 6-month customer revenue

**Metrics:**
- R² Score
- MAE
- RMSE

### 2️⃣ Classification Model (Customer Value Segment)
- Binary classification (High vs Low CLV)
- Used for interpretability and business insights

**Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## 📈 Evaluation & Business Metrics

The system also computes:
- **Decile Lift Analysis**
- **Gain Curve**
- **Elite Customer Identification (Top X%)**

These metrics help evaluate how well the model prioritizes high-value customers.

---

## 🧪 Tech Stack

### Backend
- Python 3.11
- FastAPI
- Scikit-learn
- Pandas / NumPy
- Joblib
- Docker

### Frontend
- Streamlit
- Requests
- Pandas

### Deployment
- **Backend:** Render (Dockerized FastAPI)
- **Frontend:** Streamlit Community Cloud

---

