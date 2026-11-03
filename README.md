# 🏢 Employee Retention Prediction Pipeline

[![FastAPI](https://shields.io)](https://tiangolo.com)
[![LightGBM](https://shields.io)](https://readthedocs.io)
[![Render](https://shields.io)](https://render.com)

An end-to-end machine learning system that predicts employee attrition risks and job-change probabilities. This project transitions an experimental data science workflow from a Jupyter Notebook into a production-ready asynchronous web server.

🚀 **Live API Testing Dashboard (Swagger UI):** [https://employee-retention-api.onrender.com/docs](https://employee-retention-api.onrender.com/docs)
💻 **Interactive Development Notebook:** `Employee Retention Prediction.ipynb`

---

## 📌 Project Overview & Architecture

Predicting employee turnover is a critical challenge for HR departments to minimize recruitment costs and retain top talent. This project implements a high-performance **LightGBM Classifier** optimized to identify high-risk employees based on demographic, developmental, and professional history features.

### ⚙️ Production Deployment Lifecycle
```text
[ Jupyter Notebook (Research) ] ➔ [ Joblib Artifacts (.pkl) ] ➔ [ FastAPI Production Script (app.py) ] ➔ [ GitHub Push ] ➔ [ Render Cloud Hosting (Live) ]
```

---

## 📊 Key Highlights & Machine Learning Pipeline

1. **Exploratory Data Analysis & Preprocessing:** Handles missing entries across structured records using a dedicated 'Missing' label strategy to ensure real-time API inputs never cause runtime script crashes.
2. **Feature Engineering:** Converts customized domain-specific text values into clean continuous structures (`experience_num`).
3. **Imbalance Optimization (SMOTE):** Addresses severe target class distribution skewedness using Synthetic Minority Over-sampling Technique to maximize minority group classification capabilities.
4. **Model Performance Metrics:** 
   * **Overall Accuracy:** `79.4%`
   * **ROC-AUC Score:** `0.812`

---

## 📁 Repository Structure

```text
employee-retention-api/
│
├── Employee Retention Prediction.ipynb  # Core training notebook (EDA, SMOTE, Training)
├── app.py                              # Async FastAPI application (Production backend)
├── requirements.txt                    # Environment freeze tracking system libraries
├── lightgbm_model.pkl                  # Frozen pre-trained LightGBM classifier weights
├── scaler.pkl                          # Serialized training StandardScaler parameters
└── label_encoder.pkl                   # Serialized categorical transformer mapping
```

---

## 🛠️ Local Implementation Blueprint

To spin up this prediction engine locally on your command terminal:

### 1. Position into Workspace & Load Dependencies
```bash
cd employee-retention-api
pip install -r requirements.txt
```

### 2. Launch Local Web Server
```bash
uvicorn app:app --reload
```
Once initialized, navigate your local browser to `http://127.0.0` to access the interactive validation sandbox.

---

## 📥 Sample API JSON Schema Payload & Response

### HTTP POST `/predict` Route
```json
{
  "city": "city_103",
  "city_development_index": 0.92,
  "gender": "Male",
  "relevent_experience": "Has relevent experience",
  "enrolled_university": "no_enrollment",
  "education_level": "Graduate",
  "major_discipline": "STEM",
  "company_size": "50-99",
  "company_type": "Pvt Ltd",
  "last_new_job": "1",
  "training_hours": 36,
  "experience": ">20"
}
```

### Dynamic Backend API Output
```json
{
  "prediction": 0,
  "outcome": "Stay",
  "probability_score": {
    "stay_probability": 0.8285,
    "job_change_probability": 0.1715
  }
}
```
