from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

# Initialize FastAPI application
app = FastAPI(
    title="Employee Retention Prediction API",
    description="Production endpoint utilizing the trained LightGBM model to predict attrition risk.",
    version="1.0.0"
)

# Load the saved model and preprocessors from your notebook session
try:
    model = joblib.load('lightgbm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except Exception as e:
    raise RuntimeError(f"Error loading system assets: {str(e)}")

# Define input structure matching the exact feature sequence from your notebook
class EmployeePayload(BaseModel):
    city: str = Field(..., example="city_103")
    city_development_index: float = Field(..., example=0.920)
    gender: str = Field(..., example="Male")
    relevent_experience: str = Field(..., example="Has relevent experience")
    enrolled_university: str = Field(..., example="no_enrollment")
    education_level: str = Field(..., example="Graduate")
    major_discipline: str = Field(..., example="STEM")
    company_size: str = Field(..., example="50-99")
    company_type: str = Field(..., example="Pvt Ltd")
    last_new_job: str = Field(..., example="1")
    training_hours: int = Field(..., example=36)
    experience: str = Field(..., example=">20") # Passed as original text string just like raw source data

# Health Check Route
@app.get("/")
def health_check():
    return {
        "status": "Online",
        "engine": "LightGBM Classifier",
        "metrics": {"Accuracy": "79.4%", "ROC-AUC": "0.812"}
    }

# Prediction Route
@app.post("/predict")
def predict_attrition(payload: EmployeePayload):
    try:
        # 1. Parse incoming payload into a local dictionary
        raw_data = payload.model_dump()
        
        # 2. Extract and handle 'experience' string conversion exactly as designed in your custom clean_exp function
        exp_string = raw_data['experience']
        if exp_string == '>20':
            exp_numeric = 21.0
        elif exp_string == '<1':
            exp_numeric = 0.0
        elif exp_string == 'Missing':
            exp_numeric = 5.0  # Median value from your dataset profile
        else:
            try:
                exp_numeric = float(exp_string)
            except ValueError:
                exp_numeric = 5.0  # Safe fallback to dataset median
                
        # 3. Create the new column and drop the original text column to preserve matrix order
        raw_data['experience_num'] = exp_numeric
        del raw_data['experience']
        
        # 4. Construct DataFrame to safely keep feature names and column order intact
        input_df = pd.DataFrame([raw_data])
        
        # 5. Apply Label Encoding across matching categorical features
        cat_cols = ['city', 'gender', 'relevent_experience', 'enrolled_university', 
                    'education_level', 'major_discipline', 'company_size', 
                    'company_type', 'last_new_job']
        
        for col in cat_cols:
            current_value = str(input_df[col].iloc[0])
            # Handle unseen test features dynamically by falling back to 'Missing' group
            if current_value in label_encoder.classes_:
                input_df[col] = label_encoder.transform([current_value])
            else:
                input_df[col] = label_encoder.transform(['Missing'])
                
        # 6. Apply feature scaling using the loaded StandardScaler from your notebook
        scaled_features = scaler.transform(input_df)
        
        # 7. Generate inference probabilities using LightGBM
        prediction_class = int(model.predict(scaled_features)[0])
        probabilities = model.predict_proba(scaled_features)[0]
        churn_risk = float(probabilities[1])
        
        status_message = "Look for Job Change" if prediction_class == 1 else "Stay"
        
        return {
            "prediction": prediction_class,
            "outcome": status_message,
            "probability_score": {
                "stay_probability": round(float(probabilities[0]), 4),
                "job_change_probability": round(churn_risk, 4)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing Error: {str(e)}")
