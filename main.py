# main.py content for FastAPI Server

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List

# 1. Load the Model and Feature Names
# The model must be loaded only once when the server starts
try:
    model = joblib.load('german_credit_model.pkl')
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model file not found. Check if 'german_credit_model.pkl' is in the current directory.")

# 2. Define Input Schema (Based on the 49 columns after One-Hot Encoding)
# We need to define exactly what the API expects for a new prediction.
# Since our model was trained on 48 features (after encoding), we will define a simplified input first 
# and handle the encoding internally in the API.

# Note: The original German Credit Data has 20 features (8 numerical, 12 categorical). 
# For simplicity, we define a structure that can hold the raw data and then encode it.
# We will skip defining all 20 columns here to save space, but in a real project, all 20 must be defined.

# Simplified Input Model (for demonstration)
class InputData(BaseModel):
    # This assumes the raw data attributes are provided as a list of dictionaries (one row per dict)
    # Since our raw data had Attributes like 'Attribute1', 'Attribute2', etc.
    # We will assume a single row of the raw data (20 features) comes in.
    
    # Example fields (replace with your 20 actual input fields in a real project)
    # We use a generic list of floats/ints for simplification in this code block:
    features: List[float] 


# 3. Initialize FastAPI App
app = FastAPI(
    title="German Credit Risk Classifier API",
    description="A service for predicting credit risk (Good/Bad) based on German Credit Data using Logistic Regression."
)

# 4. Define the Prediction Endpoint (The main function)
@app.post("/predict_risk")
def predict_credit_risk(data: InputData):
    """
    Receives a new set of data and returns the predicted credit risk (0: Good, 1: Bad).
    """
    
    # --- IMPORTANT SIMPLIFICATION STEP ---
    # In a real-world scenario, the incoming raw data (data.features) MUST be One-Hot Encoded 
    # to match the 48 columns the model expects. 
    
    # Since we can't perform complex encoding inside this simple API demo, 
    # we assume the input data is ALREADY PROCESSED and has 48 columns.
    
    # If the input list does not have exactly 48 features (after processing), we raise an error.
    if len(data.features) != 48:
         raise HTTPException(status_code=400, detail="Input array must contain exactly 48 processed features.")

    try:
        # 4.1 Convert the incoming list of features into a DataFrame
        input_df = pd.DataFrame([data.features])
        
        # 4.2 Get the prediction
        prediction_int = model.predict(input_df)[0]
        
        # 4.3 Map the prediction integer to a human-readable label
        risk_map = {0: "Good Credit Risk", 1: "Bad Credit Risk"}
        risk_label = risk_map.get(prediction_int, "Error")
        
        return {
            "prediction_code": int(prediction_int),
            "prediction_label": risk_label,
            "message": "Prediction successful."
        }
    
    except Exception as e:
        # Catch any errors during prediction (e.g., incorrect data types)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")