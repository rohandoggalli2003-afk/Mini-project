import joblib
import pandas as pd
import numpy as np


MODEL_PATH = r"D:\AI-Connected-Vehicle-Analytics\models\xgboost_vehicle_failure.pkl"


class VehicleFailurePredictor:

    def __init__(self):
        model_data = joblib.load(MODEL_PATH)

        self.model = model_data["model"]
        self.features = model_data["features"]

        print("XGBoost model loaded successfully.")
        print(f"Expected features: {len(self.features)}")

    def predict(self, record):
        # Create DataFrame with exactly the same feature order
        X = pd.DataFrame([record])

        X = X[self.features]

        # Handle missing/infinite values
        X = X.replace([np.inf, -np.inf], np.nan)

        X = X.fillna(X.median(numeric_only=True))

        # Failure probability
        probability = self.model.predict_proba(X)[0][1]

        # Default threshold
        prediction = int(probability >= 0.5)

        if probability >= 0.75:
            status = "CRITICAL"
        elif probability >= 0.5:
            status = "WARNING"
        else:
            status = "NORMAL"

        return {
            "failure_probability": float(probability),
            "prediction": prediction,
            "status": status
        }