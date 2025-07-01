import mlflow.sklearn
from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures, PredictionResponse
import pandas as pd

app = FastAPI()

# ✅ Replace <run_id> with the actual ID from your MLflow folder
LOCAL_MODEL_PATH = "C:\\Users\\Specter\\Documents\\Tenx_Academy\\Week-5\\credit-risk-model\\notebooks\\mlruns\\0\\models\\m-bc56b4a70362464382733ded4e174138\\artifacts"
model = mlflow.sklearn.load_model(LOCAL_MODEL_PATH)


@app.get("/")
def read_root():
    return {"status": "API is running!"}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(data: CustomerFeatures):
    df = pd.DataFrame([data.dict()])
    probability = model.predict_proba(df)[:, 1][0]
    prediction = int(probability >= 0.5)
    return PredictionResponse(risk_probability=round(probability, 4), is_high_risk=prediction)
