from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("model.joblib")

app = FastAPI(
    title="Iris ML Classifier API",
    description="API for predicting Iris flower species using a trained RandomForest model.",
    version="1.0.0",
)

artifact = None
if MODEL_PATH.exists():
    artifact = joblib.load(MODEL_PATH)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str


@app.get("/")
def health_check():
    return {"message": "ML API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    if artifact is None:
        raise HTTPException(
            status_code=500,
            detail="Model file not found. Run 'python train.py' first.",
        )

    model = artifact["model"]
    target_names = artifact["target_names"]

    input_data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]

    pred_id = int(model.predict(input_data)[0])
    return PredictionResponse(
        class_id=pred_id,
        class_name=target_names[pred_id],
    )