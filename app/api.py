# REST API Endpoints. All API-related code goes here
from fastapi import APIRouter, HTTPException
from app.core.pipeline import run_pipeline

router = APIRouter()

@router.post("/predict")
async def predict(text: str):
    """
    Accept raw text and return phishing likelihood and explanation.
    Example: POST /api/predict?text="Free gift card!"
    """
    try:
        result = run_pipeline(text)
        return {
            "label": result["label"],
            "probability": result["probability"],
            "explanation": result["explanation"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
