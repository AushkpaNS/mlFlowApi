from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import uvicorn
import os
from src.prediction import get_prediction
import time

# создаем FastAPI приложение
app = FastAPI(
    title="ML Model API",
    description="API для получения предсказаний от ML модели",
    version="1.0.0"
)

# модели данных для Pydantic
class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: int
    probabilities: Dict[str, float]
    model_type: str
    features_used: List[str]
    status: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str

@app.get("/")
async def root():
    return {"message": "ML Model API is running"}

@app.get("/healthchecks")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/prediction", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    # получем предсказание
    try:
        result = get_prediction(request.features)
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=500, 
                detail=f"Prediction error: {result.get('error', 'Unknown error')}"
            )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# openAPI конфигурация
@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_json():
    return app.openapi()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
