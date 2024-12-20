from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import uvicorn
import os
from src.models.rag_model import RAGModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Get environment variables for Vertex AI
AIP_PREDICT_ROUTE = os.getenv('AIP_PREDICT_ROUTE', '/predict')
AIP_HEALTH_ROUTE = os.getenv('AIP_HEALTH_ROUTE', '/health')

# Model singleton for reuse
class ModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = RAGModel(gcs_bucket='research-paper-rag-data')
        return cls._instance

# Pydantic models
class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]

class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]

# Health check endpoint required by Vertex AI
@app.get(AIP_HEALTH_ROUTE)
@app.get("/health")
async def health():
    return {"status": "healthy"}

# Vertex AI prediction endpoint
@app.post(AIP_PREDICT_ROUTE)
@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        model = ModelSingleton.get_instance()
        predictions = []
        
        for instance in request.instances:
            query = instance.get("query", "")
            max_tokens = instance.get("max_tokens", 150)
            num_papers = instance.get("num_papers", 3)
            
            result = model.generate_response(
                query=query,
                max_new_tokens=max_tokens,
                num_papers=num_papers
            )
            predictions.append(result)

        return {"predictions": predictions}

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add Vertex AI specific route
@app.get("/v1/endpoints/{endpoint_id}/deployedModels/{model_id}")
async def vertex_model_status(endpoint_id: str, model_id: str):
    return {
        "endpoint_id": endpoint_id,
        "model_id": model_id,
        "status": "deployed"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )