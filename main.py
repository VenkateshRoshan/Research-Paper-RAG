from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import uvicorn
from src.models.rag_model import RAGModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

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
@app.get("/health")
async def health():
    return {"status": "healthy"}

# Vertex AI prediction endpoint
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)