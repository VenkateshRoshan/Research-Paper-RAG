# src/api/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import logging
import uvicorn
from src.models.rag_model import RAGModel

from src.monitoring.metrics import track_query_metrics
from prometheus_client import make_asgi_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Research Paper RAG API",
    description="API for research paper retrieval and analysis using RAG",
    version="1.0.0"
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Initialize RAG model
model = None

def get_model():
    global model
    if model is None:
        model = RAGModel(gcs_bucket='research-paper-rag-data')
    return model

# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    style: Optional[str] = Field("academic", pattern="^(academic|simplified|technical)$")
    max_tokens: Optional[int] = Field(150, ge=50, le=500)
    num_papers: Optional[int] = Field(3, ge=1, le=10)

class PaperReference(BaseModel):
    title: str
    authors: List[str]
    categories: str
    relevance_score: float
    citation: str

class QueryResponse(BaseModel):
    query: str
    response: str
    references: List[PaperReference]
    metadata: Dict
    timestamp: datetime

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

# Main query endpoint
@app.post("/query", response_model=QueryResponse)
@track_query_metrics
async def process_query(request: QueryRequest):
    try:

        model = get_model()
        # Generate response using RAG model
        result = model.generate_response(
            query=request.query,
            max_new_tokens=request.max_tokens,
            # style=request.style,
            num_papers=request.num_papers
        )

        # Add timestamp
        result["timestamp"] = datetime.now()

        return result

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

"""
    curl -X POST "http://localhost:8080/query"   -H "Content-Type: application/json"   -d '{
        "query": "can we fine-tune LLM with less memory by using LoRA",
        "max_tokens": 400,
        "num_papers": 1
    }'

"""