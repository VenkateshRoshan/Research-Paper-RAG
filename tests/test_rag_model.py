import pytest
import asyncio
from unittest.mock import Mock, patch
from src.models.rag_model import RAGModel
from src.data.loader import DataLoader

@pytest.fixture
async def rag_model():
    """
        Fixture for RAG model instance
    """
    model = RAGModel(
        model_name="google/flan-t5-small",  # Use smaller model for testing
        max_length=512,
        quantization=None
    )
    yield model

@pytest.fixture
def sample_papers():
    """
        Fixture for sample papers
    """
    return [
        {
            "title": "Sample Paper 1",
            "authors": ["Author 1", "Author 2"],
            "categories": "cs.CL",
            "abstract": "This is a sample paper abstract.",
            "score": 0.8
        },
        {
            "title": "Sample Paper 2",
            "authors": ["Author 3", "Author 4"],
            "categories": "cs.LG",
            "abstract": "This is another sample paper abstract.",
            "score": 0.6
        }
    ]

@pytest.mark.asyncio
async def test_generate_response(rag_model, sample_papers):
    """Test response generation"""
    with patch.object(DataLoader, 'search', return_value=sample_papers):
        response = await rag_model.generate_response(
            query="What is machine learning?",
            max_new_tokens=100
        )
        
        assert isinstance(response, dict)
        assert "query" in response
        assert "response" in response
        assert "references" in response
        assert len(response["references"]) == len(sample_papers)

@pytest.mark.asyncio
async def test_error_handling(rag_model):
    """Test error handling in response generation"""
    with patch.object(DataLoader, 'search', side_effect=Exception("Test error")):
        response = await rag_model.generate_response(
            query="What is machine learning?",
            max_new_tokens=100
        )
        
        assert "error" in response["metadata"]
        assert response["metadata"]["status"] == "error"

@pytest.mark.asyncio
async def test_empty_query(rag_model):
    """Test handling of empty queries"""
    response = await rag_model.generate_response(
        query="",
        max_new_tokens=100
    )
    
    assert "error" in response["metadata"]

@pytest.mark.asyncio
async def test_long_query(rag_model, sample_papers):
    """Test handling of long queries"""
    with patch.object(DataLoader, 'search', return_value=sample_papers):
        long_query = "test " * 1000
        response = await rag_model.generate_response(
            query=long_query,
            max_new_tokens=100
        )
        
        assert isinstance(response, dict)
        assert "response" in response