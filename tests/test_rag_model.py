# tests/test_rag_model.py
import pytest
from unittest.mock import patch, Mock

from src.models.rag_model import RAGModel
import torch


@pytest.fixture
def mock_tokenizer():
    with patch('transformers.AutoTokenizer.from_pretrained') as mock:
        mock_instance = Mock()
        mock_instance.pad_token = None
        mock_instance.eos_token = '[EOS]'
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_model():
    with patch('transformers.T5ForConditionalGeneration.from_pretrained') as mock:
        mock_instance = Mock()
        mock_instance.config = Mock()
        mock_instance.config.eos_token_id = 1
        mock_instance.config.max_length = 2048
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def rag_model(mock_tokenizer, mock_model):
    with patch('src.data.loader.DataLoader') as mock_loader:
        model = RAGModel(
            model_name="google/flan-t5-base",
            gcs_bucket="test-bucket"
        )
        return model

def test_generate_response(rag_model):
    sample_query = "What are transformers?"
    mock_papers = [{
        'title': 'Test Paper',
        'abstract': 'Test Abstract',
        'authors': ['Test Author'],
        'categories': 'cs.AI',
        'score': 0.95
    }]
    
    # Mock the search results
    rag_model.loader.search.return_value = mock_papers
    
    # Mock the model generation
    rag_model.model.generate.return_value = torch.tensor([[1, 2, 3]])
    rag_model.tokenizer.decode.return_value = "Test response"
    
    response = rag_model.generate_response(
        query=sample_query,
        max_new_tokens=150,
        num_papers=1
    )
    
    assert isinstance(response, dict)
    assert response['query'] == sample_query
    assert 'response' in response
    assert len(response['references']) == 1