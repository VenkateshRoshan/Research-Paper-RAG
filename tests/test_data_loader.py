# tests/test_loader.py
import pytest
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np

from src.data.loader import DataLoader

@pytest.fixture
def mock_sentence_transformer():
    with patch('sentence_transformers.SentenceTransformer') as mock:
        mock_instance = Mock()
        mock_instance.encode.return_value = np.random.rand(1, 384)
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def data_loader(mock_gcp_credentials, mock_sentence_transformer):
    return DataLoader(bucket_name="test-bucket")

def test_search_functionality(data_loader):
    # Mock the necessary data
    data_loader.papers_data = pd.DataFrame({
        'title': ['Test Paper'],
        'abstract': ['Test Abstract'],
        'authors': [['Test Author']],
        'categories': ['cs.AI']
    })
    
    # Mock FAISS index
    with patch('faiss.IndexFlatL2') as mock_index:
        data_loader.index = mock_index
        mock_index.search.return_value = (
            np.array([[0.5]]),  # distances
            np.array([[0]])     # indices
        )
        
        results = data_loader.search("test query", k=1)
        
        assert len(results) == 1
        assert results[0]['title'] == 'Test Paper'
        assert results[0]['score'] > 0