# tests/test_data_loader.py
import pytest
from src.data.loader import DataLoader

@pytest.fixture
def data_loader():
    return DataLoader()

def test_data_loader_initialization(data_loader):
    """Test DataLoader initialization"""
    assert data_loader.embedder is not None
    assert data_loader.cache_dir is not None

@pytest.mark.asyncio
async def test_paper_search(data_loader):
    """Test paper search functionality"""
    papers = await data_loader.search("machine learning", k=2)
    
    assert isinstance(papers, list)
    assert len(papers) <= 2
    for paper in papers:
        assert "title" in paper
        assert "authors" in paper
        assert "abstract" in paper
        assert "categories" in paper
        assert "score" in paper
