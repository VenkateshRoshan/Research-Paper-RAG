import pytest
from unittest.mock import Mock, patch
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(autouse=True)
def mock_gcp_credentials():
    """Mock GCP credentials for all tests"""
    with patch('google.cloud.storage.Client') as mock_storage:
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        
        # Setup mock chain
        mock_storage.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        yield mock_storage