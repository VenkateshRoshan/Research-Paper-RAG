from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import os
import pandas as pd
from google.cloud import storage
import tempfile
from google.oauth2 import service_account
import google.auth
from google.api_core import retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, 
                 bucket_name: str = None, 
                 cache_dir: str = "data/processed",
                 credentials_path: str = "/app/credentials.json",
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the DataLoader with support for both local and Vertex AI environments
        
        Args:
            bucket_name (str): Name of the GCS bucket
            cache_dir (str): Local directory for cache
            credentials_path (str): Optional path to credentials file (for local development)
            embedding_model (str): Name of the SentenceTransformer model to use
        """
        self.bucket_name = bucket_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage client
        self.storage_client = self._initialize_storage_client(credentials_path)
        self.bucket = self.storage_client.bucket(self.bucket_name)
        
        # Initialize SBERT model for embeddings
        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.papers_data = None

        # save data to GCS
        # self.load_and_process_data()

    def _initialize_storage_client(self, credentials_path: str = None) -> storage.Client:
        """Initialize storage client with support for both environments"""
        try:
            # First try: Check if running on Vertex AI
            try:
                from google.cloud import aiplatform
                logger.info("Running on Vertex AI, using default credentials")
                return storage.Client()
            except ImportError:
                pass

            # Second try: Use provided credentials file
            if credentials_path and os.path.isfile(credentials_path):
                logger.info(f"Using provided credentials file: {credentials_path}")
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path)
                return storage.Client(credentials=credentials)

            # Third try: Use application default credentials
            logger.info("Attempting to use application default credentials")
            credentials, project = google.auth.default()
            return storage.Client(credentials=credentials, project=project)

        except Exception as e:
            logger.error(f"Error initializing storage client: {e}")
            raise

    @retry.Retry()
    def save_to_gcs(self, source_path: str, destination_blob_name: str) -> None:
        """
        Upload a file to Google Cloud Storage with retry logic
        
        Args:
            source_path (str): Local path to the file to upload
            destination_blob_name (str): Destination path in GCS
        """
        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_path)
            logger.info(f"File {source_path} uploaded to {destination_blob_name}")
        except Exception as e:
            logger.error(f"Error uploading to GCS: {e}")
            raise

    @retry.Retry()
    def download_from_gcs(self, blob_name: str, destination_path: str) -> None:
        """
        Download a file from Google Cloud Storage with retry logic
        
        Args:
            blob_name (str): Path to the file in GCS
            destination_path (str): Local path where to save the file
        """
        try:
            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(destination_path)
            logger.info(f"Downloaded {blob_name} to {destination_path}")
        except Exception as e:
            logger.error(f"Error downloading from GCS: {e}")
            raise

    def load_and_process_data(self, sample_size: Optional[int] = None) -> None:
        """
        Load arXiv dataset, process it and upload to GCS
        
        Args:
            sample_size (int, optional): Number of papers to sample for processing
        """
        try:
            logger.info("Loading arXiv dataset")
            dataset = load_dataset("CCRss/arXiv_dataset")

            papers = dataset['train'].to_pandas()
            if sample_size:
                papers = papers.sample(n=sample_size, random_state=42)

            # Clean and prepare data
            papers['authors'] = papers['authors'].apply(self._process_authors)
            self.papers_data = papers[['title', 'abstract', 'authors', 'categories']].copy()

            # Create embeddings
            logger.info("Creating embeddings...")
            abstracts = self.papers_data['abstract'].tolist()
            embeddings = self.embedder.encode(abstracts, show_progress_bar=True)

            logger.info("Creating FAISS index...")

            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings).astype('float32'))

            logger.info("Data processed successfully")

            # Save files using temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                papers_path = Path(temp_dir) / 'papers_data.pkl'
                embeddings_path = Path(temp_dir) / 'embeddings.pkl'
                index_path = Path(temp_dir) / 'faiss.index'

                # Save files locally
                self.papers_data.to_pickle(str(papers_path))
                # with open(embeddings_path, 'wb') as f:
                #     pickle.dump(embeddings, f)
                faiss.write_index(self.index, str(index_path))

                # Upload to GCS
                logger.info("Uploading data to GCS...")
                self.save_to_gcs(str(papers_path), 'papers_data.pkl')
                logger.info("Uploading index...")
                self.save_to_gcs(str(index_path), 'faiss.index')

            logger.info("Data processed and uploaded to GCS successfully")

        except Exception as e:
            logger.error(f"Error processing and uploading data: {e}")
            raise

    def load_processed_data(self) -> None:
        """Load processed data from GCS"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                papers_path = Path(temp_dir) / 'papers_data.pkl'
                index_path = Path(temp_dir) / 'faiss.index'
                
                self.download_from_gcs('papers_data.pkl', str(papers_path))
                self.download_from_gcs('faiss.index', str(index_path))

                # Load the data
                self.papers_data = pd.read_pickle(papers_path)
                self.index = faiss.read_index(str(index_path))
                
                logger.info("Data loaded successfully from GCS")
                
        except Exception as e:
            logger.error(f"Error loading processed data from GCS: {e}")
            raise

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for relevant papers given a query
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of relevant papers with their details and scores
        """
        try:
            if not self.index or self.papers_data is None:
                raise ValueError("Data not loaded. Call load_processed_data() first.")

            # Create query embeddings
            query_vector = self.embedder.encode([query])

            # Search in FAISS index
            distances, indices = self.index.search(
                np.array(query_vector).astype('float32'), k
            )

            # Get paper details
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                paper = self.papers_data.iloc[idx]
                results.append({
                    'title': paper['title'],
                    'abstract': paper['abstract'],
                    'authors': paper['authors'],
                    'categories': paper['categories'],
                    'score': float(1/(1+dist))  # Convert distance to similarity score
                })

            return results
        
        except Exception as e:
            logger.error(f"Error searching for papers: {e}")
            raise

    @staticmethod
    def _process_authors(authors_str: str) -> List[str]:
        """
        Process authors string into list of names
        
        Args:
            authors_str (str): String containing author names
            
        Returns:
            List[str]: List of processed author names
        """
        try:
            if not isinstance(authors_str, str):
                return []
                
            authors = authors_str.strip('[]').split(',')
            authors = [author.strip().strip("'\"") for author in authors]
            authors = [author for author in authors if author]
            return authors
        except Exception:
            return []

    def clear_cache(self) -> None:
        """Clear the local cache directory"""
        try:
            if self.cache_dir.exists():
                for file in self.cache_dir.glob('*'):
                    file.unlink()
                self.cache_dir.rmdir()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise