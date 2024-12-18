from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import os
import pandas as pd
from google.cloud import storage
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, bucket_name: str = None, cache_dir: str = "data/processed"):
        self.bucket_name = bucket_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GCS client
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

        # Initialize SBERT model for embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.papers_data = None

        # self.save_to_gcs((Path(__file__).resolve().parent / "../../data/processed/papers_data.pkl").resolve(), 'papers_data.pkl')
        # self.save_to_gcs((Path(__file__).resolve().parent / "../../data/processed/embeddings.pkl").resolve(), 'embeddings.pkl')
        # self.save_to_gcs((Path(__file__).resolve().parent / "../../data/processed/faiss.index").resolve(), 'faiss.index')

    def save_to_gcs(self, source_path: str, destination_blob_name: str) -> None:
        """Upload a file to Google Cloud Storage"""
        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_path)
            logger.info(f"File {source_path} uploaded to {destination_blob_name}")
        except Exception as e:
            logger.error(f"Error uploading to GCS: {e}")
            raise

    def download_from_gcs(self, blob_name: str, destination_path: str) -> None:
        """Download a file from Google Cloud Storage"""
        try:
            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(destination_path)
            logger.info(f"Downloaded {blob_name} to {destination_path}")
        except Exception as e:
            logger.error(f"Error downloading from GCS: {e}")
            raise

    def load_and_process_data(self, sample_size: int = None) -> None:
        """Load arXiv dataset, process it and upload to GCS"""
        try:
            logger.info("Loading arXiv dataset")
            dataset = load_dataset("CCRss/arXiv_dataset")

            papers = dataset['train'].to_pandas()
            if sample_size:
                papers = papers.sample(n=sample_size, random_state=42)

            papers['authors'] = papers['authors'].apply(self._process_authors)

            # Clean and prepare data
            self.papers_data = papers[['title', 'abstract', 'authors', 'categories']].copy()

            # Create embeddings
            logger.info("Creating embeddings....")
            abstracts = self.papers_data['abstract'].tolist()
            embeddings = self.embedder.encode(abstracts, show_progress_bar=True)

            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings).astype('float32'))

            # Save files locally first
            temp_dir = tempfile.mkdtemp()
            
            papers_path = Path(temp_dir) / 'papers_data.pkl'
            embeddings_path = Path(temp_dir) / 'embeddings.pkl'
            index_path = Path(temp_dir) / 'faiss.index'

            # Save files locally
            self.papers_data.to_pickle(str(papers_path))
            with open(embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f)
            faiss.write_index(self.index, str(index_path))

            # Upload to GCS
            self.save_to_gcs(str(papers_path), 'papers_data.pkl')
            self.save_to_gcs(str(embeddings_path), 'embeddings.pkl')
            self.save_to_gcs(str(index_path), 'faiss.index')

            logger.info("Data processed and uploaded to GCS successfully")

        except Exception as e:
            logger.error(f"Error processing and uploading data: {e}")
            raise

    def load_processed_data(self) -> None:
        """Load processed data from GCS"""
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Download files from GCS
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
        """Search for relevant papers given a query"""
        try:
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
                    'score': float(1/(1+dist))
                })

            return results
        
        except Exception as e:
            logger.error(f"Error searching for papers: {e}")
            raise

    def _process_authors(self, authors_str: str) -> List[str]:
        """Process authors string into list of names"""
        try:
            authors = authors_str.strip('[]').split(',')
            authors = [author.strip().strip("'\"") for author in authors]
            authors = [author for author in authors if author]
            return authors
        except:
            return []