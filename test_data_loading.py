from src.data.loader import DataLoader
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize DataLoader
        loader = DataLoader()
        loader.load_and_process_data(sample_size=1000) # testing sample size of 1000

        # test search
        test_query = 'transformer architecture in deep learning'
        results = loader.search(test_query, k=1)

        # print results
        print(f'\nTest Query: {test_query}')
        print('\nRelevant Papers:')
        for idx, paper in enumerate(results, 1):
            print(f'\n{idx}. Title: {paper["title"]}')
            print(f'\nAuthors: {paper["authors"]}')
            print(f'\nAbstract: {paper["abstract"]}')
            print(f'\nCategories: {paper["categories"]}')
            print(f'\nSimilarity Score: {paper["score"]}')

    except Exception as e:
        logger.error(f"Error loading and processing arXiv dataset: {e}")
        raise e
    
if __name__ == '__main__':
    main()