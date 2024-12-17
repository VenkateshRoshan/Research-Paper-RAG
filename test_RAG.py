from src.models.rag_model import RAGModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize RAG model
        rag_model = RAGModel()

        # Load processed data
        rag_model.load_data()

        # Test queries
        query = "What are the recent advances in transformer architectures?"

        # Generate responses
        logger.info(f"Query: {query}")
        result = rag_model.generate_response(query, max_new_tokens=500, style='all', num_papers=5)
        formatted_output = rag_model.format_output(result)

        # print results
        print(f'\nQuery: {formatted_output["query"]}')
        print(f'\nResponse: {formatted_output["response"]}')
        print(f'\nReferences:')
        for ref in formatted_output['references']:
                print(f"- {ref['title']} (Score: {ref['relevance_score']:.2f})")
                print(f"  Authors: {', '.join(ref['authors'])}")

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise e

if __name__ == '__main__':
    main()