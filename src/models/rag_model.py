from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoForCausalLM , AutoModelForSeq2SeqLM , T5ForConditionalGeneration
import torch
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import logging
from src.data.loader import DataLoader
from datetime import datetime
from transformers import BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGModel:
    def __init__(self,
                 model_name: str = "google/flan-t5-base",
                 max_length: int = 2048,
                 quantization: Optional[str] = None,
                 gcs_bucket: str = None,
                 credentials_path: str = "/app/research-paper-rag-0a8819b735b9.json"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Configure quantization
        if self.device.type == "cuda" and quantization:
            quantization_config = self._get_quantization_config(quantization)
        else:
            quantization_config = None
            
        # Initialize model
        try:
            self.model = self._initialize_model(model_name, max_length, quantization_config)
        except Exception as e:
            logger.error(f"Error initializing model with quantization: {e}")
            logger.info("Falling back to default initialization")
            self.model = self._initialize_model(model_name, max_length, None)

        # Configure model parameters
        self._configure_model_parameters(max_length)
        
        # Initialize data loader
        self.loader = DataLoader(bucket_name=gcs_bucket,
                                 credentials_path=credentials_path)
        try:
            self.loader.load_processed_data()
        except Exception as e:
            logger.warning(f"Could not load processed data: {e}")
            logger.info("Processing new data...")
            self.loader.load_and_process_data(sample_size=1000)

    def _get_quantization_config(self, quantization: str) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration based on specified type"""
        if quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=True
            )
        elif quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        return None

    def _initialize_model(self, model_name: str, max_length: int, quantization_config: Optional[BitsAndBytesConfig]):
        """Initialize model with proper configuration"""
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize model
        model_class = T5ForConditionalGeneration if "t5" in model_name.lower() else AutoModelForSeq2SeqLM
        return model_class.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.device.type == "cuda" else None,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        )

    def _configure_model_parameters(self, max_length: int):
        """Configure model parameters"""
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.config.max_length = max_length
        self.max_length = max_length

    @retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_response(self,
                         query: str,
                         max_new_tokens: int = 150,
                         temperature: float = 0.7,
                         num_papers: int = 1,
                         style: str = 'academic') -> Dict:
        """Generate response for the query using RAG with retry mechanism"""
        try:
            # Input validation
            if not query or len(query.strip()) == 0:
                raise ValueError("Query cannot be empty")

            # Retrieve papers
            retrieved_papers = self.loader.search(query, k=num_papers)
            if not retrieved_papers:
                raise ValueError("No relevant papers found")

            # Generate prompt and get model response
            prompt = self.generate_prompt(query, retrieved_papers)
            response = self._generate_model_response(prompt, max_new_tokens, temperature)

            # Format and return result
            return self._format_result(query, response, retrieved_papers, style, max_new_tokens, num_papers)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response(query, str(e))

    @torch.no_grad()
    def _generate_model_response(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Generate model response with proper error handling"""
        try:
            inputs = self.tokenizer(prompt,
                                  return_tensors="pt",
                                  max_length=self.max_length,
                                  truncation=True,
                                  padding=True).to(self.device)

            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                no_repeat_ngram_size=3,
                num_beams=2,
                early_stopping=True
            )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            logger.error(f"Error in model generation: {e}")
            raise

    def load_data(self):
        """Load processed data with retry mechanism"""
        try:
            self.loader.load_processed_data()
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _format_result(self, query: str, response: str, papers: List[Dict], 
                      style: str, max_tokens: int, num_papers: int) -> Dict:
        """Format the final response"""
        references = [{
            "title": paper["title"],
            "authors": paper["authors"],
            "categories": paper["categories"],
            "relevance_score": paper["score"],
            "citation": f"{', '.join(paper['authors'])}. \"{paper['title']}\". {paper['categories']}."
        } for paper in papers]

        return {
            "query": query,
            "response": response,
            "references": references,
            "metadata": {
                "model": self.model.config.name_or_path,
                "style": style,
                "max_tokens": max_tokens,
                "num_papers": num_papers,
            },
            "timestamp": datetime.now()
        }

    def _generate_fallback_response(self, query: str, error_msg: str) -> Dict:
        """Generate fallback response when errors occur"""
        return {
            "query": query,
            "response": "I apologize, but I'm unable to generate a proper response at the moment. "
                       "Please try again in a few moments.",
            "references": [],
            "metadata": {
                "status": "error",
                "error_type": error_msg
            },
            "timestamp": datetime.now()
        }

    def generate_prompt(self, query: str, retrieved_papers: List[Dict]) -> str:
        """Generate prompt for the model"""
        # Your existing generate_prompt implementation remains the same
        prompt = (
            "You are a research assistant tasked with explaining recent advances in machine learning "
            "based on academic papers. Provide a clear, technical, yet accessible explanation.\n\n"
        )

        prompt += f"Question: {query}\n\n"
        prompt += "I'll help you understand this topic based on the following research papers:\n\n"
        
        for i, paper in enumerate(retrieved_papers, 1):
            prompt += f"Paper {i}:\n"
            prompt += f"Title: {paper['title']}\n"
            prompt += f"Authors: {', '.join(paper['authors'])}\n"
            prompt += f"Research Area: {paper['categories']}\n"
            prompt += f"Key Findings: {paper['abstract']}\n\n"
        
        prompt += (
            "Based on the research papers provided, here's a comprehensive analysis of the "
            "recent advances in this area:\n\n"
        )

        return prompt