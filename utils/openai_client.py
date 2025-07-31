"""
Azure OpenAI Client Module for MCP AI Agent
===========================================

This module provides a central connector to Azure OpenAI services for:
- Text embeddings generation
- Document summarization
- RAG-based question answering
- Content title generation

Dependencies:
- openai (Azure OpenAI SDK)
- tiktoken (for token counting)
- tenacity (for retry logic)
"""

import os
import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    from openai import AzureOpenAI
    import tiktoken
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError as e:
    raise ImportError(f"Required dependencies missing: {e}. Please install: pip install openai tiktoken tenacity")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants - Model configurations
EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-large")
COMPLETION_MODEL = os.getenv("AZURE_COMPLETION_MODEL", "gpt-4o")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))  # text-embedding-3-large default
MAX_EMBEDDING_TOKENS = 8192
MAX_COMPLETION_TOKENS = 128000  # GPT-4o context limit
MAX_OUTPUT_TOKENS = 4096

# Model-specific embedding dimensions mapping
EMBEDDING_DIMENSIONS_MAP = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}

def get_embedding_dimensions(model_name: str) -> int:
    """Get embedding dimensions for a specific model"""
    return EMBEDDING_DIMENSIONS_MAP.get(model_name, EMBEDDING_DIMENSIONS)

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = "https://weez-openai-resource.openai.azure.com/"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
    raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables are required")

@dataclass
class SummaryConfig:
    """Configuration for different summary types"""
    short: Dict[str, Any] = None
    general: Dict[str, Any] = None
    detailed: Dict[str, Any] = None
    
    def __post_init__(self):
        self.short = {
            "max_tokens": 500,
            "temperature": 0.3,
            "instruction": "Provide a concise summary in 3-5 bullet points highlighting key information."
        }
        self.general = {
            "max_tokens": 1500,
            "temperature": 0.4,
            "instruction": "Provide a comprehensive summary covering main topics, key findings, and important details."
        }
        self.detailed = {
            "max_tokens": 3000,
            "temperature": 0.2,
            "instruction": "Provide an in-depth analysis with detailed explanations, context, and comprehensive coverage of all significant points."
        }

class AzureOpenAIClient:
    """
    Central client for Azure OpenAI services in MCP AI Agent
    """
    
    def __init__(self):
        """Initialize Azure OpenAI client with configuration"""
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.summary_config = SummaryConfig()
        
        # Get dynamic embedding dimensions
        self.embedding_dimensions = get_embedding_dimensions(EMBEDDING_MODEL)
        
        logger.info(f"Azure OpenAI Client initialized successfully with {EMBEDDING_MODEL} ({self.embedding_dimensions} dimensions)")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using character approximation")
            logger.warning("Using approximate token count due to tokenizer error")
            return len(text) // 4  # Rough approximation
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        if self.count_tokens(text) <= max_tokens:
            return text
        
        # Binary search for optimal truncation
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for input text using Azure OpenAI
        
        Args:
            text (str): Input text to embed
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            ValueError: If text is empty or too long
            Exception: If API call fails after retries
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        # Normalize and truncate text
        normalized_text = text.strip()
        if self.count_tokens(normalized_text) > MAX_EMBEDDING_TOKENS:
            normalized_text = self.truncate_text(normalized_text, MAX_EMBEDDING_TOKENS)
            logger.warning(f"Text truncated to {MAX_EMBEDDING_TOKENS} tokens for embedding")
        
        try:
            logger.debug(f"Generating embedding for text of length: {len(normalized_text)}")
            
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=normalized_text,
                dimensions=self.embedding_dimensions
            )
            
            if not response.data:
                raise ValueError("No embedding data received from API")
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def _prepare_chunks_for_completion(self, chunks: List[str], max_tokens: int) -> str:
        """
        Prepare and truncate chunks to fit within token limit using proportional allocation
        
        Args:
            chunks (List[str]): List of text chunks
            max_tokens (int): Maximum tokens allowed
            
        Returns:
            str: Combined and truncated chunks
        """
        if not chunks:
            return ""
        
        # First, get token counts for all chunks
        chunk_tokens = [self.count_tokens(chunk) for chunk in chunks]
        total_tokens = sum(chunk_tokens)
        
        # If total fits within limit, return as-is
        if total_tokens <= max_tokens:
            return "\n\n".join(f"Document {i+1}:\n{chunk}" for i, chunk in enumerate(chunks))
        
        # Calculate proportional allocation
        truncated_chunks = []
        for i, (chunk, chunk_token_count) in enumerate(zip(chunks, chunk_tokens)):
            # Allocate tokens proportionally based on original chunk size
            allocated_tokens = int((chunk_token_count / total_tokens) * max_tokens)
            
            # Ensure minimum allocation and prevent zero-token chunks
            allocated_tokens = max(allocated_tokens, min(50, chunk_token_count))
            
            if chunk_token_count > allocated_tokens:
                truncated_chunk = self.truncate_text(chunk, allocated_tokens)
                truncated_chunks.append(truncated_chunk)
                logger.warning(f"Chunk {i+1} truncated from {chunk_token_count} to {allocated_tokens} tokens")
            else:
                truncated_chunks.append(chunk)
        
        return "\n\n".join(f"Document {i+1}:\n{chunk}" for i, chunk in enumerate(truncated_chunks))
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def summarize_chunks(self, chunks: List[str], summary_type: str = "general",  query: Optional[str] = None) -> str:
        """
        Summarize multiple text chunks using GPT-4o
        
        Args:
            chunks (List[str]): List of text chunks to summarize
            summary_type (str): Type of summary ('short', 'general', 'detailed')
            
        Returns:
            str: Generated summary
            
        Raises:
            ValueError: If chunks is empty or summary_type is invalid
            Exception: If API call fails after retries
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")
        
        if summary_type not in ['short', 'general', 'detailed']:
            raise ValueError(f"Invalid summary_type: {summary_type}. Must be 'short', 'general', or 'detailed'")
        
        config = getattr(self.summary_config, summary_type)
        
        # Prepare content within token limits
        max_content_tokens = MAX_COMPLETION_TOKENS - 1000  # Reserve tokens for prompt and response
        content = self._prepare_chunks_for_completion(chunks, max_content_tokens)

        system_prompt = f"""You are a document summarization AI. Your task is to analyze the provided documents and create a {summary_type} summary according to the user query.

{config['instruction']}

Guidelines:
- Focus on key information, main topics, and important findings
- Maintain accuracy and avoid hallucination
- Use clear, professional language
- If information is insufficient, clearly state limitations
- Organize information logically"""

        user_prompt = f"""Please summarize the following documents:

{content}

Provide a {summary_type} summary following the guidelines above."""

        try:
            logger.debug(f"Generating {summary_type} summary for {len(chunks)} chunks")
            
            response = self.client.chat.completions.create(
                model=COMPLETION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=config['max_tokens'],
                temperature=config['temperature'],
                top_p=0.95
            )
            
            if not response.choices:
                raise ValueError("No response choices received from model")
            
            summary = response.choices[0].message.content
            if not summary:
                raise ValueError("Empty response content from model")
            
            summary = summary.strip()
            logger.debug(f"Generated {summary_type} summary of length: {len(summary)}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def answer_query_rag(self, chunks: List[str], query: str, stream: bool = False) -> str:
        """
        Answer query using Retrieval-Augmented Generation (RAG)
        
        Args:
            chunks (List[str]): Retrieved document chunks as context
            query (str): User query to answer
            stream (bool): Whether to stream the response
            
        Returns:
            str: Generated answer based on provided context
            
        Raises:
            ValueError: If query is empty or chunks is empty
            Exception: If API call fails after retries
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not chunks:
            raise ValueError("Chunks list cannot be empty")
        
        # Prepare context within token limits
        max_context_tokens = MAX_COMPLETION_TOKENS - 2000  # Reserve tokens for prompt and response
        context = self._prepare_chunks_for_completion(chunks, max_context_tokens)
        
        system_prompt = """You are an AI assistant specialized in answering questions based on provided document context. Your role is to provide accurate, helpful answers using only the information available in the given documents.

Guidelines:
- Answer ONLY based on the provided context
- If the context doesn't contain sufficient information, clearly state this limitation
- Quote relevant parts of the documents when appropriate
- Provide reasoning for your answers
- If uncertain, express the degree of uncertainty
- Do not hallucinate or add information not present in the context
- Structure your response clearly with proper explanations"""

        user_prompt = f"""Context Documents:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to fully answer the question, please indicate what information is missing or uncertain."""

        try:
            logger.debug(f"Generating RAG answer for query: {query[:100]}...")
            
            response = self.client.chat.completions.create(
                model=COMPLETION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.3,
                top_p=0.95,
                stream=stream
            )
            
            if stream:
                # Return generator for streaming
                return self._process_streaming_response(response)
            else:
                if not response.choices:
                    raise ValueError("No response choices received from model")
                
                answer = response.choices[0].message.content
                if not answer:
                    raise ValueError("Empty response content from model")
                
                answer = answer.strip()
                logger.debug(f"Generated RAG answer of length: {len(answer)}")
                
                return answer
            
        except Exception as e:
            logger.error(f"RAG answer generation failed: {e}")
            raise
    
    def _process_streaming_response(self, response):
        """Process streaming response from OpenAI"""
        full_response = ""
        try:
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
        except Exception as e:
            logger.error(f"Streaming response processing failed: {e}")
            raise
        
        logger.debug(f"Completed streaming response of length: {len(full_response)}")
    
    def summarize_file_chunks(self, file_id: str, user_id: str, summary_type: str = "general") -> str:
        """
        Summarize chunks for a specific file by retrieving from cosmos_client
        
        Args:
            file_id (str): ID of the file to summarize
            user_id (str): ID of the user
            summary_type (str): Type of summary ('short', 'general', 'detailed')
            
        Returns:
            str: Generated summary
            
        Raises:
            ImportError: If cosmos_client is not available
            ValueError: If no chunks found for the file
        """
        try:
            # Import cosmos_client dynamically to avoid circular imports
            from cosmos_client import get_file_chunks
            
            # Retrieve chunks for the file
            chunks = get_file_chunks(file_id, user_id)
            
            if not chunks:
                raise ValueError(f"No chunks found for file_id: {file_id}")
            
            # Extract text content from chunk documents
            chunk_texts = []
            for chunk in chunks:
                if isinstance(chunk, dict) and 'content' in chunk:
                    chunk_texts.append(chunk['content'])
                elif isinstance(chunk, str):
                    chunk_texts.append(chunk)
                else:
                    logger.warning(f"Unexpected chunk format: {type(chunk)}")
            
            if not chunk_texts:
                raise ValueError(f"No valid chunk content found for file_id: {file_id}")
            
            logger.info(f"Summarizing {len(chunk_texts)} chunks for file {file_id}")
            
            # Generate summary
            return self.summarize_chunks(chunk_texts, summary_type)
            
        except ImportError:
            raise ImportError("cosmos_client module not available. Please ensure it's implemented.")
        except Exception as e:
            logger.error(f"File summarization failed for file {file_id}: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def generate_title(self, text: str) -> str:
        """
        Generate a smart title for content using GPT-4o
        
        Args:
            text (str): Content to generate title for
            
        Returns:
            str: Generated title
            
        Raises:
            ValueError: If text is empty
            Exception: If API call fails after retries
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Truncate text if too long
        max_content_tokens = MAX_COMPLETION_TOKENS - 500  # Reserve tokens for prompt and response
        content = self.truncate_text(text.strip(), max_content_tokens)
        
        system_prompt = """You are a title generation AI. Create concise, descriptive titles that capture the main topic and essence of the provided content.

Guidelines:
- Keep titles between 3-15 words
- Make titles informative and specific
- Avoid generic or vague terms
- Focus on the main subject matter
- Use professional, clear language"""

        user_prompt = f"""Generate a descriptive title for the following content:

{content}

Title:"""

        try:
            logger.debug(f"Generating title for content of length: {len(content)}")
            
            response = self.client.chat.completions.create(
                model=COMPLETION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=100,
                temperature=0.4,
                top_p=0.95
            )
            
            if not response.choices:
                raise ValueError("No response choices received from model")
            
            title = response.choices[0].message.content
            if not title:
                raise ValueError("Empty response content from model")
            
            # Clean up title (remove quotes, extra formatting)
            title = title.strip().strip('"\'').strip()
            
            logger.debug(f"Generated title: {title}")
            
            return title
            
        except Exception as e:
            logger.error(f"Title generation failed: {e}")
            raise

# Global client instance
_client_instance = None

def get_client() -> AzureOpenAIClient:
    """Get singleton Azure OpenAI client instance"""
    global _client_instance
    if _client_instance is None:
        _client_instance = AzureOpenAIClient()
    return _client_instance

# Convenience functions for direct usage
def get_embedding(text: str) -> List[float]:
    """Generate embedding for text"""
    try:
        return get_client().get_embedding(text)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise

def summarize_chunks(chunks: List[str], summary_type: str = "general", query: Optional[str] = None) -> str:
    """Summarize text chunks"""
    try:
        return get_client().summarize_chunks(chunks, summary_type, query)
    except Exception as e:
        logger.error(f"Chunk summarization failed: {e}")
        raise

def answer_query_rag(chunks: List[str], query: str, stream: bool = True) -> str:
    """Answer query using RAG"""
    try:
        return get_client().answer_query_rag(chunks, query, stream)
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise

def generate_title(text: str) -> str:
    """Generate title for content"""
    try:
        return get_client().generate_title(text)
    except Exception as e:
        logger.error(f"Title generation failed: {e}")
        raise

def summarize_file_chunks(file_id: str, user_id: str, summary_type: str = "general") -> str:
    """Summarize chunks for a specific file"""
    try:
        return get_client().summarize_file_chunks(file_id, user_id, summary_type)
    except Exception as e:
        logger.error(f"File summarization failed: {e}")
        raise

# Example usage and testing
if __name__ == "__main__":
    # Basic functionality test
    try:
        client = get_client()
        
        # Test embedding
        sample_text = "This is a test document for embedding generation."
        embedding = client.get_embedding(sample_text)
        print(f"Embedding generated with {len(embedding)} dimensions")
        
        # Test summarization
        sample_chunks = [
            "This document discusses artificial intelligence and machine learning.",
            "The key findings show that AI can improve business processes significantly.",
            "Machine learning algorithms require large datasets for training."
        ]
        summary = client.summarize_chunks(sample_chunks, "short")
        print(f"Summary generated: {summary}")
        
        # Test RAG
        query = "What are the key findings about AI?"
        answer = client.answer_query_rag(sample_chunks, query)
        print(f"RAG answer: {answer}")
        
        # Test title generation
        title = client.generate_title(" ".join(sample_chunks))
        print(f"Generated title: {title}")
        
    except Exception as e:
        print(f"Test failed: {e}")
