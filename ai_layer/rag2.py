"""
Production-grade Retrieval-Augmented Generation (RAG) system module.

This module implements a complete RAG pipeline for answering questions about user files
stored in chunked form in Azure Cosmos DB with advanced features like token management,
conversation context, caching, confidence scoring, streaming support, and file metadata.

Integrates with the search.py module for document retrieval.
"""

import logging
import hashlib
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

# Import from search module
from .search2 import SearchEngine, SearchError, create_search_engine
from utils.openai_client import get_embedding, answer_query_rag

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_TOP_K = 10
MAX_TOKEN_LIMIT = 8000  # Conservative limit for most models
CACHE_EXPIRY_HOURS = 24
CONFIDENCE_THRESHOLD = 0.7
SIMILARITY_WEIGHT = 0.6
CHUNK_DENSITY_WEIGHT = 0.4


@dataclass
class ConversationContext:
    """Conversation context for multi-turn RAG."""
    user_id: str
    conversation_id: str
    previous_queries: List[str]
    previous_answers: List[str]
    context_chunks: List[dict]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class CacheEntry:
    """Cache entry for query-answer pairs."""
    query_hash: str
    answer: str
    chunks_used: int
    source_files: List[Dict[str, Any]]  # Enhanced to include file metadata
    confidence_score: float
    timestamp: datetime
    user_id: str
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.utcnow() - self.timestamp > timedelta(hours=CACHE_EXPIRY_HOURS)


@dataclass
class FileMetadata:
    """File metadata structure for source files."""
    file_id: str
    fileName: str
    sas_url: Optional[str]
    platform: str
    mime_type: Optional[str] = None
    department: Optional[List[str]] = None
    visibility: Optional[str] = None
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class RagResult:
    """Enhanced RAG result with confidence, streaming support, and file metadata."""
    answer: str
    query: str
    chunks_used: int
    source_files: List[Dict[str, Any]]  # Enhanced to include complete file metadata
    confidence_score: float
    context_used: bool
    cached: bool
    processing_time: float
    token_count: int


class RagError(Exception):
    """Custom exception for RAG-related errors."""
    
    def __init__(self, message: str, error_code: str = "RAG_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class TokenManager:
    """Manages token limits and chunk truncation."""
    
    def __init__(self, max_tokens: int = MAX_TOKEN_LIMIT):
        self.max_tokens = max_tokens
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def truncate_chunks_by_tokens(self, chunks: List[str], similarities: List[float]) -> Tuple[List[str], int]:
        """
        Truncate chunks based on token limits, prioritizing by similarity.
        
        Args:
            chunks: List of text chunks
            similarities: Corresponding similarity scores
            
        Returns:
            Tuple of (truncated_chunks, total_tokens)
        """
        logger.debug(f"Truncating {len(chunks)} chunks with token limit {self.max_tokens}")
        
        # Sort chunks by similarity (descending)
        chunk_similarity_pairs = list(zip(chunks, similarities))
        chunk_similarity_pairs.sort(key=lambda x: x[1], reverse=True)
        
        selected_chunks = []
        total_tokens = 0
        
        for chunk, similarity in chunk_similarity_pairs:
            chunk_tokens = self.estimate_tokens(chunk)
            
            if total_tokens + chunk_tokens <= self.max_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
                logger.debug(f"Added chunk (similarity: {similarity:.3f}, tokens: {chunk_tokens})")
            else:
                logger.debug(f"Skipped chunk (would exceed token limit)")
        
        logger.info(f"Selected {len(selected_chunks)} chunks totaling {total_tokens} tokens")
        return selected_chunks, total_tokens


class ConversationManager:
    """Manages conversation context for multi-turn RAG."""
    
    def __init__(self):
        self.contexts: Dict[str, ConversationContext] = {}
        self.max_context_length = 5  # Keep last 5 exchanges
    
    def get_context_key(self, user_id: str, conversation_id: str) -> str:
        """Generate context key."""
        return f"{user_id}:{conversation_id}"
    
    def get_context(self, user_id: str, conversation_id: str) -> Optional[ConversationContext]:
        """Retrieve conversation context."""
        key = self.get_context_key(user_id, conversation_id)
        context = self.contexts.get(key)
        
        if context and datetime.utcnow() - context.timestamp > timedelta(hours=1):
            # Context expired
            del self.contexts[key]
            return None
            
        return context
    
    def update_context(self, user_id: str, conversation_id: str, query: str, 
                      answer: str, chunks: List[dict]) -> None:
        """Update conversation context."""
        key = self.get_context_key(user_id, conversation_id)
        
        if key in self.contexts:
            context = self.contexts[key]
            context.previous_queries.append(query)
            context.previous_answers.append(answer)
            context.context_chunks.extend(chunks)
            context.timestamp = datetime.utcnow()
            
            # Keep only recent exchanges
            if len(context.previous_queries) > self.max_context_length:
                context.previous_queries = context.previous_queries[-self.max_context_length:]
                context.previous_answers = context.previous_answers[-self.max_context_length:]
        else:
            context = ConversationContext(
                user_id=user_id,
                conversation_id=conversation_id,
                previous_queries=[query],
                previous_answers=[answer],
                context_chunks=chunks,
                timestamp=datetime.utcnow()
            )
            self.contexts[key] = context
    
    def build_context_prompt(self, context: ConversationContext) -> str:
        """Build context prompt from conversation history."""
        if not context.previous_queries:
            return ""
        
        context_parts = ["Previous conversation context:"]
        for i, (q, a) in enumerate(zip(context.previous_queries[:-1], context.previous_answers[:-1])):
            context_parts.append(f"Q{i+1}: {q}")
            context_parts.append(f"A{i+1}: {a[:200]}...")  # Truncate previous answers
        
        return "\n".join(context_parts) + "\n\nCurrent question:"


class CacheManager:
    """Manages caching for query-answer pairs."""
    
    def __init__(self):
        self.cache: Dict[str, CacheEntry] = {}
    
    def generate_cache_key(self, query: str, user_id: str, filters: Optional[Dict] = None) -> str:
        """Generate cache key for query."""
        cache_data = {
            "query": query.lower().strip(),
            "user_id": user_id,
            "filters": filters or {}
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get_cached_answer(self, query: str, user_id: str, filters: Optional[Dict] = None) -> Optional[CacheEntry]:
        """Retrieve cached answer if available and not expired."""
        cache_key = self.generate_cache_key(query, user_id, filters)
        entry = self.cache.get(cache_key)
        
        if entry and not entry.is_expired():
            logger.info(f"Cache hit for query: {query[:50]}...")
            return entry
        elif entry:
            # Remove expired entry
            del self.cache[cache_key]
            
        return None
    
    def cache_answer(self, query: str, user_id: str, answer: str, chunks_used: int,
                    source_files: List[Dict[str, Any]], confidence_score: float, 
                    filters: Optional[Dict] = None) -> None:
        """Cache answer for future use."""
        cache_key = self.generate_cache_key(query, user_id, filters)
        
        entry = CacheEntry(
            query_hash=cache_key,
            answer=answer,
            chunks_used=chunks_used,
            source_files=source_files,
            confidence_score=confidence_score,
            timestamp=datetime.utcnow(),
            user_id=user_id
        )
        
        self.cache[cache_key] = entry
        logger.debug(f"Cached answer for query: {query[:50]}...")
    
    def clear_expired_cache(self) -> None:
        """Clear expired cache entries."""
        expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
        for key in expired_keys:
            del self.cache[key]
        logger.info(f"Cleared {len(expired_keys)} expired cache entries")


class ConfidenceCalculator:
    """Calculates confidence scores for RAG results."""
    
    def calculate_confidence(self, similarities: List[float], chunks: List[dict]) -> float:
        """
        Calculate confidence score based on similarity scores and chunk density.
        
        Args:
            similarities: List of cosine similarity scores
            chunks: List of retrieved chunks
            
        Returns:
            Confidence score between 0 and 1
        """
        if not similarities:
            return 0.0
        
        # Average similarity score
        avg_similarity = np.mean(similarities)
        
        # Chunk density score (how many high-quality chunks)
        high_quality_chunks = sum(1 for sim in similarities if sim > 0.7)
        chunk_density = high_quality_chunks / len(similarities)
        
        # Combined confidence score
        confidence = (SIMILARITY_WEIGHT * avg_similarity + 
                     CHUNK_DENSITY_WEIGHT * chunk_density)
        
        logger.debug(f"Confidence calculation: avg_sim={avg_similarity:.3f}, "
                    f"chunk_density={chunk_density:.3f}, final={confidence:.3f}")
        
        return min(confidence, 1.0)


# Global instances
token_manager = TokenManager()
conversation_manager = ConversationManager()
cache_manager = CacheManager()
confidence_calculator = ConfidenceCalculator()

# Create a global search engine instance
search_engine = create_search_engine()


def validate_intent(intent: Dict[str, Any]) -> None:
    """
    Validate the intent dictionary structure and required fields.
    
    Args:
        intent: Dictionary containing query intent and parameters
        
    Raises:
        RagError: If validation fails
    """
    logger.debug(f"Validating intent: {intent}")
    
    # Check if intent is a dictionary
    if not isinstance(intent, dict):
        raise RagError("Intent must be a dictionary", "INVALID_INTENT_TYPE")
    
    # Required fields
    required_fields = ["action", "query_text", "user_id"]
    for field in required_fields:
        if field not in intent:
            raise RagError(f"Missing required field: {field}", "MISSING_REQUIRED_FIELD")
        if not intent[field]:
            raise RagError(f"Required field '{field}' cannot be empty", "EMPTY_REQUIRED_FIELD")
    
    # Validate action
    if intent["action"] != "rag":
        raise RagError(f"Invalid action: {intent['action']}. Expected 'rag'", "INVALID_ACTION")
    
    # Validate query_text
    if not isinstance(intent["query_text"], str):
        raise RagError("query_text must be a string", "INVALID_QUERY_TEXT_TYPE")
    
    # Validate user_id
    if not isinstance(intent["user_id"], str):
        raise RagError("user_id must be a string", "INVALID_USER_ID_TYPE")
    
    # Validate optional fields if present
    if "top_k" in intent and intent["top_k"] is not None:
        if not isinstance(intent["top_k"], int) or intent["top_k"] <= 0:
            raise RagError("top_k must be a positive integer", "INVALID_TOP_K")
    
    if "file_id" in intent and intent["file_id"] is not None:
        if not isinstance(intent["file_id"], str):
            raise RagError("file_id must be a string", "INVALID_FILE_ID_TYPE")
    
    if "platform" in intent and intent["platform"] is not None:
        if not isinstance(intent["platform"], str):
            raise RagError("platform must be a string", "INVALID_PLATFORM_TYPE")
    
    logger.info("Intent validation successful")


def build_search_intent(intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build search intent dictionary compatible with search.py SearchEngine from RAG intent.
    
    Args:
        intent: RAG intent dictionary
        
    Returns:
        Search intent dictionary compatible with SearchEngine.search method
    """
    logger.debug("Building search intent from RAG intent")
    
    search_intent = {
        "query_text": intent["query_text"],
        "user_id": intent["user_id"]
    }
    
    # Add optional fields that match search.py expectations
    if intent.get("platform"):
        search_intent["platform"] = intent["platform"]
    
    if intent.get("file_type"):
        search_intent["file_type"] = intent["file_type"]
    
    if intent.get("time_range"):
        search_intent["time_range"] = intent["time_range"]
    
    # Handle offset and limit for pagination
    if intent.get("offset"):
        search_intent["offset"] = intent["offset"]
    
    # Set limit based on top_k parameter
    top_k = intent.get("top_k", DEFAULT_TOP_K)
    search_intent["limit"] = top_k
    
    logger.debug(f"Built search intent: {search_intent}")
    return search_intent


def extract_text_and_similarities(chunks: List[dict]) -> Tuple[List[str], List[float]]:
    """
    Extract text content and similarity scores from chunk documents returned by search.py.
    
    Args:
        chunks: List of enhanced chunk documents from SearchEngine.search method
        
    Returns:
        Tuple of (text_chunks, similarity_scores)
        
    Raises:
        RagError: If no valid text content is found
    """
    logger.debug(f"Extracting text and similarities from {len(chunks)} chunks")
    
    text_chunks = []
    similarities = []
    
    # Fields expected from search.py enhanced results
    text_fields = ["text", "content", "chunk_text", "body", "text_content"]
    
    for i, chunk in enumerate(chunks):
        # Extract text content
        chunk_text = None
        for field in text_fields:
            if field in chunk and chunk[field]:
                chunk_text = str(chunk[field]).strip()
                break
        
        # Extract similarity score from search.py enhanced results
        similarity = chunk.get('_similarity', 0.0)  # search.py adds _similarity field
        
        if chunk_text:
            text_chunks.append(chunk_text)
            similarities.append(similarity)
            logger.debug(f"Extracted chunk {i}: {len(chunk_text)} chars, similarity: {similarity:.3f}")
        else:
            logger.warning(f"No valid text content found in chunk {i}")
    
    if not text_chunks:
        raise RagError("No valid text content found in any chunks", "NO_TEXT_CONTENT")
    
    logger.info(f"Successfully extracted {len(text_chunks)} chunks with similarities")
    return text_chunks, similarities


def extract_file_metadata(chunks: List[dict]) -> List[Dict[str, Any]]:
    """
    Extract comprehensive file metadata from chunk documents with search.py enhanced structure.
    
    Args:
        chunks: List of enhanced chunk documents from search.py SearchEngine
        
    Returns:
        List of unique file metadata dictionaries with complete information
    """
    logger.debug(f"Extracting file metadata from {len(chunks)} chunks")
    
    # Use a dictionary to track unique files by file_id
    unique_files = {}
    
    for chunk in chunks:
        # Extract file access information from search.py enhanced results
        file_access = chunk.get('_file_access', {})
        
        # Extract file_id (could be in different fields)
        file_id = (chunk.get('file_id') or 
                  chunk.get('id') or 
                  chunk.get('fileId') or 
                  file_access.get('file_path'))
        
        if not file_id:
            logger.warning(f"No file_id found in chunk: {list(chunk.keys())}")
            continue
        
        # Skip if we already have this file
        if file_id in unique_files:
            continue
        
        # Extract file metadata using search.py enhanced structure
        file_metadata = {
            'file_id': file_id,
            'fileName': (
                chunk.get('fileName') or 
                chunk.get('filename') or 
                chunk.get('title') or 
                file_access.get('file_name') or
                'Unknown File'
            ),
            'platform': (
                chunk.get('platform') or 
                file_access.get('platform') or
                'unknown'
            ),
            'sas_url': (
                chunk.get('sas_url') or
                file_access.get('sas_url')
            ),
            'mime_type': (
                chunk.get('mime_type') or
                file_access.get('mime_type')
            ),
            'department': (
                chunk.get('department') or
                file_access.get('departments')
            ),
            'visibility': (
                chunk.get('visibility') or
                file_access.get('visibility')
            ),
            'created_at': chunk.get('created_at'),
            'user_id': chunk.get('user_id'),
            'similarity_score': chunk.get('_similarity', 0.0),
            'file_category': chunk.get('_file_category', 'other'),
            'can_download': file_access.get('can_download', False)
        }
        
        # Clean up None values except for sas_url which might legitimately be None
        cleaned_metadata = {}
        for key, value in file_metadata.items():
            if value is not None or key == 'sas_url':
                cleaned_metadata[key] = value
        
        unique_files[file_id] = cleaned_metadata
        
        logger.debug(f"Extracted metadata for file: {cleaned_metadata.get('fileName')} (ID: {file_id})")
    
    result = list(unique_files.values())
    logger.info(f"Found {len(result)} unique source files with metadata")
    
    return result


def extract_source_files(chunks: List[dict]) -> List[str]:
    """
    Extract and deduplicate source file names from chunk metadata.
    Kept for backward compatibility, but now extract_file_metadata is preferred.
    
    Args:
        chunks: List of enhanced chunk documents from search.py
        
    Returns:
        List of unique source file names
    """
    logger.debug(f"Extracting source files from {len(chunks)} chunks")
    
    source_files = set()
    filename_fields = ["fileName", "filename", "title", "source_file", "file_name", "document_name"]
    
    for chunk in chunks:
        # Also check _file_access from search.py enhanced results
        file_access = chunk.get('_file_access', {})
        
        for field in filename_fields:
            if field in chunk and chunk[field]:
                filename = str(chunk[field]).strip()
                if filename:
                    source_files.add(filename)
                    break
        else:
            # Try file_access if no direct filename found
            if file_access.get('file_name'):
                source_files.add(str(file_access['file_name']))
    
    result = list(source_files)
    logger.info(f"Found {len(result)} unique source files: {result}")
    return result


def search_similar_documents_by_file_id(file_id: str, user_id: str, top_k: int = 10) -> List[dict]:
    """
    Search for similar documents to a specific file using search.py SearchEngine.
    
    Args:
        file_id: ID of the reference file
        user_id: User identifier
        top_k: Number of similar documents to return
        
    Returns:
        List of similar document chunks
        
    Raises:
        RagError: If search fails
    """
    logger.debug(f"Searching for similar documents to file {file_id}")
    
    try:
        # Use SearchEngine's search_similar_documents method
        result = search_engine.search_similar_documents(
            document_id=file_id,
            user_id=user_id,
            limit=top_k
        )
        
        if result['status'] == 'success':
            return result['results']
        else:
            raise RagError(f"Similar document search failed: {result.get('error', 'Unknown error')}", "SIMILAR_SEARCH_ERROR")
            
    except SearchError as e:
        logger.error(f"SearchError in similar documents search: {e}")
        raise RagError(f"Failed to search similar documents: {str(e)}", "SIMILAR_SEARCH_ERROR")
    except Exception as e:
        logger.error(f"Unexpected error in similar documents search: {e}")
        raise RagError(f"Unexpected error searching similar documents: {str(e)}", "SIMILAR_SEARCH_ERROR")


def answer_query_with_rag(intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to answer a query using RAG with advanced features and file metadata.
    
    Args:
        intent: Dictionary containing:
            - action (str): Must be "rag"
            - query_text (str): The question to answer
            - user_id (str): User identifier
            - conversation_id (str, optional): For multi-turn conversations
            - file_id (str, optional): Filter by specific file
            - platform (str, optional): Filter by platform
            - file_type (str, optional): Filter by file type
            - time_range (str|dict, optional): Filter by time range
            - top_k (int, optional): Number of chunks to retrieve (default: 10)
            - use_cache (bool, optional): Enable caching (default: True)
            - stream (bool, optional): Enable streaming (default: False)
            - include_confidence (bool, optional): Include confidence score (default: True)
            - include_file_metadata (bool, optional): Include detailed file metadata (default: True)
    
    Returns:
        Dictionary containing:
            - answer (str): Generated answer
            - query (str): Original query text
            - chunks_used (int): Number of chunks used
            - source_files (list): List of file metadata dictionaries with id, fileName, sas_url, platform
            - confidence_score (float): Confidence score (0-1)
            - context_used (bool): Whether conversation context was used
            - cached (bool): Whether result was cached
            - processing_time (float): Processing time in seconds
            - token_count (int): Total tokens used
    
    Raises:
        RagError: If any step in the RAG process fails
    """
    start_time = time.time()
    logger.info(f"Starting enhanced RAG query for user: {intent.get('user_id')}")
    
    try:
        # Step 1: Validate intent
        validate_intent(intent)
        
        query_text = intent["query_text"]
        user_id = intent["user_id"]
        conversation_id = intent.get("conversation_id")
        top_k = intent.get("top_k", DEFAULT_TOP_K)
        use_cache = intent.get("use_cache", True)
        stream = intent.get("stream", False)
        include_confidence = intent.get("include_confidence", True)
        include_file_metadata = intent.get("include_file_metadata", True)
        
        logger.info(f"Processing query: '{query_text[:100]}...' for user: {user_id}")
        
        # Step 2: Check cache first
        search_intent = build_search_intent(intent)
        cached_result = None
        
        if use_cache:
            cached_result = cache_manager.get_cached_answer(query_text, user_id, search_intent)
            if cached_result:
                processing_time = time.time() - start_time
                return {
                    "answer": cached_result.answer,
                    "query": query_text,
                    "chunks_used": cached_result.chunks_used,
                    "source_files": cached_result.source_files,
                    "confidence_score": cached_result.confidence_score,
                    "context_used": False,
                    "cached": True,
                    "processing_time": processing_time,
                    "token_count": token_manager.estimate_tokens(cached_result.answer)
                }
        
        # Step 3: Get conversation context
        context = None
        context_used = False
        if conversation_id:
            context = conversation_manager.get_context(user_id, conversation_id)
            context_used = context is not None
        
        # Step 4: Enhance query with context if available
        enhanced_query = query_text
        if context:
            context_prompt = conversation_manager.build_context_prompt(context)
            enhanced_query = f"{context_prompt}\n{query_text}"
            # Update search intent with enhanced query
            search_intent["query_text"] = enhanced_query
        
        # Step 5: Retrieve relevant chunks using search.py SearchEngine
        logger.debug(f"Searching for similar documents with top_k={top_k}")
        try:
            if intent.get("file_id"):
                # Search within a specific file using similar documents
                chunks = search_similar_documents_by_file_id(
                    file_id=intent["file_id"],
                    user_id=user_id,
                    top_k=top_k
                )
            else:
                # General document search using SearchEngine
                search_result = search_engine.search(search_intent)
                
                if search_result['status'] == 'success':
                    chunks = search_result['results']
                    logger.info(f"Retrieved {len(chunks)} chunks from SearchEngine")
                else:
                    raise RagError("Search failed", "SEARCH_FAILED")
            
        except SearchError as e:
            logger.error(f"Search error: {e}")
            raise RagError(f"Failed to retrieve documents: {str(e)}", "SEARCH_ERROR")
        except Exception as e:
            logger.error(f"Unexpected search error: {e}")
            raise RagError(f"Unexpected error during search: {str(e)}", "SEARCH_ERROR")
        
        # Step 6: Check if chunks were found
        if not chunks:
            logger.warning("No relevant chunks found for the query")
            raise RagError("No relevant documents found for your query", "NO_CHUNKS_FOUND")
        
        # Step 7: Extract text and similarities
        text_chunks, similarities = extract_text_and_similarities(chunks)
        
        # Step 8: Apply token limits with smart ranking
        final_chunks, token_count = token_manager.truncate_chunks_by_tokens(text_chunks, similarities)
        final_similarities = similarities[:len(final_chunks)]
        final_chunk_docs = chunks[:len(final_chunks)]  # Keep corresponding chunk documents
        
        # Step 9: Calculate confidence score
        confidence_score = 0.0
        if include_confidence:
            confidence_score = confidence_calculator.calculate_confidence(final_similarities, final_chunk_docs)
        
        # Step 10: Extract comprehensive file metadata
        if include_file_metadata:
            source_files = extract_file_metadata(final_chunk_docs)
        else:
            # Fallback to simple file names for backward compatibility
            source_file_names = extract_source_files(final_chunk_docs)
            source_files = [{"fileName": name} for name in source_file_names]
        
        # Step 11: Generate RAG answer
        logger.debug("Generating RAG answer using OpenAI client")
        try:
            # Handle streaming vs non-streaming properly
            if stream:
                # For streaming, we need to handle differently
                answer_generator = answer_query_rag(
                    chunks=final_chunks,
                    query=query_text,
                    stream=True
                )
                # Collect all chunks into a single answer
                answer_parts = []
                for chunk in answer_generator:
                    if isinstance(chunk, str):
                        answer_parts.append(chunk)
                    elif isinstance(chunk, dict) and 'content' in chunk:
                        answer_parts.append(chunk['content'])
                answer = ''.join(answer_parts)
            else:
                # Non-streaming mode
                answer = answer_query_rag(
                    chunks=final_chunks,
                    query=query_text,
                    stream=False
                )
            
            # Ensure answer is a string
            if not isinstance(answer, str):
                raise RagError("Generated answer is not a string", "INVALID_ANSWER_TYPE")
            
            logger.info("RAG answer generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise RagError(f"Failed to generate answer: {str(e)}", "ANSWER_GENERATION_ERROR")
        
        # Step 12: Update conversation context
        if conversation_id:
            conversation_manager.update_context(user_id, conversation_id, query_text, answer, final_chunk_docs)
        
        # Step 13: Cache the result (only if confidence is high enough)
        if use_cache and confidence_score > CONFIDENCE_THRESHOLD:
            cache_manager.cache_answer(
                query_text, user_id, answer, len(final_chunks), 
                source_files, confidence_score, search_intent
            )

        # Step 14: Measure processing time
        processing_time = time.time() - start_time
        
        # Step 15: Build and return result
        result = RagResult(
            answer=answer,
            query=query_text,
            chunks_used=len(final_chunks),
            source_files=source_files,
            confidence_score=confidence_score,
            context_used=context_used,
            cached=False,
            processing_time=processing_time,
            token_count=token_count
        )
        
        logger.info(f"RAG query completed successfully in {processing_time:.2f}s")
        
        # Return as dictionary
        return {
            "answer": result.answer,
            "query": result.query,
            "chunks_used": result.chunks_used,
            "source_files": result.source_files,
            "confidence_score": result.confidence_score,
            "context_used": result.context_used,
            "cached": result.cached,
            "processing_time": result.processing_time,
            "token_count": result.token_count
        }
        
    except RagError as e:
        # Log and re-raise RAG-specific errors
        logger.error(f"RAG error: {e.message} (Code: {e.error_code})")
        processing_time = time.time() - start_time
        raise e
    
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in RAG pipeline: {str(e)}")
        processing_time = time.time() - start_time
        raise RagError(f"Unexpected error in RAG pipeline: {str(e)}", "UNEXPECTED_ERROR")


def stream_rag_answer(intent: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Stream RAG answer chunks for real-time response delivery.
    
    Args:
        intent: Same as answer_query_with_rag but with stream=True implied
        
    Yields:
        Dictionary chunks with:
            - type (str): "chunk", "metadata", or "complete"
            - content (str): Text content for chunks
            - metadata (dict): For metadata and completion info
    """
    start_time = time.time()
    logger.info(f"Starting streaming RAG query for user: {intent.get('user_id')}")
    
    try:
        # Force streaming mode
        intent["stream"] = True
        
        # Step 1: Validate intent
        validate_intent(intent)
        
        query_text = intent["query_text"]
        user_id = intent["user_id"]
        conversation_id = intent.get("conversation_id")
        top_k = intent.get("top_k", DEFAULT_TOP_K)
        use_cache = intent.get("use_cache", True)
        include_confidence = intent.get("include_confidence", True)
        include_file_metadata = intent.get("include_file_metadata", True)
        
        # Step 2: Check cache first
        search_intent = build_search_intent(intent)
        cached_result = None
        
        if use_cache:
            cached_result = cache_manager.get_cached_answer(query_text, user_id, search_intent)
            if cached_result:
                # Stream cached result as chunks
                words = cached_result.answer.split()
                for i in range(0, len(words), 5):  # Stream 5 words at a time
                    chunk = " ".join(words[i:i+5])
                    if i + 5 < len(words):
                        chunk += " "
                    
                    yield {
                        "type": "chunk",
                        "content": chunk
                    }
                
                # Send completion metadata
                processing_time = time.time() - start_time
                yield {
                    "type": "complete",
                    "metadata": {
                        "query": query_text,
                        "chunks_used": cached_result.chunks_used,
                        "source_files": cached_result.source_files,
                        "confidence_score": cached_result.confidence_score,
                        "context_used": False,
                        "cached": True,
                        "processing_time": processing_time,
                        "token_count": token_manager.estimate_tokens(cached_result.answer)
                    }
                }
                return
        
        # Step 3: Get conversation context
        context = None
        context_used = False
        if conversation_id:
            context = conversation_manager.get_context(user_id, conversation_id)
            context_used = context is not None
        
        # Step 4: Enhance query with context if available
        enhanced_query = query_text
        if context:
            context_prompt = conversation_manager.build_context_prompt(context)
            enhanced_query = f"{context_prompt}\n{query_text}"
            search_intent["query_text"] = enhanced_query
        
        # Step 5: Retrieve relevant chunks
        logger.debug(f"Searching for similar documents with top_k={top_k}")
        try:
            if intent.get("file_id"):
                chunks = search_similar_documents_by_file_id(
                    file_id=intent["file_id"],
                    user_id=user_id,
                    top_k=top_k
                )
            else:
                search_result = search_engine.search(search_intent)
                
                if search_result['status'] == 'success':
                    chunks = search_result['results']
                else:
                    raise RagError("Search failed", "SEARCH_FAILED")
            
        except (SearchError, Exception) as e:
            logger.error(f"Search error in streaming: {e}")
            yield {
                "type": "error",
                "content": f"Failed to retrieve documents: {str(e)}"
            }
            return
        
        # Step 6: Check if chunks were found
        if not chunks:
            yield {
                "type": "error",
                "content": "No relevant documents found for your query"
            }
            return
        
        # Step 7: Extract text and similarities
        text_chunks, similarities = extract_text_and_similarities(chunks)
        
        # Step 8: Apply token limits
        final_chunks, token_count = token_manager.truncate_chunks_by_tokens(text_chunks, similarities)
        final_similarities = similarities[:len(final_chunks)]
        final_chunk_docs = chunks[:len(final_chunks)]
        
        # Step 9: Calculate confidence score
        confidence_score = 0.0
        if include_confidence:
            confidence_score = confidence_calculator.calculate_confidence(final_similarities, final_chunk_docs)
        
        # Step 10: Extract file metadata
        if include_file_metadata:
            source_files = extract_file_metadata(final_chunk_docs)
        else:
            source_file_names = extract_source_files(final_chunk_docs)
            source_files = [{"fileName": name} for name in source_file_names]
        
        # Send metadata first
        yield {
            "type": "metadata",
            "metadata": {
                "query": query_text,
                "chunks_used": len(final_chunks),
                "source_files": source_files,
                "confidence_score": confidence_score,
                "context_used": context_used
            }
        }
        
        # Step 11: Generate and stream RAG answer
        logger.debug("Generating streaming RAG answer")
        try:
            answer_generator = answer_query_rag(
                chunks=final_chunks,
                query=query_text,
                stream=True
            )
            
            full_answer = ""
            for chunk in answer_generator:
                content = ""
                if isinstance(chunk, str):
                    content = chunk
                elif isinstance(chunk, dict) and 'content' in chunk:
                    content = chunk['content']
                
                if content:
                    full_answer += content
                    yield {
                        "type": "chunk",
                        "content": content
                    }
            
            # Step 12: Update conversation context
            if conversation_id:
                conversation_manager.update_context(user_id, conversation_id, query_text, full_answer, final_chunk_docs)
            
            # Step 13: Cache the result if confidence is high enough
            if use_cache and confidence_score > CONFIDENCE_THRESHOLD:
                cache_manager.cache_answer(
                    query_text, user_id, full_answer, len(final_chunks),
                    source_files, confidence_score, search_intent
                )
            
            # Step 14: Send completion
            processing_time = time.time() - start_time
            yield {
                "type": "complete",
                "metadata": {
                    "processing_time": processing_time,
                    "token_count": token_count,
                    "cached": False
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate streaming answer: {e}")
            yield {
                "type": "error",
                "content": f"Failed to generate answer: {str(e)}"
            }
            return
        
    except RagError as e:
        logger.error(f"RAG error in streaming: {e.message}")
        yield {
            "type": "error",
            "content": e.message
        }
    except Exception as e:
        logger.error(f"Unexpected error in streaming RAG: {str(e)}")
        yield {
            "type": "error",
            "content": f"Unexpected error: {str(e)}"
        }


def get_conversation_history(user_id: str, conversation_id: str) -> Dict[str, Any]:
    """
    Retrieve conversation history for a user and conversation.
    
    Args:
        user_id: User identifier
        conversation_id: Conversation identifier
        
    Returns:
        Dictionary containing conversation history or empty if not found
    """
    logger.debug(f"Retrieving conversation history for user: {user_id}, conversation: {conversation_id}")
    
    context = conversation_manager.get_context(user_id, conversation_id)
    
    if not context:
        return {
            "status": "not_found",
            "conversation_id": conversation_id,
            "user_id": user_id,
            "history": []
        }
    
    # Build conversation history
    history = []
    for i, (query, answer) in enumerate(zip(context.previous_queries, context.previous_answers)):
        history.append({
            "turn": i + 1,
            "query": query,
            "answer": answer[:500] + "..." if len(answer) > 500 else answer,  # Truncate long answers
            "timestamp": context.timestamp.isoformat()
        })
    
    return {
        "status": "success",
        "conversation_id": conversation_id,
        "user_id": user_id,
        "history": history,
        "total_turns": len(history),
        "last_updated": context.timestamp.isoformat()
    }


def clear_conversation_history(user_id: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Clear conversation history for a user.
    
    Args:
        user_id: User identifier
        conversation_id: Optional specific conversation ID to clear
        
    Returns:
        Dictionary with operation status
    """
    logger.info(f"Clearing conversation history for user: {user_id}")
    
    if conversation_id:
        # Clear specific conversation
        key = conversation_manager.get_context_key(user_id, conversation_id)
        if key in conversation_manager.contexts:
            del conversation_manager.contexts[key]
            logger.info(f"Cleared conversation {conversation_id} for user {user_id}")
            return {
                "status": "success",
                "message": f"Cleared conversation {conversation_id}",
                "cleared_conversations": 1
            }
        else:
            return {
                "status": "not_found",
                "message": f"Conversation {conversation_id} not found",
                "cleared_conversations": 0
            }
    else:
        # Clear all conversations for user
        keys_to_remove = [k for k in conversation_manager.contexts.keys() if k.startswith(f"{user_id}:")]
        for key in keys_to_remove:
            del conversation_manager.contexts[key]
        
        logger.info(f"Cleared {len(keys_to_remove)} conversations for user {user_id}")
        return {
            "status": "success",
            "message": f"Cleared all conversations for user {user_id}",
            "cleared_conversations": len(keys_to_remove)
        }


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics and performance metrics.
    
    Returns:
        Dictionary containing cache statistics
    """
    logger.debug("Retrieving cache statistics")
    
    total_entries = len(cache_manager.cache)
    expired_entries = sum(1 for entry in cache_manager.cache.values() if entry.is_expired())
    active_entries = total_entries - expired_entries
    
    # Calculate cache size and average confidence
    total_confidence = 0.0
    if active_entries > 0:
        active_cache_entries = [entry for entry in cache_manager.cache.values() if not entry.is_expired()]
        total_confidence = sum(entry.confidence_score for entry in active_cache_entries)
        avg_confidence = total_confidence / active_entries
    else:
        avg_confidence = 0.0
    
    return {
        "total_entries": total_entries,
        "active_entries": active_entries,
        "expired_entries": expired_entries,
        "average_confidence": round(avg_confidence, 3),
        "cache_expiry_hours": CACHE_EXPIRY_HOURS
    }


def clear_cache(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Clear cache entries, optionally filtered by user.
    
    Args:
        user_id: Optional user ID to filter cache clearing
        
    Returns:
        Dictionary with operation status
    """
    logger.info(f"Clearing cache entries" + (f" for user {user_id}" if user_id else ""))
    
    if user_id:
        # Clear cache for specific user
        keys_to_remove = [k for k, v in cache_manager.cache.items() if v.user_id == user_id]
        for key in keys_to_remove:
            del cache_manager.cache[key]
        
        cleared_count = len(keys_to_remove)
    else:
        # Clear all cache
        cleared_count = len(cache_manager.cache)
        cache_manager.cache.clear()
    
    return {
        "status": "success",
        "message": f"Cleared {cleared_count} cache entries",
        "cleared_entries": cleared_count
    }


def maintenance_cleanup() -> Dict[str, Any]:
    """
    Perform maintenance cleanup of expired cache entries and old conversations.
    
    Returns:
        Dictionary with cleanup statistics
    """
    logger.info("Starting maintenance cleanup")
    
    # Clear expired cache entries
    initial_cache_size = len(cache_manager.cache)
    cache_manager.clear_expired_cache()
    cache_cleared = initial_cache_size - len(cache_manager.cache)
    
    # Clear old conversation contexts (older than 24 hours)
    initial_conversations = len(conversation_manager.contexts)
    cutoff_time = datetime.utcnow() - timedelta(hours=24)
    
    old_conversations = [
        key for key, context in conversation_manager.contexts.items()
        if context.timestamp < cutoff_time
    ]
    
    for key in old_conversations:
        del conversation_manager.contexts[key]
    
    conversations_cleared = len(old_conversations)
    
    logger.info(f"Cleanup completed: {cache_cleared} cache entries, {conversations_cleared} conversations")
    
    return {
        "status": "success",
        "cache_entries_cleared": cache_cleared,
        "conversations_cleared": conversations_cleared,
        "remaining_cache_entries": len(cache_manager.cache),
        "remaining_conversations": len(conversation_manager.contexts)
    }


def health_check() -> Dict[str, Any]:
    """
    Perform health check on RAG system components.
    
    Returns:
        Dictionary with health status of all components
    """
    logger.debug("Performing RAG system health check")
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {}
    }
    
    # Check search engine
    try:
        # Test search engine with a simple query
        test_result = search_engine.search({
            "query_text": "test",
            "user_id": "health_check",
            "limit": 1
        })
        
        health_status["components"]["search_engine"] = {
            "status": "healthy" if test_result["status"] == "success" else "unhealthy",
            "message": "Search engine responding normally"
        }
    except Exception as e:
        health_status["components"]["search_engine"] = {
            "status": "unhealthy",
            "message": f"Search engine error: {str(e)}"
        }
        health_status["status"] = "degraded"
    
    # Check token manager
    try:
        test_tokens = token_manager.estimate_tokens("test text")
        health_status["components"]["token_manager"] = {
            "status": "healthy",
            "message": f"Token estimation working (test: {test_tokens} tokens)"
        }
    except Exception as e:
        health_status["components"]["token_manager"] = {
            "status": "unhealthy",
            "message": f"Token manager error: {str(e)}"
        }
        health_status["status"] = "degraded"
    
    # Check cache manager
    try:
        cache_stats = get_cache_stats()
        health_status["components"]["cache_manager"] = {
            "status": "healthy",
            "message": f"Cache active with {cache_stats['active_entries']} entries"
        }
    except Exception as e:
        health_status["components"]["cache_manager"] = {
            "status": "unhealthy",
            "message": f"Cache manager error: {str(e)}"
        }
        health_status["status"] = "degraded"
    
    # Check conversation manager
    try:
        active_conversations = len(conversation_manager.contexts)
        health_status["components"]["conversation_manager"] = {
            "status": "healthy",
            "message": f"Managing {active_conversations} active conversations"
        }
    except Exception as e:
        health_status["components"]["conversation_manager"] = {
            "status": "unhealthy",
            "message": f"Conversation manager error: {str(e)}"
        }
        health_status["status"] = "degraded"
    
    # Check confidence calculator
    try:
        test_confidence = confidence_calculator.calculate_confidence([0.8, 0.7], [{}, {}])
        health_status["components"]["confidence_calculator"] = {
            "status": "healthy",
            "message": f"Confidence calculation working (test: {test_confidence:.3f})"
        }
    except Exception as e:
        health_status["components"]["confidence_calculator"] = {
            "status": "unhealthy",
            "message": f"Confidence calculator error: {str(e)}"
        }
        health_status["status"] = "degraded"
    
    logger.info(f"Health check completed with status: {health_status['status']}")
    return health_status


# Utility functions for external integrations
def get_system_stats() -> Dict[str, Any]:
    """
    Get comprehensive system statistics.
    
    Returns:
        Dictionary with system statistics
    """
    return {
        "cache_stats": get_cache_stats(),
        "active_conversations": len(conversation_manager.contexts),
        "token_limit": token_manager.max_tokens,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "default_top_k": DEFAULT_TOP_K,
        "cache_expiry_hours": CACHE_EXPIRY_HOURS
    }


# Export main functions for external use
__all__ = [
    'answer_query_with_rag',
    'stream_rag_answer',
    'get_conversation_history',
    'clear_conversation_history',
    'get_cache_stats',
    'clear_cache',
    'maintenance_cleanup',
    'health_check',
    'get_system_stats',
    'RagError',
    'RagResult',
    'ConversationContext',
    'FileMetadata'
]

logger.info("RAG module initialized successfully")
