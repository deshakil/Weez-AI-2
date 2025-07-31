"""
Production-grade summarization module for Weez MCP AI Agent.

This module provides intelligent document summarization capabilities for files
stored across cloud platforms like Google Drive, OneDrive, Slack, and local storage.
Supports both file-based and query-based summarization with configurable detail levels.
Enhanced with file metadata (id, fileName, sas_url, platform) for direct file access.

Updated to properly integrate with the search.py module using SearchEngine class.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Set
from collections import defaultdict
from .search2 import SearchEngine, SearchError, create_search_engine
from utils.openai_client import summarize_chunks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationError(Exception):
    """Custom exception for summarization-related errors."""
    pass


def validate_intent(intent: Dict[str, Any]) -> None:
    """
    Validate the intent dictionary for summarization requirements.
    
    Args:
        intent: Dictionary containing user intent with required fields
        
    Raises:
        SummarizationError: If validation fails
    """
    if not isinstance(intent, dict):
        raise SummarizationError("Intent must be a dictionary")
    
    # Check if action is summarize
    action = intent.get("action")
    if action != "summarize":
        raise SummarizationError(f"Invalid action '{action}'. Expected 'summarize'")
    
    # Check if either file_id or query_text is present
    file_id = intent.get("file_id")
    query_text = intent.get("query_text")
    
    if not file_id and not query_text:
        raise SummarizationError("Either 'file_id' or 'query_text' must be provided")
    
    # Validate user_id is present
    user_id = intent.get("user_id")
    if not user_id:
        raise SummarizationError("'user_id' is required for summarization")
    
    logger.info(f"Intent validation passed for user {user_id}")


def determine_summarization_type(intent: Dict[str, Any]) -> str:
    """
    Determine the type of summarization based on the intent.
    
    Args:
        intent: Dictionary containing user intent
        
    Returns:
        String indicating summarization type: "file-based" or "query-based"
    """
    file_id = intent.get("file_id")
    query_text = intent.get("query_text")
    
    if file_id and not query_text:
        return "file-based"
    elif query_text and not file_id:
        return "query-based"
    elif file_id and query_text:
        # If both are present, prioritize file-based summarization with query context
        logger.info("Both file_id and query_text present, using file-based summarization with query context")
        return "file-based"
    else:
        # This should not happen due to validation, but handle it gracefully
        raise SummarizationError("Unable to determine summarization type")


def extract_file_metadata(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract comprehensive file metadata from chunks for file referencing.
    
    Args:
        chunks: List of chunk dictionaries containing metadata
        
    Returns:
        Dictionary containing file metadata fields
    """
    if not chunks:
        return {}
    
    # Get metadata from first chunk (all chunks from same file should have same file metadata)
    first_chunk = chunks[0]
    
    # Extract core file metadata fields
    file_metadata = {}
    
    # File ID - try multiple possible field names
    file_id_fields = ["file_id", "fileId", "id"]
    for field in file_id_fields:
        if field in first_chunk and first_chunk[field]:
            file_metadata["file_id"] = first_chunk[field]
            break
    
    # File Name - try multiple possible field names
    filename_fields = ["fileName", "file_name", "filename", "name", "title"]
    for field in filename_fields:
        if field in first_chunk and first_chunk[field]:
            file_metadata["fileName"] = first_chunk[field]
            break
    
    # SAS URL for direct file access
    sas_url_fields = ["sas_url", "sasUrl", "downloadUrl", "url"]
    for field in sas_url_fields:
        if field in first_chunk and first_chunk[field]:
            file_metadata["sas_url"] = first_chunk[field]
            break
    
    # Platform information
    platform_fields = ["platform", "source", "provider"]
    for field in platform_fields:
        if field in first_chunk and first_chunk[field]:
            file_metadata["platform"] = first_chunk[field]
            break
    
    # Additional useful metadata fields
    metadata_fields = {
        "mime_type": ["mime_type", "mimeType", "contentType", "content_type"],
        "created_at": ["created_at", "createdAt", "dateCreated", "created"],
        "file_size": ["fileSize", "file_size", "size"],
        "file_path": ["filePath", "file_path", "path"],
        "department": ["department", "departments"],
        "visibility": ["visibility", "access_level", "accessLevel"],
        "shared_with": ["shared_with", "sharedWith", "collaborators"],
        "created_by": ["created_by", "createdBy", "author", "owner"]
    }
    
    for target_field, possible_fields in metadata_fields.items():
        for field in possible_fields:
            if field in first_chunk and first_chunk[field] is not None:
                file_metadata[target_field] = first_chunk[field]
                break
    
    # Extract platform-specific metadata if available
    if "platform_metadata" in first_chunk and first_chunk["platform_metadata"]:
        file_metadata["platform_metadata"] = first_chunk["platform_metadata"]
    
    logger.info(f"Extracted file metadata: {list(file_metadata.keys())}")
    return file_metadata


def aggregate_file_metadata(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Aggregate metadata from multiple files when doing query-based summarization.
    
    Args:
        chunks: List of chunk dictionaries from potentially multiple files
        
    Returns:
        Dictionary mapping file_ids to their metadata and chunks
    """
    file_aggregation = defaultdict(lambda: {
        "metadata": {},
        "chunks": [],
        "chunk_count": 0
    })
    
    for chunk in chunks:
        # Determine file identifier
        file_id = None
        file_id_fields = ["file_id", "fileId", "id"]
        for field in file_id_fields:
            if field in chunk and chunk[field]:
                file_id = chunk[field]
                break
        
        # If no file_id found, create a temporary one based on filename
        if not file_id:
            filename = None
            filename_fields = ["fileName", "file_name", "filename", "name"]
            for field in filename_fields:
                if field in chunk and chunk[field]:
                    filename = chunk[field]
                    break
            file_id = f"unknown_{filename}" if filename else f"unknown_{hash(str(chunk))}"
        
        # Add chunk to file aggregation
        file_aggregation[file_id]["chunks"].append(chunk)
        file_aggregation[file_id]["chunk_count"] += 1
        
        # Extract and store file metadata (only once per file)
        if not file_aggregation[file_id]["metadata"]:
            file_aggregation[file_id]["metadata"] = extract_file_metadata([chunk])
    
    logger.info(f"Aggregated chunks from {len(file_aggregation)} files")
    return dict(file_aggregation)


def search_chunks_for_summarization(intent: Dict[str, Any], search_engine: Optional[SearchEngine] = None) -> List[Dict[str, Any]]:
    """
    Search for document chunks using the SearchEngine class.
    
    Args:
        intent: Dictionary containing user intent
        search_engine: Optional SearchEngine instance, will create new one if not provided
        
    Returns:
        List of chunk dictionaries from search results
        
    Raises:
        SummarizationError: If search fails
    """
    if search_engine is None:
        search_engine = create_search_engine()
    
    user_id = intent["user_id"]
    summarization_type = determine_summarization_type(intent)
    
    # Set limit based on summarization needs (more chunks for detailed summaries)
    summary_type = intent.get("summary_type", "short")
    limit = 30 if summary_type == "detailed" else 15
    
    logger.info(f"Starting {summarization_type} search with limit={limit}")
    
    try:
        if summarization_type == "file-based":
            # For file-based summarization, we need to search within a specific file
            # Since the search module doesn't have a direct file_id search, we'll use
            # a targeted search approach
            file_id = intent["file_id"]
            query_text = intent.get("query_text", "")
            
            logger.info(f"Searching within file {file_id} for user {user_id}")
            
            # Create search intent for file-based search
            # We'll use a broad query if no specific query_text is provided
            if not query_text:
                query_text = "content summary overview"  # Generic query to get file content
            
            search_intent = {
                "query_text": query_text,
                "user_id": user_id,
                "limit": limit,
                "offset": 0
            }
            
            # Add optional filters from original intent
            if intent.get("platform"):
                search_intent["platform"] = intent["platform"]
            if intent.get("file_type"):
                search_intent["file_type"] = intent["file_type"]
            if intent.get("time_range"):
                search_intent["time_range"] = intent["time_range"]
            
            # Execute search
            search_result = search_engine.search(search_intent)
            raw_chunks = search_result.get("results", [])
            
            # Filter results to only include chunks from the specified file
            file_chunks = []
            for chunk in raw_chunks:
                chunk_file_id = (chunk.get("file_id") or 
                               chunk.get("fileId") or 
                               chunk.get("id"))
                if chunk_file_id == file_id:
                    file_chunks.append(chunk)
            
            if not file_chunks:
                # If no chunks found for specific file_id, try to get all chunks
                # This might happen if file_id field name doesn't match
                logger.warning(f"No chunks found for file_id {file_id}, using all search results")
                file_chunks = raw_chunks
            
            chunks = file_chunks
            
        else:
            # Query-based summarization
            query_text = intent["query_text"]
            
            # Create search intent
            search_intent = {
                "query_text": query_text,
                "user_id": user_id,
                "limit": limit,
                "offset": 0
            }
            
            # Add optional filters
            if intent.get("platform"):
                search_intent["platform"] = intent["platform"]
            if intent.get("file_type"):
                search_intent["file_type"] = intent["file_type"]
            if intent.get("time_range"):
                search_intent["time_range"] = intent["time_range"]
            
            logger.info(f"Searching documents for query: '{query_text}'")
            
            # Execute search
            search_result = search_engine.search(search_intent)
            chunks = search_result.get("results", [])
        
        logger.info(f"Found {len(chunks)} chunks from search")
        return chunks
        
    except SearchError as e:
        logger.error(f"Search error: {str(e)}")
        raise SummarizationError(f"Failed to search documents: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in search: {str(e)}")
        raise SummarizationError(f"Unexpected error during document search: {str(e)}")


def prepare_chunks_for_summary(chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Prepare chunks for summarization by extracting text content.
    
    Args:
        chunks: List of chunk dictionaries from search results
        
    Returns:
        List of text strings ready for summarization
    """
    text_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Extract text content from various possible fields
        text_content = None
        
        # Try different field names for chunk content (based on search.py structure)
        content_fields = ["text", "content", "chunk_text", "body", "data"]
        
        for field in content_fields:
            if field in chunk and chunk[field]:
                text_content = chunk[field]
                break
        
        if text_content:
            # Clean and prepare the text
            cleaned_text = str(text_content).strip()
            if cleaned_text:
                text_chunks.append(cleaned_text)
            else:
                logger.warning(f"Empty text content after cleaning in chunk {i}")
        else:
            logger.warning(f"No text content found in chunk {i}: {list(chunk.keys())}")
    
    logger.info(f"Prepared {len(text_chunks)} text chunks for summarization")
    return text_chunks


def validate_chunks_quality(chunks: List[Dict[str, Any]], intent: Dict[str, Any]) -> None:
    """
    Validate that the retrieved chunks are suitable for summarization.
    
    Args:
        chunks: List of chunk dictionaries
        intent: Original intent dictionary
        
    Raises:
        SummarizationError: If chunks are not suitable for summarization
    """
    if not chunks:
        summarization_type = determine_summarization_type(intent)
        if summarization_type == "file-based":
            error_msg = f"No content found for file_id: {intent.get('file_id')}"
        else:
            error_msg = f"No relevant content found for query: {intent.get('query_text')}"
        raise SummarizationError(error_msg)
    
    # Check if chunks have reasonable similarity scores (if available)
    scored_chunks = [c for c in chunks if '_similarity' in c and c['_similarity'] is not None]
    if scored_chunks:
        avg_similarity = sum(c['_similarity'] for c in scored_chunks) / len(scored_chunks)
        if avg_similarity < 0.3:  # Very low similarity threshold
            logger.warning(f"Low average similarity score: {avg_similarity:.3f}")
    
    # Check total content length
    total_content_length = 0
    for chunk in chunks:
        text_content = chunk.get('text') or chunk.get('content', '')
        total_content_length += len(str(text_content))
    
    if total_content_length < 100:  # Minimum content length
        raise SummarizationError("Retrieved content is too short for meaningful summarization")
    
    logger.info(f"Chunks validation passed: {len(chunks)} chunks, {total_content_length} total characters")


def create_summary_context(intent: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create context information for the summary with enhanced file metadata.
    
    Args:
        intent: Original intent dictionary
        chunks: Retrieved chunks
        
    Returns:
        Dictionary with context information including file metadata
    """
    summarization_type = determine_summarization_type(intent)
    
    context = {
        "summarization_type": summarization_type,
        "chunks_count": len(chunks),
        "user_query": intent.get("query_text"),
        "summary_type": intent.get("summary_type", "short")
    }
    
    # Add file metadata based on summarization type
    if summarization_type == "file-based":
        # Single file summarization - extract metadata from chunks
        file_metadata = extract_file_metadata(chunks)
        context.update(file_metadata)
    else:
        # Query-based summarization - aggregate metadata from multiple files
        file_aggregation = aggregate_file_metadata(chunks)
        context["source_files"] = {}
        
        for file_id, file_data in file_aggregation.items():
            context["source_files"][file_id] = {
                "metadata": file_data["metadata"],
                "chunk_count": file_data["chunk_count"]
            }
        
        # Also create a summary of unique files
        context["unique_files_count"] = len(file_aggregation)
        context["files_summary"] = []
        
        for file_id, file_data in file_aggregation.items():
            file_summary = {
                "file_id": file_id,
                "fileName": file_data["metadata"].get("fileName", "Unknown"),
                "platform": file_data["metadata"].get("platform", "Unknown"),
                "chunk_count": file_data["chunk_count"]
            }
            
            # Add SAS URL if available for direct access
            if "sas_url" in file_data["metadata"]:
                file_summary["sas_url"] = file_data["metadata"]["sas_url"]
            
            context["files_summary"].append(file_summary)
    
    # Add platform and time range if specified
    if intent.get("platform"):
        context["platform_filter"] = intent["platform"]
    if intent.get("time_range"):
        context["time_range"] = intent["time_range"]
    
    # Calculate average similarity if available
    scored_chunks = [c for c in chunks if '_similarity' in c and c['_similarity'] is not None]
    if scored_chunks:
        context["average_similarity"] = sum(c['_similarity'] for c in scored_chunks) / len(scored_chunks)
    
    return context


def summarize_document(intent: Dict[str, Any], search_engine: Optional[SearchEngine] = None) -> Dict[str, Any]:
    """
    Main function to perform document summarization based on user intent.
    Enhanced with comprehensive file metadata for direct file access.
    
    Args:
        intent: Dictionary containing structured user intent with fields:
            - action: Must be "summarize"
            - file_id: Optional file identifier for file-based summarization
            - query_text: Optional query text for query-based summarization
            - summary_type: Optional summary detail level ("short" or "detailed")
            - user_id: Required user identifier
            - platform: Optional platform identifier
            - file_type: Optional file type filter
            - time_range: Optional time range filter
        search_engine: Optional SearchEngine instance, will create new one if not provided
            
    Returns:
        Dictionary containing:
            - summary: Generated summary text
            - summary_type: Type of summary generated
            - chunks_used: Number of chunks used for summarization
            - file_metadata: File metadata for direct access (file-based)
            - source_files: Source files metadata (query-based)
            - context: Additional context information
            
    Raises:
        SummarizationError: If summarization fails
    """
    try:
        # Validate input intent
        validate_intent(intent)
        
        # Set default summary type if not specified
        summary_type = intent.get("summary_type", "short")
        if summary_type not in ["short", "detailed"]:
            logger.warning(f"Invalid summary_type '{summary_type}', defaulting to 'short'")
            summary_type = "short"
            intent["summary_type"] = summary_type
        
        # Determine summarization type
        summarization_type = determine_summarization_type(intent)
        
        logger.info(f"Starting {summarization_type} summarization for user {intent['user_id']}")
        
        # Create search engine if not provided
        if search_engine is None:
            search_engine = create_search_engine()
        
        # Search for relevant chunks using the search module
        try:
            chunks = search_chunks_for_summarization(intent, search_engine)
        except Exception as e:
            logger.error(f"Error searching for chunks: {str(e)}")
            raise SummarizationError(f"Failed to retrieve document chunks: {str(e)}")
        
        # Validate chunks quality
        validate_chunks_quality(chunks, intent)
        
        # Prepare chunks for summarization
        text_chunks = prepare_chunks_for_summary(chunks)
        
        if not text_chunks:
            raise SummarizationError("No valid text content found in retrieved chunks")
        
        # Create context for summary with enhanced file metadata
        context = create_summary_context(intent, chunks)
        
        # Generate summary using OpenAI
        try:
            query_for_summary = intent.get("query_text") if summarization_type == "query-based" else None
            summary = summarize_chunks(text_chunks, summary_type, query_for_summary)
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise SummarizationError(f"Failed to generate summary: {str(e)}")
        
        # Prepare enhanced response with file metadata
        response = {
            "summary": summary,
            "summary_type": summary_type,
            "chunks_used": len(text_chunks),
            "context": context
        }
        
        # Add file metadata based on summarization type
        if summarization_type == "file-based":
            # Single file - extract comprehensive metadata
            file_metadata = extract_file_metadata(chunks)
            response["file_metadata"] = file_metadata
            
            # Add backward compatibility fields
            if "fileName" in file_metadata:
                response["fileName"] = file_metadata["fileName"]
            if "file_id" in file_metadata:
                response["file_id"] = file_metadata["file_id"]
            if "sas_url" in file_metadata:
                response["sas_url"] = file_metadata["sas_url"]
            if "platform" in file_metadata:
                response["platform"] = file_metadata["platform"]
                
        else:
            # Query-based - multiple files metadata
            file_aggregation = aggregate_file_metadata(chunks)
            response["source_files"] = []
            
            for file_id, file_data in file_aggregation.items():
                file_info = file_data["metadata"].copy()
                file_info["chunk_count"] = file_data["chunk_count"]
                response["source_files"].append(file_info)
            
            # Add summary statistics
            response["unique_files_count"] = len(file_aggregation)
            response["total_source_files"] = len(file_aggregation)
        
        # Add search metadata if available
        if chunks and '_search_metadata' in chunks[0]:
            response["search_metadata"] = chunks[0]['_search_metadata']
        
        logger.info(f"Successfully generated {summary_type} summary using {len(text_chunks)} chunks")
        
        return response
        
    except SummarizationError:
        # Re-raise custom errors
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in summarization: {str(e)}")
        raise SummarizationError(f"Unexpected error during summarization: {str(e)}")


def summarize_search_results(search_results: List[Dict[str, Any]], 
                           summary_type: str = "short",
                           query_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Summarize pre-existing search results with enhanced file metadata.
    This function allows for a two-step process: search first, then summarize.
    
    Args:
        search_results: List of search result dictionaries from SearchEngine.search()
        summary_type: Type of summary ("short" or "detailed")
        query_context: Optional query context for focused summarization
        
    Returns:
        Dictionary containing summarization results with file metadata
        
    Raises:
        SummarizationError: If summarization fails
    """
    try:
        if not search_results:
            raise SummarizationError("No search results provided for summarization")
        
        if summary_type not in ["short", "detailed"]:
            logger.warning(f"Invalid summary_type '{summary_type}', defaulting to 'short'")
            summary_type = "short"
        
        logger.info(f"Summarizing {len(search_results)} pre-existing search results")
        
        # Prepare chunks for summarization
        text_chunks = prepare_chunks_for_summary(search_results)
        
        if not text_chunks:
            raise SummarizationError("No valid text content found in search results")
        
        # Generate summary using OpenAI
        try:
            summary = summarize_chunks(text_chunks, summary_type, query_context)
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise SummarizationError(f"Failed to generate summary: {str(e)}")
        
        # Aggregate file metadata from search results
        file_aggregation = aggregate_file_metadata(search_results)
        
        # Create context information with file metadata
        context = {
            "summarization_type": "search-results-based",
            "chunks_count": len(search_results),
            "text_chunks_used": len(text_chunks),
            "summary_type": summary_type,
            "unique_files_count": len(file_aggregation)
        }
        
        # Add query context if provided
        if query_context:
            context["query_context"] = query_context
        
        # Calculate average similarity if available
        scored_results = [r for r in search_results if '_similarity' in r and r['_similarity'] is not None]
        if scored_results:
            context["average_similarity"] = sum(r['_similarity'] for r in scored_results) / len(scored_results)
        
        # Prepare source files metadata
        source_files = []
        for file_id, file_data in file_aggregation.items():
            file_info = file_data["metadata"].copy()
            file_info["chunk_count"] = file_data["chunk_count"]
            source_files.append(file_info)
        
        # Prepare enhanced response
        response = {
            "summary": summary,
            "summary_type": summary_type,
            "chunks_used": len(text_chunks),
            "context": context,
            "source_files": source_files,
            "unique_files_count": len(file_aggregation),
            "total_source_files": len(file_aggregation)
        }
        
        logger.info(f"Successfully generated {summary_type} summary from search results")
        
        return response
        
    except SummarizationError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error summarizing search results: {str(e)}")
        raise SummarizationError(f"Unexpected error during search results summarization: {str(e)}")


def summarize_file(file_id: str, user_id: str, summary_type: str = "short", 
                  platform: Optional[str] = None, query_context: Optional[str] = None,
                  search_engine: Optional[SearchEngine] = None) -> Dict[str, Any]:
    """
    Convenience function for file-based summarization with enhanced metadata.
    
    Args:
        file_id: Identifier of the file to summarize
        user_id: User identifier
        summary_type: Type of summary ("short" or "detailed")
        platform: Optional platform identifier
        query_context: Optional query for focused summarization within the file
        search_engine: Optional SearchEngine instance
        
    Returns:
        Dictionary containing summarization results with file metadata
    """
    intent = {
        "action": "summarize",
        "file_id": file_id,
        "user_id": user_id,
        "summary_type": summary_type
    }
    
    if platform:
        intent["platform"] = platform
    
    if query_context:
        intent["query_text"] = query_context
    
    return summarize_document(intent, search_engine)


def summarize_query(query_text: str, user_id: str, summary_type: str = "short",
                   platform: Optional[str] = None, file_type: Optional[str] = None,
                   time_range: Optional[str] = None,
                   search_engine: Optional[SearchEngine] = None) -> Dict[str, Any]:
    """
    Convenience function for query-based summarization with source files metadata.
    
    Args:
        query_text: Query text to search and summarize
        user_id: User identifier
        summary_type: Type of summary ("short" or "detailed")
        platform: Optional platform identifier
        file_type: Optional file type filter
        time_range: Optional time range filter
        search_engine: Optional SearchEngine instance
        
    Returns:
        Dictionary containing summarization results with source files metadata
    """
    intent = {
        "action": "summarize",
        "query_text": query_text,
        "user_id": user_id,
        "summary_type": summary_type
    }
    
    if platform:
        intent["platform"] = platform
    
    if file_type:
        intent["file_type"] = file_type
    
    if time_range:
        intent["time_range"] = time_range
    
    return summarize_document(intent, search_engine)


# Workflow functions for search-then-summarize pattern
def search_and_summarize(search_intent: Dict[str, Any], summary_type: str = "short",
                        search_engine: Optional[SearchEngine] = None) -> Dict[str, Any]:
    """
    Combined workflow: search documents then summarize the results with file metadata.
    This enables a two-step process where search results can be reviewed before summarization.
    
    Args:
        search_intent: Dictionary for SearchEngine.search() function
        summary_type: Type of summary to generate
        search_engine: Optional SearchEngine instance
        
    Returns:
        Dictionary containing both search results and summary with file metadata
    """
    try:
        # Create search engine if not provided
        if search_engine is None:
            search_engine = create_search_engine()
        
        logger.info(f"Starting search-and-summarize workflow for user {search_intent.get('user_id')}")
        
        # Perform search using SearchEngine
        try:
            search_result = search_engine.search(search_intent)
            search_results = search_result.get("results", [])
        except SearchError as e:
            raise SummarizationError(f"Search failed: {str(e)}")
        
        if not search_results:
            raise SummarizationError("No search results found for summarization")
        
        # Summarize the search results
        query_context = search_intent.get('query_text')
        summary_result = summarize_search_results(search_results, summary_type, query_context)
        
        # Combine results with enhanced metadata
        combined_result = {
            "search_results": search_results,
            "search_metadata": search_result.get("metrics", {}),
            "summary": summary_result["summary"],
            "summary_type": summary_result["summary_type"],
            "chunks_used": summary_result["chunks_used"],
            "context": summary_result["context"],
            "source_files": summary_result["source_files"],
            "unique_files_count": summary_result["unique_files_count"],
            "total_source_files": summary_result["total_source_files"]
        }
        
        # Add search timing information if available
        if "search_time" in search_result:
            combined_result["search_time"] = search_result["search_time"]
        
        logger.info(f"Successfully completed search-and-summarize workflow with {len(search_results)} results")
        
        return combined_result
        
    except SummarizationError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in search-and-summarize workflow: {str(e)}")
        raise SummarizationError(f"Search-and-summarize workflow failed: {str(e)}")


def batch_summarize_files(file_ids: List[str], user_id: str, 
                         summary_type: str = "short",
                         platform: Optional[str] = None,
                         search_engine: Optional[SearchEngine] = None) -> Dict[str, Any]:
    """
    Batch summarization for multiple files with comprehensive metadata.
    
    Args:
        file_ids: List of file identifiers to summarize
        user_id: User identifier
        summary_type: Type of summary ("short" or "detailed")
        platform: Optional platform identifier
        search_engine: Optional SearchEngine instance
        
    Returns:
        Dictionary containing batch summarization results with file metadata
    """
    try:
        if not file_ids:
            raise SummarizationError("No file IDs provided for batch summarization")
        
        logger.info(f"Starting batch summarization for {len(file_ids)} files")
        
        # Create search engine if not provided
        if search_engine is None:
            search_engine = create_search_engine()
        
        successful_summaries = []
        failed_summaries = []
        
        for file_id in file_ids:
            try:
                # Summarize individual file
                result = summarize_file(
                    file_id=file_id,
                    user_id=user_id,
                    summary_type=summary_type,
                    platform=platform,
                    search_engine=search_engine
                )
                
                successful_summaries.append({
                    "file_id": file_id,
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"Failed to summarize file {file_id}: {str(e)}")
                failed_summaries.append({
                    "file_id": file_id,
                    "error": str(e)
                })
        
        # Aggregate results
        batch_result = {
            "batch_summary_type": summary_type,
            "total_files": len(file_ids),
            "successful_count": len(successful_summaries),
            "failed_count": len(failed_summaries),
            "successful_summaries": successful_summaries,
            "failed_summaries": failed_summaries
        }
        
        # Create combined summary if multiple files were successfully processed
        if len(successful_summaries) > 1:
            try:
                # Combine all text chunks from successful summaries
                all_summaries = [s["result"]["summary"] for s in successful_summaries]
                combined_text = "\n\n".join(all_summaries)
                
                # Generate a meta-summary
                meta_summary = summarize_chunks([combined_text], summary_type, 
                                              f"Combined summary of {len(successful_summaries)} files")
                
                batch_result["combined_summary"] = meta_summary
                
            except Exception as e:
                logger.warning(f"Failed to create combined summary: {str(e)}")
                batch_result["combined_summary_error"] = str(e)
        
        logger.info(f"Batch summarization completed: {len(successful_summaries)} successful, {len(failed_summaries)} failed")
        
        return batch_result
        
    except Exception as e:
        logger.error(f"Batch summarization failed: {str(e)}")
        raise SummarizationError(f"Batch summarization failed: {str(e)}")


def get_summary_statistics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate statistics about the chunks used for summarization.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Dictionary containing statistics about the chunks
    """
    if not chunks:
        return {"error": "No chunks provided"}
    
    stats = {
        "total_chunks": len(chunks),
        "total_characters": 0,
        "total_words": 0,
        "unique_files": set(),
        "platforms": set(),
        "similarity_scores": [],
        "chunk_sizes": []
    }
    
    for chunk in chunks:
        # Text statistics
        text_content = chunk.get('text') or chunk.get('content', '')
        text_str = str(text_content)
        
        stats["total_characters"] += len(text_str)
        stats["total_words"] += len(text_str.split())
        stats["chunk_sizes"].append(len(text_str))
        
        # File tracking
        file_id = (chunk.get("file_id") or 
                  chunk.get("fileId") or 
                  chunk.get("id"))
        if file_id:
            stats["unique_files"].add(file_id)
        
        # Platform tracking
        platform = chunk.get("platform")
        if platform:
            stats["platforms"].add(platform)
        
        # Similarity scores
        if '_similarity' in chunk and chunk['_similarity'] is not None:
            stats["similarity_scores"].append(chunk['_similarity'])
    
    # Convert sets to lists for JSON serialization
    stats["unique_files"] = list(stats["unique_files"])
    stats["platforms"] = list(stats["platforms"])
    stats["unique_files_count"] = len(stats["unique_files"])
    stats["platforms_count"] = len(stats["platforms"])
    
    # Calculate averages
    if stats["chunk_sizes"]:
        stats["average_chunk_size"] = sum(stats["chunk_sizes"]) / len(stats["chunk_sizes"])
        stats["min_chunk_size"] = min(stats["chunk_sizes"])
        stats["max_chunk_size"] = max(stats["chunk_sizes"])
    
    if stats["similarity_scores"]:
        stats["average_similarity"] = sum(stats["similarity_scores"]) / len(stats["similarity_scores"])
        stats["min_similarity"] = min(stats["similarity_scores"])
        stats["max_similarity"] = max(stats["similarity_scores"])
    
    return stats


def export_summary_report(summary_result: Dict[str, Any], 
                         format_type: str = "markdown") -> str:
    """
    Export a formatted summary report with metadata.
    
    Args:
        summary_result: Result dictionary from summarization functions
        format_type: Export format ("markdown", "text", or "json")
        
    Returns:
        Formatted report string
    """
    try:
        if format_type == "json":
            import json
            return json.dumps(summary_result, indent=2, default=str)
        
        elif format_type == "markdown":
            return _export_markdown_report(summary_result)
        
        elif format_type == "text":
            return _export_text_report(summary_result)
        
        else:
            raise SummarizationError(f"Unsupported export format: {format_type}")
            
    except Exception as e:
        logger.error(f"Failed to export summary report: {str(e)}")
        raise SummarizationError(f"Export failed: {str(e)}")


def _export_markdown_report(summary_result: Dict[str, Any]) -> str:
    """Generate a markdown-formatted summary report."""
    
    lines = []
    lines.append("# Document Summary Report")
    lines.append("")
    
    # Summary section
    lines.append("## Summary")
    lines.append(summary_result.get("summary", "No summary available"))
    lines.append("")
    
    # Metadata section
    lines.append("## Summary Details")
    lines.append(f"- **Summary Type**: {summary_result.get('summary_type', 'Unknown')}")
    lines.append(f"- **Chunks Used**: {summary_result.get('chunks_used', 0)}")
    
    context = summary_result.get("context", {})
    if "summarization_type" in context:
        lines.append(f"- **Summarization Type**: {context['summarization_type']}")
    
    if "average_similarity" in context:
        lines.append(f"- **Average Similarity**: {context['average_similarity']:.3f}")
    
    lines.append("")
    
    # File information
    if "file_metadata" in summary_result:
        # Single file summary
        lines.append("## File Information")
        metadata = summary_result["file_metadata"]
        
        if "fileName" in metadata:
            lines.append(f"- **File Name**: {metadata['fileName']}")
        if "platform" in metadata:
            lines.append(f"- **Platform**: {metadata['platform']}")
        if "file_id" in metadata:
            lines.append(f"- **File ID**: {metadata['file_id']}")
        if "sas_url" in metadata:
            lines.append(f"- **Direct Access**: [Download File]({metadata['sas_url']})")
        
        lines.append("")
    
    elif "source_files" in summary_result:
        # Multiple files summary
        lines.append("## Source Files")
        lines.append(f"Total files: {summary_result.get('unique_files_count', 0)}")
        lines.append("")
        
        source_files = summary_result.get("source_files", [])
        for i, file_info in enumerate(source_files, 1):
            lines.append(f"### File {i}")
            
            if "fileName" in file_info:
                lines.append(f"- **Name**: {file_info['fileName']}")
            if "platform" in file_info:
                lines.append(f"- **Platform**: {file_info['platform']}")
            if "chunk_count" in file_info:
                lines.append(f"- **Chunks**: {file_info['chunk_count']}")
            if "sas_url" in file_info:
                lines.append(f"- **Direct Access**: [Download File]({file_info['sas_url']})")
            
            lines.append("")
    
    # Search metadata if available
    if "search_metadata" in summary_result:
        lines.append("## Search Information")
        search_meta = summary_result["search_metadata"]
        
        for key, value in search_meta.items():
            lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        
        lines.append("")
    
    return "\n".join(lines)


def _export_text_report(summary_result: Dict[str, Any]) -> str:
    """Generate a plain text summary report."""
    
    lines = []
    lines.append("DOCUMENT SUMMARY REPORT")
    lines.append("=" * 50)
    lines.append("")
    
    # Summary
    lines.append("SUMMARY:")
    lines.append("-" * 20)
    lines.append(summary_result.get("summary", "No summary available"))
    lines.append("")
    
    # Details
    lines.append("DETAILS:")
    lines.append("-" * 20)
    lines.append(f"Summary Type: {summary_result.get('summary_type', 'Unknown')}")
    lines.append(f"Chunks Used: {summary_result.get('chunks_used', 0)}")
    
    context = summary_result.get("context", {})
    if "summarization_type" in context:
        lines.append(f"Summarization Type: {context['summarization_type']}")
    
    if "average_similarity" in context:
        lines.append(f"Average Similarity: {context['average_similarity']:.3f}")
    
    lines.append("")
    
    # File information
    if "file_metadata" in summary_result:
        lines.append("FILE INFORMATION:")
        lines.append("-" * 20)
        metadata = summary_result["file_metadata"]
        
        for key, value in metadata.items():
            if value and key not in ['platform_metadata']:  # Skip complex nested data
                lines.append(f"{key.replace('_', ' ').title()}: {value}")
        
        lines.append("")
    
    elif "source_files" in summary_result:
        lines.append("SOURCE FILES:")
        lines.append("-" * 20)
        lines.append(f"Total files: {summary_result.get('unique_files_count', 0)}")
        lines.append("")
        
        source_files = summary_result.get("source_files", [])
        for i, file_info in enumerate(source_files, 1):
            lines.append(f"File {i}:")
            
            for key, value in file_info.items():
                if value and key not in ['platform_metadata']:
                    lines.append(f"  {key.replace('_', ' ').title()}: {value}")
            
            lines.append("")
    
    return "\n".join(lines)


# Utility functions for integration testing and debugging
def validate_summarization_pipeline(user_id: str, test_query: str = "test document", 
                                  search_engine: Optional[SearchEngine] = None) -> Dict[str, Any]:
    """
    Validate the entire summarization pipeline for testing purposes.
    
    Args:
        user_id: User identifier for testing
        test_query: Test query to use
        search_engine: Optional SearchEngine instance
        
    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        "pipeline_status": "unknown",
        "components_tested": [],
        "errors": [],
        "warnings": []
    }
    
    try:
        # Test search engine creation
        if search_engine is None:
            try:
                search_engine = create_search_engine()
                validation_results["components_tested"].append("search_engine_creation")
            except Exception as e:
                validation_results["errors"].append(f"Search engine creation failed: {str(e)}")
                return validation_results
        
        # Test query-based summarization
        try:
            test_intent = {
                "action": "summarize",
                "query_text": test_query,
                "user_id": user_id,
                "summary_type": "short"
            }
            
            validate_intent(test_intent)
            validation_results["components_tested"].append("intent_validation")
            
            # Test search functionality
            chunks = search_chunks_for_summarization(test_intent, search_engine)
            validation_results["components_tested"].append("search_functionality")
            
            if chunks:
                # Test chunk preparation
                text_chunks = prepare_chunks_for_summary(chunks)
                validation_results["components_tested"].append("chunk_preparation")
                
                # Test metadata extraction
                file_metadata = extract_file_metadata(chunks)
                validation_results["components_tested"].append("metadata_extraction")
                
                # Test context creation
                context = create_summary_context(test_intent, chunks)
                validation_results["components_tested"].append("context_creation")
                
                validation_results["test_data"] = {
                    "chunks_found": len(chunks),
                    "text_chunks_prepared": len(text_chunks),
                    "metadata_fields": list(file_metadata.keys()),
                    "context_fields": list(context.keys())
                }
            else:
                validation_results["warnings"].append("No chunks found for test query")
        
        except Exception as e:
            validation_results["errors"].append(f"Pipeline test failed: {str(e)}")
        
        # Determine overall status
        if not validation_results["errors"]:
            if validation_results["warnings"]:
                validation_results["pipeline_status"] = "warning"
            else:
                validation_results["pipeline_status"] = "success"
        else:
            validation_results["pipeline_status"] = "error"
        
        logger.info(f"Pipeline validation completed with status: {validation_results['pipeline_status']}")
        
    except Exception as e:
        validation_results["pipeline_status"] = "error"
        validation_results["errors"].append(f"Validation process failed: {str(e)}")
    
    return validation_results


# Export all main functions for external use
__all__ = [
    # Main functions
    'summarize_document',
    'summarize_search_results',
    'summarize_file',
    'summarize_query',
    'search_and_summarize',
    'batch_summarize_files',
    
    # Utility functions
    'validate_intent',
    'determine_summarization_type',
    'extract_file_metadata',
    'aggregate_file_metadata',
    'get_summary_statistics',
    'export_summary_report',
    'validate_summarization_pipeline',
    
    # Exception classes
    'SummarizationError'
]


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1:
        test_user_id = sys.argv[1]
        
        try:
            # Test pipeline validation
            validation = validate_summarization_pipeline(test_user_id)
            print("Pipeline Validation Results:")
            print(f"Status: {validation['pipeline_status']}")
            print(f"Components Tested: {validation['components_tested']}")
            
            if validation['errors']:
                print(f"Errors: {validation['errors']}")
            
            if validation['warnings']:
                print(f"Warnings: {validation['warnings']}")
                
        except Exception as e:
            print(f"Validation failed: {str(e)}")
    else:
        print("Usage: python summarizer.py <user_id>")
        print("This will run pipeline validation tests.")
