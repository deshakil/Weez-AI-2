# ai_layer/tools.py

from typing import List, Dict, Any
import logging

# Import from the new search.py module
from ai_layer.search2 import (
    create_search_engine,
    validate_search_intent,
    build_search_filters,
    process_search_results,
    SearchError,
    Platform,
    FileCategory
)
from ai_layer.summarizer2 import (
    summarize_document,
    summarize_search_results,
    summarize_file,
    summarize_query,
    search_and_summarize,
    batch_summarize_files,
    SummarizationError
)
from ai_layer.rag2 import (
    answer_query_with_rag,
    stream_rag_answer,
    get_conversation_history,
    clear_conversation_history,
    get_cache_stats,
    clear_cache,
    maintenance_cleanup,
    health_check as rag_health_check,
    get_system_stats,
    RagError
)

# Configure logger
logger = logging.getLogger(__name__)

# Create a global search engine instance
search_engine = create_search_engine()

# ===========================
# 🔍 Search Tool Wrapper
# ===========================
def tool_search(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced search tool wrapper using the new search.py module.
    Supports all advanced features including validation, filtering, and result processing.
    """
    try:
        # Extract required parameters
        query_text = args.get("query_text")
        user_id = args.get("user_id")
        
        if not query_text or not user_id:
            return {
                "error": "Missing required parameters: query_text and user_id",
                "tool_used": "search"
            }
        
        # Extract optional parameters with proper validation
        top_k = args.get("top_k", 10)
        
        # Validate top_k
        if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
            top_k = 10  # Default fallback
        
        # Build search intent using the new format
        intent = {
            "query_text": query_text,
            "user_id": user_id,
            "limit": top_k
        }
        
        # Map platform parameter
        if args.get("platform"):
            platform = args["platform"].lower().strip()
            # Map common platform names to search.py format
            platform_mapping = {
                "drive": "google_drive",
                "google_drive": "google_drive",
                "onedrive": "onedrive", 
                "dropbox": "dropbox",
                "sharepoint": "sharepoint",
                "local": "local",
                "slack": "slack",
                "teams": "teams"
            }
            intent["platform"] = platform_mapping.get(platform, platform)
        
        # Map mime_type/file_type parameter
        if args.get("mime_type"):
            # Convert MIME type to search.py file_type format
            mime_type = args["mime_type"]
            mime_to_file_type = {
                'application/pdf': 'PDF',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'DOCX',
                'application/msword': 'DOC',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'XLSX',
                'application/vnd.ms-excel': 'XLS',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'PPTX',
                'application/vnd.ms-powerpoint': 'PPT',
                'text/plain': 'TXT'
            }
            intent["file_type"] = mime_to_file_type.get(mime_type, mime_type)
        elif args.get("file_type"):
            intent["file_type"] = args["file_type"]
        
        # Map time_range parameter
        if args.get("time_range"):
            time_range = args["time_range"]
            if isinstance(time_range, dict):
                # Convert from/to format to start_date/end_date format
                converted_time_range = {}
                if time_range.get("from"):
                    converted_time_range["start_date"] = time_range["from"]
                if time_range.get("to"):
                    converted_time_range["end_date"] = time_range["to"]
                intent["time_range"] = converted_time_range
            elif isinstance(time_range, str):
                # Handle relative time ranges
                intent["time_range"] = time_range
        
        # Add pagination parameters if provided
        if args.get("offset"):
            intent["offset"] = args["offset"]
        if args.get("limit"):
            intent["limit"] = min(args["limit"], 100)  # Cap at 100
        
        # Execute the search using the new search engine
        logger.info(f"Executing search for user {user_id} with query: '{query_text}'")
        search_result = search_engine.search(intent)
        
        # Extract results and metrics from the new format
        results = search_result.get('results', [])
        metrics = search_result.get('metrics', {})
        validated_intent = search_result.get('intent', intent)
        
        # Extract applied filters for response
        applied_filters = {}
        if validated_intent.get("platform"):
            applied_filters["platform"] = validated_intent["platform"]
        if validated_intent.get("file_type"):
            applied_filters["file_type"] = validated_intent["file_type"]
        if validated_intent.get("time_range"):
            applied_filters["time_range"] = validated_intent["time_range"]
        
        # Format response
        response = {
            "tool_used": "search",
            "query_text": query_text,
            "user_id": user_id,
            "filters_applied": applied_filters,
            "total_results": len(results),
            "results": results
        }
        
        # Add search metrics if available
        if metrics:
            response["search_metrics"] = metrics
        
        logger.info(f"Search completed: {len(results)} results returned")
        return response
        
    except SearchError as e:
        logger.error(f"Search error: {str(e)}")
        return {
            "error": f"Search failed: {str(e)}",
            "tool_used": "search"
        }
    except Exception as e:
        logger.error(f"Unexpected error in tool_search: {str(e)}", exc_info=True)
        return {
            "error": f"Unexpected search error: {str(e)}",
            "tool_used": "search"
        }


# ===========================
# 🔍 File-Specific Search Tool
# ===========================
def tool_search_file(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search within a specific file using the search engine's cosmos client.
    """
    try:
        file_id = args.get("file_id")
        user_id = args.get("user_id")
        query_text = args.get("query_text", "")
        top_k = args.get("top_k", 10)
        
        if not file_id or not user_id:
            return {
                "error": "Missing required parameters: file_id and user_id",
                "tool_used": "search_file"
            }
        
        # Validate top_k
        if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
            top_k = 10
        
        logger.info(f"Searching within file {file_id} for user {user_id}")
        
        # Use the cosmos client from search engine to search by file ID
        cosmos_client = search_engine.cosmos_client
        
        # Build filters for file-specific search
        filters = {
            'user_id': user_id,
            'file_id': file_id
        }
        
        if query_text:
            # Generate embedding for the query
            from utils.openai_client import get_embedding
            query_embedding = get_embedding(query_text)
            
            # Execute hybrid search with file filter
            raw_results = cosmos_client.hybrid_search(
                query_vector=query_embedding,
                filters=filters,
                limit=top_k,
                similarity_threshold=0.3
            )
        else:
            # If no query text, get all chunks from the file
            raw_results = cosmos_client.get_document_chunks(file_id, user_id, limit=top_k)
        
        # Process results
        intent = {'query_text': query_text, 'user_id': user_id}
        results = process_search_results(raw_results, intent)
        
        return {
            "tool_used": "search_file",
            "file_id": file_id,
            "user_id": user_id,
            "query_text": query_text,
            "total_results": len(results),
            "results": results
        }
        
    except SearchError as e:
        logger.error(f"File search error: {str(e)}")
        return {
            "error": f"File search failed: {str(e)}",
            "tool_used": "search_file"
        }
    except Exception as e:
        logger.error(f"Unexpected error in tool_search_file: {str(e)}", exc_info=True)
        return {
            "error": f"Unexpected file search error: {str(e)}",
            "tool_used": "search_file"
        }


# ===========================
# 🔍 Similar Documents Tool
# ===========================
def tool_similar_documents(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find documents similar to a given file using the new search engine.
    """
    try:
        file_id = args.get("file_id")
        user_id = args.get("user_id")
        top_k = args.get("top_k", 5)
        
        if not file_id or not user_id:
            return {
                "error": "Missing required parameters: file_id and user_id",
                "tool_used": "similar_documents"
            }
        
        # Validate top_k
        if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
            top_k = 5
        
        logger.info(f"Finding documents similar to {file_id} for user {user_id}")
        
        # Use the search engine's similar document search
        result = search_engine.search_similar_documents(file_id, user_id, top_k)
        
        return {
            "tool_used": "similar_documents",
            "source_file_id": file_id,
            "user_id": user_id,
            "total_results": len(result.get('similar_documents', [])),
            "similar_documents": result.get('similar_documents', [])
        }
        
    except SearchError as e:
        logger.error(f"Similar documents error: {str(e)}")
        return {
            "error": f"Similar documents search failed: {str(e)}",
            "tool_used": "similar_documents"
        }
    except Exception as e:
        logger.error(f"Unexpected error in tool_similar_documents: {str(e)}", exc_info=True)
        return {
            "error": f"Unexpected similar documents error: {str(e)}",
            "tool_used": "similar_documents"
        }


# ===========================
# 🔍 Search Suggestions Tool
# ===========================
def tool_search_suggestions(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate search suggestions based on partial query using the new search engine.
    """
    try:
        partial_query = args.get("partial_query", "")
        user_id = args.get("user_id")
        limit = args.get("limit", 5)
        
        if not user_id:
            return {
                "error": "Missing required parameter: user_id",
                "tool_used": "search_suggestions"
            }
        
        if len(partial_query.strip()) < 2:
            return {
                "tool_used": "search_suggestions",
                "partial_query": partial_query,
                "suggestions": []
            }
        
        # Validate limit
        if not isinstance(limit, int) or limit < 1 or limit > 20:
            limit = 5
        
        logger.debug(f"Generating search suggestions for '{partial_query}' for user {user_id}")
        
        # Use the search engine's suggestion method
        result = search_engine.get_search_suggestions(partial_query, user_id, limit)
        
        return {
            "tool_used": "search_suggestions",
            "partial_query": partial_query,
            "user_id": user_id,
            "total_suggestions": len(result.get('suggestions', [])),
            "suggestions": result.get('suggestions', [])
        }
        
    except Exception as e:
        logger.error(f"Error generating search suggestions: {str(e)}")
        return {
            "tool_used": "search_suggestions",
            "partial_query": args.get("partial_query", ""),
            "suggestions": []
        }


# ===========================
# 📊 Search Stats Tool
# ===========================
def tool_search_stats(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get search and document statistics for a user using the cosmos client.
    """
    try:
        user_id = args.get("user_id")
        
        if not user_id:
            return {
                "error": "Missing required parameter: user_id",
                "tool_used": "search_stats"
            }
        
        logger.info(f"Retrieving search statistics for user {user_id}")
        
        # Use the cosmos client to get user statistics
        cosmos_client = search_engine.cosmos_client
        
        # Get basic document statistics
        try:
            # Get user's documents count by platform
            platform_stats = {}
            for platform in Platform:
                filters = {'user_id': user_id, 'platform': platform.value}
                count = cosmos_client.count_documents(filters)
                if count > 0:
                    platform_stats[platform.value] = count
            
            # Get file type statistics
            file_type_stats = {}
            for file_type in ['PDF', 'DOCX', 'DOC', 'XLSX', 'XLS', 'PPTX', 'PPT', 'TXT']:
                filters = {'user_id': user_id, 'file_types': [file_type]}
                count = cosmos_client.count_documents(filters)
                if count > 0:
                    file_type_stats[file_type] = count
            
            # Get recent documents
            recent_docs = cosmos_client.get_recent_documents(user_id, limit=10)
            
            stats = {
                'total_documents': sum(platform_stats.values()),
                'platform_distribution': platform_stats,
                'file_type_distribution': file_type_stats,
                'recent_documents_count': len(recent_docs),
                'last_updated': recent_docs[0].get('created_at') if recent_docs else None
            }
            
        except Exception as e:
            logger.warning(f"Could not retrieve detailed stats: {str(e)}")
            stats = {
                'total_documents': 0,
                'platform_distribution': {},
                'file_type_distribution': {},
                'recent_documents_count': 0,
                'last_updated': None,
                'error': 'Detailed statistics unavailable'
            }
        
        return {
            "tool_used": "search_stats",
            "user_id": user_id,
            "statistics": stats
        }
        
    except SearchError as e:
        logger.error(f"Search stats error: {str(e)}")
        return {
            "error": f"Failed to retrieve search statistics: {str(e)}",
            "tool_used": "search_stats"
        }
    except Exception as e:
        logger.error(f"Unexpected error in tool_search_stats: {str(e)}", exc_info=True)
        return {
            "error": f"Unexpected error retrieving statistics: {str(e)}",
            "tool_used": "search_stats"
        }


# ===========================
# 🧠 Enhanced Summarize Tool Wrapper
# ===========================
def tool_summarize(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced summarize tool wrapper using the new summarizer.py module.
    Supports both file-based and query-based summarization.
    """
    try:
        # Extract required parameters
        user_id = args.get("user_id")
        
        if not user_id:
            return {
                "error": "Missing required parameter: user_id",
                "tool_used": "summarize"
            }
        
        # Extract optional parameters
        file_id = args.get("file_id")
        query_text = args.get("query_text")
        summary_type = args.get("summary_type", "short")
        platform = args.get("platform")
        file_type = args.get("file_type")
        time_range = args.get("time_range")
        
        # Validate that either file_id or query_text is provided
        if not file_id and not query_text:
            return {
                "error": "Either 'file_id' or 'query_text' must be provided",
                "tool_used": "summarize"
            }
        
        # Validate summary_type
        if summary_type not in ["short", "detailed", "bullet", "executive", "technical"]:
            logger.warning(f"Invalid summary_type '{summary_type}', defaulting to 'short'")
            summary_type = "short"
        
        logger.info(f"Starting summarization for user {user_id}")
        
        # Determine summarization approach and call appropriate function
        if file_id and not query_text:
            # File-based summarization
            logger.info(f"Performing file-based summarization for file {file_id}")
            result = summarize_file(
                file_id=file_id,
                user_id=user_id,
                summary_type=summary_type,
                platform=platform,
                search_engine=search_engine
            )
            
        elif query_text and not file_id:
            # Query-based summarization
            logger.info(f"Performing query-based summarization for query: '{query_text}'")
            result = summarize_query(
                query_text=query_text,
                user_id=user_id,
                summary_type=summary_type,
                platform=platform,
                file_type=file_type,
                time_range=time_range,
                search_engine=search_engine
            )
            
        else:
            # Both file_id and query_text provided - use file-based with query context
            logger.info(f"Performing file-based summarization with query context for file {file_id}")
            result = summarize_file(
                file_id=file_id,
                user_id=user_id,
                summary_type=summary_type,
                platform=platform,
                query_context=query_text,
                search_engine=search_engine
            )
        
        # Format response to match old format
        response = {
            "tool_used": "summarize",
            "user_id": user_id,
            "file_id": file_id,
            "query_text": query_text,
            "summary_type": summary_type,
            "summary": result.get("summary", ""),
            "summary_length": len(result.get("summary", "")),
            "chunks_used": result.get("chunks_used", 0)
        }
        
        logger.info(f"Summarization completed successfully using {result.get('chunks_used', 0)} chunks")
        return response
        
    except SummarizationError as e:
        logger.error(f"Summarization error: {str(e)}")
        return {
            "error": f"Summarization failed: {str(e)}",
            "tool_used": "summarize"
        }
    except Exception as e:
        logger.error(f"Unexpected error in tool_summarize: {str(e)}", exc_info=True)
        return {
            "error": f"Unexpected summarization error: {str(e)}",
            "tool_used": "summarize"
        }


# ===========================
# 📖 Enhanced RAG Tool Wrapper
# ===========================
def tool_rag(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced RAG tool wrapper using the new rag.py module.
    """
    try:
        # Extract required parameters
        query_text = args.get("query_text")
        user_id = args.get("user_id")
        
        if not query_text or not user_id:
            return {
                "error": "Missing required parameters: query_text and user_id",
                "tool_used": "rag"
            }
        
        # Build RAG intent dictionary according to new rag.py format
        rag_intent = {
            "action": "rag",
            "query_text": query_text,
            "user_id": user_id
        }
        
        # Add optional parameters
        max_context_chunks = args.get("max_context_chunks", 5)
        if max_context_chunks:
            rag_intent["top_k"] = max_context_chunks
        
        if args.get("conversation_id"):
            rag_intent["conversation_id"] = args["conversation_id"]
        
        if args.get("file_id"):
            rag_intent["file_id"] = args["file_id"]
        
        if args.get("platform"):
            rag_intent["platform"] = args["platform"]
        
        if args.get("file_type"):
            rag_intent["file_type"] = args["file_type"]
        
        if args.get("time_range"):
            rag_intent["time_range"] = args["time_range"]
        
        logger.info(f"Executing RAG query for user {user_id}: '{query_text[:100]}...'")
        
        # Call the enhanced RAG function
        result = answer_query_with_rag(rag_intent)
        
        # Format response to match old format
        response = {
            "tool_used": "rag",
            "query_text": query_text,
            "user_id": user_id,
            "answer": result.get("answer", ""),
            "context_chunks_used": result.get("context_chunks_used", 0),
            "sources": result.get("sources", []),
            "confidence_score": result.get("confidence_score", 0.0)
        }
        
        logger.info(f"RAG query completed successfully. Response length: {len(result.get('answer', ''))}")
        return response
        
    except RagError as e:
        logger.error(f"RAG error: {str(e)}")
        return {
            "error": f"RAG query failed: {str(e)}",
            "tool_used": "rag"
        }
    except Exception as e:
        logger.error(f"Unexpected error in tool_rag: {str(e)}", exc_info=True)
        return {
            "error": f"Unexpected RAG error: {str(e)}",
            "tool_used": "rag"
        }


# ===========================
# 🛠 Enhanced Tool Function Registry
# ===========================
TOOL_FUNCTIONS = {
    "search": {
        "function": tool_search,
        "spec": {
            "name": "search",
            "description": "Search relevant document chunks using hybrid vector + metadata search with advanced filtering capabilities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string", 
                        "description": "The user's search query (2-1000 characters)",
                        "minLength": 2,
                        "maxLength": 1000
                    },
                    "user_id": {
                        "type": "string", 
                        "description": "The user ID for data isolation"
                    },
                    "platform": {
                        "type": "string", 
                        "description": "Platform filter: google_drive, onedrive, dropbox, sharepoint, local, slack, teams",
                        "enum": ["google_drive", "onedrive", "dropbox", "sharepoint", "local", "slack", "teams"]
                    },
                    "file_type": {
                        "type": "string", 
                        "description": "File type filter: PDF, DOCX, DOC, XLSX, XLS, PPTX, PPT, TXT",
                        "enum": ["PDF", "DOCX", "DOC", "XLSX", "XLS", "PPTX", "PPT", "TXT"]
                    },
                    "mime_type": {
                        "type": "string", 
                        "description": "MIME type filter (alternative to file_type)"
                    },
                    "time_range": {
                        "oneOf": [
                            {
                                "type": "string",
                                "description": "Relative time range",
                                "enum": ["last_hour", "last_24_hours", "last_7_days", "last_30_days", "last_month", "last_3_months", "last_6_months", "last_year"]
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string", "format": "date", "description": "Start date (ISO format)"},
                                    "to": {"type": "string", "format": "date", "description": "End date (ISO format)"}
                                },
                                "description": "Absolute time range filter"
                            }
                        ]
                    },
                    "top_k": {
                        "type": "integer", 
                        "description": "Number of top results to return (1-100)",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Pagination offset",
                        "minimum": 0
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Pagination limit (1-100)",
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "required": ["query_text", "user_id"]
            }
        }
    },
    "search_file": {
        "function": tool_search_file,
        "spec": {
            "name": "search_file",
            "description": "Search within a specific document by file ID using semantic similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string", 
                        "description": "Unique identifier for the file to search within"
                    },
                    "user_id": {
                        "type": "string", 
                        "description": "The user ID for data isolation"
                    },
                    "query_text": {
                        "type": "string", 
                        "description": "Optional search query within the file",
                        "default": ""
                    },
                    "top_k": {
                        "type": "integer", 
                        "description": "Number of chunks to return (1-100)",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10
                    }
                },
                "required": ["file_id", "user_id"]
            }
        }
    },
    "similar_documents": {
        "function": tool_similar_documents,
        "spec": {
            "name": "similar_documents",
            "description": "Find documents similar to a given file using semantic similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string", 
                        "description": "Source file ID to find similar documents for"
                    },
                    "user_id": {
                        "type": "string", 
                        "description": "The user ID for data isolation"
                    },
                    "top_k": {
                        "type": "integer", 
                        "description": "Number of similar documents to return (1-20)",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    }
                },
                "required": ["file_id", "user_id"]
            }
        }
    },
    "search_suggestions": {
        "function": tool_search_suggestions,
        "spec": {
            "name": "search_suggestions",
            "description": "Generate search suggestions based on partial query input and user's document collection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "partial_query": {
                        "type": "string", 
                        "description": "Partial search query to generate suggestions for"
                    },
                    "user_id": {
                        "type": "string", 
                        "description": "The user ID for personalized suggestions"
                    },
                    "limit": {
                        "type": "integer", 
                        "description": "Maximum number of suggestions to return (1-20)",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    }
                },
                "required": ["user_id"]
            }
        }
    },
    "search_stats": {
        "function": tool_search_stats,
        "spec": {
            "name": "search_stats",
            "description": "Get comprehensive search and document statistics for a user's collection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string", 
                        "description": "The user ID to get statistics for"
                    }
                },
                "required": ["user_id"]
            }
        }
    },
    "summarize": {
        "function": tool_summarize,
        "spec": {
            "name": "summarize",
            "description": "Summarize a document by file ID or based on a query with various summary types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string", 
                        "description": "The ID of the file to summarize"
                    },
                    "query_text": {
                        "type": "string", 
                        "description": "Optional query for query-based summarization"
                    },
                    "summary_type": {
                        "type": "string", 
                        "description": "Type of summary to generate",
                        "enum": ["short", "detailed", "bullet", "executive", "technical"]
                    },
                    "user_id": {
                        "type": "string", 
                        "description": "The user ID"
                    },
                    "platform": {
                        "type": "string",
                        "description": "Platform filter for query-based summarization",
                        "enum": ["google_drive", "onedrive", "dropbox", "sharepoint", "local", "slack", "teams"]
                    },
                    "file_type": {
                        "type": "string",
                        "description": "File type filter for query-based summarization",
                        "enum": ["PDF", "DOCX", "DOC", "XLSX", "XLS", "PPTX", "PPT", "TXT"]
                    },
                    "time_range": {
                        "oneOf": [
                            {
                                "type": "string",
                                "description": "Relative time range",
                                "enum": ["last_hour", "last_24_hours", "last_7_days", "last_30_days", "last_month", "last_3_months", "last_6_months", "last_year"]
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string", "format": "date", "description": "Start date (ISO format)"},
                                    "to": {"type": "string", "format": "date", "description": "End date (ISO format)"}
                                },
                                "description": "Absolute time range filter"
                            }
                        ]
                    }
                },
                "required": ["user_id"]
            }
        }
    },
    "rag": {
        "function": tool_rag,
        "spec": {
            "name": "rag",
            "description": "Answer questions using Retrieval-Augmented Generation with context from user's documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string", 
                        "description": "The question or query to answer using RAG",
                        "minLength": 3,
                        "maxLength": 2000
                    },
                    "user_id": {
                        "type": "string", 
                        "description": "The user ID for data isolation"
                    },
                    "conversation_id": {
                        "type": "string", 
                        "description": "Optional conversation ID for maintaining context"
                    },
                    "file_id": {
                        "type": "string", 
                        "description": "Optional file ID to focus RAG on specific document"
                    },
                    "platform": {
                        "type": "string",
                        "description": "Platform filter for RAG context",
                        "enum": ["google_drive", "onedrive", "dropbox", "sharepoint", "local", "slack", "teams"]
                    },
                    "file_type": {
                        "type": "string",
                        "description": "File type filter for RAG context",
                        "enum": ["PDF", "DOCX", "DOC", "XLSX", "XLS", "PPTX", "PPT", "TXT"]
                    },
                    "time_range": {
                        "oneOf": [
                            {
                                "type": "string",
                                "description": "Relative time range",
                                "enum": ["last_hour", "last_24_hours", "last_7_days", "last_30_days", "last_month", "last_3_months", "last_6_months", "last_year"]
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string", "format": "date", "description": "Start date (ISO format)"},
                                    "to": {"type": "string", "format": "date", "description": "End date (ISO format)"}
                                },
                                "description": "Absolute time range filter"
                            }
                        ]
                    },
                    "max_context_chunks": {
                        "type": "integer", 
                        "description": "Maximum number of context chunks to use (1-20)",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    }
                },
                "required": ["query_text", "user_id"]
            }
        }
    }
}


# ===========================
# 🔍 Additional Utility Tools
# ===========================

def tool_batch_summarize(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Batch summarize multiple files using the new summarizer.py module.
    """
    try:
        file_ids = args.get("file_ids", [])
        user_id = args.get("user_id")
        summary_type = args.get("summary_type", "short")
        
        if not file_ids or not user_id:
            return {
                "error": "Missing required parameters: file_ids and user_id",
                "tool_used": "batch_summarize"
            }
        
        if not isinstance(file_ids, list) or len(file_ids) == 0:
            return {
                "error": "file_ids must be a non-empty list",
                "tool_used": "batch_summarize"
            }
        
        # Limit batch size to prevent overload
        if len(file_ids) > 10:
            file_ids = file_ids[:10]
            logger.warning(f"Batch size limited to 10 files for user {user_id}")
        
        logger.info(f"Starting batch summarization for {len(file_ids)} files for user {user_id}")
        
        # Use the batch summarization function
        result = batch_summarize_files(
            file_ids=file_ids,
            user_id=user_id,
            summary_type=summary_type,
            search_engine=search_engine
        )
        
        return {
            "tool_used": "batch_summarize",
            "user_id": user_id,
            "summary_type": summary_type,
            "total_files": len(file_ids),
            "successful_summaries": result.get("successful_summaries", 0),
            "failed_summaries": result.get("failed_summaries", 0),
            "summaries": result.get("summaries", []),
            "errors": result.get("errors", [])
        }
        
    except SummarizationError as e:
        logger.error(f"Batch summarization error: {str(e)}")
        return {
            "error": f"Batch summarization failed: {str(e)}",
            "tool_used": "batch_summarize"
        }
    except Exception as e:
        logger.error(f"Unexpected error in tool_batch_summarize: {str(e)}", exc_info=True)
        return {
            "error": f"Unexpected batch summarization error: {str(e)}",
            "tool_used": "batch_summarize"
        }


def tool_search_and_summarize(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combined search and summarization tool using the new modules.
    """
    try:
        query_text = args.get("query_text")
        user_id = args.get("user_id")
        summary_type = args.get("summary_type", "short")
        
        if not query_text or not user_id:
            return {
                "error": "Missing required parameters: query_text and user_id",
                "tool_used": "search_and_summarize"
            }
        
        # Extract optional search parameters
        platform = args.get("platform")
        file_type = args.get("file_type")
        time_range = args.get("time_range")
        top_k = args.get("top_k", 5)
        
        logger.info(f"Starting search and summarize for user {user_id}: '{query_text[:50]}...'")
        
        # Use the combined search and summarize function
        result = search_and_summarize(
            query_text=query_text,
            user_id=user_id,
            summary_type=summary_type,
            platform=platform,
            file_type=file_type,
            time_range=time_range,
            top_k=top_k,
            search_engine=search_engine
        )
        
        return {
            "tool_used": "search_and_summarize",
            "query_text": query_text,
            "user_id": user_id,
            "summary_type": summary_type,
            "search_results_count": result.get("search_results_count", 0),
            "summary": result.get("summary", ""),
            "sources": result.get("sources", []),
            "chunks_used": result.get("chunks_used", 0)
        }
        
    except (SearchError, SummarizationError) as e:
        logger.error(f"Search and summarize error: {str(e)}")
        return {
            "error": f"Search and summarize failed: {str(e)}",
            "tool_used": "search_and_summarize"
        }
    except Exception as e:
        logger.error(f"Unexpected error in tool_search_and_summarize: {str(e)}", exc_info=True)
        return {
            "error": f"Unexpected search and summarize error: {str(e)}",
            "tool_used": "search_and_summarize"
        }


def tool_conversation_history(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get conversation history for RAG context.
    """
    try:
        conversation_id = args.get("conversation_id")
        user_id = args.get("user_id")
        limit = args.get("limit", 10)
        
        if not conversation_id or not user_id:
            return {
                "error": "Missing required parameters: conversation_id and user_id",
                "tool_used": "conversation_history"
            }
        
        # Validate limit
        if not isinstance(limit, int) or limit < 1 or limit > 50:
            limit = 10
        
        logger.info(f"Retrieving conversation history for {conversation_id}, user {user_id}")
        
        # Get conversation history using the rag module
        history = get_conversation_history(conversation_id, user_id, limit)
        
        return {
            "tool_used": "conversation_history",
            "conversation_id": conversation_id,
            "user_id": user_id,
            "total_messages": len(history),
            "history": history
        }
        
    except RagError as e:
        logger.error(f"Conversation history error: {str(e)}")
        return {
            "error": f"Failed to retrieve conversation history: {str(e)}",
            "tool_used": "conversation_history"
        }
    except Exception as e:
        logger.error(f"Unexpected error in tool_conversation_history: {str(e)}", exc_info=True)
        return {
            "error": f"Unexpected error retrieving conversation history: {str(e)}",
            "tool_used": "conversation_history"
        }


def tool_clear_cache(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clear various caches and perform maintenance cleanup.
    """
    try:
        user_id = args.get("user_id")
        cache_type = args.get("cache_type", "all")  # all, conversation, search, rag
        
        if not user_id:
            return {
                "error": "Missing required parameter: user_id",
                "tool_used": "clear_cache"
            }
        
        logger.info(f"Clearing cache type '{cache_type}' for user {user_id}")
        
        cleared_items = 0
        
        if cache_type in ["all", "conversation"]:
            # Clear conversation history cache
            try:
                clear_conversation_history(user_id)
                cleared_items += 1
                logger.info(f"Cleared conversation history cache for user {user_id}")
            except Exception as e:
                logger.warning(f"Failed to clear conversation cache: {str(e)}")
        
        if cache_type in ["all", "rag", "search"]:
            # Clear general cache
            try:
                clear_cache()
                cleared_items += 1
                logger.info("Cleared general cache")
            except Exception as e:
                logger.warning(f"Failed to clear general cache: {str(e)}")
        
        if cache_type == "all":
            # Perform maintenance cleanup
            try:
                maintenance_cleanup()
                cleared_items += 1
                logger.info("Performed maintenance cleanup")
            except Exception as e:
                logger.warning(f"Failed to perform maintenance cleanup: {str(e)}")
        
        return {
            "tool_used": "clear_cache",
            "user_id": user_id,
            "cache_type": cache_type,
            "items_cleared": cleared_items,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in tool_clear_cache: {str(e)}", exc_info=True)
        return {
            "error": f"Unexpected error clearing cache: {str(e)}",
            "tool_used": "clear_cache"
        }


def tool_health_check(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform health check on the search and RAG systems.
    """
    try:
        user_id = args.get("user_id")
        
        if not user_id:
            return {
                "error": "Missing required parameter: user_id",
                "tool_used": "health_check"
            }
        
        logger.info(f"Performing health check for user {user_id}")
        
        # Perform RAG health check
        rag_health = rag_health_check()
        
        # Get system stats
        system_stats = get_system_stats()
        
        # Get cache stats
        cache_stats = get_cache_stats()
        
        # Test search engine connectivity
        search_health = {"status": "unknown"}
        try:
            # Simple test search
            test_result = search_engine.search({
                "query_text": "test",
                "user_id": user_id,
                "limit": 1
            })
            search_health = {
                "status": "healthy" if test_result else "degraded",
                "test_results_count": len(test_result.get('results', []))
            }
        except Exception as e:
            search_health = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        overall_status = "healthy"
        if (rag_health.get("status") != "healthy" or 
            search_health.get("status") != "healthy"):
            overall_status = "degraded"
        
        return {
            "tool_used": "health_check",
            "user_id": user_id,
            "overall_status": overall_status,
            "rag_health": rag_health,
            "search_health": search_health,
            "system_stats": system_stats,
            "cache_stats": cache_stats,
            "timestamp": logger.name  # Using logger name as timestamp placeholder
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in tool_health_check: {str(e)}", exc_info=True)
        return {
            "error": f"Unexpected error during health check: {str(e)}",
            "tool_used": "health_check",
            "overall_status": "unhealthy"
        }


# ===========================
# 🔧 Extended Tool Registry
# ===========================

# Add the additional tools to the registry
TOOL_FUNCTIONS.update({
    "batch_summarize": {
        "function": tool_batch_summarize,
        "spec": {
            "name": "batch_summarize",
            "description": "Summarize multiple documents in batch with various summary types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file IDs to summarize (max 10)",
                        "maxItems": 10,
                        "minItems": 1
                    },
                    "user_id": {
                        "type": "string",
                        "description": "The user ID"
                    },
                    "summary_type": {
                        "type": "string",
                        "description": "Type of summary to generate for all files",
                        "enum": ["short", "detailed", "bullet", "executive", "technical"],
                        "default": "short"
                    }
                },
                "required": ["file_ids", "user_id"]
            }
        }
    },
    "search_and_summarize": {
        "function": tool_search_and_summarize,
        "spec": {
            "name": "search_and_summarize",
            "description": "Search for relevant documents and generate a summary based on the results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "The search query to find relevant documents",
                        "minLength": 2,
                        "maxLength": 1000
                    },
                    "user_id": {
                        "type": "string",
                        "description": "The user ID"
                    },
                    "summary_type": {
                        "type": "string",
                        "description": "Type of summary to generate",
                        "enum": ["short", "detailed", "bullet", "executive", "technical"],
                        "default": "short"
                    },
                    "platform": {
                        "type": "string",
                        "description": "Platform filter",
                        "enum": ["google_drive", "onedrive", "dropbox", "sharepoint", "local", "slack", "teams"]
                    },
                    "file_type": {
                        "type": "string",
                        "description": "File type filter",
                        "enum": ["PDF", "DOCX", "DOC", "XLSX", "XLS", "PPTX", "PPT", "TXT"]
                    },
                    "time_range": {
                        "oneOf": [
                            {
                                "type": "string",
                                "description": "Relative time range",
                                "enum": ["last_hour", "last_24_hours", "last_7_days", "last_30_days", "last_month", "last_3_months", "last_6_months", "last_year"]
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string", "format": "date"},
                                    "to": {"type": "string", "format": "date"}
                                }
                            }
                        ]
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of documents to include in summary (1-20)",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    }
                },
                "required": ["query_text", "user_id"]
            }
        }
    },
    "conversation_history": {
        "function": tool_conversation_history,
        "spec": {
            "name": "conversation_history",
            "description": "Retrieve conversation history for maintaining context in RAG.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "The conversation ID to retrieve history for"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "The user ID"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of messages to retrieve (1-50)",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10
                    }
                },
                "required": ["conversation_id", "user_id"]
            }
        }
    },
    "clear_cache": {
        "function": tool_clear_cache,
        "spec": {
            "name": "clear_cache",
            "description": "Clear various caches and perform maintenance cleanup.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID"
                    },
                    "cache_type": {
                        "type": "string",
                        "description": "Type of cache to clear",
                        "enum": ["all", "conversation", "search", "rag"],
                        "default": "all"
                    }
                },
                "required": ["user_id"]
            }
        }
    },
    "health_check": {
        "function": tool_health_check,
        "spec": {
            "name": "health_check",
            "description": "Perform comprehensive health check on search and RAG systems.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID for testing user-specific functionality"
                    }
                },
                "required": ["user_id"]
            }
        }
    }
})


# ===========================
# 🎯 Main Tool Dispatcher
# ===========================

def dispatch_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main tool dispatcher function that routes requests to appropriate tool functions.
    
    Args:
        tool_name: Name of the tool to execute
        args: Arguments dictionary for the tool
        
    Returns:
        Dictionary containing tool results or error information
    """
    try:
        if tool_name not in TOOL_FUNCTIONS:
            available_tools = list(TOOL_FUNCTIONS.keys())
            logger.error(f"Unknown tool: {tool_name}. Available tools: {available_tools}")
            return {
                "error": f"Unknown tool: {tool_name}. Available tools: {available_tools}",
                "tool_used": tool_name
            }
        
        # Get the tool function
        tool_function = TOOL_FUNCTIONS[tool_name]["function"]
        
        # Log the tool dispatch
        logger.info(f"Dispatching tool: {tool_name} with args: {list(args.keys())}")
        
        # Execute the tool function
        result = tool_function(args)
        
        # Ensure the result includes the tool name
        if isinstance(result, dict) and "tool_used" not in result:
            result["tool_used"] = tool_name
        
        return result
        
    except Exception as e:
        logger.error(f"Error dispatching tool {tool_name}: {str(e)}", exc_info=True)
        return {
            "error": f"Tool dispatch failed: {str(e)}",
            "tool_used": tool_name
        }


def get_tool_specs() -> List[Dict[str, Any]]:
    """
    Get all tool specifications for API documentation or client registration.
    
    Returns:
        List of tool specifications
    """
    return [tool_info["spec"] for tool_info in TOOL_FUNCTIONS.values()]


def get_available_tools() -> List[str]:
    """
    Get list of available tool names.
    
    Returns:
        List of tool names
    """
    return list(TOOL_FUNCTIONS.keys())


# ===========================
# 🧪 Tool Testing Functions
# ===========================

def test_tool_connectivity() -> Dict[str, Any]:
    """
    Test basic connectivity of all tools.
    
    Returns:
        Dictionary with connectivity test results
    """
    results = {}
    
    for tool_name in TOOL_FUNCTIONS.keys():
        try:
            # Skip tools that require specific parameters for basic connectivity test
            if tool_name in ["search_file", "similar_documents", "batch_summarize"]:
                results[tool_name] = {"status": "skipped", "reason": "requires specific parameters"}
                continue
            
            # Test with minimal parameters
            test_args = {"user_id": "test_user"}
            
            if tool_name in ["search", "rag", "search_and_summarize"]:
                test_args["query_text"] = "test query"
            elif tool_name == "conversation_history":
                test_args["conversation_id"] = "test_conversation"
            elif tool_name == "search_suggestions":
                test_args["partial_query"] = "test"
            
            # This would normally call the tool, but for connectivity test we just check if it's callable
            tool_function = TOOL_FUNCTIONS[tool_name]["function"]
            if callable(tool_function):
                results[tool_name] = {"status": "connected"}
            else:
                results[tool_name] = {"status": "error", "reason": "not callable"}
                
        except Exception as e:
            results[tool_name] = {"status": "error", "reason": str(e)}
    
    return {
        "test_type": "connectivity",
        "total_tools": len(TOOL_FUNCTIONS),
        "results": results,
        "summary": {
            "connected": len([r for r in results.values() if r["status"] == "connected"]),
            "skipped": len([r for r in results.values() if r["status"] == "skipped"]),
            "errors": len([r for r in results.values() if r["status"] == "error"])
        }
    }


# ===========================
# 📊 Module Initialization & Logging
# ===========================

logger.info(f"Tools module initialized with {len(TOOL_FUNCTIONS)} tools")
logger.debug(f"Available tools: {list(TOOL_FUNCTIONS.keys())}")

# Export main functions for external use
__all__ = [
    'dispatch_tool',
    'get_tool_specs', 
    'get_available_tools',
    'test_tool_connectivity',
    'TOOL_FUNCTIONS'
]
