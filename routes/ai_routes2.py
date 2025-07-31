import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List, Any
from openai import AzureOpenAI

from ai_layer.brain2 import initialize_brain, reason_and_act  # Central decision engine

# Import the new unified tool system
from ai_layer.tools2 import (
    dispatch_tool,
    get_tool_specs,
    get_available_tools,
    test_tool_connectivity,
    TOOL_FUNCTIONS
)

router = APIRouter()

# Initialize the brain on module load
def _initialize_brain_if_needed():
    """Initialize the brain with Azure OpenAI client if not already done."""
    try:
        # Check if brain is already initialized by attempting a dummy call
        # This will raise RuntimeError if not initialized
        from ai_layer.brain import _brain_instance
        if _brain_instance is None:
            # Get Azure OpenAI credentials from environment variables
            api_key = os.getenv("OPENAI_API_KEY")
            endpoint = os.getenv("OPENAI_ENDPOINT", "https://weez-openai-resource.openai.azure.com/")
            api_version = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")
            chat_deployment = os.getenv("OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

            if not api_key or not endpoint:
                raise ValueError("Missing Azure OpenAI credentials. Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.")
            
            # Create Azure OpenAI client
            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version
            )
            
            # Initialize the brain
            initialize_brain(
                azure_openai_client=client,
                chat_deployment=chat_deployment
            )
            
    except Exception as e:
        raise RuntimeError(f"Failed to initialize brain: {str(e)}")

# Initialize brain when module is imported
_initialize_brain_if_needed()


# === Pydantic Schemas ===

class IntentRequest(BaseModel):
    action: str
    query_text: Optional[str] = None
    file_id: Optional[str] = None
    user_id: str
    summary_type: Optional[str] = "short"
    platform: Optional[str] = None
    time_range: Optional[Dict[str, str]] = None
    top_k: Optional[int] = 10
    file_type: Optional[str] = None
    mime_type: Optional[str] = None
    offset: Optional[int] = None
    limit: Optional[int] = None
    conversation_id: Optional[str] = None
    max_context_chunks: Optional[int] = 5


class QueryRequest(BaseModel):
    query: str
    user_id: str


class SearchRequest(BaseModel):
    query_text: str
    user_id: str
    platform: Optional[str] = None
    file_type: Optional[str] = None
    mime_type: Optional[str] = None
    time_range: Optional[str] = None
    top_k: Optional[int] = 10
    offset: Optional[int] = None
    limit: Optional[int] = None


class FileSearchRequest(BaseModel):
    file_id: str
    user_id: str
    query_text: Optional[str] = ""
    top_k: Optional[int] = 10


class SimilarDocumentsRequest(BaseModel):
    file_id: str
    user_id: str
    top_k: Optional[int] = 5


class SearchSuggestionsRequest(BaseModel):
    partial_query: str
    user_id: str
    limit: Optional[int] = 5


class BatchSummarizeRequest(BaseModel):
    file_ids: List[str]
    user_id: str
    summary_type: Optional[str] = "short"


class SearchAndSummarizeRequest(BaseModel):
    query_text: str
    user_id: str
    summary_type: Optional[str] = "short"
    platform: Optional[str] = None
    file_type: Optional[str] = None
    time_range: Optional[str] = None
    top_k: Optional[int] = 5


class ConversationHistoryRequest(BaseModel):
    conversation_id: str
    user_id: str
    limit: Optional[int] = 10


class ClearCacheRequest(BaseModel):
    user_id: str
    cache_type: Optional[str] = "all"


class ToolRequest(BaseModel):
    tool_name: str
    args: Dict[str, Any]


# === Legacy Routes (Backward Compatibility) ===

@router.post("/summarize")
async def summarize_endpoint(intent: IntentRequest):
    """
    Summarize a document or based on query intent.
    """
    try:
        # Convert intent to tool arguments
        args = intent.dict()
        
        # Use the unified tool system
        result = dispatch_tool("summarize", args)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")


@router.post("/rag")
async def rag_endpoint(intent: IntentRequest):
    """
    Answer a question using RAG over stored document chunks.
    """
    try:
        # Convert intent to tool arguments
        args = intent.dict()
        
        # Use the unified tool system
        result = dispatch_tool("rag", args)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")


@router.post("/ask")
async def ask_endpoint(request: QueryRequest):
    """
    Ask the AI agent a question using the ReAct reasoning engine.
    """
    try:
        # Ensure brain is initialized before processing request
        _initialize_brain_if_needed()
        
        response = reason_and_act(user_id=request.user_id, user_input=request.query)
        return {"response": response, "user_id": request.user_id}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Brain initialization error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Agent error: {str(e)}")


# === Enhanced Search Routes ===

@router.post("/search")
async def search_endpoint(request: SearchRequest):
    """
    Search for documents using semantic similarity and metadata filters.
    """
    try:
        # Convert request to tool arguments
        args = request.dict(exclude_none=True)
        
        # Use the unified tool system
        result = dispatch_tool("search", args)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "results": result.get("results", []),
            "total_found": result.get("total_results", 0),
            "query": result.get("query_text", request.query_text),
            "user_id": result.get("user_id", request.user_id),
            "filters_applied": result.get("filters_applied", {}),
            "search_metrics": result.get("search_metrics", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal search error: {str(e)}")


@router.post("/search/file")
async def search_file_endpoint(request: FileSearchRequest):
    """
    Search within a specific document by file ID.
    """
    try:
        # Convert request to tool arguments
        args = request.dict(exclude_none=True)
        
        # Use the unified tool system
        result = dispatch_tool("search_file", args)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "results": result.get("results", []),
            "total_found": result.get("total_results", 0),
            "file_id": result.get("file_id", request.file_id),
            "query": result.get("query_text", request.query_text),
            "user_id": result.get("user_id", request.user_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal file search error: {str(e)}")


@router.post("/search/similar")
async def similar_documents_endpoint(request: SimilarDocumentsRequest):
    """
    Find documents similar to a given file.
    """
    try:
        # Convert request to tool arguments
        args = request.dict(exclude_none=True)
        
        # Use the unified tool system
        result = dispatch_tool("similar_documents", args)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "similar_documents": result.get("similar_documents", []),
            "total_found": result.get("total_results", 0),
            "source_file_id": result.get("source_file_id", request.file_id),
            "user_id": result.get("user_id", request.user_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal similar documents error: {str(e)}")


@router.post("/search/suggestions")
async def search_suggestions_endpoint(request: SearchSuggestionsRequest):
    """
    Get search suggestions based on partial query input.
    """
    try:
        # Convert request to tool arguments
        args = request.dict(exclude_none=True)
        
        # Use the unified tool system
        result = dispatch_tool("search_suggestions", args)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "suggestions": result.get("suggestions", []),
            "partial_query": result.get("partial_query", request.partial_query),
            "user_id": result.get("user_id", request.user_id),
            "total_suggestions": result.get("total_suggestions", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal suggestions error: {str(e)}")


@router.get("/search/stats/{user_id}")
async def search_stats_endpoint(user_id: str):
    """
    Get search and document statistics for a user.
    """
    try:
        # Use the unified tool system
        result = dispatch_tool("search_stats", {"user_id": user_id})
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "stats": result.get("statistics", {}),
            "user_id": result.get("user_id", user_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal stats error: {str(e)}")


# === New Enhanced Routes ===

@router.post("/batch/summarize")
async def batch_summarize_endpoint(request: BatchSummarizeRequest):
    """
    Batch summarize multiple documents.
    """
    try:
        # Convert request to tool arguments
        args = request.dict(exclude_none=True)
        
        # Use the unified tool system
        result = dispatch_tool("batch_summarize", args)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch summarization error: {str(e)}")


@router.post("/search-and-summarize")
async def search_and_summarize_endpoint(request: SearchAndSummarizeRequest):
    """
    Search for relevant documents and generate a summary.
    """
    try:
        # Convert request to tool arguments
        args = request.dict(exclude_none=True)
        
        # Use the unified tool system
        result = dispatch_tool("search_and_summarize", args)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search and summarize error: {str(e)}")


@router.post("/conversation/history")
async def conversation_history_endpoint(request: ConversationHistoryRequest):
    """
    Get conversation history for maintaining context.
    """
    try:
        # Convert request to tool arguments
        args = request.dict(exclude_none=True)
        
        # Use the unified tool system
        result = dispatch_tool("conversation_history", args)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversation history error: {str(e)}")


@router.post("/cache/clear")
async def clear_cache_endpoint(request: ClearCacheRequest):
    """
    Clear various caches and perform maintenance.
    """
    try:
        # Convert request to tool arguments
        args = request.dict(exclude_none=True)
        
        # Use the unified tool system
        result = dispatch_tool("clear_cache", args)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear error: {str(e)}")


# === Generic Tool Endpoint ===

@router.post("/tool")
async def generic_tool_endpoint(request: ToolRequest):
    """
    Generic endpoint for executing any available tool.
    """
    try:
        # Validate tool name
        available_tools = get_available_tools()
        if request.tool_name not in available_tools:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown tool: {request.tool_name}. Available tools: {available_tools}"
            )
        
        # Use the unified tool system
        result = dispatch_tool(request.tool_name, request.args)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool execution error: {str(e)}")


# === System and Admin Routes ===

@router.get("/tools")
async def list_tools_endpoint():
    """
    List all available tools and their specifications.
    """
    try:
        return {
            "available_tools": get_available_tools(),
            "total_tools": len(get_available_tools()),
            "tool_specs": get_tool_specs()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing tools: {str(e)}")


@router.get("/tools/connectivity")
async def test_tools_connectivity_endpoint():
    """
    Test connectivity of all tools.
    """
    try:
        result = test_tool_connectivity()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connectivity test error: {str(e)}")


@router.get("/health")
async def health():
    """
    Check the health status of the entire AI system.
    """
    try:
        # Use the health check tool with a system user
        result = dispatch_tool("health_check", {"user_id": "system"})
        
        # If there's an error, still return some basic health info
        if "error" in result:
            return {
                "overall_status": "degraded",
                "error": result["error"],
                "tools_available": len(get_available_tools()),
                "brain_initialized": True  # If we got here, brain is initialized
            }
        
        # Enhance the health check result with additional system info
        result.update({
            "tools_available": len(get_available_tools()),
            "brain_initialized": True,
            "api_version": "2.0",  # Version of the enhanced API
            "features": [
                "unified_tool_system",
                "enhanced_search",
                "batch_operations",
                "conversation_context",
                "cache_management",
                "health_monitoring"
            ]
        })
        
        return result
        
    except Exception as e:
        return {
            "overall_status": "unhealthy",
            "error": f"Health check failed: {str(e)}",
            "tools_available": len(get_available_tools()) if get_available_tools() else 0,
            "brain_initialized": False
        }


@router.get("/version")
async def version_endpoint():
    """
    Get API version and feature information.
    """
    return {
        "api_version": "2.0",
        "description": "Enhanced AI Layer API with unified tool system",
        "features": {
            "unified_tool_system": "All AI operations through consistent tool interface",
            "enhanced_search": "Advanced semantic search with filtering",
            "batch_operations": "Batch summarization and processing",
            "conversation_context": "Conversation history management",
            "cache_management": "Intelligent caching and cleanup",
            "health_monitoring": "Comprehensive system health checks",
            "backward_compatibility": "Legacy endpoint support"
        },
        "available_tools": get_available_tools(),
        "total_tools": len(get_available_tools())
    }


# === Error Handlers ===

