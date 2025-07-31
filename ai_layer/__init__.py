# ai_layer/__init__.py

from ai_layer.intent_parser import parse_user_intent
from utils.openai_client import get_embedding
from ai_layer.search import (
    #search_documents_by_similarity,
    #get_document_by_id,
    #search_within_file,
    search_documents
)
from ai_layer.summarizer import (
    summarize_document,
    summarize_file,
    summarize_query,
)
from ai_layer.rag import answer_query_with_rag

__all__ = [
    "parse_user_intent",
    "get_embedding",
    #"search_documents_by_similarity",
    #"get_document_by_id",
    #"search_within_file",
    "search_documents",
    "summarize_document",
    "summarize_file",
    "summarize_query",
    "answer_query_with_rag",
]
