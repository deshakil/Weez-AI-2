import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from azure.cosmos import CosmosClient, exceptions
from azure.cosmos.partition_key import PartitionKey

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CosmosVectorClient:
    """
    Azure Cosmos DB client with native vector search capabilities
    for AI assistant file chunk retrieval and similarity search.
    """
    
    def __init__(self, cosmos_endpoint: str = None, cosmos_key: str = None, 
                 database_name: str = "weezyai", container_name: str = "files"):
        """
        Initialize Cosmos DB client with vector search capabilities
        
        Args:
            cosmos_endpoint: Azure Cosmos DB endpoint URL
            cosmos_key: Azure Cosmos DB primary key
            database_name: Name of the database
            container_name: Name of the container
        """
        self.cosmos_endpoint = cosmos_endpoint or os.getenv('COSMOS_ENDPOINT')
        self.cosmos_key = cosmos_key or os.getenv('COSMOS_KEY')
        self.database_name = database_name
        self.container_name = container_name
        
        if not self.cosmos_endpoint or not self.cosmos_key:
            raise ValueError("Cosmos DB endpoint and key must be provided")
        
        # Initialize Cosmos client
        self.client = CosmosClient(self.cosmos_endpoint, self.cosmos_key)
        self.database = self.client.get_database_client(self.database_name)
        self.container = self.database.get_container_client(self.container_name)
        
        # Vector search configuration
        self.vector_embedding_policy = {
            "vectorEmbeddings": [
                {
                    "path": "/embedding",
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": 3072  # OpenAI text-embedding-3-large dimensions
                }
            ]
        }
        
        # File type mappings for filtering
        self.file_type_mappings = {
            'PDF': 'application/pdf',
            'DOCX': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'DOC': 'application/msword',
            'XLSX': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'XLS': 'application/vnd.ms-excel',
            'PPTX': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'PPT': 'application/vnd.ms-powerpoint',
            'TXT': 'text/plain'
        }
        
        logger.info(f"Initialized CosmosVectorClient for database: {database_name}, container: {container_name}")

    def _build_filter_conditions(self, filters: Dict[str, Any]) -> tuple:
        """
        Build SQL WHERE conditions and parameters from filters
        
        Args:
            filters: Dictionary containing filter criteria
            
        Returns:
            Tuple of (conditions_list, parameters_list)
        """
        conditions = []
        parameters = []
        
        # User ID filter (required for partition key efficiency)
        if 'user_id' in filters and filters['user_id']:
            conditions.append("c.user_id = @user_id")
            parameters.append({"name": "@user_id", "value": filters['user_id']})
        
        # Platform filter
        if 'platforms' in filters and filters['platforms']:
            platform_conditions = []
            for i, platform in enumerate(filters['platforms']):
                param_name = f"@platform_{i}"
                platform_conditions.append(f"c.platform = {param_name}")
                parameters.append({"name": param_name, "value": platform})
            
            if platform_conditions:
                conditions.append(f"({' OR '.join(platform_conditions)})")
        
        # File type filter (converted to MIME types)
        if 'file_types' in filters and filters['file_types']:
            mime_conditions = []
            for i, file_type in enumerate(filters['file_types']):
                mime_type = self.file_type_mappings.get(file_type.upper())
                if mime_type:
                    param_name = f"@mime_type_{i}"
                    mime_conditions.append(f"c.mime_type = {param_name}")
                    parameters.append({"name": param_name, "value": mime_type})
            
            if mime_conditions:
                conditions.append(f"({' OR '.join(mime_conditions)})")
        
        # Time range filter
        if 'time_range' in filters and filters['time_range']:
            try:
                # Validate ISO8601 format
                datetime.fromisoformat(filters['time_range'].replace('Z', '+00:00'))
                conditions.append("c.created_at >= @time_range")
                parameters.append({"name": "@time_range", "value": filters['time_range']})
            except ValueError:
                logger.warning(f"Invalid time_range format: {filters['time_range']}")
        
        # Additional filters for chunk documents
        conditions.append("IS_DEFINED(c.chunk_index)")  # Ensure it's a chunk document
        conditions.append("IS_DEFINED(c.embedding)")    # Ensure it has embedding
        conditions.append("IS_ARRAY(c.embedding)")      # Ensure embedding is an array
        
        return conditions, parameters

    def vector_search_cosmos(self, embedding: List[float], filters: Dict[str, Any], 
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform native vector similarity search with filters
        
        Args:
            embedding: Query embedding vector (list of floats)
            filters: Dictionary with filter criteria:
                - user_id: string (required for optimal performance)
                - platforms: list of strings (e.g., ["google_drive", "local"])
                - file_types: list of strings (e.g., ["PDF", "DOCX"])
                - time_range: ISO8601 string for created_at >= filtering
            top_k: Number of top results to return
            
        Returns:
            List of matching chunk documents with similarity scores
        """
        try:
            # Validate inputs
            if not embedding or not isinstance(embedding, list):
                raise ValueError("Embedding must be a non-empty list of floats")
            
            if not isinstance(filters, dict):
                raise ValueError("Filters must be a dictionary")
            
            if top_k <= 0 or top_k > 100:
                raise ValueError("top_k must be between 1 and 100")
            
            # Build filter conditions
            conditions, parameters = self._build_filter_conditions(filters)
            
            # Construct the SQL query with vector search
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            # Use native vector search with VECTOR function
            query = f"""
                SELECT 
                    c.id,
                    c.file_id,
                    c.fileName,
                    c.chunk_index,
                    c.text,
                    c.platform,
                    c.mime_type,
                    c.created_at,
                    c.metadata,
                    VectorDistance(c.embedding, @embedding) AS score
                FROM c
                WHERE {where_clause}
                ORDER BY VectorDistance(c.embedding, @embedding)
                OFFSET 0 LIMIT @top_k
            """
            
            # Add embedding parameter
            parameters.append({"name": "@embedding", "value": embedding})
            parameters.append({"name": "@top_k", "value": top_k})
            
            logger.info(f"Executing vector search query with {len(parameters)} parameters")
            logger.debug(f"Query: {query}")
            
            # Execute query with vector search enabled
            results = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
                max_item_count=top_k
            ))
            
            # Process results
            processed_results = []
            for item in results:
                # Extract relevant fields and clean up
                result = {
                    'id': item.get('id'),
                    'file_id': item.get('file_id'),
                    'fileName': item.get('fileName'),
                    'chunk_index': item.get('chunk_index', 0),
                    'text': item.get('text', ''),
                    'platform': item.get('platform'),
                    'mime_type': item.get('mime_type'),
                    'created_at': item.get('created_at'),
                    'score': item.get('score', 0.0),
                    'metadata': item.get('metadata', {})
                }
                
                # Truncate text for response efficiency
                if len(result['text']) > 500:
                    result['text'] = result['text'][:500] + '...'
                
                processed_results.append(result)
            
            logger.info(f"Vector search returned {len(processed_results)} results")
            return processed_results
            
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Cosmos DB HTTP error during vector search: {e}")
            logger.error(f"Status code: {e.status_code}, Message: {e.message}")
            return []
        except exceptions.CosmosResourceNotFoundError as e:
            logger.error(f"Cosmos DB resource not found: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during vector search: {str(e)}")
            return []

    def get_chunk_by_id(self, chunk_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by its ID
        
        Args:
            chunk_id: The chunk document ID
            user_id: User ID (partition key)
            
        Returns:
            Chunk document or None if not found
        """
        try:
            response = self.container.read_item(
                item=chunk_id,
                partition_key=user_id
            )
            return response
        except exceptions.CosmosResourceNotFoundError:
            logger.warning(f"Chunk not found: {chunk_id} for user: {user_id}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {str(e)}")
            return None

    def get_file_chunks(self, file_id: str, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific file
        
        Args:
            file_id: The file identifier
            user_id: User ID (partition key)
            
        Returns:
            List of chunk documents for the file
        """
        try:
            query = """
                SELECT c.id, c.chunk_index, c.text, c.metadata, c.created_at
                FROM c
                WHERE c.user_id = @user_id 
                AND c.file_id = @file_id
                AND IS_DEFINED(c.chunk_index)
                ORDER BY c.chunk_index
            """
            
            parameters = [
                {"name": "@user_id", "value": user_id},
                {"name": "@file_id", "value": file_id}
            ]
            
            results = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            logger.info(f"Retrieved {len(results)} chunks for file {file_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving chunks for file {file_id}: {str(e)}")
            return []

    def hybrid_search(self, embedding: List[float], text_query: str, 
                     filters: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and text search
        
        Args:
            embedding: Query embedding vector
            text_query: Text query for keyword matching
            filters: Filter criteria
            top_k: Number of results to return
            
        Returns:
            List of matching documents with combined scores
        """
        try:
            # Build filter conditions
            conditions, parameters = self._build_filter_conditions(filters)
            
            # Add text search condition
            if text_query:
                conditions.append("CONTAINS(LOWER(c.text), LOWER(@text_query))")
                parameters.append({"name": "@text_query", "value": text_query})
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            # Hybrid search query with both vector and text relevance
            query = f"""
                SELECT 
                    c.id,
                    c.file_id,
                    c.fileName,
                    c.chunk_index,
                    c.text,
                    c.platform,
                    c.mime_type,
                    c.created_at,
                    c.metadata,
                    VectorDistance(c.embedding, @embedding) AS vector_score,
                    (CASE 
                        WHEN CONTAINS(LOWER(c.text), LOWER(@text_query)) THEN 1.0
                        ELSE 0.0
                    END) AS text_score
                FROM c
                WHERE {where_clause}
                ORDER BY (VectorDistance(c.embedding, @embedding) * 0.7 + 
                         (CASE WHEN CONTAINS(LOWER(c.text), LOWER(@text_query)) THEN 0.0 ELSE 1.0 END) * 0.3)
                OFFSET 0 LIMIT @top_k
            """
            
            # Add parameters
            parameters.extend([
                {"name": "@embedding", "value": embedding},
                {"name": "@text_query", "value": text_query},
                {"name": "@top_k", "value": top_k}
            ])
            
            results = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
                max_item_count=top_k
            ))
            
            # Process results
            processed_results = []
            for item in results:
                result = {
                    'id': item.get('id'),
                    'file_id': item.get('file_id'),
                    'fileName': item.get('fileName'),
                    'chunk_index': item.get('chunk_index', 0),
                    'text': item.get('text', '')[:500] + ('...' if len(item.get('text', '')) > 500 else ''),
                    'platform': item.get('platform'),
                    'mime_type': item.get('mime_type'),
                    'created_at': item.get('created_at'),
                    'vector_score': item.get('vector_score', 0.0),
                    'text_score': item.get('text_score', 0.0),
                    'metadata': item.get('metadata', {})
                }
                processed_results.append(result)
            
            logger.info(f"Hybrid search returned {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error during hybrid search: {str(e)}")
            return []

    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics about user's files and chunks
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with user statistics
        """
        try:
            # Get chunk statistics
            chunk_query = """
                SELECT 
                    COUNT(1) as total_chunks,
                    COUNT(DISTINCT c.file_id) as total_files,
                    c.platform,
                    c.mime_type
                FROM c
                WHERE c.user_id = @user_id 
                AND IS_DEFINED(c.chunk_index)
                GROUP BY c.platform, c.mime_type
            """
            
            parameters = [{"name": "@user_id", "value": user_id}]
            
            results = list(self.container.query_items(
                query=chunk_query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            # Process statistics
            stats = {
                'user_id': user_id,
                'total_chunks': sum(r.get('total_chunks', 0) for r in results),
                'total_files': len(set(r.get('total_files', 0) for r in results)),
                'platform_distribution': {},
                'file_type_distribution': {},
                'generated_at': datetime.utcnow().isoformat() + 'Z'
            }
            
            for result in results:
                platform = result.get('platform', 'unknown')
                mime_type = result.get('mime_type', 'unknown')
                chunk_count = result.get('total_chunks', 0)
                
                stats['platform_distribution'][platform] = stats['platform_distribution'].get(platform, 0) + chunk_count
                stats['file_type_distribution'][mime_type] = stats['file_type_distribution'].get(mime_type, 0) + chunk_count
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting user statistics: {str(e)}")
            return {
                'user_id': user_id,
                'error': str(e),
                'generated_at': datetime.utcnow().isoformat() + 'Z'
            }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Cosmos DB connection
        
        Returns:
            Health status dictionary
        """
        try:
            # Test database connection
            database_info = self.database.read()
            container_info = self.container.read()
            
            # Test query execution
            test_query = "SELECT VALUE COUNT(1) FROM c"
            test_results = list(self.container.query_items(
                query=test_query,
                enable_cross_partition_query=True,
                max_item_count=1
            ))
            
            return {
                'status': 'healthy',
                'database_id': database_info.get('id'),
                'container_id': container_info.get('id'),
                'total_documents': test_results[0] if test_results else 0,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }


# Example usage and utility functions
def create_cosmos_client(cosmos_endpoint: str = None, cosmos_key: str = None) -> CosmosVectorClient:
    """
    Factory function to create a CosmosVectorClient instance
    
    Args:
        cosmos_endpoint: Cosmos DB endpoint URL
        cosmos_key: Cosmos DB primary key
        
    Returns:
        Configured CosmosVectorClient instance
    """
    return CosmosVectorClient(cosmos_endpoint, cosmos_key)


# Example usage
if __name__ == "__main__":
    # Example usage of the CosmosVectorClient
    try:
        # Initialize client
        client = create_cosmos_client()
        
        # Example vector search
        query_embedding = [0.1] * 1536  # Example embedding
        
        results = client.vector_search_cosmos(
            embedding=query_embedding,
            filters={
                "user_id": "sayyadshakiltajoddin@gmail.com",
                "platforms": ["google_drive"],
                "file_types": ["pdf"],
                "time_range": "2024-01-01"
            },
            top_k=5
        )
        
        print(f"Found {len(results)} matching chunks")
        for result in results:
            print(f"File: {result['fileName']}, Score: {result['score']:.4f}")
            
    except Exception as e:
        logger.error(f"Example execution failed: {str(e)}")
