"""
Production-grade memory module for Weez MCP Agent.
Handles conversation storage and retrieval using Azure Cosmos DB with conversation threading.
"""

import os
import uuid
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from azure.cosmos import CosmosClient, exceptions
from azure.cosmos.database import DatabaseProxy
from azure.cosmos.container import ContainerProxy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CosmosMemoryError(Exception):
    """Custom exception for Cosmos DB memory operations."""
    pass

class CosmosMemoryManager:
    """
    Production-grade memory manager for AI Agent conversations using Azure Cosmos DB.
    Supports conversation threading with conversation_id.
    """
    
    def __init__(self, 
                 cosmos_endpoint: Optional[str] = None,
                 cosmos_key: Optional[str] = None,
                 database_name: str = "users",
                 container_name: str = "users_conversations"):
        """
        Initialize the Cosmos DB memory manager.
        
        Args:
            cosmos_endpoint: Cosmos DB endpoint URL (optional, uses env var if not provided)
            cosmos_key: Cosmos DB primary key (optional, uses env var if not provided)
            database_name: Database name (default: "users")
            container_name: Container name (default: "users_conversations")
        """
        self.database_name = database_name
        self.container_name = container_name
        
        # Get endpoint and key from environment or parameters
        self.cosmos_endpoint = cosmos_endpoint or os.getenv("COSMOS_ENDPOINT_2")
        self.cosmos_key = cosmos_key or os.getenv("COSMOS_KEY_2")
        
        if not self.cosmos_endpoint:
            raise CosmosMemoryError(
                "Cosmos DB endpoint not provided. Set COSMOS_ENDPOINT_2 environment variable."
            )
        
        if not self.cosmos_key:
            raise CosmosMemoryError(
                "Cosmos DB key not provided. Set COSMOS_KEY_2 environment variable."
            )
        
        # Initialize client and container
        self._initialize_cosmos_client()
    
    def _initialize_cosmos_client(self) -> None:
        """Initialize Cosmos DB client, database, and container references."""
        try:
            self.client = CosmosClient(self.cosmos_endpoint, self.cosmos_key)
            self.database = self.client.get_database_client(self.database_name)
            self.container = self.database.get_container_client(self.container_name)
            
            # Test connection with a simple query
            list(self.container.query_items(
                query="SELECT TOP 1 * FROM c",
                enable_cross_partition_query=True
            ))
            
            logger.info(f"Successfully connected to Cosmos DB: {self.cosmos_endpoint}")
        except Exception as e:
            logger.error(f"Failed to initialize Cosmos DB client: {str(e)}")
            raise CosmosMemoryError(f"Failed to connect to Cosmos DB: {str(e)}")
    
    def _retry_with_backoff(self, func, *args, max_retries: int = 3, **kwargs) -> Any:
        """
        Execute function with exponential backoff retry logic.
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retry attempts
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Function result
            
        Raises:
            CosmosMemoryError: If all retry attempts fail
        """
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except exceptions.CosmosResourceExistsError:
                # Don't retry on duplicate key errors
                raise
            except exceptions.CosmosHttpResponseError as e:
                if attempt == max_retries - 1:
                    logger.error(f"All retry attempts failed: {str(e)}")
                    raise CosmosMemoryError(f"Cosmos DB operation failed after {max_retries} attempts: {str(e)}")
                
                # Exponential backoff: 1s, 2s, 4s...
                wait_time = 2 ** attempt
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise CosmosMemoryError(f"Unexpected error during Cosmos DB operation: {str(e)}")
    
    def store_conversation(self, user_id: str, conversation_id: str, user_query: str, agent_response: str, timestamp: Optional[str] = None) -> None:
        """
        Store a conversation interaction in Cosmos DB with conversation threading.
        
        Args:
            user_id: Unique identifier for the user (email)
            conversation_id: Unique identifier for the conversation thread
            user_query: User's query/message
            agent_response: Agent's response
            timestamp: Optional ISO8601 timestamp (auto-generated if not provided)
            
        Raises:
            CosmosMemoryError: If storage operation fails
        """
        try:
            # Validate inputs
            if not user_id:
                raise ValueError("user_id is required and cannot be empty")
            if not conversation_id:
                raise ValueError("conversation_id is required and cannot be empty")
            if not user_query:
                raise ValueError("user_query is required and cannot be empty")
            if not agent_response:
                raise ValueError("agent_response is required and cannot be empty")
            
            # Generate timestamp if not provided
            if timestamp is None:
                timestamp = datetime.now(timezone.utc).isoformat()
            
            # Ensure timestamp is in ISO format
            if not isinstance(timestamp, str):
                timestamp = str(timestamp)
            
            # Create document with conversation_id
            document = {
                "id": str(uuid.uuid4()),
                "user_id": str(user_id),
                "conversation_id": str(conversation_id),
                "user_query": str(user_query),
                "agent_response": str(agent_response),
                "timestamp": timestamp
            }
            
            # Store with retry logic
            self._retry_with_backoff(
                self.container.create_item,
                body=document,
                enable_automatic_id_generation=False
            )
            logger.info(f"Successfully stored conversation for user {user_id} in conversation {conversation_id}")
            
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise CosmosMemoryError(f"Invalid input: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to store conversation: {str(e)}")
            raise CosmosMemoryError(f"Failed to store conversation: {str(e)}")
    
    def get_conversation_history(self, 
                               user_id: str, 
                               conversation_id: str, 
                               limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a specific conversation thread.
        
        Args:
            user_id: Unique identifier for the user (email)
            conversation_id: Unique identifier for the conversation thread
            limit: Optional limit on number of messages to retrieve (most recent first)
            
        Returns:
            List of conversation messages sorted by timestamp (oldest first)
            
        Raises:
            CosmosMemoryError: If retrieval operation fails
        """
        try:
            # Validate inputs
            if not user_id:
                raise ValueError("user_id is required and cannot be empty")
            if not conversation_id:
                raise ValueError("conversation_id is required and cannot be empty")
            
            # Build query
            query = """
            SELECT * FROM c 
            WHERE c.user_id = @user_id AND c.conversation_id = @conversation_id 
            ORDER BY c.timestamp ASC
            """
            
            parameters = [
                {"name": "@user_id", "value": str(user_id)},
                {"name": "@conversation_id", "value": str(conversation_id)}
            ]
            
            # Execute query with retry logic
            items = self._retry_with_backoff(
                lambda: list(self.container.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True
                ))
            )
            
            # Apply limit if specified (take most recent)
            if limit and limit > 0:
                items = items[-limit:]
            
            logger.info(f"Retrieved {len(items)} messages for user {user_id} in conversation {conversation_id}")
            return items
            
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise CosmosMemoryError(f"Invalid input: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {str(e)}")
            raise CosmosMemoryError(f"Failed to retrieve conversation history: {str(e)}")
    
    def get_user_conversations(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all conversation threads for a user with summary information.
        
        Args:
            user_id: Unique identifier for the user (email)
            limit: Optional limit on number of conversations to retrieve
            
        Returns:
            List of conversation summaries with metadata
            
        Raises:
            CosmosMemoryError: If retrieval operation fails
        """
        try:
            # Validate inputs
            if not user_id:
                raise ValueError("user_id is required and cannot be empty")
            
            # Query to get conversation summaries
            query = """
            SELECT c.conversation_id, 
                   MIN(c.timestamp) as first_message_time,
                   MAX(c.timestamp) as last_message_time,
                   COUNT(1) as message_count
            FROM c 
            WHERE c.user_id = @user_id 
            GROUP BY c.conversation_id
            ORDER BY MAX(c.timestamp) DESC
            """
            
            parameters = [{"name": "@user_id", "value": str(user_id)}]
            
            # Execute query with retry logic
            items = self._retry_with_backoff(
                lambda: list(self.container.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True
                ))
            )
            
            # Apply limit if specified
            if limit and limit > 0:
                items = items[:limit]
            
            logger.info(f"Retrieved {len(items)} conversations for user {user_id}")
            return items
            
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise CosmosMemoryError(f"Invalid input: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to retrieve user conversations: {str(e)}")
            raise CosmosMemoryError(f"Failed to retrieve user conversations: {str(e)}")
    
    def search_conversations(self, 
                           user_id: str, 
                           search_term: str, 
                           conversation_id: Optional[str] = None,
                           limit: Optional[int] = 10) -> List[Dict[str, Any]]:
        """
        Search through conversation history for specific terms.
        
        Args:
            user_id: Unique identifier for the user (email)
            search_term: Term to search for in queries and responses
            conversation_id: Optional specific conversation to search within
            limit: Maximum number of results to return (default: 10)
            
        Returns:
            List of matching conversation messages
            
        Raises:
            CosmosMemoryError: If search operation fails
        """
        try:
            # Validate inputs
            if not user_id:
                raise ValueError("user_id is required and cannot be empty")
            if not search_term:
                raise ValueError("search_term is required and cannot be empty")
            
            # Build query based on whether conversation_id is specified
            if conversation_id:
                query = """
                SELECT * FROM c 
                WHERE c.user_id = @user_id 
                AND c.conversation_id = @conversation_id
                AND (CONTAINS(LOWER(c.user_query), LOWER(@search_term)) 
                     OR CONTAINS(LOWER(c.agent_response), LOWER(@search_term)))
                ORDER BY c.timestamp DESC
                """
                parameters = [
                    {"name": "@user_id", "value": str(user_id)},
                    {"name": "@conversation_id", "value": str(conversation_id)},
                    {"name": "@search_term", "value": str(search_term)}
                ]
            else:
                query = """
                SELECT * FROM c 
                WHERE c.user_id = @user_id 
                AND (CONTAINS(LOWER(c.user_query), LOWER(@search_term)) 
                     OR CONTAINS(LOWER(c.agent_response), LOWER(@search_term)))
                ORDER BY c.timestamp DESC
                """
                parameters = [
                    {"name": "@user_id", "value": str(user_id)},
                    {"name": "@search_term", "value": str(search_term)}
                ]
            
            # Execute query with retry logic
            items = self._retry_with_backoff(
                lambda: list(self.container.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True
                ))
            )
            
            # Apply limit if specified
            if limit and limit > 0:
                items = items[:limit]
            
            logger.info(f"Found {len(items)} messages matching '{search_term}' for user {user_id}")
            return items
            
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise CosmosMemoryError(f"Invalid input: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to search conversations: {str(e)}")
            raise CosmosMemoryError(f"Failed to search conversations: {str(e)}")
    
    def delete_conversation(self, user_id: str, conversation_id: str) -> int:
        """
        Delete an entire conversation thread.
        
        Args:
            user_id: Unique identifier for the user (email)
            conversation_id: Unique identifier for the conversation thread to delete
            
        Returns:
            Number of messages deleted
            
        Raises:
            CosmosMemoryError: If deletion operation fails
        """
        try:
            # Validate inputs
            if not user_id:
                raise ValueError("user_id is required and cannot be empty")
            if not conversation_id:
                raise ValueError("conversation_id is required and cannot be empty")
            
            # First, get all messages in the conversation
            messages = self.get_conversation_history(user_id, conversation_id)
            
            # Delete each message
            deleted_count = 0
            for message in messages:
                try:
                    self._retry_with_backoff(
                        self.container.delete_item,
                        item=message["id"],
                        partition_key=message["user_id"]
                    )
                    deleted_count += 1
                except exceptions.CosmosResourceNotFoundError:
                    # Message already deleted, continue
                    logger.warning(f"Message {message['id']} not found during deletion")
                    continue
            
            logger.info(f"Successfully deleted {deleted_count} messages from conversation {conversation_id}")
            return deleted_count
            
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise CosmosMemoryError(f"Invalid input: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to delete conversation: {str(e)}")
            raise CosmosMemoryError(f"Failed to delete conversation: {str(e)}")
    
    def get_recent_context(self, 
                          user_id: str, 
                          conversation_id: str, 
                          context_limit: int = 5) -> str:
        """
        Get recent conversation context as formatted string for AI prompts.
        
        Args:
            user_id: Unique identifier for the user (email)
            conversation_id: Unique identifier for the conversation thread
            context_limit: Number of recent message pairs to include (default: 5)
            
        Returns:
            Formatted conversation context string
            
        Raises:
            CosmosMemoryError: If retrieval operation fails
        """
        try:
            # Get recent conversation history
            messages = self.get_conversation_history(user_id, conversation_id, limit=context_limit * 2)
            
            if not messages:
                return "No previous conversation history found."
            
            # Format as conversation context
            context_parts = []
            for message in messages:
                timestamp = message.get("timestamp", "")
                user_query = message.get("user_query", "")
                agent_response = message.get("agent_response", "")
                
                context_parts.append(f"User: {user_query}")
                context_parts.append(f"Assistant: {agent_response}")
                context_parts.append("---")
            
            # Remove the last separator
            if context_parts and context_parts[-1] == "---":
                context_parts.pop()
            
            context_string = "\n".join(context_parts)
            logger.info(f"Generated context with {len(messages)} messages for conversation {conversation_id}")
            
            return context_string
            
        except Exception as e:
            logger.error(f"Failed to get recent context: {str(e)}")
            raise CosmosMemoryError(f"Failed to get recent context: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the Cosmos DB connection.
        
        Returns:
            Dictionary with health check results
            
        Raises:
            CosmosMemoryError: If health check fails
        """
        try:
            start_time = time.time()
            
            # Test connection with a simple query
            test_result = list(self.container.query_items(
                query="SELECT VALUE COUNT(1) FROM c",
                enable_cross_partition_query=True
            ))
            
            end_time = time.time()
            response_time = round((end_time - start_time) * 1000, 2)  # Convert to milliseconds
            
            health_status = {
                "status": "healthy",
                "database": self.database_name,
                "container": self.container_name,
                "response_time_ms": response_time,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_documents": test_result[0] if test_result else 0
            }
            
            logger.info(f"Health check passed - Response time: {response_time}ms")
            return health_status
            
        except Exception as e:
            error_status = {
                "status": "unhealthy",
                "database": self.database_name,
                "container": self.container_name,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.error(f"Health check failed: {str(e)}")
            raise CosmosMemoryError(f"Health check failed: {str(e)}")
    
    def cleanup_old_conversations(self, 
                                 user_id: str, 
                                 days_old: int = 90) -> int:
        """
        Clean up conversations older than specified days.
        
        Args:
            user_id: Unique identifier for the user (email)
            days_old: Number of days to keep (default: 90)
            
        Returns:
            Number of messages deleted
            
        Raises:
            CosmosMemoryError: If cleanup operation fails
        """
        try:
            # Validate inputs
            if not user_id:
                raise ValueError("user_id is required and cannot be empty")
            if days_old <= 0:
                raise ValueError("days_old must be greater than 0")
            
            # Calculate cutoff date
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
            cutoff_iso = cutoff_date.isoformat()
            
            # Query for old messages
            query = """
            SELECT c.id, c.user_id FROM c 
            WHERE c.user_id = @user_id AND c.timestamp < @cutoff_date
            """
            
            parameters = [
                {"name": "@user_id", "value": str(user_id)},
                {"name": "@cutoff_date", "value": cutoff_iso}
            ]
            
            # Get old messages
            old_messages = self._retry_with_backoff(
                lambda: list(self.container.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True
                ))
            )
            
            # Delete old messages
            deleted_count = 0
            for message in old_messages:
                try:
                    self._retry_with_backoff(
                        self.container.delete_item,
                        item=message["id"],
                        partition_key=message["user_id"]
                    )
                    deleted_count += 1
                except exceptions.CosmosResourceNotFoundError:
                    # Message already deleted, continue
                    logger.warning(f"Message {message['id']} not found during cleanup")
                    continue
            
            logger.info(f"Cleaned up {deleted_count} old messages for user {user_id}")
            return deleted_count
            
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise CosmosMemoryError(f"Invalid input: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to cleanup old conversations: {str(e)}")
            raise CosmosMemoryError(f"Failed to cleanup old conversations: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    import json
    from datetime import timedelta
    
    # Example usage
    try:
        # Initialize memory manager
        memory = CosmosMemoryManager()
        
        # Test health check
        health = memory.health_check()
        print("Health Check:", json.dumps(health, indent=2))
        
        # Example user and conversation IDs
        test_user_id = "test@example.com"
        test_conversation_id = "conv_001"
        
        # Store some test conversations
        memory.store_conversation(
            user_id=test_user_id,
            conversation_id=test_conversation_id,
            user_query="What is machine learning?",
            agent_response="Machine learning is a subset of artificial intelligence..."
        )
        
        # Retrieve conversation history
        history = memory.get_conversation_history(test_user_id, test_conversation_id)
        print(f"\nConversation History: {len(history)} messages")
        
        # Get recent context
        context = memory.get_recent_context(test_user_id, test_conversation_id)
        print(f"\nRecent Context:\n{context}")
        
        # Search conversations
        search_results = memory.search_conversations(test_user_id, "machine learning")
        print(f"\nSearch Results: {len(search_results)} matches")
        
        # Get user conversations
        user_convs = memory.get_user_conversations(test_user_id)
        print(f"\nUser Conversations: {len(user_convs)} conversations")
        
    except CosmosMemoryError as e:
        print(f"Memory Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
