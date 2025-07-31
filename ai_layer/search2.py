"""
search2.py - Production-grade intelligent document search module with RBAC for AI assistant.

This module provides hybrid vector + metadata search capabilities over document chunks
stored in Azure Cosmos DB with comprehensive Role-Based Access Control (RBAC).

Key Features:
- Role-Based Access Control (RBAC) integration with Supabase
- Structured intent-based search with comprehensive validation
- Hybrid vector + metadata search over Azure Cosmos DB
- Enhanced result processing with metadata enrichment including SAS URLs
- Production-ready error handling and logging
- Modular design with clear separation of concerns

Author: AI Assistant
Version: 2.0.0 - Added comprehensive RBAC support
"""

import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import os

# Import external dependencies
from utils.openai_client import get_embedding
from utils.cosmos_client import CosmosVectorClient

# Configure module logger
logger = logging.getLogger(__name__)

def create_cosmos_client(cosmos_endpoint: str = os.getenv('COSMOS_ENDPOINT'), cosmos_key: str = os.getenv('COSMOS_KEY')) -> CosmosVectorClient:
    """
    Factory function to create a CosmosVectorClient instance
    
    Args:
        cosmos_endpoint: Cosmos DB endpoint URL
        cosmos_key: Cosmos DB primary key
        
    Returns:
        Configured CosmosVectorClient instance
    """
    return CosmosVectorClient(cosmos_endpoint, cosmos_key)

def generate_sas_url(file_path: str) -> Optional[str]:
    """
    Generate SAS URL for blob with 1-year expiration
    
    Args:
        file_path: Path to the file in blob storage
        
    Returns:
        SAS URL string or None if generation fails
    """
    try:
        from azure.storage.blob import generate_blob_sas, BlobSasPermissions
        from datetime import timedelta
        
        # Extract account name and key from connection string
        conn_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING_1')
        if not conn_string:
            logger.error("AZURE_STORAGE_CONNECTION_STRING_1 not found in environment")
            return None
            
        conn_parts = dict(item.split('=', 1) for item in conn_string.split(';') if '=' in item)
        account_name = conn_parts.get('AccountName')
        account_key = conn_parts.get('AccountKey')
        
        if not account_name or not account_key:
            logger.error("Could not extract account credentials from connection string")
            return None
        
        # Clean the file path - remove any leading slashes
        blob_name = file_path.lstrip('/')
        container_name = "weezyaifiles"
        
        # Generate SAS token with 1 year expiration
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(days=365)  # 1 year
        )
        
        # Construct full SAS URL
        sas_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
        return sas_url
        
    except Exception as e:
        logger.error(f"Error generating SAS URL for {file_path}: {e}")
        return None

class SearchError(Exception):
    """
    Custom exception for search-related errors.
    
    Raised for all predictable search issues including validation failures,
    embedding generation errors, and search execution problems.
    """
    pass

class RBACError(Exception):
    """
    Custom exception for RBAC-related errors.
    
    Raised for access control violations, permission errors, and authorization failures.
    """
    pass

class Platform(Enum):
    """Supported platform types for document storage."""
    GOOGLE_DRIVE = "google_drive"
    ONEDRIVE = "onedrive"
    DROPBOX = "dropbox"
    SHAREPOINT = "sharepoint"
    LOCAL = "local"
    SLACK = "slack"
    TEAMS = "teams"
    PLATFORM_SYNC = "platform_sync"

class FileCategory(Enum):
    """Document file categories for enhanced metadata."""
    TEXT = "text"
    PDF = "pdf"
    DOCUMENT = "document"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    ARCHIVE = "archive"
    OTHER = "other"

class Role(Enum):
    """User roles in the system."""
    ADMIN = "admin"
    TEAM_LEAD = "team_lead"
    EMPLOYEE = "employee"
    VIEWER = "viewer"

class Permission(Enum):
    """Available permissions in the system."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    MANAGE_ROLES = "manage_roles"
    MANAGE_USERS = "manage_users"

class ResourceScope(Enum):
    """Resource access scopes."""
    ALL = "all"
    TEAM = "team"
    OWN = "own"
    DEPARTMENT = "department"

@dataclass
class UserPermissions:
    """Container for user's effective permissions."""
    user_id: str
    roles: List[str]
    permissions: Dict[str, str]  # permission -> resource_scope
    teams: List[str]
    departments: List[str]
    can_access_all: bool
    
    def has_permission(self, permission: str, resource_scope: str = "own") -> bool:
        """Check if user has specific permission with given scope."""
        if self.can_access_all:
            return True
            
        user_permission_scope = self.permissions.get(permission)
        if not user_permission_scope:
            return False
            
        # Permission hierarchy: all > department > team > own
        scope_hierarchy = {"all": 4, "department": 3, "team": 2, "own": 1}
        required_level = scope_hierarchy.get(resource_scope, 1)
        user_level = scope_hierarchy.get(user_permission_scope, 1)
        
        return user_level >= required_level

@dataclass
class SearchMetrics:
    """Container for search performance metrics."""
    total_results: int
    search_time_ms: float
    embedding_time_ms: float
    filter_count: int
    top_similarity_score: float
    rbac_filter_time_ms: float
    accessible_documents: int

class RBACManager:
    """
    Manages Role-Based Access Control for search operations.
    
    Integrates with Supabase to determine user permissions and filter
    search results based on access rights.
    """
    
    def __init__(self, supabase_client=None):
        """
        Initialize RBAC manager with Supabase client.
        
        Args:
            supabase_client: Configured Supabase client instance
        """
        self.supabase = supabase_client
        if not self.supabase:
            try:
                from supabase import create_client
                supabase_url = os.getenv('SUPABASE_URL')
                supabase_key = os.getenv('SUPABASE_ANON_KEY')
                if supabase_url and supabase_key:
                    self.supabase = create_client(supabase_url, supabase_key)
                else:
                    logger.warning("Supabase credentials not found, RBAC will be disabled")
                    self.supabase = None
            except ImportError:
                logger.warning("Supabase client not available, RBAC will be disabled")
                self.supabase = None
        
        logger.info(f"RBACManager initialized with Supabase: {bool(self.supabase)}")
    
    def get_user_permissions(self, user_id: str) -> UserPermissions:
        """
        Retrieve comprehensive user permissions from Supabase.
        
        Args:
            user_id: User ID to get permissions for
            
        Returns:
            UserPermissions object containing all user access rights
            
        Raises:
            RBACError: If permission retrieval fails
        """
        if not self.supabase:
            logger.warning("RBAC disabled, granting basic permissions")
            return UserPermissions(
                user_id=user_id,
                roles=["employee"],
                permissions={"read": "own", "write": "own"},
                teams=[],
                departments=[],
                can_access_all=False
            )
        
        try:
            # Get user roles
            user_roles_response = self.supabase.table('user_roles').select(
                'role, profiles(department)'
            ).eq('user_id', user_id).execute()
            
            if not user_roles_response.data:
                logger.warning(f"No roles found for user {user_id}")
                return UserPermissions(
                    user_id=user_id,
                    roles=[],
                    permissions={},
                    teams=[],
                    departments=[],
                    can_access_all=False
                )
            
            # Extract roles and departments
            roles = [role['role'] for role in user_roles_response.data]
            departments = list(set([
                role['profiles']['department'] 
                for role in user_roles_response.data 
                if role['profiles'] and role['profiles'].get('department')
            ]))
            
            # Get permissions for these roles
            permissions_response = self.supabase.table('permissions').select(
                'permission, resource'
            ).in_('role', roles).execute()
            
            # Build permissions dictionary
            permissions = {}
            for perm in permissions_response.data:
                permission_type = perm['permission']
                resource_scope = perm['resource']
                
                # Keep the highest level permission for each type
                if permission_type not in permissions:
                    permissions[permission_type] = resource_scope
                else:
                    current_scope = permissions[permission_type]
                    scope_hierarchy = {"all": 4, "department": 3, "team": 2, "own": 1}
                    if scope_hierarchy.get(resource_scope, 1) > scope_hierarchy.get(current_scope, 1):
                        permissions[permission_type] = resource_scope
            
            # Get user's teams
            teams_response = self.supabase.table('team_employees').select(
                'team_id'
            ).eq('user_id', user_id).execute()
            
            teams = [team['team_id'] for team in teams_response.data]
            
            # Check if user has admin privileges
            can_access_all = 'admin' in roles or permissions.get('read') == 'all'
            
            user_permissions = UserPermissions(
                user_id=user_id,
                roles=roles,
                permissions=permissions,
                teams=teams,
                departments=departments,
                can_access_all=can_access_all
            )
            
            logger.debug(f"Retrieved permissions for user {user_id}: {len(roles)} roles, {len(permissions)} permissions")
            return user_permissions
            
        except Exception as e:
            logger.error(f"Failed to retrieve user permissions for {user_id}: {str(e)}")
            raise RBACError(f"Permission retrieval failed: {str(e)}")
    
    def filter_accessible_documents(self, user_permissions: UserPermissions, documents: List[dict]) -> List[dict]:
        """
        Filter documents based on user's access permissions.
        
        Args:
            user_permissions: User's permission object
            documents: List of documents to filter
            
        Returns:
            List of documents user has access to
        """
        if user_permissions.can_access_all:
            logger.debug(f"User {user_permissions.user_id} has access to all documents")
            return documents
        
        accessible_docs = []
        
        for doc in documents:
            if self._can_access_document(user_permissions, doc):
                accessible_docs.append(doc)
        
        logger.debug(f"Filtered {len(documents)} documents to {len(accessible_docs)} accessible for user {user_permissions.user_id}")
        return accessible_docs
    
    def _can_access_document(self, user_permissions: UserPermissions, document: dict) -> bool:
        """
        Check if user can access a specific document.
        
        Args:
            user_permissions: User's permission object
            document: Document to check access for
            
        Returns:
            True if user can access the document
        """
        # Check if user has read permission
        if not user_permissions.has_permission('read'):
            return False
        
        # Admin or all-access users can access everything
        if user_permissions.can_access_all:
            return True
        
        # Check document ownership
        doc_owner = document.get('user_id') or document.get('owner_id') or document.get('uploaded_by')
        if doc_owner == user_permissions.user_id:
            return True
        
        # Check team access
        doc_teams = document.get('teams', [])
        if isinstance(doc_teams, str):
            doc_teams = [doc_teams]
        
        if any(team in user_permissions.teams for team in doc_teams):
            read_scope = user_permissions.permissions.get('read', 'own')
            if read_scope in ['team', 'department', 'all']:
                return True
        
        # Check department access
        doc_departments = document.get('departments', []) or document.get('department', [])
        if isinstance(doc_departments, str):
            doc_departments = [doc_departments]
        
        if any(dept in user_permissions.departments for dept in doc_departments):
            read_scope = user_permissions.permissions.get('read', 'own')
            if read_scope in ['department', 'all']:
                return True
        
        # Check document visibility
        doc_visibility = document.get('visibility', 'private')
        if doc_visibility == 'public':
            return True
        
        # Check if document is shared with user's teams or departments
        shared_with = document.get('shared_with', [])
        if isinstance(shared_with, str):
            shared_with = [shared_with]
        
        # Check if any of user's teams/departments are in shared_with
        user_access_groups = user_permissions.teams + user_permissions.departments + [user_permissions.user_id]
        if any(group in shared_with for group in user_access_groups):
            return True
        
        return False
    
    def get_rbac_filters(self, user_permissions: UserPermissions) -> Dict[str, Any]:
        """
        Generate database filters based on user permissions.
        
        Args:
            user_permissions: User's permission object
            
        Returns:
            Dictionary of filters for database queries
        """
        if user_permissions.can_access_all:
            return {}  # No additional filters needed
        
        # Build OR conditions for accessible documents
        access_conditions = []
        
        # Own documents
        access_conditions.append(f"user_id = '{user_permissions.user_id}'")
        access_conditions.append(f"owner_id = '{user_permissions.user_id}'")
        access_conditions.append(f"uploaded_by = '{user_permissions.user_id}'")
        
        # Team documents (if user has team access)
        if user_permissions.teams and user_permissions.has_permission('read', 'team'):
            team_conditions = [f"teams @> '[{team}]'" for team in user_permissions.teams]
            access_conditions.extend(team_conditions)
        
        # Department documents (if user has department access)
        if user_permissions.departments and user_permissions.has_permission('read', 'department'):
            dept_conditions = [f"departments @> '[{dept}]'" for dept in user_permissions.departments]
            access_conditions.extend(dept_conditions)
        
        # Public documents
        access_conditions.append("visibility = 'public'")
        
        # Shared documents
        user_access_groups = user_permissions.teams + user_permissions.departments + [user_permissions.user_id]
        shared_conditions = [f"shared_with @> '[{group}]'" for group in user_access_groups]
        access_conditions.extend(shared_conditions)
        
        return {
            'rbac_filter': f"({' OR '.join(access_conditions)})"
        }

class SearchValidator:
    """
    Validates and normalizes search intent parameters with RBAC integration.
    
    Ensures all required fields are present and valid, normalizes optional fields,
    validates user permissions, and provides helpful error messages for validation failures.
    """
    
    # Supported MIME types for document search (matching cosmos_client expectations)
    SUPPORTED_MIME_TYPES = {
        'text/plain',
        'text/markdown',
        'text/html',
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # .docx
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',  # .pptx
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # .xlsx
        'application/msword',  # .doc
        'application/vnd.ms-powerpoint',  # .ppt
        'application/vnd.ms-excel',  # .xls
        'application/json',
        'application/xml',
        'text/csv'
    }
    
    # File type shortcuts that will be converted to MIME types
    FILE_TYPE_SHORTCUTS = {
        'pdf': 'PDF',
        'doc': 'DOC',
        'docx': 'DOCX',
        'xls': 'XLS',
        'xlsx': 'XLSX',
        'ppt': 'PPT',
        'pptx': 'PPTX',
        'txt': 'TXT'
    }
    
    # Supported relative time ranges
    SUPPORTED_TIME_RANGES = {
        'last_hour', 'last_24_hours', 'last_7_days', 'last_30_days', 
        'last_month', 'last_3_months', 'last_6_months', 'last_year'
    }
    
    def __init__(self, rbac_manager: Optional[RBACManager] = None):
        """
        Initialize validator with RBAC manager.
        
        Args:
            rbac_manager: RBAC manager for permission validation
        """
        self.rbac_manager = rbac_manager
    
    def validate_intent(self, intent: dict) -> Tuple[dict, UserPermissions]:
        """
        Validates and normalizes the search intent dictionary with RBAC checks.
        
        Args:
            intent: Raw search intent dictionary from client
            
        Returns:
            Tuple of (normalized intent dictionary, user permissions)
            
        Raises:
            SearchError: If validation fails for any required or optional field
            RBACError: If user lacks required permissions
        """
        if not isinstance(intent, dict):
            raise SearchError("Intent must be a dictionary")
        
        # Validate required fields
        self._validate_required_fields(intent)
        
        # Get user permissions
        user_permissions = None
        if self.rbac_manager:
            try:
                user_permissions = self.rbac_manager.get_user_permissions(intent['user_id'])
                
                # Check if user has read permission
                if not user_permissions.has_permission('read'):
                    raise RBACError(f"User {intent['user_id']} lacks read permission")
                    
            except Exception as e:
                logger.error(f"RBAC validation failed for user {intent['user_id']}: {str(e)}")
                if isinstance(e, RBACError):
                    raise
                raise RBACError(f"Permission validation failed: {str(e)}")
        else:
            # Create basic permissions if RBAC is disabled
            user_permissions = UserPermissions(
                user_id=intent['user_id'],
                roles=["employee"],
                permissions={"read": "own", "write": "own"},
                teams=[],
                departments=[],
                can_access_all=False
            )
        
        # Create normalized copy
        normalized = intent.copy()
        
        # Normalize and validate optional fields
        self._normalize_query_text(normalized)
        self._validate_platform(normalized)
        self._validate_file_type(normalized)
        self._validate_time_range(normalized)
        self._validate_pagination(normalized)
        
        logger.debug(f"Intent validation successful for user: {normalized['user_id']}")
        return normalized, user_permissions
    
    @classmethod
    def _validate_required_fields(cls, intent: dict) -> None:
        """Validates required fields in intent dictionary."""
        if 'query_text' not in intent:
            raise SearchError("Missing required field: query_text")
        
        if not intent['query_text'] or not isinstance(intent['query_text'], str):
            raise SearchError("query_text must be a non-empty string")
        
        if 'user_id' not in intent:
            raise SearchError("Missing required field: user_id")
        
        if not intent['user_id'] or not isinstance(intent['user_id'], str):
            raise SearchError("user_id must be a non-empty string")
    
    @classmethod
    def _normalize_query_text(cls, intent: dict) -> None:
        """Normalizes query text by trimming whitespace and validating length."""
        query_text = intent['query_text'].strip()
        
        if len(query_text) < 2:
            raise SearchError("query_text must be at least 2 characters long")
        
        if len(query_text) > 1000:
            raise SearchError("query_text cannot exceed 1000 characters")
        
        intent['query_text'] = query_text
    
    @classmethod
    def _validate_platform(cls, intent: dict) -> None:
        """Validates and normalizes platform field."""
        if 'platform' not in intent or not intent['platform']:
            return
        
        platform = intent['platform'].lower().strip()
        valid_platforms = {p.value for p in Platform}
        
        if platform not in valid_platforms:
            logger.warning(f"Unknown platform: {platform}. Supported: {valid_platforms}")
            # Don't raise error, just log warning to allow flexibility
        
        intent['platform'] = platform
    
    @classmethod
    def _validate_file_type(cls, intent: dict) -> None:
        """Validates file type field and converts to cosmos_client expected format."""
        if 'file_type' not in intent or not intent['file_type']:
            return
        
        file_type = intent['file_type'].lower().strip()
        
        # Convert shortcut to cosmos_client format
        if file_type in cls.FILE_TYPE_SHORTCUTS:
            intent['file_type'] = cls.FILE_TYPE_SHORTCUTS[file_type]
        elif file_type.upper() in ['PDF', 'DOC', 'DOCX', 'XLS', 'XLSX', 'PPT', 'PPTX', 'TXT']:
            intent['file_type'] = file_type.upper()
        elif file_type in cls.SUPPORTED_MIME_TYPES:
            # If it's already a MIME type, convert it to cosmos_client format
            mime_to_type = {
                'application/pdf': 'PDF',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'DOCX',
                'application/msword': 'DOC',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'XLSX',
                'application/vnd.ms-excel': 'XLS',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'PPTX',
                'application/vnd.ms-powerpoint': 'PPT',
                'text/plain': 'TXT'
            }
            intent['file_type'] = mime_to_type.get(file_type, file_type.upper())
        else:
            logger.warning(f"Unknown file type: {file_type}. Will be passed as-is.")
            intent['file_type'] = file_type.upper()
    
    @classmethod
    def _validate_time_range(cls, intent: dict) -> None:
        """Validates time range specification."""
        if 'time_range' not in intent or not intent['time_range']:
            return
        
        time_range = intent['time_range']
        
        if isinstance(time_range, str):
            cls._validate_relative_time_range(time_range)
        elif isinstance(time_range, dict):
            cls._validate_absolute_time_range(time_range)
        else:
            raise SearchError("time_range must be a string or dictionary")
    
    @classmethod
    def _validate_relative_time_range(cls, time_range: str) -> None:
        """Validates relative time range strings."""
        if time_range not in cls.SUPPORTED_TIME_RANGES:
            raise SearchError(f"Invalid time range: {time_range}. Supported: {cls.SUPPORTED_TIME_RANGES}")
    
    @classmethod
    def _validate_absolute_time_range(cls, time_range: dict) -> None:
        """Validates absolute time range dictionaries."""
        if 'start_date' not in time_range and 'end_date' not in time_range:
            raise SearchError("Absolute time range must contain at least start_date or end_date")
        
        for date_field in ['start_date', 'end_date']:
            if date_field in time_range:
                try:
                    datetime.fromisoformat(time_range[date_field])
                except ValueError:
                    raise SearchError(f"Invalid {date_field} format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
    
    @classmethod
    def _validate_pagination(cls, intent: dict) -> None:
        """Validates pagination parameters."""
        if 'offset' in intent:
            if not isinstance(intent['offset'], int) or intent['offset'] < 0:
                raise SearchError("offset must be a non-negative integer")
        
        if 'limit' in intent:
            if not isinstance(intent['limit'], int) or intent['limit'] < 1 or intent['limit'] > 100:
                raise SearchError("limit must be an integer between 1 and 100")

class FilterBuilder:
    """
    Constructs metadata filters for Cosmos DB search from validated intent with RBAC integration.
    
    Translates high-level search intent into specific database query filters
    compatible with cosmos_client.py filter structure, including access control filters.
    """
    
    def __init__(self, rbac_manager: Optional[RBACManager] = None):
        """
        Initialize filter builder with RBAC manager.
        
        Args:
            rbac_manager: RBAC manager for permission-based filtering
        """
        self.rbac_manager = rbac_manager
    
    def build_filters(self, intent: dict, user_permissions: UserPermissions) -> dict:
        """
        Builds comprehensive metadata filters compatible with CosmosVectorClient including RBAC.
        
        Args:
            intent: Validated search intent dictionary
            user_permissions: User's permission object
            
        Returns:
            Dictionary of filters compatible with cosmos_client.py structure:
            - user_id: string (required)
            - platforms: list of strings (optional)
            - file_types: list of strings (optional, in cosmos_client format)
            - time_range: ISO8601 string (optional)
            - rbac_filters: access control filters
        """
        filters = {}
        
        # Always include user_id for data isolation and security (required by cosmos_client)
        filters['user_id'] = intent['user_id']
        
        # Add RBAC filters
        if self.rbac_manager and not user_permissions.can_access_all:
            rbac_filters = self.rbac_manager.get_rbac_filters(user_permissions)
            filters.update(rbac_filters)
        
        # Platform-specific filtering - convert to list format expected by cosmos_client
        if intent.get('platform'):
            filters['platforms'] = [intent['platform']]
        elif intent.get('platforms'):  # Support both single platform and list
            filters['platforms'] = intent['platforms'] if isinstance(intent['platforms'], list) else [intent['platforms']]
        
        # File type filtering - convert to list format expected by cosmos_client
        if intent.get('file_type'):
            filters['file_types'] = [intent['file_type']]
        elif intent.get('file_types'):  # Support both single file_type and list
            filters['file_types'] = intent['file_types'] if isinstance(intent['file_types'], list) else [intent['file_types']]
        
        # Time-based filtering - convert to ISO8601 string format
        if intent.get('time_range'):
            time_filter = self._build_time_filter(intent['time_range'])
            if time_filter:
                filters['time_range'] = time_filter
        
        # Add access scope information
        filters['_rbac_metadata'] = {
            'user_roles': user_permissions.roles,
            'can_access_all': user_permissions.can_access_all,
            'teams': user_permissions.teams,
            'departments': user_permissions.departments
        }
        
        logger.debug(f"Built {len(filters)} filters for cosmos_client with RBAC: {list(filters.keys())}")
        return filters
    
    @classmethod
    def _build_time_filter(cls, time_range: Union[str, dict]) -> Optional[str]:
        """
        Builds time-based filter in ISO8601 string format for cosmos_client.
        
        Args:
            time_range: Time range specification (relative string or absolute dict)
            
        Returns:
            ISO8601 time string for created_at >= filtering or None if invalid
        """
        if isinstance(time_range, str):
            return cls._build_relative_time_filter(time_range)
        elif isinstance(time_range, dict):
            return cls._build_absolute_time_filter(time_range)
        
        return None
    
    @classmethod
    def _build_relative_time_filter(cls, time_range: str) -> Optional[str]:
        """Builds ISO8601 string for relative time ranges (e.g., 'last_7_days')."""
        now = datetime.utcnow()
        
        time_deltas = {
            'last_hour': timedelta(hours=1),
            'last_24_hours': timedelta(hours=24),
            'last_7_days': timedelta(days=7),
            'last_30_days': timedelta(days=30),
            'last_month': timedelta(days=30),
            'last_3_months': timedelta(days=90),
            'last_6_months': timedelta(days=180),
            'last_year': timedelta(days=365)
        }
        
        delta = time_deltas.get(time_range)
        if delta:
            start_time = now - delta
            return start_time.isoformat() + 'Z'
        
        return None
    
    @classmethod
    def _build_absolute_time_filter(cls, time_range: dict) -> Optional[str]:
        """Builds ISO8601 string for absolute time ranges with start/end dates."""
        # For cosmos_client compatibility, we'll use start_date as the >= filter
        # More complex range filtering would need to be handled in cosmos_client
        if 'start_date' in time_range:
            try:
                start_date = datetime.fromisoformat(time_range['start_date'])
                return start_date.isoformat() + 'Z'
            except ValueError:
                logger.warning(f"Invalid start_date format: {time_range['start_date']}")
        
        return None

class ResultProcessor:
    """
    Processes and enriches search results with metadata and access control validation.
    
    Handles result ranking, metadata enrichment, SAS URL generation,
    and final access control validation.
    """
    
    def __init__(self, rbac_manager: Optional[RBACManager] = None):
        """
        Initialize result processor with RBAC manager.
        
        Args:
            rbac_manager: RBAC manager for access validation
        """
        self.rbac_manager = rbac_manager
    
    def process_results(self, raw_results: List[dict], user_permissions: UserPermissions, 
                       query_text: str, offset: int = 0, limit: int = 10) -> dict:
        """
        Processes raw search results into enriched, paginated response with RBAC filtering.
        
        Args:
            raw_results: Raw results from cosmos vector search
            user_permissions: User's permission object
            query_text: Original search query for relevance scoring
            offset: Pagination offset
            limit: Results per page limit
            
        Returns:
            Processed search response dictionary with metadata
        """
        start_time = datetime.utcnow()
        
        # Apply RBAC filtering
        if self.rbac_manager:
            accessible_results = self.rbac_manager.filter_accessible_documents(
                user_permissions, raw_results
            )
        else:
            accessible_results = raw_results
        
        # Enrich results with metadata
        enriched_results = []
        for result in accessible_results:
            enriched_result = self._enrich_result(result, query_text)
            if enriched_result:
                enriched_results.append(enriched_result)
        
        # Sort by relevance score (descending)
        enriched_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Apply pagination
        total_results = len(enriched_results)
        paginated_results = enriched_results[offset:offset + limit]
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            'results': paginated_results,
            'total_results': total_results,
            'returned_results': len(paginated_results),
            'offset': offset,
            'limit': limit,
            'has_more': offset + len(paginated_results) < total_results,
            'processing_time_ms': processing_time,
            'query': query_text,
            'rbac_applied': bool(self.rbac_manager),
            'user_permissions': {
                'roles': user_permissions.roles,
                'can_access_all': user_permissions.can_access_all
            }
        }
    
    def _enrich_result(self, result: dict, query_text: str) -> Optional[dict]:
        """
        Enriches a single search result with additional metadata and SAS URLs.
        
        Args:
            result: Raw search result from Cosmos DB
            query_text: Original search query
            
        Returns:
            Enriched result dictionary or None if processing fails
        """
        try:
            enriched = result.copy()
            
            # Generate SAS URL for file access
            if 'file_path' in result:
                sas_url = generate_sas_url(result['file_path'])
                if sas_url:
                    enriched['sas_url'] = sas_url
                    enriched['download_url'] = sas_url  # Alias for backward compatibility
            
            # Add file category based on MIME type or file extension
            enriched['file_category'] = self._determine_file_category(result)
            
            # Calculate relevance score (combination of similarity score and other factors)
            enriched['relevance_score'] = self._calculate_relevance_score(result, query_text)
            
            # Format timestamps
            for time_field in ['created_at', 'updated_at', 'last_modified']:
                if time_field in result and result[time_field]:
                    try:
                        if isinstance(result[time_field], str):
                            parsed_time = datetime.fromisoformat(result[time_field].replace('Z', '+00:00'))
                        else:
                            parsed_time = result[time_field]
                        enriched[f'{time_field}_formatted'] = parsed_time.strftime('%Y-%m-%d %H:%M:%S UTC')
                    except (ValueError, AttributeError):
                        logger.debug(f"Could not format timestamp {time_field}: {result[time_field]}")
            
            # Add snippet highlighting (basic implementation)
            if 'content' in result and query_text:
                enriched['highlighted_snippet'] = self._create_highlighted_snippet(
                    result['content'], query_text
                )
            
            # Add access metadata
            enriched['access_metadata'] = {
                'owner_id': result.get('user_id') or result.get('owner_id'),
                'visibility': result.get('visibility', 'private'),
                'shared_with': result.get('shared_with', []),
                'teams': result.get('teams', []),
                'departments': result.get('departments', [])
            }
            
            # Clean up internal fields
            for internal_field in ['vector_embedding', '_rid', '_self', '_etag', '_attachments', '_ts']:
                enriched.pop(internal_field, None)
            
            return enriched
            
        except Exception as e:
            logger.error(f"Failed to enrich result: {str(e)}")
            return None
    
    @classmethod
    def _determine_file_category(cls, result: dict) -> str:
        """Determines file category from MIME type or file extension."""
        mime_type = result.get('mime_type', '').lower()
        file_name = result.get('file_name', '').lower()
        
        # Category mapping based on MIME types
        mime_categories = {
            'text/': FileCategory.TEXT.value,
            'application/pdf': FileCategory.PDF.value,
            'application/vnd.openxmlformats-officedocument.wordprocessingml': FileCategory.DOCUMENT.value,
            'application/msword': FileCategory.DOCUMENT.value,
            'application/vnd.openxmlformats-officedocument.presentationml': FileCategory.PRESENTATION.value,
            'application/vnd.ms-powerpoint': FileCategory.PRESENTATION.value,
            'application/vnd.openxmlformats-officedocument.spreadsheetml': FileCategory.SPREADSHEET.value,
            'application/vnd.ms-excel': FileCategory.SPREADSHEET.value,
            'image/': FileCategory.IMAGE.value,
            'audio/': FileCategory.AUDIO.value,
            'video/': FileCategory.VIDEO.value,
            'application/zip': FileCategory.ARCHIVE.value,
            'application/x-rar': FileCategory.ARCHIVE.value
        }
        
        # Check MIME type first
        for mime_prefix, category in mime_categories.items():
            if mime_type.startswith(mime_prefix):
                return category
        
        # Fall back to file extension
        if file_name:
            extension = file_name.split('.')[-1] if '.' in file_name else ''
            extension_categories = {
                'pdf': FileCategory.PDF.value,
                'doc': FileCategory.DOCUMENT.value,
                'docx': FileCategory.DOCUMENT.value,
                'txt': FileCategory.TEXT.value,
                'md': FileCategory.TEXT.value,
                'ppt': FileCategory.PRESENTATION.value,
                'pptx': FileCategory.PRESENTATION.value,
                'xls': FileCategory.SPREADSHEET.value,
                'xlsx': FileCategory.SPREADSHEET.value,
                'jpg': FileCategory.IMAGE.value,
                'jpeg': FileCategory.IMAGE.value,
                'png': FileCategory.IMAGE.value,
                'gif': FileCategory.IMAGE.value,
                'mp3': FileCategory.AUDIO.value,
                'wav': FileCategory.AUDIO.value,
                'mp4': FileCategory.VIDEO.value,
                'avi': FileCategory.VIDEO.value,
                'zip': FileCategory.ARCHIVE.value,
                'rar': FileCategory.ARCHIVE.value
            }
            
            return extension_categories.get(extension, FileCategory.OTHER.value)
        
        return FileCategory.OTHER.value
    
    @classmethod
    def _calculate_relevance_score(cls, result: dict, query_text: str) -> float:
        """
        Calculates relevance score combining similarity score with other factors.
        
        Args:
            result: Search result with similarity score
            query_text: Original search query
            
        Returns:
            Relevance score between 0 and 1
        """
        base_score = result.get('similarity_score', 0.0)
        
        # Boost score based on various factors
        boost_factors = 1.0
        
        # Title/filename match boost
        title = result.get('title', '') or result.get('file_name', '')
        if title and query_text.lower() in title.lower():
            boost_factors *= 1.2
        
        # Recent documents get slight boost
        try:
            created_at = result.get('created_at')
            if created_at:
                if isinstance(created_at, str):
                    created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                else:
                    created_date = created_at
                
                days_old = (datetime.utcnow().replace(tzinfo=created_date.tzinfo) - created_date).days
                if days_old < 30:  # Documents less than 30 days old get boost
                    boost_factors *= 1.1
        except (ValueError, AttributeError, TypeError):
            pass
        
        # Apply boost but ensure score stays within [0, 1]
        final_score = min(base_score * boost_factors, 1.0)
        return round(final_score, 4)
    
    @classmethod
    def _create_highlighted_snippet(cls, content: str, query_text: str, max_length: int = 200) -> str:
        """
        Creates a highlighted snippet of content around query matches.
        
        Args:
            content: Full content text
            query_text: Search query to highlight
            max_length: Maximum snippet length
            
        Returns:
            Snippet with query terms highlighted
        """
        if not content or not query_text:
            return content[:max_length] if content else ""
        
        # Simple highlighting - find first occurrence of any query word
        query_words = query_text.lower().split()
        content_lower = content.lower()
        
        # Find the earliest position of any query word
        earliest_pos = len(content)
        for word in query_words:
            pos = content_lower.find(word)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
        
        if earliest_pos == len(content):
            # No matches found, return beginning of content
            return content[:max_length] + ("..." if len(content) > max_length else "")
        
        # Create snippet around the match
        start_pos = max(0, earliest_pos - max_length // 3)
        end_pos = min(len(content), start_pos + max_length)
        
        snippet = content[start_pos:end_pos]
        
        # Simple highlighting (wrap matches in **bold**)
        for word in query_words:
            if len(word) > 2:  # Only highlight words longer than 2 characters
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                snippet = pattern.sub(f"**{word}**", snippet)
        
        # Add ellipsis if truncated
        if start_pos > 0:
            snippet = "..." + snippet
        if end_pos < len(content):
            snippet = snippet + "..."
        
        return snippet

class SearchEngine:
    """
    Main search engine class that orchestrates the entire search process with RBAC.
    
    Integrates validation, filtering, vector search, result processing, and access control
    to provide a complete search solution with comprehensive error handling and metrics.
    """
    
    def __init__(self, cosmos_client: Optional[CosmosVectorClient] = None, 
                 rbac_manager: Optional[RBACManager] = None):
        """
        Initialize search engine with required components.
        
        Args:
            cosmos_client: Configured Cosmos DB vector client
            rbac_manager: RBAC manager for access control
        """
        self.cosmos_client = cosmos_client or create_cosmos_client()
        self.rbac_manager = rbac_manager or RBACManager()
        self.validator = SearchValidator(self.rbac_manager)
        self.filter_builder = FilterBuilder(self.rbac_manager)
        self.result_processor = ResultProcessor(self.rbac_manager)
        
        logger.info("SearchEngine initialized with RBAC support")
    
    # Fix for search2.py - Replace the vector search call in the SearchEngine.search method

    def search(self, intent: dict) -> dict:
     """
     Performs comprehensive document search with RBAC filtering.
     """
     search_start_time = datetime.utcnow()
    
     try:
        # Step 1: Validate intent and get user permissions
         logger.debug(f"Starting search for user: {intent.get('user_id')}")
         validated_intent, user_permissions = self.validator.validate_intent(intent)
        
        # Step 2: Generate query embedding
         embedding_start_time = datetime.utcnow()
         try:
             query_embedding = get_embedding(validated_intent['query_text'])
         except Exception as e:
             logger.error(f"Embedding generation failed: {str(e)}")
             raise SearchError(f"Failed to generate query embedding: {str(e)}")
        
         embedding_time_ms = (datetime.utcnow() - embedding_start_time).total_seconds() * 1000
        
        # Step 3: Build metadata filters with RBAC
         filters = self.filter_builder.build_filters(validated_intent, user_permissions)
        
        # Step 4: Execute vector search - FIXED METHOD CALL
         vector_search_start_time = datetime.utcnow()
         try:
            # Use the correct method name: vector_search_cosmos
             raw_results = self.cosmos_client.vector_search_cosmos(
                 embedding=query_embedding,
                 filters={
                    'user_id': validated_intent['user_id'],
                    'platforms': filters.get('platforms'),
                    'file_types': filters.get('file_types'),
                    'time_range': filters.get('time_range')
                },
                top_k=validated_intent.get('limit', 10) * 3  # Get more results for RBAC filtering
            )
         except Exception as e:
             logger.error(f"Vector search execution failed: {str(e)}")
             raise SearchError(f"Search execution failed: {str(e)}")
        
         vector_search_time_ms = (datetime.utcnow() - vector_search_start_time).total_seconds() * 1000
        
        # Step 5: Process and enrich results with RBAC filtering
         result_processing_start_time = datetime.utcnow()
         processed_response = self.result_processor.process_results(
            raw_results=raw_results,
            user_permissions=user_permissions,
            query_text=validated_intent['query_text'],
            offset=validated_intent.get('offset', 0),
            limit=validated_intent.get('limit', 10)
        )
        
         result_processing_time_ms = (datetime.utcnow() - result_processing_start_time).total_seconds() * 1000
        
        # Step 6: Add comprehensive metrics
         total_search_time_ms = (datetime.utcnow() - search_start_time).total_seconds() * 1000
        
         search_metrics = SearchMetrics(
            total_results=processed_response['total_results'],
            search_time_ms=total_search_time_ms,
            embedding_time_ms=embedding_time_ms,
            filter_count=len([k for k in filters.keys() if not k.startswith('_')]),
            top_similarity_score=processed_response['results'][0].get('score', 0.0) if processed_response['results'] else 0.0,
            rbac_filter_time_ms=result_processing_time_ms,
            accessible_documents=len(raw_results)
        )
        
        # Add metrics to response
         processed_response.update({
            'search_metrics': {
                'total_search_time_ms': search_metrics.search_time_ms,
                'embedding_time_ms': search_metrics.embedding_time_ms,
                'vector_search_time_ms': vector_search_time_ms,
                'result_processing_time_ms': result_processing_time_ms,
                'filter_count': search_metrics.filter_count,
                'top_similarity_score': search_metrics.top_similarity_score,
                'documents_before_rbac': search_metrics.accessible_documents,
                'documents_after_rbac': processed_response['total_results']
            },
            'filters_applied': {k: v for k, v in filters.items() if not k.startswith('_')},
            'rbac_metadata': filters.get('_rbac_metadata', {})
        })
        
         logger.info(f"Search completed successfully for user {validated_intent['user_id']}: "
                   f"{processed_response['returned_results']}/{processed_response['total_results']} results "
                   f"in {total_search_time_ms:.2f}ms")
        
         return processed_response
        
     except (SearchError, RBACError):
        # Re-raise our custom exceptions as-is
        raise
     except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}")
        raise SearchError(f"Search failed due to unexpected error: {str(e)}")


# Additional fix: Update the ResultProcessor to handle CosmosVectorClient field names
    def _enrich_result(self, result: dict, query_text: str) -> Optional[dict]:
     """
     Enriches a single search result with additional metadata and SAS URLs.
     Updated to handle CosmosVectorClient field names
     """
     try:
        enriched = result.copy()
        
        # Map CosmosVectorClient fields to expected fields
        field_mappings = {
            'fileName': 'file_name',
            'score': 'similarity_score',
            'text': 'content'
        }
        
        # Apply field mappings
        for cosmos_field, expected_field in field_mappings.items():
            if cosmos_field in result:
                enriched[expected_field] = result[cosmos_field]
        
        # Generate SAS URL for file access - use file_id if available
        file_path = result.get('file_path') or result.get('fileName') or result.get('file_name')
        if file_path:
            sas_url = generate_sas_url(file_path)
            if sas_url:
                enriched['sas_url'] = sas_url
                enriched['download_url'] = sas_url
        
        # Add file category based on MIME type
        enriched['file_category'] = self._determine_file_category(result)
        
        # Calculate relevance score (use 'score' from CosmosVectorClient)
        cosmos_score = result.get('score', 0.0)
        # Convert distance to similarity (assuming cosine distance: similarity = 1 - distance)
        similarity_score = max(0.0, 1.0 - cosmos_score) if cosmos_score <= 1.0 else cosmos_score
        enriched['relevance_score'] = self._calculate_relevance_score_from_similarity(
            similarity_score, query_text, result
        )
        enriched['similarity_score'] = similarity_score
        
        # Format timestamps
        for time_field in ['created_at', 'updated_at', 'last_modified']:
            if time_field in result and result[time_field]:
                try:
                    if isinstance(result[time_field], str):
                        parsed_time = datetime.fromisoformat(result[time_field].replace('Z', '+00:00'))
                    else:
                        parsed_time = result[time_field]
                    enriched[f'{time_field}_formatted'] = parsed_time.strftime('%Y-%m-%d %H:%M:%S UTC')
                except (ValueError, AttributeError):
                    logger.debug(f"Could not format timestamp {time_field}: {result[time_field]}")
        
        # Add snippet highlighting using 'text' field from CosmosVectorClient
        content_text = result.get('text', '') or result.get('content', '')
        if content_text and query_text:
            enriched['highlighted_snippet'] = self._create_highlighted_snippet(
                content_text, query_text
            )
        
        # Add access metadata
        enriched['access_metadata'] = {
            'owner_id': result.get('user_id') or result.get('owner_id'),
            'visibility': result.get('visibility', 'private'),
            'shared_with': result.get('shared_with', []),
            'teams': result.get('teams', []),
            'departments': result.get('departments', [])
        }
        
        # Clean up internal fields
        internal_fields = ['vector_embedding', '_rid', '_self', '_etag', '_attachments', '_ts', 'embedding']
        for field in internal_fields:
            enriched.pop(field, None)
        
        return enriched
        
     except Exception as e:
        logger.error(f"Failed to enrich result: {str(e)}")
        return None

    def _calculate_relevance_score_from_similarity(self, similarity_score: float, query_text: str, result: dict) -> float:
     """
    Calculate relevance score from similarity score with boost factors
    """
     base_score = similarity_score
    
    # Boost score based on various factors
     boost_factors = 1.0
    
    # Title/filename match boost
     title = result.get('fileName', '') or result.get('file_name', '')
     if title and query_text.lower() in title.lower():
        boost_factors *= 1.2
    
    # Recent documents get slight boost
     try:
        created_at = result.get('created_at')
        if created_at:
            if isinstance(created_at, str):
                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                created_date = created_at
            
            days_old = (datetime.utcnow().replace(tzinfo=created_date.tzinfo) - created_date).days
            if days_old < 30:
                boost_factors *= 1.1
     except (ValueError, AttributeError, TypeError):
        pass
    
    # Apply boost but ensure score stays within [0, 1]
     final_score = min(base_score * boost_factors, 1.0)
     return round(final_score, 4)
    
    def get_user_accessible_platforms(self, user_id: str) -> List[str]:
        """
        Get list of platforms accessible to a user based on their permissions.
        
        Args:
            user_id: User ID to check permissions for
            
        Returns:
            List of platform names user can access
        """
        try:
            if self.rbac_manager:
                user_permissions = self.rbac_manager.get_user_permissions(user_id)
                if user_permissions.can_access_all:
                    return [p.value for p in Platform]
                
                # For now, return all platforms - more sophisticated logic could be added
                # to restrict platforms based on teams/departments
                return [p.value for p in Platform]
            else:
                return [p.value for p in Platform]
                
        except Exception as e:
            logger.error(f"Failed to get accessible platforms for user {user_id}: {str(e)}")
            return [Platform.LOCAL.value]  # Default to local only
    
    def health_check(self) -> dict:
        """
        Performs health check on all search engine components.
        
        Returns:
            Health status dictionary
        """
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'components': {}
        }
        
        # Check Cosmos DB connection
        try:
            # Simple connectivity test
            self.cosmos_client.get_database_info()
            health_status['components']['cosmos_db'] = 'healthy'
        except Exception as e:
            health_status['components']['cosmos_db'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'degraded'
        
        # Check RBAC system
        try:
            if self.rbac_manager and self.rbac_manager.supabase:
                # Simple RBAC connectivity test
                self.rbac_manager.supabase.table('profiles').select('id').limit(1).execute()
                health_status['components']['rbac'] = 'healthy'
            else:
                health_status['components']['rbac'] = 'disabled'
        except Exception as e:
            health_status['components']['rbac'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'degraded'
        
        # Check embedding service
        try:
            test_embedding = get_embedding("health check test")
            if test_embedding and len(test_embedding) > 0:
                health_status['components']['embedding_service'] = 'healthy'
            else:
                health_status['components']['embedding_service'] = 'unhealthy: empty embedding'
                health_status['status'] = 'degraded'
        except Exception as e:
            health_status['components']['embedding_service'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'degraded'
        
        return health_status

# Factory Functions and Public API

def create_search_engine(cosmos_client: Optional[CosmosVectorClient] = None,
                        rbac_manager: Optional[RBACManager] = None) -> SearchEngine:
    """
    Factory function to create a fully configured SearchEngine instance.
    
    Args:
        cosmos_client: Optional pre-configured Cosmos client
        rbac_manager: Optional pre-configured RBAC manager
        
    Returns:
        Configured SearchEngine instance
    """
    return SearchEngine(cosmos_client=cosmos_client, rbac_manager=rbac_manager)

def validate_search_intent(intent: dict, rbac_manager: Optional[RBACManager] = None) -> Tuple[dict, UserPermissions]:
    """
    Standalone function to validate search intent with RBAC.
    
    Args:
        intent: Search intent dictionary
        rbac_manager: Optional RBAC manager for permission validation
        
    Returns:
        Tuple of (validated intent, user permissions)
        
    Raises:
        SearchError: If validation fails
        RBACError: If permission validation fails
    """
    validator = SearchValidator(rbac_manager)
    return validator.validate_intent(intent)

def build_search_filters(intent: dict, user_permissions: UserPermissions, 
                        rbac_manager: Optional[RBACManager] = None) -> dict:
    """
    Standalone function to build search filters with RBAC.
    
    Args:
        intent: Validated search intent
        user_permissions: User's permission object
        rbac_manager: Optional RBAC manager
        
    Returns:
        Dictionary of search filters
    """
    filter_builder = FilterBuilder(rbac_manager)
    return filter_builder.build_filters(intent, user_permissions)

def process_search_results(raw_results: List[dict], user_permissions: UserPermissions,
                         query_text: str, offset: int = 0, limit: int = 10,
                         rbac_manager: Optional[RBACManager] = None) -> dict:
    """
    Standalone function to process search results with RBAC.
    
    Args:
        raw_results: Raw search results from Cosmos DB
        user_permissions: User's permission object
        query_text: Original search query
        offset: Pagination offset
        limit: Results limit
        rbac_manager: Optional RBAC manager
        
    Returns:
        Processed search response
    """
    processor = ResultProcessor(rbac_manager)
    return processor.process_results(raw_results, user_permissions, query_text, offset, limit)

# Module-level convenience functions for backward compatibility

def search_documents(intent: dict, cosmos_client: Optional[CosmosVectorClient] = None) -> dict:
    """
    Convenience function for quick document search.
    
    Args:
        intent: Search intent dictionary
        cosmos_client: Optional Cosmos client
        
    Returns:
        Search response dictionary
    """
    search_engine = create_search_engine(cosmos_client=cosmos_client)
    return search_engine.search(intent)

# Export public API
__all__ = [
    # Main classes
    'SearchEngine', 'RBACManager', 'SearchValidator', 'FilterBuilder', 'ResultProcessor',
    
    # Exception classes
    'SearchError', 'RBACError',
    
    # Enums
    'Platform', 'FileCategory', 'Role', 'Permission', 'ResourceScope',
    
    # Data classes
    'UserPermissions', 'SearchMetrics',
    
    # Factory functions
    'create_search_engine', 'create_cosmos_client',
    
    # Standalone functions
    'validate_search_intent', 'build_search_filters', 'process_search_results',
    'search_documents', 'generate_sas_url',
    
    # Utility functions
    'get_embedding'  # Re-exported from utils.openai_client
]
