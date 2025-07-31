import os
import json
import uuid
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify, redirect, session, url_for
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.cosmos import CosmosClient
import requests
from requests_oauthlib import OAuth2Session
import io
import mimetypes
from urllib.parse import urlencode
import base64
import re

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'baigan77')

# Configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING_1')
AZURE_STORAGE_CONTAINER = os.environ.get('AZURE_STORAGE_CONTAINER', 'weezyaifiles')
AZURE_USER_STORAGE_CONTAINER='weez-users-info'

COSMOS_ENDPOINT = os.getenv('COSMOS_ENDPOINT')
COSMOS_KEY = os.getenv('COSMOS_KEY')
COSMOS_DATABASE = os.getenv('COSMOS_DATABASE', 'weezyai')
COSMOS_CONTAINER = os.getenv('COSMOS_CONTAINER', 'files')

# OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
MICROSOFT_CLIENT_ID = os.getenv('MICROSOFT_CLIENT_ID')
MICROSOFT_CLIENT_SECRET = os.getenv('MICROSOFT_CLIENT_SECRET')
DROPBOX_CLIENT_ID = os.getenv('DROPBOX_CLIENT_ID')
DROPBOX_CLIENT_SECRET = os.getenv('DROPBOX_CLIENT_SECRET')
SLACK_CLIENT_ID = os.getenv('SLACK_CLIENT_ID')
SLACK_CLIENT_SECRET = os.getenv('SLACK_CLIENT_SECRET')
NOTION_CLIENT_ID = os.environ.get('NOTION_CLIENT_ID')
NOTION_CLIENT_SECRET = os.environ.get('NOTION_CLIENT_SECRET')

# FIXED: Improved base URL configuration
BASE_URL = os.environ.get('BASE_URL', 'https://platform-connection-api-g0b5c3fve2dfb2ag.canadacentral-01.azurewebsites.net')

# FIXED: Better production detection
def is_production():
    """Detect if running in production environment"""
    return (
        os.environ.get('FLASK_ENV') == 'production' or 
        os.environ.get('WEBSITE_SITE_NAME') or  # Azure App Service indicator
        'azurewebsites.net' in os.environ.get('WEBSITE_HOSTNAME', '') or
        BASE_URL.startswith('https://')
    )

app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
app.config['SESSION_COOKIE_SECURE'] = is_production()
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# FIXED: Force HTTPS in production
if is_production():
    app.config['PREFERRED_URL_SCHEME'] = 'https'

def make_session_permanent():
    """Make current session permanent"""
    session.permanent = True

# FIXED: Consistent redirect URI generation for all platforms
def get_redirect_uri(endpoint_name):
    """Generate consistent redirect URI for OAuth callbacks"""
    if is_production():
        endpoint_mapping = {
            'auth_google_callback': f"{BASE_URL}/auth/google/callback",
            'auth_microsoft_callback': f"{BASE_URL}/auth/microsoft/callback", 
            'auth_dropbox_callback': f"{BASE_URL}/auth/dropbox/callback",
            'auth_notion_callback': f"{BASE_URL}/auth/notion/callback",
            'auth_slack_callback': f"{BASE_URL}/auth/slack/callback"  # ADD THIS LINE
        }
        return endpoint_mapping.get(endpoint_name)
    else:
        return url_for(endpoint_name, _external=True)


# Debug function to check redirect URI generation
def debug_redirect_uris():
    """Debug function to check what redirect URIs are being generated"""
    endpoints = ['auth_google_callback', 'auth_microsoft_callback', 'auth_dropbox_callback', 'auth_notion_callback']
    
    uris = {}
    for endpoint in endpoints:
        uris[endpoint] = get_redirect_uri(endpoint)
    
    print("=== REDIRECT URI DEBUG ===")
    for endpoint, uri in uris.items():
        print(f"{endpoint}: {uri}")
    print("===========================")
    
    return uris

def debug_oauth_config():
    config_status = {
        'GOOGLE_CLIENT_ID': bool(GOOGLE_CLIENT_ID),
        'GOOGLE_CLIENT_SECRET': bool(GOOGLE_CLIENT_SECRET),
        'MICROSOFT_CLIENT_ID': bool(MICROSOFT_CLIENT_ID),
        'MICROSOFT_CLIENT_SECRET': bool(MICROSOFT_CLIENT_SECRET),
        'DROPBOX_CLIENT_ID': bool(DROPBOX_CLIENT_ID),
        'DROPBOX_CLIENT_SECRET': bool(DROPBOX_CLIENT_SECRET),
        'NOTION_CLIENT_ID': bool(NOTION_CLIENT_ID),
        'NOTION_CLIENT_SECRET': bool(NOTION_CLIENT_SECRET),
        'BASE_URL': BASE_URL,
        'IS_PRODUCTION': is_production(),
        'FLASK_ENV': os.environ.get('FLASK_ENV'),
        'WEBSITE_SITE_NAME': os.environ.get('WEBSITE_SITE_NAME'),
        'WEBSITE_HOSTNAME': os.environ.get('WEBSITE_HOSTNAME')
    }
    print("OAuth Configuration Status:", config_status)
    return config_status

# Call this on startup
debug_oauth_config()
debug_redirect_uris()

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.xlsx', '.doc', '.ppt', '.xls'}

# Initialize Azure clients
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
database = cosmos_client.get_database_client(COSMOS_DATABASE)
container = database.get_container_client(COSMOS_CONTAINER)

# RBAC Helper Functions
def extract_departments_from_path(file_path):
    """Extract departments from file path based on directory structure"""
    departments = []
    
    # Normalize path separators
    path = file_path.replace('\\', '/').lower()
    
    # Define department patterns and keywords
    department_patterns = {
        'marketing': ['marketing', 'brand', 'advertising', 'promotion', 'campaign'],
        'engineering': ['engineering', 'development', 'dev', 'tech', 'software', 'backend', 'frontend'],
        'sales': ['sales', 'revenue', 'deals', 'prospects', 'crm'],
        'hr': ['hr', 'human-resources', 'people', 'talent', 'recruitment', 'hiring'],
        'finance': ['finance', 'accounting', 'budget', 'financial', 'accounting'],
        'operations': ['operations', 'ops', 'logistics', 'supply-chain'],
        'legal': ['legal', 'contracts', 'compliance', 'regulatory'],
        'product': ['product', 'pm', 'product-management', 'roadmap'],
        'design': ['design', 'ui', 'ux', 'creative', 'graphics'],
        'data': ['data', 'analytics', 'data-science', 'bi', 'reporting'],
        'security': ['security', 'infosec', 'cybersecurity', 'privacy'],
        'support': ['support', 'customer-service', 'help-desk', 'customer-success']
    }
    
    # Check for cross-functional patterns
    cross_functional_patterns = [
        'company-wide', 'all-hands', 'cross-functional', 'multi-department',
        'organization', 'company', 'global', 'enterprise'
    ]
    
    # Split path into segments
    path_segments = [seg.strip() for seg in path.split('/') if seg.strip()]
    
    # Check for cross-functional indicators
    is_cross_functional = any(pattern in path for pattern in cross_functional_patterns)
    
    if is_cross_functional:
        # For cross-functional files, try to identify specific departments mentioned
        for dept, keywords in department_patterns.items():
            if any(keyword in path for keyword in keywords):
                departments.append(dept)
        
        # If no specific departments found in cross-functional, mark as company-wide
        if not departments:
            departments = ['company-wide']
    else:
        # Regular department detection
        for dept, keywords in department_patterns.items():
            if any(keyword in path for keyword in keywords):
                departments.append(dept)
    
    # If no departments detected, try to infer from common folder structures
    if not departments:
        for segment in path_segments:
            for dept, keywords in department_patterns.items():
                if any(keyword in segment for keyword in keywords):
                    departments.append(dept)
                    break
    
    # Default fallback
    if not departments:
        departments = ['general']
    
    return list(set(departments))  # Remove duplicates

def normalize_user_info(email_or_id, platform, access_token=None):
    """Normalize user information across platforms to get consistent user data"""
    if not email_or_id:
        return None
    
    # If it's already an email, return it
    if '@' in str(email_or_id):
        return {
            'email': email_or_id.lower().strip(),
            'platform_id': email_or_id,
            'display_name': email_or_id.split('@')[0],
            'platform': platform
        }
    
    # For platform-specific IDs, try to resolve to email/name
    user_info = {
        'email': None,
        'platform_id': email_or_id,
        'display_name': None,
        'platform': platform
    }
    
    try:
        if platform == 'slack' and access_token:
            headers = {'Authorization': f'Bearer {access_token}'}
            user_response = requests.get(
                'https://slack.com/api/users.info',
                headers=headers,
                params={'user': email_or_id}
            )
            if user_response.status_code == 200 and user_response.json().get('ok'):
                user_data = user_response.json().get('user', {})
                profile = user_data.get('profile', {})
                user_info.update({
                    'email': profile.get('email'),
                    'display_name': profile.get('display_name') or profile.get('real_name'),
                })
    except Exception as e:
        print(f"Error resolving user info for {email_or_id} on {platform}: {e}")
    
    return user_info

def extract_enhanced_sharing_info(file_info, platform, access_token):
    """Extract comprehensive sharing information with user details"""
    sharing_info = {
        'owners': [],
        'editors': [],
        'viewers': [],
        'public_access': False,
        'link_sharing': False,
        'domain_sharing': None,
        'groups': [],
        'channels': []
    }
    
    try:
        if platform == 'google_drive':
            headers = {'Authorization': f'Bearer {access_token}'}
            file_id = file_info.get('id')
            
            permissions_url = f'https://www.googleapis.com/drive/v3/files/{file_id}/permissions'
            response = requests.get(permissions_url, headers=headers)
            
            if response.status_code == 200:
                permissions = response.json().get('permissions', [])
                
                for perm in permissions:
                    user_info = normalize_user_info(
                        perm.get('emailAddress'), platform, access_token
                    )
                    
                    role = perm.get('role')
                    perm_type = perm.get('type')
                    
                    if perm_type == 'anyone':
                        sharing_info['public_access'] = True
                        sharing_info['link_sharing'] = True
                    elif perm_type == 'domain':
                        sharing_info['domain_sharing'] = perm.get('domain')
                    elif user_info and user_info['email']:
                        if role == 'owner':
                            sharing_info['owners'].append(user_info)
                        elif role == 'writer':
                            sharing_info['editors'].append(user_info)
                        elif role == 'reader':
                            sharing_info['viewers'].append(user_info)
        
        elif platform == 'slack':
            channels = file_info.get('channels', [])
            groups = file_info.get('groups', [])
            ims = file_info.get('ims', [])
            
            sharing_info['public_access'] = file_info.get('is_public', False)
            sharing_info['channels'] = channels + groups + ims
            
            user_id = file_info.get('user')
            if user_id:
                owner_info = normalize_user_info(user_id, platform, access_token)
                if owner_info:
                    sharing_info['owners'].append(owner_info)
        
        elif platform == 'onedrive':
            headers = {'Authorization': f'Bearer {access_token}'}
            file_id = file_info.get('id')
            
            permissions_url = f'https://graph.microsoft.com/v1.0/me/drive/items/{file_id}/permissions'
            response = requests.get(permissions_url, headers=headers)
            
            if response.status_code == 200:
                permissions = response.json().get('value', [])
                
                for perm in permissions:
                    roles = perm.get('roles', [])
                    link = perm.get('link', {})
                    
                    if link:
                        sharing_info['link_sharing'] = True
                        if link.get('scope') == 'anonymous':
                            sharing_info['public_access'] = True
                    
                    granted_to = perm.get('grantedTo', {}) or perm.get('grantedToV2', {})
                    user = granted_to.get('user', {})
                    
                    if user.get('email'):
                        user_info = normalize_user_info(user['email'], platform, access_token)
                        
                        if 'owner' in roles:
                            sharing_info['owners'].append(user_info)
                        elif 'write' in roles:
                            sharing_info['editors'].append(user_info)
                        elif 'read' in roles:
                            sharing_info['viewers'].append(user_info)
        
        elif platform == 'dropbox':
            headers = {'Authorization': f'Bearer {access_token}'}
            file_path = file_info.get('path_lower')
            
            sharing_url = 'https://api.dropboxapi.com/2/sharing/list_file_members'
            response = requests.post(sharing_url, headers=headers, json={'file': file_path})
            
            if response.status_code == 200:
                members = response.json()
                
                for member in members.get('users', []):
                    user = member.get('user', {})
                    access_type = member.get('access_type', {}).get('.tag')
                    
                    if user.get('email'):
                        user_info = normalize_user_info(user['email'], platform, access_token)
                        
                        if access_type == 'owner':
                            sharing_info['owners'].append(user_info)
                        elif access_type == 'editor':
                            sharing_info['editors'].append(user_info)
                        elif access_type == 'viewer':
                            sharing_info['viewers'].append(user_info)
    
    except Exception as e:
        print(f"Error extracting enhanced sharing info: {e}")
    
    return sharing_info

def determine_enhanced_visibility(sharing_info, file_path, platform):
    """Determine file visibility based on enhanced sharing information"""
    if sharing_info.get('public_access'):
        return 'public'
    
    if sharing_info.get('link_sharing'):
        return 'link_shared'
    
    if sharing_info.get('domain_sharing'):
        return 'domain_shared'
    
    total_shared_users = (
        len(sharing_info.get('editors', [])) + 
        len(sharing_info.get('viewers', []))
    )
    
    if total_shared_users == 0:
        return 'private'
    elif total_shared_users <= 5:
        return 'team_shared'
    elif total_shared_users <= 20:
        return 'department_shared'
    else:
        return 'organization_shared'

def determine_visibility(file_path, platform, shared_with=None, created_by=None):
    """Determine file visibility based on path, platform, and sharing info"""
    path_lower = file_path.lower()
    
    # Public indicators
    public_indicators = ['public', 'open', 'everyone', 'all-access', 'external']
    if any(indicator in path_lower for indicator in public_indicators):
        return 'public'
    
    # Private indicators
    private_indicators = ['private', 'personal', 'confidential', 'restricted']
    if any(indicator in path_lower for indicator in private_indicators):
        return 'private'
    
    # Department-specific indicators
    department_indicators = ['department', 'team', 'group', 'unit']
    if any(indicator in path_lower for indicator in department_indicators):
        return 'department'
    
    # Check sharing information
    if shared_with and len(shared_with) > 0:
        if len(shared_with) > 10:  # Shared with many people
            return 'internal'
        else:
            return 'department'
    
    # Platform-based defaults
    platform_defaults = {
        'google_drive': 'internal',
        'onedrive': 'internal', 
        'dropbox': 'department',
        'notion': 'internal'
    }
    
    return platform_defaults.get(platform, 'internal')

def generate_sas_url(blob_name, container_name=None):
    """Generate SAS URL for blob with 1-year expiration"""
    try:
        if container_name is None:
            container_name = AZURE_STORAGE_CONTAINER
            
        # Extract account name and key from connection string
        conn_parts = dict(item.split('=', 1) for item in AZURE_STORAGE_CONNECTION_STRING.split(';') if '=' in item)
        account_name = conn_parts.get('AccountName')
        account_key = conn_parts.get('AccountKey')
        
        if not account_name or not account_key:
            print("Could not extract account credentials from connection string")
            return None
        
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
        print(f"Error generating SAS URL: {e}")
        return None

class PlatformIntegration:
    def __init__(self):
        pass
    
    def get_file_extension(self, filename):
        """Extract file extension from filename"""
        return os.path.splitext(filename.lower())[1]
    
    def is_supported_file(self, filename):
        """Check if file extension is supported"""
        return self.get_file_extension(filename) in SUPPORTED_EXTENSIONS
    
    def upload_to_blob_storage(self, file_content, username, filename):
        """Upload file to Azure Blob Storage"""
        try:
            blob_name = f"{username}/{filename}"
            blob_client = blob_service_client.get_blob_client(
                container=AZURE_STORAGE_CONTAINER, 
                blob=blob_name
            )
            blob_client.upload_blob(file_content, overwrite=True)
            return blob_name
        except Exception as e:
            print(f"Error uploading to blob storage: {e}")
            return None
    
    def save_enhanced_file_metadata(self, user_id, filename, blob_path, platform, original_date, 
                                   file_size=None, mime_type=None, file_info=None, access_token=None):
        """Enhanced metadata saving with comprehensive owner/sharing information"""
        try:
            sas_url = generate_sas_url(blob_path)
            departments = extract_departments_from_path(blob_path)
            sharing_info = extract_enhanced_sharing_info(file_info or {}, platform, access_token)
            visibility = determine_enhanced_visibility(sharing_info, blob_path, platform)
            
            owners = sharing_info.get('owners', [])
            all_shared_users = (
                sharing_info.get('editors', []) + 
                sharing_info.get('viewers', [])
            )
            
            syncing_user_info = normalize_user_info(user_id, platform, access_token)
            if syncing_user_info and syncing_user_info not in owners:
                if not owners:
                    owners.append(syncing_user_info)
            
            metadata = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "document_title": os.path.splitext(filename)[0],
                "fileName": filename,
                "filePath": blob_path,
                "platform": platform,
                "source": "platform_sync",
                "uploaded_at": original_date.isoformat() if isinstance(original_date, datetime) else original_date,
                "file_size": file_size,
                "mime_type": mime_type,
                "created_at": datetime.now(timezone.utc).isoformat(),
                
                "sas_url": sas_url,
                "department": departments,
                "visibility": visibility,
                
                "owners": [
                    {
                        "email": owner.get('email'),
                        "display_name": owner.get('display_name'),
                        "platform_id": owner.get('platform_id'),
                        "role": "owner"
                    } for owner in owners
                ],
                
                "shared_with": [
                    {
                        "email": user.get('email'),
                        "display_name": user.get('display_name'),
                        "platform_id": user.get('platform_id'),
                        "role": "editor" if user in sharing_info.get('editors', []) else "viewer",
                        "shared_date": datetime.now(timezone.utc).isoformat()
                    } for user in all_shared_users
                ],
                
                "access_control": {
                    "public_access": sharing_info.get('public_access', False),
                    "link_sharing": sharing_info.get('link_sharing', False),
                    "domain_sharing": sharing_info.get('domain_sharing'),
                    "groups_with_access": sharing_info.get('groups', []),
                    "channels_with_access": sharing_info.get('channels', [])
                },
                
                # Legacy fields for backwards compatibility
                "created_by": [owner.get('email') for owner in owners if owner.get('email')],
                
                "platform_metadata": {
                    **(file_info or {}),
                    "sync_timestamp": datetime.now(timezone.utc).isoformat(),
                    "sharing_last_updated": datetime.now(timezone.utc).isoformat()
                }
            }
            
            container.create_item(body=metadata)
            return metadata
            
        except Exception as e:
            print(f"Error saving enhanced metadata: {e}")
            return None


# Google Drive Integration
class GoogleDriveIntegration(PlatformIntegration):
    def __init__(self):
        super().__init__()
        self.authorization_base_url = 'https://accounts.google.com/o/oauth2/auth'
        self.token_url = 'https://oauth2.googleapis.com/token'
        self.scope = ['https://www.googleapis.com/auth/drive.readonly']
        
    def get_auth_url(self):
        """Get Google OAuth authorization URL"""
        if not GOOGLE_CLIENT_ID:
            raise ValueError("GOOGLE_CLIENT_ID not configured")
            
        redirect_uri = get_redirect_uri('auth_google_callback')
        print(f"Google redirect URI: {redirect_uri}")  # Debug log
        
        google = OAuth2Session(
            GOOGLE_CLIENT_ID,
            scope=self.scope,
            redirect_uri=redirect_uri
        )
        authorization_url, state = google.authorization_url(
            self.authorization_base_url,
            access_type="offline",
            prompt="select_account"
        )
        session['google_oauth_state'] = state
        return authorization_url

    def get_access_token(self, authorization_response):
        """Exchange authorization code for access token"""
        if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
            raise ValueError("Google OAuth credentials not configured")
            
        redirect_uri = get_redirect_uri('auth_google_callback')
        google = OAuth2Session(
            GOOGLE_CLIENT_ID,
            state=session['google_oauth_state'],
            redirect_uri=redirect_uri
        )
        token = google.fetch_token(
            self.token_url,
            authorization_response=authorization_response,
            client_secret=GOOGLE_CLIENT_SECRET
        )
        return token
    
    def get_file_sharing_info(self, file_id, access_token):
        """Get sharing information for a Google Drive file"""
        headers = {'Authorization': f'Bearer {access_token}'}
        
        try:
            # Get file permissions
            permissions_url = f'https://www.googleapis.com/drive/v3/files/{file_id}/permissions'
            response = requests.get(permissions_url, headers=headers)
            
            shared_with = []
            created_by = []
            
            if response.status_code == 200:
                permissions = response.json().get('permissions', [])
                
                for perm in permissions:
                    email = perm.get('emailAddress')
                    role = perm.get('role')
                    perm_type = perm.get('type')
                    
                    if email:
                        if role == 'owner':
                            created_by.append(email)
                        elif perm_type in ['user', 'group'] and role in ['reader', 'writer', 'commenter']:
                            shared_with.append(email)
            
            return shared_with, created_by
            
        except Exception as e:
            print(f"Error getting sharing info for file {file_id}: {e}")
            return [], []
    
    def sync_files(self, access_token, user_email):
        """Sync files from Google Drive with enhanced metadata"""
        headers = {'Authorization': f'Bearer {access_token}'}
        
        # Get files list
        url = 'https://www.googleapis.com/drive/v3/files'
        params = {
            'fields': 'files(id,name,mimeType,createdTime,size,parents,owners,webViewLink)',
            'q': 'trashed=false'
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            return {'error': 'Failed to fetch files from Google Drive'}
        
        files = response.json().get('files', [])
        synced_files = []
        
        for file_info in files:
            filename = file_info.get('name', '')
            if not self.is_supported_file(filename):
                continue
            
            # Download file content
            file_id = file_info['id']
            download_url = f'https://www.googleapis.com/drive/v3/files/{file_id}?alt=media'
            file_response = requests.get(download_url, headers=headers)
            
            if file_response.status_code == 200:
                # Get sharing information
                shared_with, created_by = self.get_file_sharing_info(file_id, access_token)
                
                # Upload to blob storage
                blob_path = self.upload_to_blob_storage(
                    file_response.content, 
                    user_email, 
                    filename
                )
                
                if blob_path:
                    # Save metadata with RBAC fields
                    original_date = file_info.get('createdTime')
                    if original_date:
                        original_date = datetime.fromisoformat(original_date.replace('Z', '+00:00'))
                    
                    # Get owners information
                    owners = file_info.get('owners', [])
                    if owners and not created_by:
                        created_by = [owner.get('emailAddress') for owner in owners if owner.get('emailAddress')]
                    
                    platform_file_info = {
                        'google_file_id': file_id,
                        'web_view_link': file_info.get('webViewLink'),
                        'parents': file_info.get('parents', [])
                    }
                    
                    metadata = self.save_file_metadata(
                        user_email,
                        filename,
                        blob_path,
                        'google_drive',
                        original_date,
                        file_info.get('size'),
                        file_info.get('mimeType'),
                        shared_with,
                        created_by,
                        platform_file_info
                    )
                    
                    if metadata:
                        synced_files.append(metadata)
        
        return {'synced_files': synced_files, 'count': len(synced_files)}

# Microsoft OneDrive Integration
class OneDriveIntegration(PlatformIntegration):
    def __init__(self):
        super().__init__()
        self.authorization_base_url = 'https://login.microsoftonline.com/common/oauth2/v2.0/authorize'
        self.token_url = 'https://login.microsoftonline.com/common/oauth2/v2.0/token'
        self.scope = ['Files.Read', 'User.Read']
    
    def get_auth_url(self):
        """Get Microsoft OAuth authorization URL"""
        if not MICROSOFT_CLIENT_ID:
            raise ValueError("MICROSOFT_CLIENT_ID not configured")
            
        redirect_uri = get_redirect_uri('auth_microsoft_callback')
        print(f"Microsoft redirect URI: {redirect_uri}")  # Debug log
        
        microsoft = OAuth2Session(
            MICROSOFT_CLIENT_ID,
            scope=self.scope,
            redirect_uri=redirect_uri
        )
        authorization_url, state = microsoft.authorization_url(self.authorization_base_url)
        session['microsoft_oauth_state'] = state
        return authorization_url

    def get_access_token(self, authorization_response):
        """Exchange authorization code for access token"""
        if not MICROSOFT_CLIENT_ID or not MICROSOFT_CLIENT_SECRET:
            raise ValueError("Microsoft OAuth credentials not configured")
            
        redirect_uri = get_redirect_uri('auth_microsoft_callback')
        microsoft = OAuth2Session(
            MICROSOFT_CLIENT_ID,
            state=session['microsoft_oauth_state'],
            redirect_uri=redirect_uri
        )
        token = microsoft.fetch_token(
            self.token_url,
            authorization_response=authorization_response,
            client_secret=MICROSOFT_CLIENT_SECRET
        )
        return token
    
    def get_file_sharing_info(self, file_id, access_token):
        """Get sharing information for a OneDrive file"""
        headers = {'Authorization': f'Bearer {access_token}'}
        
        try:
            # Get file permissions
            permissions_url = f'https://graph.microsoft.com/v1.0/me/drive/items/{file_id}/permissions'
            response = requests.get(permissions_url, headers=headers)
            
            shared_with = []
            created_by = []
            
            if response.status_code == 200:
                permissions = response.json().get('value', [])
                
                for perm in permissions:
                    granted_to = perm.get('grantedTo', {})
                    granted_to_v2 = perm.get('grantedToV2', {})
                    roles = perm.get('roles', [])
                    
                    # Extract email from different permission structures
                    email = None
                    if granted_to.get('user', {}).get('email'):
                        email = granted_to['user']['email']
                    elif granted_to_v2.get('user', {}).get('email'):
                        email = granted_to_v2['user']['email']
                    
                    if email:
                        if 'owner' in roles:
                            created_by.append(email)
                        elif any(role in roles for role in ['read', 'write', 'sp.full control']):
                            shared_with.append(email)
            
            return shared_with, created_by
            
        except Exception as e:
            print(f"Error getting sharing info for file {file_id}: {e}")
            return [], []
    
    def sync_files(self, access_token, user_email):
        """Sync files from OneDrive with enhanced metadata"""
        headers = {'Authorization': f'Bearer {access_token}'}
        
        # Get files list
        url = 'https://graph.microsoft.com/v1.0/me/drive/root/children'
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return {'error': 'Failed to fetch files from OneDrive'}
        
        files = response.json().get('value', [])
        synced_files = []
        
        for file_info in files:
            if file_info.get('folder'):  # Skip folders
                continue
                
            filename = file_info.get('name', '')
            if not self.is_supported_file(filename):
                continue
            
            # Download file content
            download_url = file_info.get('@microsoft.graph.downloadUrl')
            if download_url:
                file_response = requests.get(download_url)
                
                if file_response.status_code == 200:
                    # Get sharing information
                    file_id = file_info.get('id')
                    shared_with, created_by = self.get_file_sharing_info(file_id, access_token)
                    
                    # Upload to blob storage
                    blob_path = self.upload_to_blob_storage(
                        file_response.content, 
                        user_email, 
                        filename
                    )
                    
                    if blob_path:
                        # Save metadata with RBAC fields
                        original_date = file_info.get('createdDateTime')
                        if original_date:
                            original_date = datetime.fromisoformat(original_date.replace('Z', '+00:00'))
                        
                        # Get creator information
                        created_by_info = file_info.get('createdBy', {}).get('user', {})
                        if created_by_info.get('email') and not created_by:
                            created_by = [created_by_info['email']]
                        
                        platform_file_info = {
                            'onedrive_file_id': file_id,
                            'web_url': file_info.get('webUrl'),
                            'parent_reference': file_info.get('parentReference', {})
                        }
                        
                        metadata = self.save_file_metadata(
                            user_email,
                            filename,
                            blob_path,
                            'onedrive',
                            original_date,
                            file_info.get('size'),
                            file_info.get('file', {}).get('mimeType'),
                            shared_with,
                            created_by,
                            platform_file_info
                        )
                        
                        if metadata:
                            synced_files.append(metadata)
        
        return {'synced_files': synced_files, 'count': len(synced_files)}

# Dropbox Integration  
class DropboxIntegration(PlatformIntegration):
    def __init__(self):
        super().__init__()
        self.authorization_base_url = 'https://www.dropbox.com/oauth2/authorize'
        self.token_url = 'https://api.dropbox.com/oauth2/token'
    
    def get_auth_url(self, user_email=None):
        """Get Dropbox OAuth authorization URL"""
        if not DROPBOX_CLIENT_ID:
            raise ValueError("DROPBOX_CLIENT_ID not configured")
            
        redirect_uri = get_redirect_uri('auth_dropbox_callback')
        print(f"Dropbox redirect URI: {redirect_uri}")  # Debug log
        
        state = str(uuid.uuid4())
        
        # Store user_email in session with state
        session[f'dropbox_oauth_state_{state}'] = {
            'state': state,
            'user_email': user_email
        }
        
        params = {
            'client_id': DROPBOX_CLIENT_ID,
            'redirect_uri': redirect_uri,
            'response_type': 'code',
            'state': state
        }
        return f"{self.authorization_base_url}?{urlencode(params)}"
    
    def get_access_token(self, authorization_code, state):
        """Exchange authorization code for access token"""
        if not DROPBOX_CLIENT_ID or not DROPBOX_CLIENT_SECRET:
            raise ValueError("Dropbox OAuth credentials not configured")
            
        # Verify state and get user_email
        state_data = session.get(f'dropbox_oauth_state_{state}')
        if not state_data or state_data['state'] != state:
            raise ValueError("State mismatch - possible CSRF attack")
            
        redirect_uri = get_redirect_uri('auth_dropbox_callback')
        data = {
            'code': authorization_code,
            'grant_type': 'authorization_code',
            'client_id': DROPBOX_CLIENT_ID,
            'client_secret': DROPBOX_CLIENT_SECRET,
            'redirect_uri': redirect_uri
        }
        
        response = requests.post(self.token_url, data=data)
        if response.status_code != 200:
            raise ValueError(f"Failed to get access token: {response.text}")
        
        # Clean up session
        del session[f'dropbox_oauth_state_{state}']
        
        return response.json(), state_data['user_email']
    
    def get_file_sharing_info(self, file_path, access_token):
        """Get sharing information for a Dropbox file"""
        headers = {'Authorization': f'Bearer {access_token}'}
        
        try:
            # Get file sharing information
            sharing_url = 'https://api.dropboxapi.com/2/sharing/list_file_members'
            data = {'file': file_path}
            
            response = requests.post(sharing_url, headers=headers, json=data)
            
            shared_with = []
            created_by = []
            
            if response.status_code == 200:
                members = response.json()
                
                # Get users who have access
                for member in members.get('users', []):
                    email = member.get('user', {}).get('email')
                    access_type = member.get('access_type', {}).get('.tag')
                    
                    if email:
                        if access_type == 'owner':
                            created_by.append(email)
                        elif access_type in ['editor', 'viewer']:
                            shared_with.append(email)
                
                # Get groups who have access
                for group in members.get('groups', []):
                    group_name = group.get('group', {}).get('group_name')
                    access_type = group.get('access_type', {}).get('.tag')
                    
                    if group_name and access_type in ['editor', 'viewer']:
                        shared_with.append(f"group:{group_name}")
            
            return shared_with, created_by
            
        except Exception as e:
            print(f"Error getting sharing info for file {file_path}: {e}")
            return [], []
    
    def sync_files(self, access_token, user_email):
        """Sync files from Dropbox with enhanced metadata"""
        headers = {'Authorization': f'Bearer {access_token}'}
        
        # Get files list
        url = 'https://api.dropboxapi.com/2/files/list_folder'
        data = {'path': '', 'recursive': True}
        
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            return {'error': 'Failed to fetch files from Dropbox'}
        
        files = response.json().get('entries', [])
        synced_files = []
        
        for file_info in files:
            if file_info.get('.tag') != 'file':  # Skip folders
                continue
                
            filename = file_info.get('name', '')
            if not self.is_supported_file(filename):
                continue
            
            # Download file content
            file_path = file_info.get('path_lower')
            download_url = 'https://content.dropboxapi.com/2/files/download'
            download_headers = {
                **headers,
                'Dropbox-API-Arg': json.dumps({'path': file_path})
            }
            
            file_response = requests.post(download_url, headers=download_headers)
            
            if file_response.status_code == 200:
                # Get sharing information
                shared_with, created_by = self.get_file_sharing_info(file_path, access_token)
                
                # Upload to blob storage
                blob_path = self.upload_to_blob_storage(
                    file_response.content, 
                    user_email, 
                    filename
                )
                
                if blob_path:
                    # Save metadata with RBAC fields
                    original_date = file_info.get('client_modified')
                    if original_date:
                        original_date = datetime.fromisoformat(original_date.replace('Z', '+00:00'))
                    
                    platform_file_info = {
                        'dropbox_file_id': file_info.get('id'),
                        'path_lower': file_path,
                        'path_display': file_info.get('path_display'),
                        'content_hash': file_info.get('content_hash')
                    }
                    
                    metadata = self.save_file_metadata(
                        user_email,
                        filename,
                        blob_path,
                        'dropbox',
                        original_date,
                        file_info.get('size'),
                        None,  # Dropbox doesn't provide mime type directly
                        shared_with,
                        created_by,
                        platform_file_info
                    )
                    
                    if metadata:
                        synced_files.append(metadata)
        
        return {'synced_files': synced_files, 'count': len(synced_files)}

# Notion Integration
class NotionIntegration(PlatformIntegration):
    def __init__(self):
        super().__init__()
        self.authorization_base_url = 'https://api.notion.com/v1/oauth/authorize'
        self.token_url = 'https://api.notion.com/v1/oauth/token'
    
    def get_auth_url(self, user_email=None):
        """Get Notion OAuth authorization URL"""
        if not NOTION_CLIENT_ID:
            raise ValueError("NOTION_CLIENT_ID not configured")
            
        redirect_uri = get_redirect_uri('auth_notion_callback')
        print(f"Notion redirect URI: {redirect_uri}")  # Debug log
        
        state = str(uuid.uuid4())
        
        # Store user_email in session with state
        session[f'notion_oauth_state_{state}'] = {
            'state': state,
            'user_email': user_email
        }
        
        params = {
            'client_id': NOTION_CLIENT_ID,
            'redirect_uri': redirect_uri,
            'response_type': 'code',
            'owner': 'user',
            'state': state
        }
        return f"{self.authorization_base_url}?{urlencode(params)}"
    
    def get_access_token(self, authorization_code, state):
        """Exchange authorization code for access token"""
        if not NOTION_CLIENT_ID or not NOTION_CLIENT_SECRET:
            raise ValueError("Notion OAuth credentials not configured")
            
        # Verify state and get user_email
        state_data = session.get(f'notion_oauth_state_{state}')
        if not state_data or state_data['state'] != state:
            raise ValueError("State mismatch - possible CSRF attack")
            
        redirect_uri = get_redirect_uri('auth_notion_callback')
        
        # Encode credentials for basic auth
        credentials = base64.b64encode(f"{NOTION_CLIENT_ID}:{NOTION_CLIENT_SECRET}".encode()).decode()
        
        headers = {
            'Authorization': f'Basic {credentials}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'grant_type': 'authorization_code',
            'code': authorization_code,
            'redirect_uri': redirect_uri
        }
        
        response = requests.post(self.token_url, headers=headers, json=data)
        if response.status_code != 200:
            raise ValueError(f"Failed to get access token: {response.text}")
        
        # Clean up session
        del session[f'notion_oauth_state_{state}']
        
        return response.json(), state_data['user_email']
    
    def export_page_as_pdf(self, page_id, access_token):
        """Export Notion page as PDF"""
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Notion-Version': '2022-06-28'
        }
        
        try:
            # Get page content
            page_url = f'https://api.notion.com/v1/pages/{page_id}'
            page_response = requests.get(page_url, headers=headers)
            
            if page_response.status_code != 200:
                return None, None
            
            page_data = page_response.json()
            page_title = page_data.get('properties', {}).get('title', {}).get('title', [{}])[0].get('plain_text', 'Untitled')
            
            # Get page blocks
            blocks_url = f'https://api.notion.com/v1/blocks/{page_id}/children'
            blocks_response = requests.get(blocks_url, headers=headers)
            
            if blocks_response.status_code != 200:
                return None, None
            
            blocks = blocks_response.json().get('results', [])
            
            # Convert to simple text (in a real implementation, you'd want to use a proper PDF generator)
            content = f"# {page_title}\n\n"
            for block in blocks:
                block_type = block.get('type')
                if block_type == 'paragraph':
                    text = ''
                    for rich_text in block.get('paragraph', {}).get('rich_text', []):
                        text += rich_text.get('plain_text', '')
                    content += text + '\n\n'
                elif block_type == 'heading_1':
                    text = ''
                    for rich_text in block.get('heading_1', {}).get('rich_text', []):
                        text += rich_text.get('plain_text', '')
                    content += f"# {text}\n\n"
                elif block_type == 'heading_2':
                    text = ''
                    for rich_text in block.get('heading_2', {}).get('rich_text', []):
                        text += rich_text.get('plain_text', '')
                    content += f"## {text}\n\n"
            
            # Convert text to PDF-like format (simplified)
            pdf_content = content.encode('utf-8')
            filename = f"{page_title}.txt"  # In reality, you'd generate a proper PDF
            
            return pdf_content, filename
            
        except Exception as e:
            print(f"Error exporting page {page_id}: {e}")
            return None, None
    
    def sync_files(self, access_token, user_email):
        """Sync files from Notion with enhanced metadata"""
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Notion-Version': '2022-06-28'
        }
        
        # Search for pages
        search_url = 'https://api.notion.com/v1/search'
        data = {
            'filter': {
                'property': 'object',
                'value': 'page'
            }
        }
        
        response = requests.post(search_url, headers=headers, json=data)
        if response.status_code != 200:
            return {'error': 'Failed to fetch pages from Notion'}
        
        pages = response.json().get('results', [])
        synced_files = []
        
        for page_info in pages:
            page_id = page_info.get('id')
            
            # Export page as text/PDF
            content, filename = self.export_page_as_pdf(page_id, access_token)
            
            if content and filename:
                # Upload to blob storage
                blob_path = self.upload_to_blob_storage(
                    content, 
                    user_email, 
                    filename
                )
                
                if blob_path:
                    # Save metadata with RBAC fields
                    original_date = page_info.get('created_time')
                    if original_date:
                        original_date = datetime.fromisoformat(original_date.replace('Z', '+00:00'))
                    
                    # Get creator information
                    created_by = []
                    created_by_info = page_info.get('created_by', {})
                    if created_by_info.get('type') == 'person':
                        person_email = created_by_info.get('person', {}).get('email')
                        if person_email:
                            created_by = [person_email]
                    
                    platform_file_info = {
                        'notion_page_id': page_id,
                        'notion_url': page_info.get('url'),
                        'parent': page_info.get('parent', {}),
                        'archived': page_info.get('archived', False)
                    }
                    
                    metadata = self.save_file_metadata(
                        user_email,
                        filename,
                        blob_path,
                        'notion',
                        original_date,
                        len(content),
                        'text/plain',
                        [],  # Notion doesn't have traditional sharing like other platforms
                        created_by,
                        platform_file_info
                    )
                    
                    if metadata:
                        synced_files.append(metadata)
        
        return {'synced_files': synced_files, 'count': len(synced_files)}
    
class SlackIntegration(PlatformIntegration):
    def __init__(self):
        super().__init__()
        self.authorization_base_url = 'https://slack.com/oauth/v2/authorize'
        self.token_url = 'https://slack.com/api/oauth.v2.access'
        self.scope = ['files:read', 'channels:read', 'groups:read', 'im:read', 'mpim:read', 'users:read', 'users:read.email']
    
    def get_auth_url(self, user_email=None):
        """Get Slack OAuth authorization URL"""
        if not SLACK_CLIENT_ID:
            raise ValueError("SLACK_CLIENT_ID not configured")
            
        redirect_uri = get_redirect_uri('auth_slack_callback')
        print(f"Slack redirect URI: {redirect_uri}")
        
        state = str(uuid.uuid4())
        
        session[f'slack_oauth_state_{state}'] = {
            'state': state,
            'user_email': user_email
        }
        
        params = {
            'client_id': SLACK_CLIENT_ID,
            'redirect_uri': redirect_uri,
            'scope': ' '.join(self.scope),
            'state': state,
            'response_type': 'code'
        }
        return f"{self.authorization_base_url}?{urlencode(params)}"
    
    def get_access_token(self, authorization_code, state):
        """Exchange authorization code for access token"""
        if not SLACK_CLIENT_ID or not SLACK_CLIENT_SECRET:
            raise ValueError("Slack OAuth credentials not configured")
            
        state_data = session.get(f'slack_oauth_state_{state}')
        if not state_data or state_data['state'] != state:
            raise ValueError("State mismatch - possible CSRF attack")
            
        redirect_uri = get_redirect_uri('auth_slack_callback')
        
        data = {
            'client_id': SLACK_CLIENT_ID,
            'client_secret': SLACK_CLIENT_SECRET,
            'code': authorization_code,
            'redirect_uri': redirect_uri
        }
        
        response = requests.post(self.token_url, data=data)
        if response.status_code != 200:
            raise ValueError(f"Failed to get access token: {response.text}")
        
        token_data = response.json()
        if not token_data.get('ok'):
            raise ValueError(f"Slack OAuth error: {token_data.get('error', 'Unknown error')}")
        
        del session[f'slack_oauth_state_{state}']
        
        return token_data, state_data['user_email']
    
    def get_file_sharing_info(self, file_info, access_token):
        """Get sharing information for a Slack file"""
        return extract_enhanced_sharing_info(file_info, 'slack', access_token)
    
    def download_slack_file(self, file_url, access_token):
        """Download file from Slack using the private download URL"""
        headers = {'Authorization': f'Bearer {access_token}'}
        
        try:
            response = requests.get(file_url, headers=headers)
            if response.status_code == 200:
                return response.content
            else:
                print(f"Failed to download file from {file_url}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading file from Slack: {e}")
            return None
    
    def sync_files(self, access_token, user_email):
        """Sync files from Slack with enhanced metadata"""
        headers = {'Authorization': f'Bearer {access_token}'}
        
        synced_files = []
        
        try:
            files_list_url = 'https://slack.com/api/files.list'
            params = {
                'count': 100,
                'types': 'all'
            }
            
            response = requests.get(files_list_url, headers=headers, params=params)
            if response.status_code != 200:
                return {'error': 'Failed to fetch files from Slack'}
            
            response_data = response.json()
            if not response_data.get('ok'):
                return {'error': f"Slack API error: {response_data.get('error', 'Unknown error')}"}
            
            files = response_data.get('files', [])
            
            for file_info in files:
                filename = file_info.get('name', '')
                
                file_content = None
                is_supported = self.is_supported_file(filename)
                
                if is_supported:
                    private_url = file_info.get('url_private_download') or file_info.get('url_private')
                    if private_url:
                        file_content = self.download_slack_file(private_url, access_token)
                else:
                    file_info_text = f"Slack File Metadata\n"
                    file_info_text += f"Original filename: {filename}\n"
                    file_info_text += f"File type: {file_info.get('filetype', 'unknown')}\n"
                    file_info_text += f"Size: {file_info.get('size', 0)} bytes\n"
                    file_info_text += f"Title: {file_info.get('title', 'No title')}\n"
                    file_info_text += f"URL: {file_info.get('permalink', 'No URL')}\n"
                    
                    file_content = file_info_text.encode('utf-8')
                    filename = f"{os.path.splitext(filename)[0]}_metadata.txt"
                
                if file_content:
                    sharing_info = self.get_file_sharing_info(file_info, access_token)
                    
                    blob_path = self.upload_to_blob_storage(
                        file_content, 
                        user_email, 
                        filename
                    )
                    
                    if blob_path:
                        created_timestamp = file_info.get('created')
                        original_date = None
                        if created_timestamp:
                            original_date = datetime.fromtimestamp(created_timestamp, tz=timezone.utc)
                        
                        platform_file_info = {
                            'slack_file_id': file_info.get('id'),
                            'slack_team_id': file_info.get('team'),
                            'permalink': file_info.get('permalink'),
                            'channels': file_info.get('channels', []),
                            'groups': file_info.get('groups', []),
                            'ims': file_info.get('ims', []),
                            'original_filetype': file_info.get('filetype'),
                            'is_external': file_info.get('is_external', False),
                            'is_public': file_info.get('is_public', False)
                        }
                        
                        metadata = self.save_enhanced_file_metadata(
                            user_email,
                            filename,
                            blob_path,
                            'slack',
                            original_date,
                            file_info.get('size'),
                            file_info.get('mimetype'),
                            file_info,
                            access_token
                        )
                        
                        if metadata:
                            synced_files.append(metadata)
        
        except Exception as e:
            print(f"Error syncing Slack files: {e}")
            return {'error': str(e)}
        
        return {'synced_files': synced_files, 'count': len(synced_files)}



# Initialize platform integrations
google_drive = GoogleDriveIntegration()
onedrive = OneDriveIntegration()
dropbox = DropboxIntegration()
notion = NotionIntegration()
slack = SlackIntegration()

# Helper function to save user platform tokens
def save_user_platform_token(user_email, platform, token_data):
    """Save user's platform access token to blob storage"""
    try:
        user_blob_client = blob_service_client.get_blob_client(
            container=AZURE_USER_STORAGE_CONTAINER,
            blob=f"{user_email}/{platform}_token.json"
        )
        
        token_json = json.dumps(token_data)
        user_blob_client.upload_blob(token_json, overwrite=True)
        return True
    except Exception as e:
        print(f"Error saving token for {user_email} - {platform}: {e}")
        return False

def get_user_platform_token(user_email, platform):
    """Retrieve user's platform access token from blob storage"""
    try:
        user_blob_client = blob_service_client.get_blob_client(
            container=AZURE_USER_STORAGE_CONTAINER,
            blob=f"{user_email}/{platform}_token.json"
        )
        
        blob_data = user_blob_client.download_blob()
        token_data = json.loads(blob_data.readall())
        return token_data
    except Exception as e:
        print(f"Error retrieving token for {user_email} - {platform}: {e}")
        return None

# Flask Routes
@app.route('/')
def index():
    """Main index route"""
    return jsonify({
        'message': 'Platform Connection API',
        'version': '1.0',
        'supported_platforms': ['google_drive', 'onedrive', 'dropbox', 'notion'],
        'endpoints': {
            'auth': {
                'google': '/auth/google',
                'microsoft': '/auth/microsoft', 
                'dropbox': '/auth/dropbox',
                'notion': '/auth/notion'
            },
            'sync': {
                'google': '/sync/google',
                'microsoft': '/sync/microsoft',
                'dropbox': '/sync/dropbox', 
                'notion': '/sync/notion'
            }
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now(timezone.utc).isoformat()})

# Google Drive Routes
@app.route('/auth/google')
def auth_google():
    """Initiate Google Drive OAuth"""
    try:
        user_email = request.args.get('user_email')
        if not user_email:
            return jsonify({'error': 'user_email parameter required'}), 400
        
        session['user_email'] = user_email
        make_session_permanent()
        
        auth_url = google_drive.get_auth_url()
        return redirect(auth_url)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/auth/google/callback')
def auth_google_callback():
    """Handle Google Drive OAuth callback"""
    try:
        user_email = session.get('user_email')
        if not user_email:
            return jsonify({'error': 'Session expired'}), 400
        
        token = google_drive.get_access_token(request.url)
        
        # Save token
        if save_user_platform_token(user_email, 'google_drive', token):
            return jsonify({
                'message': 'Google Drive connected successfully',
                'user_email': user_email,
                'platform': 'google_drive'
            })
        else:
            return jsonify({'error': 'Failed to save token'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sync/google')
def sync_google():
    """Sync files from Google Drive"""
    try:
        user_email = request.args.get('user_email')
        if not user_email:
            return jsonify({'error': 'user_email parameter required'}), 400
        
        # Get stored token
        token_data = get_user_platform_token(user_email, 'google_drive')
        if not token_data:
            return jsonify({'error': 'No Google Drive token found. Please authenticate first.'}), 401
        
        access_token = token_data.get('access_token')
        result = google_drive.sync_files(access_token, user_email)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Microsoft OneDrive Routes
@app.route('/auth/microsoft')
def auth_microsoft():
    """Initiate Microsoft OneDrive OAuth"""
    try:
        user_email = request.args.get('user_email')
        if not user_email:
            return jsonify({'error': 'user_email parameter required'}), 400
        
        session['user_email'] = user_email
        make_session_permanent()
        
        auth_url = onedrive.get_auth_url()
        return redirect(auth_url)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/auth/microsoft/callback')
def auth_microsoft_callback():
    """Handle Microsoft OneDrive OAuth callback"""
    try:
        user_email = session.get('user_email')
        if not user_email:
            return jsonify({'error': 'Session expired'}), 400
        
        token = onedrive.get_access_token(request.url)
        
        # Save token
        if save_user_platform_token(user_email, 'onedrive', token):
            return jsonify({
                'message': 'Microsoft OneDrive connected successfully',
                'user_email': user_email,
                'platform': 'onedrive'
            })
        else:
            return jsonify({'error': 'Failed to save token'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sync/microsoft')
def sync_microsoft():
    """Sync files from Microsoft OneDrive"""
    try:
        user_email = request.args.get('user_email')
        if not user_email:
            return jsonify({'error': 'user_email parameter required'}), 400
        
        # Get stored token
        token_data = get_user_platform_token(user_email, 'onedrive')
        if not token_data:
            return jsonify({'error': 'No OneDrive token found. Please authenticate first.'}), 401
        
        access_token = token_data.get('access_token')
        result = onedrive.sync_files(access_token, user_email)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Dropbox Routes
@app.route('/auth/dropbox')
def auth_dropbox():
    """Initiate Dropbox OAuth"""
    try:
        user_email = request.args.get('user_email')
        if not user_email:
            return jsonify({'error': 'user_email parameter required'}), 400
        
        auth_url = dropbox.get_auth_url(user_email)
        return redirect(auth_url)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/auth/dropbox/callback')
def auth_dropbox_callback():
    """Handle Dropbox OAuth callback"""
    try:
        code = request.args.get('code')
        state = request.args.get('state')
        
        if not code or not state:
            return jsonify({'error': 'Missing authorization code or state'}), 400
        
        token_data, user_email = dropbox.get_access_token(code, state)
        
        # Save token
        if save_user_platform_token(user_email, 'dropbox', token_data):
            return jsonify({
                'message': 'Dropbox connected successfully',
                'user_email': user_email,
                'platform': 'dropbox'
            })
        else:
            return jsonify({'error': 'Failed to save token'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sync/dropbox')
def sync_dropbox():
    """Sync files from Dropbox"""
    try:
        user_email = request.args.get('user_email')
        if not user_email:
            return jsonify({'error': 'user_email parameter required'}), 400
        
        # Get stored token
        token_data = get_user_platform_token(user_email, 'dropbox')
        if not token_data:
            return jsonify({'error': 'No Dropbox token found. Please authenticate first.'}), 401
        
        access_token = token_data.get('access_token')
        result = dropbox.sync_files(access_token, user_email)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Notion Routes
@app.route('/auth/notion')
def auth_notion():
    """Initiate Notion OAuth"""
    try:
        user_email = request.args.get('user_email')
        if not user_email:
            return jsonify({'error': 'user_email parameter required'}), 400
        
        auth_url = notion.get_auth_url(user_email)
        return redirect(auth_url)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/auth/notion/callback')
def auth_notion_callback():
    """Handle Notion OAuth callback"""
    try:
        code = request.args.get('code')
        state = request.args.get('state')
        
        if not code or not state:
            return jsonify({'error': 'Missing authorization code or state'}), 400
        
        token_data, user_email = notion.get_access_token(code, state)
        
        # Save token
        if save_user_platform_token(user_email, 'notion', token_data):
            return jsonify({
                'message': 'Notion connected successfully',
                'user_email': user_email,
                'platform': 'notion'
            })
        else:
            return jsonify({'error': 'Failed to save token'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sync/notion')
def sync_notion():
    """Sync files from Notion"""
    try:
        user_email = request.args.get('user_email')
        if not user_email:
            return jsonify({'error': 'user_email parameter required'}), 400
        
        # Get stored token
        token_data = get_user_platform_token(user_email, 'notion')
        if not token_data:
            return jsonify({'error': 'No Notion token found. Please authenticate first.'}), 401
        
        access_token = token_data.get('access_token')
        result = notion.sync_files(access_token, user_email)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Utility Routes
@app.route('/platforms/status')
def platforms_status():
    """Get status of all platform connections for a user"""
    try:
        user_email = request.args.get('user_email')
        if not user_email:
            return jsonify({'error': 'user_email parameter required'}), 400
        
        platforms = ['google_drive', 'onedrive', 'dropbox', 'notion']
        status = {}
        
        for platform in platforms:
            token_data = get_user_platform_token(user_email, platform)
            status[platform] = {
                'connected': bool(token_data),
                'last_updated': token_data.get('created_at') if token_data else None
            }
        
        return jsonify({
            'user_email': user_email,
            'platforms': status
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/files/user/<user_email>')
def get_user_files():
    """Get all files for a specific user"""
    try:
        user_email = request.path_values.get('user_email')
        if not user_email:
            return jsonify({'error': 'user_email required'}), 400
        
        # Query Cosmos DB for user files
        query = "SELECT * FROM c WHERE c.user_id = @user_email"
        parameters = [{"name": "@user_email", "value": user_email}]
        
        files = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        
        return jsonify({
            'user_email': user_email,
            'files': files,
            'count': len(files)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sync/all')
def sync_all_platforms():
    """Sync files from all connected platforms for a user"""
    try:
        user_email = request.args.get('user_email')
        if not user_email:
            return jsonify({'error': 'user_email parameter required'}), 400
        
        results = {}
        platforms = [
            ('google_drive', google_drive),
            ('onedrive', onedrive), 
            ('dropbox', dropbox),
            ('notion', notion)
        ]
        
        for platform_name, platform_obj in platforms:
            try:
                token_data = get_user_platform_token(user_email, platform_name)
                if token_data:
                    access_token = token_data.get('access_token')
                    result = platform_obj.sync_files(access_token, user_email)
                    results[platform_name] = result
                else:
                    results[platform_name] = {'error': 'Not connected'}
            except Exception as e:
                results[platform_name] = {'error': str(e)}
        
        return jsonify({
            'user_email': user_email,
            'sync_results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/auth/slack')
def auth_slack():
    """Initiate Slack OAuth"""
    try:
        user_email = request.args.get('user_email')
        if not user_email:
            return jsonify({'error': 'user_email parameter required'}), 400
        
        auth_url = slack.get_auth_url(user_email)
        return redirect(auth_url)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/auth/slack/callback')
def auth_slack_callback():
    """Handle Slack OAuth callback"""
    try:
        code = request.args.get('code')
        state = request.args.get('state')
        error = request.args.get('error')
        
        if error:
            return jsonify({'error': f'Slack OAuth error: {error}'}), 400
        
        if not code or not state:
            return jsonify({'error': 'Missing authorization code or state'}), 400
        
        token_data, user_email = slack.get_access_token(code, state)
        
        if save_user_platform_token(user_email, 'slack', token_data):
            return jsonify({
                'message': 'Slack connected successfully',
                'user_email': user_email,
                'platform': 'slack',
                'team_name': token_data.get('team', {}).get('name', 'Unknown team')
            })
        else:
            return jsonify({'error': 'Failed to save token'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sync/slack')
def sync_slack():
    """Sync files from Slack"""
    try:
        user_email = request.args.get('user_email')
        if not user_email:
            return jsonify({'error': 'user_email parameter required'}), 400
        
        token_data = get_user_platform_token(user_email, 'slack')
        if not token_data:
            return jsonify({'error': 'No Slack token found. Please authenticate first.'}), 401
        
        access_token = token_data.get('access_token')
        result = slack.sync_files(access_token, user_email)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/files/owned-by/<owner_email>')
def get_files_owned_by(owner_email):
    """API endpoint to get files owned by a specific user"""
    try:
        user_id = request.args.get('user_id')
        
        query = """
        SELECT * FROM c 
        WHERE ARRAY_CONTAINS(c.owners, {'email': @owner_email}, true)
        """
        parameters = [{"name": "@owner_email", "value": owner_email}]
        
        if user_id:
            query += " AND c.user_id = @user_id"
            parameters.append({"name": "@user_id", "value": user_id})
        
        files = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        
        return jsonify({
            'owner_email': owner_email,
            'files': files,
            'count': len(files)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/files/shared-with/<user_email>')  
def get_files_shared_with(user_email):
    """API endpoint to get files shared with a specific user"""
    try:
        user_id = request.args.get('user_id')
        
        query = """
        SELECT * FROM c 
        WHERE ARRAY_CONTAINS(c.shared_with, {'email': @user_email}, true)
        """
        parameters = [{"name": "@user_email", "value": user_email}]
        
        if user_id:
            query += " AND c.user_id = @user_id"
            parameters.append({"name": "@user_id", "value": user_id})
        
        files = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        
        return jsonify({
            'shared_with': user_email,
            'files': files,
            'count': len(files)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/files/by-visibility/<visibility>')
def get_files_by_visibility_level(visibility):
    """API endpoint to get files by visibility level"""
    try:
        user_id = request.args.get('user_id')
        
        query = "SELECT * FROM c WHERE c.visibility = @visibility"
        parameters = [{"name": "@visibility", "value": visibility}]
        
        if user_id:
            query += " AND c.user_id = @user_id"
            parameters.append({"name": "@user_id", "value": user_id})
        
        files = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        
        return jsonify({
            'visibility': visibility,
            'files': files,
            'count': len(files)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Run the application
    port = int(os.environ.get('PORT', 5000))
    debug = not is_production()
    
    print(f"Starting Platform Connection API on port {port}")
    print(f"Production mode: {is_production()}")
    print(f"Base URL: {BASE_URL}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
