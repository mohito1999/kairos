from fastapi import Depends, HTTPException, status, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from supabase_py_async import create_client
from supabase_py_async.lib.client_options import ClientOptions

from app.core.config import settings
from app.core.security import oauth2_scheme
from app.database import get_db
from app.models.user import User
from app.services.user_service import get_user_by_supabase_id
from app.schemas.user import UserCreate
from app.services.user_service import create_user_with_organization
from app.services.sdk_service import get_agent_from_api_key
from app.models.agent import Agent

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def get_agent_from_sdk_auth(
    api_key: str = Security(api_key_header),
    db: AsyncSession = Depends(get_db),
) -> Agent:
    """
    Dependency to authenticate and retrieve an agent via an API key.
    The key is expected in the "Authorization" header, e.g., "Authorization: kai_...".
    Note: We don't use "Bearer" here for SDK keys to distinguish them from user JWTs.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing",
        )
    
    agent = await get_agent_from_api_key(db, api_key)
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or inactive API Key",
        )
    return agent


# This is a FastAPI dependency that will act as a "gatekeeper" for protected endpoints.
async def get_current_user(
    db: AsyncSession = Depends(get_db),
    token: str = Depends(oauth2_scheme) # Extracts "Bearer <token>" from header
) -> User:
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Not authenticated"
        )
    
    try:
        # 1. Create an async Supabase client using the public ANON key.
        #    This client doesn't need admin rights; it only needs to verify a token.
        supabase_client = await create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_ANON_KEY,
            options=ClientOptions(auto_refresh_token=False, persist_session=False)
        )
        
        # 2. Ask Supabase to validate the token. If it's invalid, expired, or
        #    malformed, Supabase will raise an error, which our `except` block will catch.
        auth_response = await supabase_client.auth.get_user(token)
        auth_user = auth_response.user
        
        if not auth_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid token"
            )

        # 3. The token is valid! Now, use the Supabase user's ID to find the
        #    corresponding user in our OWN `public.users` table. This is where
        #    we store app-specific data like their role and organization.
        local_user = await get_user_by_supabase_id(db, supabase_id=auth_user.id)
        
        if not local_user:
            # This is a special case: the user is valid in Supabase Auth,
            # but doesn't have a profile in our application's database yet.
            # We will handle this in the next step by creating a profile for them.
            # For now, we deny access.
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail="User profile not found in application. Please complete signup."
            )
        
        # 4. Return our application's full User object to the endpoint.
        return local_user
        
    except Exception:
        # Catch any other errors (e.g., from Supabase client) and return a generic auth error.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    

async def get_current_user_with_provisioining(
    db: AsyncSession = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    A dependency that gets the current user.
    If the user is authenticated with Supabase but doesn't exist in our
    local DB, it creates (provisions) a new user and organization for them.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Not authenticated"
        )
    
    try:
        supabase_client = await create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_ANON_KEY,
            options=ClientOptions(auto_refresh_token=False, persist_session=False)
        )
        auth_response = await supabase_client.auth.get_user(token)
        auth_user = auth_response.user
        
        if not auth_user or not auth_user.email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid token"
            )

        # Try to fetch the user from our local DB
        local_user = await get_user_by_supabase_id(db, supabase_id=auth_user.id)
        
        if not local_user:
            # --- THIS IS THE NEW LOGIC ---
            # User is valid but doesn't exist locally. Let's create them.
            print(f"Provisioning new user for email: {auth_user.email}")
            user_create_schema = UserCreate(
                supabase_auth_id=auth_user.id,
                email=auth_user.email,
                full_name=auth_user.user_metadata.get("full_name")
            )
            local_user = await create_user_with_organization(db, user_in=user_create_schema)

        return local_user
        
    except Exception as e:
        print(f"Auth error: {e}") # For debugging
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_agent_from_sdk_auth(
    api_key: str = Security(api_key_header),
    db: AsyncSession = Depends(get_db),
) -> Agent:
    """
    Dependency to authenticate and retrieve an agent via an API key.
    The key is expected in the "Authorization" header, e.g., "Authorization: kai_...".
    Note: We don't use "Bearer" here for SDK keys to distinguish them from user JWTs.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing",
        )
    
    agent = await get_agent_from_api_key(db, api_key)
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or inactive API Key",
        )
    return agent
