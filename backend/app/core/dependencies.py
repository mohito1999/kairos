from fastapi import Depends, HTTPException, status, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from supabase import create_client, Client 
from supabase.lib.client_options import ClientOptions

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

supabase_client: Client = create_client(
    settings.SUPABASE_URL,
    settings.SUPABASE_ANON_KEY,
    options=ClientOptions(auto_refresh_token=False, persist_session=False)
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


async def get_current_user(
    db: AsyncSession = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    user = await get_current_user_with_provisioining(db, token)
    if not user:
         raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail="User profile not found in application. Please complete signup."
            )
    return user

    

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
        # The get_user method on the client is asynchronous
        auth_response = await supabase_client.auth.get_user(token)
        auth_user = auth_response.user
        
        if not auth_user or not auth_user.email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid token"
            )

        local_user = await get_user_by_supabase_id(db, supabase_id=auth_user.id)
        
        if not local_user:
            print(f"Provisioning new user for email: {auth_user.email}")
            user_create_schema = UserCreate(
                supabase_auth_id=auth_user.id,
                email=auth_user.email,
                full_name=auth_user.user_metadata.get("full_name")
            )
            local_user = await create_user_with_organization(db, user_in=user_create_schema)

        return local_user
        
    except Exception as e:
        print(f"Auth error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
