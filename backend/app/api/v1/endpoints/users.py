from fastapi import APIRouter, Depends
from app.schemas.user import User
from app.core.dependencies import get_current_user_with_provisioining

router = APIRouter()

@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user_with_provisioining)):
    """
    Get the profile of the currently authenticated user.
    If the user does not exist in our DB, they will be created.
    """
    return current_user