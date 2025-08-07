import json
import uuid
from io import StringIO
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.historical_upload import HistoricalUpload
from app.core.dependencies import get_current_user_with_provisioining as get_current_user
from app.database import get_db
# Correctly import the module
from app.services import agent_service
# Correctly import the task
from app.background.tasks import process_historical_upload_task

router = APIRouter()

@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_historical_data(
    agent_id: uuid.UUID = Form(...),
    data_mapping: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Accepts a historical data file, reads it into memory,
    and queues it for processing.
    """
    # Ensure the agent exists and belongs to the user's organization
    # Correctly call the function from the imported module
    agent = await agent_service.get_agent_by_id(db, agent_id, current_user.organization_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    try:
        mapping = json.loads(data_mapping)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid data_mapping JSON format.")

    # Read the entire file into memory as bytes
    file_content_bytes = await file.read()

    # Create a record for the upload in our database
    new_upload = HistoricalUpload(
        agent_id=agent_id,
        organization_id=current_user.organization_id,
        filename=file.filename,
        status="PROCESSING",
    )
    db.add(new_upload)
    await db.commit()
    await db.refresh(new_upload)

    # Queue the background job, passing the file content and mapping directly
    process_historical_upload_task.delay(str(new_upload.id), file_content_bytes, mapping)
    
    return {"message": "File upload accepted and is being processed.", "upload_id": new_upload.id}