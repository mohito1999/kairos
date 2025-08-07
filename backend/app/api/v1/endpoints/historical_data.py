from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
import json

from app.models.user import User
from app.models.historical_upload import HistoricalUpload
from app.core.dependencies import get_current_user_with_provisioining as get_current_user
from app.database import get_db
from app.background.tasks import process_historical_upload_task

router = APIRouter()

@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_historical_data(
    agent_id: uuid.UUID = Form(...),
    data_mapping: str = Form(...), # Receive data_mapping as a JSON string
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Accepts a historical data file upload and queues it for processing.
    """
    # TODO: In a real app, we would stream this file to a cloud storage (like S3)
    # For now, we'll just log its details and assume it's stored.
    print(f"Received file: {file.filename} for agent {agent_id}")
    print(f"File content type: {file.content_type}")
    
    try:
        mapping = json.loads(data_mapping)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid data_mapping JSON format.")

    # Create a record for the upload in our database
    new_upload = HistoricalUpload(
        agent_id=agent_id,
        organization_id=current_user.organization_id,
        filename=file.filename,
        status="UPLOADING",
        # total_interactions will be populated after parsing
    )
    db.add(new_upload)
    await db.commit()
    await db.refresh(new_upload)

    # Queue the background job to process the file
    process_historical_upload_task.delay(str(new_upload.id))
    
    new_upload.status = "PROCESSING"
    await db.commit()

    return {"message": "File upload accepted and is being processed.", "upload_id": new_upload.id}