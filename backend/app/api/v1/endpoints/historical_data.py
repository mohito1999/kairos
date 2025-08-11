import json
import uuid
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.historical_upload import HistoricalUpload
from app.core.dependencies import get_current_user_with_provisioining as get_current_user
from app.database import get_db
from app.services import agent_service
from app.core.celery_app import celery_app

# --- CRITICAL FIX ---
# REMOVE the direct import of the task function
# from app.background.tasks import process_historical_upload_task 

# ADD an import for the celery_app instance instead
from app.core.celery_app import celery_app

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
    agent = await agent_service.get_agent_by_id(db, agent_id, current_user.organization_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    try:
        mapping = json.loads(data_mapping)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid data_mapping JSON format.")

    file_content_bytes = await file.read()

    new_upload = HistoricalUpload(
        agent_id=agent_id,
        organization_id=current_user.organization_id,
        filename=file.filename,
        status="PROCESSING", # The initial status before parsing
    )
    db.add(new_upload)
    await db.commit()
    await db.refresh(new_upload)

    # --- CRITICAL FIX ---
    # REPLACE the .delay() call with the decoupled send_task method.
    # We pass the full string path to the task.
    task_name = "app.background.tasks.process_historical_upload_task"
    celery_app.send_task(
        task_name,
        args=[str(new_upload.id), file_content_bytes, mapping]
    )
    
    return {"message": "File upload accepted and is being processed.", "upload_id": new_upload.id}