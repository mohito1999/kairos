# backend/app/core/celery_app.py

from celery import Celery

# Note: We've removed the os.getenv for now to keep it simple and rely on the docker-compose default.
# The app name is also changed as suggested.
celery_app = Celery(
    "kairos_tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0",
)

# Configure Celery settings using his recommendations for a production-ready setup
celery_app.conf.update(
    imports=(
        'app.background.tasks',
    ),
    task_track_started=True,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True, # Good for reliability
    result_expires=3600, # Results expire after 1 hour
)