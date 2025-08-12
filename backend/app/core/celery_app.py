# backend/app/core/celery_app.py
from celery import Celery
import os

celery_app = Celery(
    "kairos_tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0",
)

# PRODUCTION-READY CELERY CONFIGURATION
celery_app.conf.update(
    # Basic settings
    imports=(
        'app.background.tasks',
    ),
    task_track_started=True,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # CRITICAL: Process-safe worker configuration
    # Use 'solo' pool for development/debugging or 'threads' for production
    # This avoids the fork-related async issues entirely
    worker_pool='threads',  # Alternative: 'solo' for single-threaded debugging
    worker_concurrency=4,   # Adjust based on your server capacity
    
    # Reliability settings
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
    task_soft_time_limit=1800,  # 30 minutes
    task_time_limit=2400,       # 40 minutes (hard limit)
    
    # Connection settings
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    
    # Memory management
    worker_max_tasks_per_child=50,  # Restart workers after 50 tasks to prevent memory leaks
    worker_disable_rate_limits=True,
    
    # Async-friendly settings
    worker_lost_wait=10.0,
    worker_send_task_events=True,
    task_send_sent_event=True,
)