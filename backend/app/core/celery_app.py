from celery import Celery

celery_app = Celery(
    "tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0"
)

# Tell celery to find tasks in our tasks file
celery_app.autodiscover_tasks(['app.background.tasks'])

celery_app.conf.update(
    task_track_started=True,
)