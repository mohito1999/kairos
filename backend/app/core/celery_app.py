from celery import Celery

# We assume Redis is running on the default host and port.
# In a production setup, this would come from the config settings.
celery_app = Celery(
    "tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0"
)

celery_app.conf.update(
    task_track_started=True,
)