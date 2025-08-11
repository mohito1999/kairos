# backend/app/worker.py

"""
Celery worker entry point.
This module is specifically designed to be used by the Celery worker command.
It ensures that the app is configured and all task modules are imported
so that the @task decorators are registered.
"""

# This line is crucial for making sure the app is configured.
from app.core.celery_app import celery_app

# By importing the tasks module here, we ensure that the worker process
# runs the @celery_app.task decorators and registers the tasks.
import app.background.tasks

# The celery command line tool will inspect this file and find this instance.
# We don't need the setup function he suggested, a direct import is cleaner and sufficient.