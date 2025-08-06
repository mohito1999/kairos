# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install poetry and its essential 'export' plugin
RUN pip install --no-cache-dir poetry
RUN poetry self add poetry-plugin-export

# Copy only the dependency definition files to leverage Docker's build cache
COPY pyproject.toml poetry.lock* ./

# Generate a requirements.txt from Poetry’s lockfile and install with pip
# This installs dependencies system-wide within the container.
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's source code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# This command will be used by docker-compose for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]