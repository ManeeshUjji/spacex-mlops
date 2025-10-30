# Dockerfile — FastAPI + Uvicorn on 8081
FROM python:3.11-slim

# Keep Python predictable in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8081

# All project files will live under /app inside the image
WORKDIR /app

# If any packages need OS build tools, uncomment below and add as needed
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code and the model registry snapshot
# This makes the image self-contained and lets your app load the champion on boot.
COPY apps ./apps
COPY registry ./registry

# Optional: run as non-root (uncomment if you prefer)
# RUN useradd -m appuser
# USER appuser

# Expose the app port and start Uvicorn
EXPOSE 8081
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8081"]
