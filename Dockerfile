FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY apps /app/apps
COPY registry /app/registry
ENV MODEL_DIR_OVERRIDE=""
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["bash", "-lc", "python -m uvicorn apps.api.main:app --host 0.0.0.0 --port ${PORT}"]
