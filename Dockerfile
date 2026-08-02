# Inference service for the streamflow forecast API.
# Serving only - tuning and orchestration stay outside the image.

FROM python:3.12-slim

# xgboost needs libgomp at runtime; curl is for the healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dependencies first so code edits don't invalidate the layer
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY streamflow/ ./streamflow/
COPY config/ ./config/

# model comes from a mounted mlruns dir; override to point elsewhere.
# mlflow 3.x refuses a file store unless this opt-out is set - fine for a
# course project, but a real deployment would use a tracking server.
ENV MLFLOW_TRACKING_URI=file:/app/mlruns \
    MLFLOW_ALLOW_FILE_STORE=true \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "streamflow.serve:app", "--host", "0.0.0.0", "--port", "8000"]
