# Inference service only - tuning and orchestration stay outside the image.

FROM python:3.12-slim

# libgomp is an xgboost runtime dependency; curl backs the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source so code edits reuse the cached install layer.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY streamflow/ ./streamflow/
COPY config/ ./config/

# The model is mounted, not baked in, so a retrain is picked up without a rebuild.
ENV MLFLOW_TRACKING_URI=file:/app/mlruns \
    MLFLOW_ALLOW_FILE_STORE=true \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "streamflow.serve:app", "--host", "0.0.0.0", "--port", "8000"]
