FROM python:3.12.15-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
COPY deployment/requirements.txt /app/deployment/requirements.txt
RUN pip install --no-cache-dir -r deployment/requirements.txt && useradd --create-home appuser
COPY module5 /app/module5
COPY deployment /app/deployment
USER appuser
EXPOSE 8000
CMD ["uvicorn", "deployment.cloud_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
