FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PREFECT_SERVER_ANALYTICS_ENABLED=false DO_NOT_TRACK=1 \
    PREFECT_HOME=/tmp/prefect
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 analyst
COPY src/ ./src/
COPY flows/ ./flows/
COPY manifests/ ./manifests/
COPY tests/ ./tests/
COPY pytest.ini ./
RUN mkdir -p /app/data/processed && chown -R analyst:analyst /app
USER analyst
ENTRYPOINT ["python", "-m", "flows.m5_pipeline"]
