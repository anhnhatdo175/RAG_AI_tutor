FROM python:3.11-slim AS service-test

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements-service-test.txt ./requirements-service-test.txt
RUN pip install --no-cache-dir -r requirements-service-test.txt

COPY lmsrag ./lmsrag

CMD ["python", "-m", "pytest", "lmsrag/tests/test_service.py", "lmsrag/tests/test_layer5_6.py", "-q"]

FROM python:3.11-slim AS service

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY lmsrag ./lmsrag

EXPOSE 8000

CMD ["uvicorn", "lmsrag.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
