FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN groupadd -r appuser -g 10001 && useradd -u 10001 -r -g appuser -m -d /home/appuser appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --default-timeout=100 --retries=10 --no-cache-dir -r requirements.txt

COPY app.py .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
