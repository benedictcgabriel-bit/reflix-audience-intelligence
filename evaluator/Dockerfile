FROM python:3.11-slim

RUN groupadd -r appuser -g 10001 && useradd -u 10001 -r -g appuser -m -d /home/appuser appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --default-timeout=100 --retries=10 --no-cache-dir -r requirements.txt

COPY evaluate.py .

USER appuser

CMD ["python", "evaluate.py"]
