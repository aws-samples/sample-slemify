FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/app/deps -r requirements.txt

FROM python:3.12-slim

RUN groupadd -r slemify && useradd -r -g slemify -u 1000 slemify
WORKDIR /app

COPY --from=builder /app/deps /app/deps
COPY . .

ENV PYTHONPATH=/app/deps
USER 1000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import sys; sys.exit(0)"]

ENTRYPOINT ["python", "main.py"]
