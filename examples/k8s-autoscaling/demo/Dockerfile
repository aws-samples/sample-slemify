FROM python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    uvicorn==0.34.2 \
    httpx==0.28.1 \
    opensearch-py==2.8.0 \
    boto3==1.38.12 \
    langgraph==1.2.5

COPY server.py .

EXPOSE 8000

USER 1000:1000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
