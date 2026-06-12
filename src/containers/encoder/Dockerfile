FROM python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3

WORKDIR /app

COPY requirements.txt .
# Install CPU-only torch first (the default torch wheel drags ~1.5GB of CUDA
# libraries we don't need — this image runs CPU inference only).
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Model is downloaded at startup (not baked in); cache to a writable dir.
ENV HF_HOME=/tmp/hf-cache

EXPOSE 8080

USER 1000:1000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
