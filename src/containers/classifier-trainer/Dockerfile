FROM python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3

WORKDIR /app

COPY requirements.txt .
# CPU-only torch (the default wheel drags ~1.5GB of CUDA libs we don't need).
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY train.py .

# Models are downloaded at run time; cache to a writable dir.
ENV HF_HOME=/tmp/hf-cache

USER 1000:1000

ENTRYPOINT ["python", "train.py"]
