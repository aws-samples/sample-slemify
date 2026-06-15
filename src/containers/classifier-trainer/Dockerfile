FROM python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3

WORKDIR /app

COPY requirements.txt .
# Install CPU torch from the PyTorch CPU index first, then the rest. torch is
# pinned to the same version in requirements.txt so the second resolve sees it
# already satisfied and does not pull the ~1.5GB CUDA wheel.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY extract_common.py .
COPY train.py .

# Models are downloaded at run time; cache to a writable dir.
ENV HF_HOME=/tmp/hf-cache

USER 1000:1000

ENTRYPOINT ["python", "train.py"]
