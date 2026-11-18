FROM public.ecr.aws/docker/library/python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3

WORKDIR /app

COPY requirements.txt .
# Install CPU torch from the PyTorch CPU index first, then the rest. The torch
# version is read from requirements.txt (single source of truth) so a Dependabot
# bump there can't drift from this line; with torch already satisfied at the
# pinned version, the second resolve does not pull the ~1.5GB default CUDA wheel.
RUN TORCH_VERSION="$(sed -n 's/^torch==//p' requirements.txt)" \
    && pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bake the default encoder into the image at a pinned revision, so a training
# Job needs no Hugging Face access at run time. train.py uses this directory
# when EMBEDDING_MODEL_NAME is the default.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-base-en-v1.5', revision='a5beb1e3e68b9ab74eb54cfd186867f64f240e1a', local_dir='/opt/encoder/bge-base-en-v1.5')"

# Run offline: the encoder is already present.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY extract_common.py .
COPY train.py .

# Models are downloaded at run time; cache to a writable dir. USER/HOME are set
# so libraries that resolve the current user (e.g. torch 2.12's dynamo/inductor
# cache via getpass.getuser()) don't fail on the numeric uid 1000, which has no
# /etc/passwd entry; all caches stay under the writable /tmp.
ENV HF_HOME=/tmp/hf-cache \
    HOME=/tmp \
    USER=slemify \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

USER 1000:1000

ENTRYPOINT ["python", "train.py"]
