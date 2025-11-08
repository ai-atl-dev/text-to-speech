# Dockerfile (OVERWRITE your existing Dockerfile with this exact content)
FROM nvidia/cuda:12.4.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ffmpeg ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# ensure python
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# copy repo into image
COPY . /app

# Install Python deps (use requirements.txt provided by repo)
RUN pip3 install --upgrade pip setuptools wheel
# Install huggingface_hub explicitly (needed for snapshot_download)
RUN pip3 install huggingface_hub

# Install repo requirements
RUN if [ -f requirements.txt ]; then pip3 install -r requirements.txt || true; fi

# Install the package (if setup.py exists)
RUN if [ -f setup.py ]; then pip3 install -e . || true; fi

# Build arg/secret for Hugging Face token (passed from Cloud Build secret)
ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
# Put HF cache in an accessible path baked into the image
ENV HF_HOME=/opt/hf_cache
ENV HF_DATASETS_CACHE=$HF_HOME
ENV TRANSFORMERS_CACHE=$HF_HOME

# Pre-download the csm-1b repo into HF_HOME during build so the image contains weights.
# This avoids runtime downloads and cold starts.
RUN python -c "import os, sys; from huggingface_hub import snapshot_download; token=os.environ.get('HUGGINGFACE_TOKEN'); snapshot_download(repo_id='sesame/csm-1b', cache_dir=os.environ.get('HF_HOME', '/opt/hf_cache'), token=token, allow_patterns=None); print('Downloaded sesame/csm-1b into', os.environ.get('HF_HOME'))"

# Expose port and run
EXPOSE 8080
ENV PORT=8080
ENV NO_TORCH_COMPILE=1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
