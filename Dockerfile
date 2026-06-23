FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.9 python3.9-venv python3.9-distutils python3-pip \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

WORKDIR /opt/schemafree
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade "pip==23.3.2" \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -m pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m", "schemafree.sheets.pretrain"]
