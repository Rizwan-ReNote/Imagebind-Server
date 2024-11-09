FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

RUN export DEBIAN_FRONTEND=noninteractive \
  && apt-get update \
  && apt-get -y install git libgomp1 wget \
  && rm -rf /var/lib/apt/lists/*

# RUN pip install --no-cache-dir --upgrade pip setuptools

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH="/app:/app/ImageBind"

RUN python3 download.py

ENTRYPOINT ["/bin/sh", "-c"]

CMD ["uvicorn app:app --host 0.0.0.0 --port 8080"]