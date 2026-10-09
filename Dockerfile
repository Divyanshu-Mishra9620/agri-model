FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

COPY . .

ARG MODEL_DOWNLOAD_URL
RUN bash scripts/fetch_model.sh

ENV PORT=8500 \
    ML_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2
EXPOSE 8500

CMD ["sh", "-c", "python -m uvicorn serve:app --host 0.0.0.0 --port $PORT --log-level info"]
