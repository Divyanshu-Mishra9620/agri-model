# Serving-only image for serve.py (see render.yaml / .env.example for the
# equivalent Render config this mirrors). Deliberately NOT used for training —
# see requirements-render.txt for why the dependency set is this narrow.
#
# Railway note: this file exists specifically to bypass Railpack's Python
# auto-builder, which expects to own venv creation end-to-end and breaks
# ("copy /app/.venv": not found) when RAILPACK_INSTALL_CMD is overridden.
# A Dockerfile at the repo root takes priority over Railpack automatically.
FROM python:3.11-slim

WORKDIR /app

# libglib2.0-0: common opencv-python-headless runtime dependency on slim
# Debian images. curl: needed by scripts/fetch_model.sh.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Torch's CPU-only wheel needs its own index (see requirements-render.txt's
# header comment) — installed separately from the rest, before copying the
# full source tree, so this layer stays cached across ordinary code changes.
#
# torch and torchvision MUST be installed together in one command: pip
# resolves compatible versions when both are requested at once, but
# installing torch alone here and letting timm (in requirements-render.txt)
# pull in torchvision later as a side effect picks an uncoordinated version —
# confirmed locally as "RuntimeError: operator torchvision::nms does not
# exist" at model-load time, a version-mismatch error, not a code bug.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

COPY . .

# Bake the model into the image at BUILD time, not container start: a cold
# container (Railway sleeping an idle service, or any fresh deploy) would
# otherwise have to download ~112MB over the network before it could answer
# its first request — confirmed in production as a 30s+ hang that tripped
# the RAG service's ML-call timeout. Railway auto-passes service Variables
# as Docker build args for any ARG declared here, so this reuses the same
# MODEL_DOWNLOAD_URL Variable already set for this service — nothing new to
# configure.
ARG MODEL_DOWNLOAD_URL
RUN bash scripts/fetch_model.sh

ENV PORT=8500
EXPOSE 8500

CMD ["sh", "-c", "python -m uvicorn serve:app --host 0.0.0.0 --port $PORT --log-level info"]
