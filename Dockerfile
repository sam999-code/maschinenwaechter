# MachineGuard Dockerfile - every line explained (Lesson 12)
#
# WHAT DOCKER IS (plain language):
# A container is like a SHIPPING CONTAINER for software. It packs our
# app + Python + all libraries + the model into one box that runs
# IDENTICALLY on any machine - your laptop, a German factory server,
# or a cloud VM. "Works on my machine" becomes "works on every machine".

# ---------------------------------------------------------------------------
# 1. START FROM A KNOWN BASE
# python:3.12-slim = a minimal Linux with Python 3.12 already installed.
# "slim" = small (no extra tools) -> faster downloads, smaller image.
# LESSON LEARNED LIVE: we started with 3.11, but our venv pins numpy
# 2.5.3 which needs Python >= 3.12 -> build failed. The base image's
# Python version must match the version the dependency pins were
# created with ("works on my machine" bites again!).
FROM python:3.12-slim

# 2. CPU-ONLY torch wheel (the default PyPI torch bundles CUDA/GPU
#    libraries ~2GB we don't need). Must be set BEFORE pip install.
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# 3. WORKDIR = the folder inside the container where we work.
#    Everything after this happens in /app of the CONTAINER filesystem
#    (totally separate from your Windows disk).
WORKDIR /app

# 4. Install dependencies in TWO steps on purpose:
#    Docker caches each step, and torch (~200MB) dwarfs everything
#    else. Splitting means a timeout/retry resumes from a completed
#    layer instead of re-downloading all ("layer caching" - same idea
#    as copying requirements before code). Versions match our venv.
RUN pip install --no-cache-dir numpy==2.5.3 pandas==3.0.6 matplotlib==3.11.2 pyyaml==6.0.3
RUN pip install --no-cache-dir fastapi==0.141.1 uvicorn==0.54.0 httpx==0.28.1 python-multipart==0.0.32 "torch==2.14.0+cpu"

# 5. NOW copy the app code and the trained model bundle.
#    (Both change more often than dependencies.)
COPY lessons/machineguard ./machineguard
COPY models/machineguard_v1_weights.pt models/machineguard_v1_config.json ./models/

# 6. Tell the app where the model bundle lives INSIDE the container
#    (service.py reads this env var - see the 12-factor comment there)
ENV MACHINEGUARD_MODELS=/app/models/machineguard_v1

# 7. DOCUMENT the port (informational; -p flag actually publishes it)
EXPOSE 8000

# 7. The command that starts when the container runs: same uvicorn
#    command you already know, just inside the box.
#    --host 0.0.0.0 = accept connections from OUTSIDE the container
#    (without this, only the container itself could reach the API!).
CMD ["python", "-m", "uvicorn", "machineguard.service:app", "--host", "0.0.0.0", "--port", "8000"]

# BUILD & RUN (after installing Docker Desktop):
#   docker build -t machineguard:v1 .
#   docker run -p 8000:8000 machineguard:v1
# then:  curl http://localhost:8000/health   ->  {"status":"ok",...}
