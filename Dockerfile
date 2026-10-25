# Packages the mobile app (FastAPI backend + static frontend) for
# deployment. Build context is this directory (cordia_ai/), since
# mobile_app/backend/main.py imports from ../src and reads
# ../checkpoints, ../datasets/uci at runtime.
#
# NOT tested with an actual `docker build` on the machine this was written
# on -- Docker itself isn't installed there, and Docker Desktop's usual
# WSL2/virtualization setup plus a required reboot was judged too
# disruptive to install autonomously. The Dockerfile follows the project's
# real, working `pip install -r requirements.txt` + `uvicorn` setup exactly
# (see README.md's Setup / Mobile app sections), so it should build cleanly,
# but treat the first real build as the actual verification step.

# Stage 1: build the 3D digital twin (heart_digital_twin_3d/) into static
# files; main.py serves them at /twin/ when heart_digital_twin_3d/dist exists.
FROM node:20-slim AS twin
WORKDIR /twin
COPY heart_digital_twin_3d/package.json heart_digital_twin_3d/package-lock.json ./
RUN npm ci
COPY heart_digital_twin_3d/ ./
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

# System deps for scientific-Python wheels (scipy, torch's runtime libs).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Only what the app actually needs at runtime -- not the ECG datasets or the
# Colab notebooks. The ECG checkpoint IS needed: main.py's startup loads it
# for the ECG upload endpoint, so the container would fail to start without it.
COPY src/ src/
COPY mobile_app/ mobile_app/
COPY checkpoints/ft_transformer.pt checkpoints/ft_transformer.pt
COPY checkpoints/ecg_vit.pt checkpoints/ecg_vit.pt
COPY datasets/uci/heart_disease_raw.csv datasets/uci/heart_disease_raw.csv
COPY --from=twin /twin/dist heart_digital_twin_3d/dist

EXPOSE 8000
WORKDIR /app/mobile_app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
