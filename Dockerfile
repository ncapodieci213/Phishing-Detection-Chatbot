# ===============================
# Stage 1: Base image
# ===============================
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (optional: for URL parsing, SSL, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Stage 2: Install dependencies
# ===============================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===============================
# Stage 3: Copy application code
# ===============================
COPY . .

# Expose Gradio/FastAPI port
EXPOSE 7860

# ===============================
# Stage 4: Run application
# ===============================
# Use Uvicorn to serve FastAPI + Gradio hybrid app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]


# ======================================================
# GPU-optimized Dockerfile for FastAPI + Gradio chatbot
# ======================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# System setup
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose app port
EXPOSE 7860

# Optional: Environment variables for Gradio/FastAPI
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV APP_ENV=production

# Start app with GPU support
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
