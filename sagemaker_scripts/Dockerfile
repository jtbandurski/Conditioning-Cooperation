# Dockerfile for MeltingPot MARL Training on SageMaker
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/ml/code

# Copy project files
COPY . /opt/ml/code/

# Install dmlab2d first (requires special handling)
RUN pip install --no-cache-dir dmlab2d

# Install other dependencies from requirements file
COPY sagemaker_scripts/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt

# Apply Ray patches
RUN chmod +x ray_patch.sh && sh ray_patch.sh

# Set environment variables for SageMaker
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# SageMaker expects the training script at this location
ENV SAGEMAKER_PROGRAM=train_sagemaker.py

# Use shell form to ignore extra arguments SageMaker passes
ENTRYPOINT ["/bin/bash", "-c", "python sagemaker_scripts/train_sagemaker.py"]
