#!/bin/bash

# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install required system packages
sudo apt-get install -y \
    python3-full \
    python3-pip \
    python3-venv \
    python3-dev \
    git

# Install CUDA dependencies if using GPU instance
if lspci | grep -i nvidia > /dev/null; then
    # Install NVIDIA drivers and CUDA
    sudo apt-get install -y nvidia-driver-470 nvidia-cuda-toolkit
fi

# Use the current Imagenet directory
PROJECT_DIR="$PWD"

# Create and activate a virtual environment in the current directory
python3 -m venv venv
. ./venv/bin/activate

# Upgrade pip in the virtual environment
pip install --upgrade pip

# Install project dependencies in the virtual environment
pip install torch torchvision tqdm boto3 botocore

# Create directories for logs and checkpoints if they don't exist
mkdir -p logs
mkdir -p checkpoints

echo "Setup completed successfully!"
echo "To activate the environment, run: . ./activate_env.sh"