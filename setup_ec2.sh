#!/bin/bash

# Update system packages
sudo apt-get update && sudo apt-get upgrade -y
#1) Update package state and try quick fixes
sudo apt update
sudo apt --fix-broken install
sudo dpkg --configure -a
sudo apt install -f
#2) Remove obvious conflicting/partially-installed NVIDIA/CUDA packages (safe cleanup)
sudo apt remove --purge '^nvidia-.*' '^libnvidia-.*' '^libcuda.*' cuda-* -y
sudo apt autoremove --purge -y
sudo apt clean
#) Make sure kernel headers are installed (needed for DKMS modules)
sudo apt install linux-headers-$(uname -r) build-essential dkms -y
#4) Add / enable the correct repositories
sudo add-apt-repository universe
sudo add-apt-repository multiverse
sudo apt update

# Install required system packages
sudo apt-get install -y \
    python3-full \
    python3-pip \
    python3-venv \
    python3-dev \
    git

# Install CUDA dependencies if using GPU instance
#sudo apt-get install -y nvidia-driver-470 nvidia-cuda-toolkit

#) Install a matching NVIDIA driver (pick a modern series: 535 / 550 / 555 / 560 etc.)
# Auto-detect and install recommended driver (Ubuntu)
sudo ubuntu-drivers autoinstall
# OR install a chosen driver, e.g. 535
sudo apt update
sudo apt install nvidia-driver-535 -y



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

#Reboot Manual 
sudo reboot

#vidia-smi
#dconfig -p | grep libcuda
#pt policy libnvidia-compute-535
