#!/bin/bash

# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and pip
sudo apt-get install -y python3-pip python3-dev

# Install CUDA dependencies if using GPU instance
if lspci | grep -i nvidia > /dev/null; then
    # Install NVIDIA drivers and CUDA
    sudo apt-get install -y nvidia-driver-470 nvidia-cuda-toolkit
fi

# Create a virtual environment
sudo apt-get install -y python3-venv
python3 -m venv venv
source venv/bin/activate

# Install project dependencies
pip install torch torchvision tqdm boto3 numpy pillow
pip install -e .

# Create directories for logs and checkpoints
mkdir -p logs
mkdir -p checkpoints

# Configure AWS credentials (if not using IAM role)
# AWS credentials should preferably be configured through IAM roles
# but this is a fallback method
mkdir -p ~/.aws
cat > ~/.aws/config << EOL
[default]
region = us-east-1
output = json
EOL

echo "Setup completed successfully!"