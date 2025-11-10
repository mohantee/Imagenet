---
title: ImageNet1K Classifier
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# ImageNet1K Image Classification

This Space hosts a trained ImageNet1K classification model. Upload an image to get predictions with confidence scores.

## Model

The model is trained on ImageNet-1K dataset and saved as `best_model.pth`.

## Usage

1. Upload an image using the interface
2. Get top 5 predictions with confidence scores

## Model Architecture

**Important**: You need to set the `MODEL_ARCH` environment variable in your Space settings to match your model architecture. Common options:
- `resnet50` (default)
- `resnet101`
- `resnet18`
- `efficientnet_b0`

To set environment variables in Hugging Face Spaces:
1. Go to your Space settings
2. Add a new variable: `MODEL_ARCH` with value matching your model

## Files

- `app.py`: Gradio interface and model inference code
- `best_model.pth`: Trained model weights
- `requirements.txt`: Python dependencies
- `imagenet_classes.txt`: ImageNet class labels (optional, will use generic labels if not provided)

## Notes

- Make sure your `best_model.pth` file is uploaded to the Space
- If you have ImageNet class labels, create `imagenet_classes.txt` with one class name per line
- The model expects RGB images and will automatically resize/crop them to 224x224