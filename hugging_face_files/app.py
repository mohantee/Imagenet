import os
import json
import urllib.request
from typing import Tuple
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import gradio as gr
from model import ResNet50   # ✅ USE YOUR MODEL, not timm!


def get_imagenet_labels():
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    filename = "imagenet_class_index.json"
    try:
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)
        with open(filename, "r") as f:
            idx = json.load(f)
        return [idx[str(i)][1] for i in range(1000)]
    except Exception:
        return [f"class_{i}" for i in range(1000)]


def load_model(ckpt_path: str, device=None) -> Tuple[torch.nn.Module, str]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ✅ Use YOUR architecture
    model = ResNet50(num_classes=1000)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # ✅ Prefer EMA weights if available
    if "ema_state_dict" in ck:
        state_dict = ck["ema_state_dict"]
    else:
        state_dict = ck.get("model_state_dict", ck)

    # ✅ Strip DataParallel prefix
    clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(clean_sd, strict=False)
    model.eval().to(device)
    return model, device


def predict(image, *, model, device, labels):
    if image is None:
        return "No image", "", {}

    pil = image.convert("RGB")
    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    x = preprocess(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits[0], dim=0)
        top5_prob, top5_idx = torch.topk(prob, 5)

    # keep values in [0,1] for Gradio
    top5 = [
        (labels[i.item()], float(top5_prob[j].item()))
        for j, i in enumerate(top5_idx)
    ]

    pred_name = top5[0][0]
    pred_conf = f"{top5[0][1] * 100:.2f}%"   # format for display

    # keep raw probabilities for bar chart
    top5_dict = {name: score for name, score in top5}

    return pred_name, pred_conf, top5_dict


def create_demo():
    labels = get_imagenet_labels()
    ckpt = os.environ.get("CKPT_PATH", "best_model.pth")

    model, device = load_model(ckpt)

    with gr.Blocks(title="ResNet-50 ImageNet Classifier") as demo:
        gr.Markdown("# 🐶🐱 ResNet-50 Classifier")
        gr.Markdown("Upload an image and get top-5 predictions.")

        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="pil", label="Upload image")
                btn = gr.Button("Classify")

            with gr.Column():
                pred = gr.Textbox(label="Prediction")
                conf = gr.Textbox(label="Confidence")
                top5 = gr.Label(label="Top-5 probabilities")

        btn.click(
            lambda img: predict(img, model=model, device=device, labels=labels),
            inputs=[img_in],
            outputs=[pred, conf, top5],
        )
    return demo


if __name__ == "__main__":
    demo = create_demo()
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
