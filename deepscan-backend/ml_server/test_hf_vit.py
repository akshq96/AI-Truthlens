import torch
import numpy as np
from transformers import ViTForImageClassification
from PIL import Image

MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"

model = ViTForImageClassification.from_pretrained(MODEL_ID).eval()

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    w, h = img.size
    short = min(w, h)
    scale = 440 / short
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - 384) / 2
    top = (new_h - 384) / 2
    img = img.crop((round(left), round(top), round(left) + 384, round(top) + 384))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def fake_prob(path):
    img = Image.open(path)
    pixel_values = preprocess(img)
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
    return torch.sigmoid(logits).item()


import sys
for path in sys.argv[1:]:
    print(path, "-> fake_prob =", fake_prob(path))
