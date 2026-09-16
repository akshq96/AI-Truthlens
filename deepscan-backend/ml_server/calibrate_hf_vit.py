import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification

MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = ViTForImageClassification.from_pretrained(MODEL_ID).eval()
    return _model


def preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    w, h = img.size
    short = min(w, h)
    scale = 440 / short
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
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
        logits = get_model()(pixel_values=pixel_values).logits
    return torch.sigmoid(logits).item()


def list_images(folder, limit):
    files = [os.path.join(folder, n) for n in sorted(os.listdir(folder))
             if os.path.splitext(n)[1].lower() in IMAGE_EXTS]
    random.Random(42).shuffle(files)
    return files[:limit] if limit else files


def evaluate_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    bal_acc = (tpr + tnr) / 2.0
    acc = (tp + tn) / max(1, y_true.size)
    return {"threshold": float(threshold), "accuracy": float(acc), "balanced_accuracy": float(bal_acc),
            "fp_rate_real_flagged_fake": float(1 - tnr), "fn_rate_fake_flagged_real": float(1 - tpr),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def score_dataset(name, real_dir, fake_dir, limit):
    real_files = list_images(real_dir, limit)
    fake_files = list_images(fake_dir, limit)
    y_true, y_prob = [], []
    for i, p in enumerate(real_files):
        y_true.append(0)
        y_prob.append(fake_prob(p))
        if (i + 1) % 100 == 0:
            print(f"  [{name}] real {i+1}/{len(real_files)}")
    for i, p in enumerate(fake_files):
        y_true.append(1)
        y_prob.append(fake_prob(p))
        if (i + 1) % 100 == 0:
            print(f"  [{name}] fake {i+1}/{len(fake_files)}")
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    current = evaluate_threshold(y_true, y_prob, 0.5)
    thresholds = np.linspace(0.0, 1.0, 1001)
    best = max((evaluate_threshold(y_true, y_prob, t) for t in thresholds),
               key=lambda m: (m["balanced_accuracy"], m["accuracy"], -abs(m["threshold"] - 0.5)))
    print(f"\n=== {name} (n={y_true.size}) ===")
    print("threshold=0.5:", json.dumps(current))
    print("best:", json.dumps(best))
    print("real_mean=", float(np.mean(y_prob[y_true == 0])), "fake_mean=", float(np.mean(y_prob[y_true == 1])))
    return {"dataset": name, "n": int(y_true.size), "at_0.5": current, "best": best,
            "real_mean_prob": float(np.mean(y_prob[y_true == 0])), "fake_mean_prob": float(np.mean(y_prob[y_true == 1]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--output-json", default="hf_vit_calibration.json")
    args = ap.parse_args()

    results = []
    results.append(score_dataset("face_dataset(calibration_data2)", "calibration_data2/real", "calibration_data2/fake", args.limit))
    results.append(score_dataset("cifake_general(finaltest)", "cifake_subsets/finaltest/real", "cifake_subsets/finaltest/fake", args.limit))

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved:", args.output_json)


if __name__ == "__main__":
    main()
