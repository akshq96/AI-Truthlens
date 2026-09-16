"""Calibrate the general (non-face) AI-image model's decision threshold."""
import argparse
import json
import os
import random

import numpy as np

from image_server import (
    general_model,
    GENERAL_POSITIVE_CLASS,
    GENERAL_DEEPFAKE_THRESHOLD,
    preprocess_image,
    run_image_inference,
    deepfake_probability,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--fake-dir", required=True)
    ap.add_argument("--limit", type=int, default=1200)
    args = ap.parse_args()

    real_files = list_images(args.real_dir, args.limit)
    fake_files = list_images(args.fake_dir, args.limit)

    y_true, y_prob = [], []
    for p in real_files:
        raw = run_image_inference(preprocess_image(p), model=general_model)
        y_true.append(0)
        y_prob.append(deepfake_probability(raw, positive_class=GENERAL_POSITIVE_CLASS))
    for p in fake_files:
        raw = run_image_inference(preprocess_image(p), model=general_model)
        y_true.append(1)
        y_prob.append(deepfake_probability(raw, positive_class=GENERAL_POSITIVE_CLASS))

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    current = evaluate_threshold(y_true, y_prob, GENERAL_DEEPFAKE_THRESHOLD)
    thresholds = np.linspace(0.0, 1.0, 1001)
    best = max((evaluate_threshold(y_true, y_prob, t) for t in thresholds),
               key=lambda m: (m["balanced_accuracy"], m["accuracy"], -abs(m["threshold"] - 0.5)))

    print("n=", y_true.size)
    print("CURRENT:", json.dumps(current, indent=2))
    print("BEST:", json.dumps(best, indent=2))
    print("real_mean_prob=", float(np.mean(y_prob[y_true == 0])), "fake_mean_prob=", float(np.mean(y_prob[y_true == 1])))


if __name__ == "__main__":
    main()
