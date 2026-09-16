"""
Head-to-head: HF Community Forensics ViT vs. our CIFAKE-tuned general model,
on REALISTIC-resolution AI-vs-real images (the earlier CIFAKE comparison used
32x32 native images, which is pathological for a 384px ViT and therefore was
not a fair basis for choosing which model handles real uploads).
"""
import argparse
import json
import os
import random

import numpy as np

from image_server import (
    GENERAL_POSITIVE_CLASS,
    deepfake_probability,
    general_model,
    preprocess_image,
    preprocess_for_hf_vit,
    run_hf_vit_inference,
    run_image_inference,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images_recursive(root, limit):
    files = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for n in sorted(filenames):
            if os.path.splitext(n)[1].lower() in IMAGE_EXTS:
                files.append(os.path.join(dirpath, n))
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
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float((tp + tn) / max(1, y_true.size)), 4),
        "balanced_accuracy": round(float((tpr + tnr) / 2.0), 4),
        "fake_caught_rate": round(float(tpr), 4),
        "real_kept_rate": round(float(tnr), 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def best_threshold(y_true, y_prob):
    return max(
        (evaluate_threshold(y_true, y_prob, t) for t in np.linspace(0.0, 1.0, 1001)),
        key=lambda m: (m["balanced_accuracy"], m["accuracy"], -abs(m["threshold"] - 0.5)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", default="realistic_data/real_dataset")
    ap.add_argument("--fake-dir", default="realistic_data/Ai_generated_dataset")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--output-json", default="realistic_comparison.json")
    args = ap.parse_args()

    real_files = list_images_recursive(args.real_dir, args.limit)
    fake_files = list_images_recursive(args.fake_dir, args.limit)
    print(f"real={len(real_files)} fake={len(fake_files)}")

    y_true, hf_probs, gen_probs = [], [], []
    for label, files in ((0, real_files), (1, fake_files)):
        for i, p in enumerate(files):
            try:
                hf_p = run_hf_vit_inference(preprocess_for_hf_vit(p))
                gen_raw = run_image_inference(preprocess_image(p), model=general_model)
                gen_p = deepfake_probability(gen_raw, positive_class=GENERAL_POSITIVE_CLASS)
            except Exception as exc:
                print(f"  skip {p}: {exc}")
                continue
            y_true.append(label)
            hf_probs.append(hf_p)
            gen_probs.append(gen_p)
            if (i + 1) % 50 == 0:
                print(f"  label={label} {i+1}/{len(files)}")

    y_true = np.array(y_true)
    hf_probs = np.array(hf_probs)
    gen_probs = np.array(gen_probs)

    report = {
        "n": int(y_true.size),
        "n_real": int(np.sum(y_true == 0)),
        "n_fake": int(np.sum(y_true == 1)),
        "hf_vit": {
            "at_deployed_0.719": evaluate_threshold(y_true, hf_probs, 0.719),
            "at_0.5": evaluate_threshold(y_true, hf_probs, 0.5),
            "best": best_threshold(y_true, hf_probs),
            "real_mean": round(float(np.mean(hf_probs[y_true == 0])), 4),
            "fake_mean": round(float(np.mean(hf_probs[y_true == 1])), 4),
        },
        "our_general_cifake": {
            "at_deployed_0.95": evaluate_threshold(y_true, gen_probs, 0.95),
            "at_0.5": evaluate_threshold(y_true, gen_probs, 0.5),
            "best": best_threshold(y_true, gen_probs),
            "real_mean": round(float(np.mean(gen_probs[y_true == 0])), 4),
            "fake_mean": round(float(np.mean(gen_probs[y_true == 1])), 4),
        },
    }

    print(json.dumps(report, indent=2))
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)
    print("saved", args.output_json)


if __name__ == "__main__":
    main()
