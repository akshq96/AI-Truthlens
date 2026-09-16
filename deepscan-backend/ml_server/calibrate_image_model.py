"""
Calibrate/evaluate the image deepfake model (model_v3.pth) against a labeled
real/fake face dataset (e.g. downloaded from Kaggle).

Usage:
  python calibrate_image_model.py --real-dir path/to/real --fake-dir path/to/fake \
      [--limit 2000] [--positive-class real] [--output-json report.json]

Computes, for every threshold in a dense grid, balanced accuracy / accuracy /
precision / recall / F1 / confusion counts, and reports the best threshold —
so DEEPSCAN_IMAGE_DEEPFAKE_THRESHOLD can be set from real evidence instead of
a guess.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List

import numpy as np

from image_server import (
    DEEPFAKE_THRESHOLD,
    POSITIVE_CLASS,
    deepfake_probability,
    preprocess_image,
    run_image_inference,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageSample:
    path: str
    label: int  # 1 = fake/ai, 0 = real
    deepfake_probability: float


def list_images(folder: str, limit: int) -> List[str]:
    files = []
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if not os.path.isfile(full):
            continue
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            files.append(full)
    random.Random(42).shuffle(files)
    return files[:limit] if limit else files


def score_image(path: str, label: int, positive_class: str) -> ImageSample:
    tensor = preprocess_image(path)
    raw = run_image_inference(tensor)
    prob = deepfake_probability(raw, positive_class=positive_class)
    return ImageSample(path=path, label=label, deepfake_probability=float(prob))


def evaluate_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tpr
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / max(1, y_true.size)
    bal_acc = (tpr + tnr) / 2.0

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate_real_flagged_fake": float(1.0 - tnr),
        "false_negative_rate_fake_flagged_real": float(1.0 - tpr),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def pick_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    thresholds = np.linspace(0.0, 1.0, 1001)
    scored = [evaluate_threshold(y_true, y_prob, float(t)) for t in thresholds]
    best = max(
        scored,
        key=lambda m: (
            m["balanced_accuracy"],
            m["accuracy"],
            m["f1"],
            -abs(m["threshold"] - 0.5),
        ),
    )
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate image-model threshold using real/fake folders")
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--fake-dir", required=True)
    parser.add_argument("--limit", type=int, default=1500, help="Max images per class (0 = all)")
    parser.add_argument("--positive-class", default=POSITIVE_CLASS, choices=["real", "deepfake"])
    parser.add_argument("--output-json", default="image_calibration.json")
    args = parser.parse_args()

    if not os.path.isdir(args.real_dir):
        raise SystemExit(f"real dir not found: {args.real_dir}")
    if not os.path.isdir(args.fake_dir):
        raise SystemExit(f"fake dir not found: {args.fake_dir}")

    real_files = list_images(args.real_dir, args.limit)
    fake_files = list_images(args.fake_dir, args.limit)

    if not real_files:
        raise SystemExit("No real images found")
    if not fake_files:
        raise SystemExit("No fake images found")

    samples: List[ImageSample] = []

    print(f"Scoring {len(real_files)} real images...")
    for i, path in enumerate(real_files):
        try:
            samples.append(score_image(path, label=0, positive_class=args.positive_class))
        except Exception as exc:
            print(f"[WARN] real skip: {os.path.basename(path)} -> {exc}")
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(real_files)}")

    print(f"Scoring {len(fake_files)} fake images...")
    for i, path in enumerate(fake_files):
        try:
            samples.append(score_image(path, label=1, positive_class=args.positive_class))
        except Exception as exc:
            print(f"[WARN] fake skip: {os.path.basename(path)} -> {exc}")
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(fake_files)}")

    if not samples:
        raise SystemExit("No samples were scored")

    y_true = np.array([s.label for s in samples], dtype=np.int32)
    y_prob = np.array([s.deepfake_probability for s in samples], dtype=np.float64)

    current = evaluate_threshold(y_true, y_prob, DEEPFAKE_THRESHOLD)
    best = pick_best_threshold(y_true, y_prob)

    report = {
        "dataset": {
            "real_dir": os.path.abspath(args.real_dir),
            "fake_dir": os.path.abspath(args.fake_dir),
            "real_count": int(np.sum(y_true == 0)),
            "fake_count": int(np.sum(y_true == 1)),
            "total_scored": int(y_true.size),
        },
        "positive_class": args.positive_class,
        "current_threshold_env": float(DEEPFAKE_THRESHOLD),
        "current_threshold_metrics": current,
        "best_threshold": best,
        "recommended_env": {
            "DEEPSCAN_IMAGE_DEEPFAKE_THRESHOLD": float(round(best["threshold"], 6)),
        },
        "prob_stats": {
            "real_mean_deepfake_prob": float(np.mean(y_prob[y_true == 0])),
            "real_median_deepfake_prob": float(np.median(y_prob[y_true == 0])),
            "fake_mean_deepfake_prob": float(np.mean(y_prob[y_true == 1])),
            "fake_median_deepfake_prob": float(np.median(y_prob[y_true == 1])),
        },
    }

    out_path = os.path.abspath(args.output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nCalibration complete")
    print(f"Total scored: {y_true.size} (real={report['dataset']['real_count']}, fake={report['dataset']['fake_count']})")
    print(f"\n--- CURRENT threshold={DEEPFAKE_THRESHOLD} ---")
    print(json.dumps(current, indent=2))
    print(f"\n--- BEST threshold={best['threshold']:.4f} ---")
    print(json.dumps(best, indent=2))
    print("\nProb stats:")
    print(json.dumps(report["prob_stats"], indent=2))
    print(f"\nSaved report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
