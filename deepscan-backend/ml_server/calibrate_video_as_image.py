import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import List

import numpy as np

from image_server import (
    VIDEO_DECISION_THRESHOLD,
    VIDEO_NUM_SAMPLES,
    VIDEO_POSITIVE_CLASS,
    deepfake_probability,
    preprocess_rgb_array,
    preprocess_video_frames_around_second,
    run_image_inference,
)


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass
class VideoSample:
    path: str
    label: int  # 1 = deepfake/ai, 0 = real
    deepfake_probability: float
    sampled_seconds: List[float]
    sampled_frame_indices: List[int]
    frames_analyzed: int
    per_frame_deepfake_probabilities: List[float]


def list_videos(folder: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in VIDEO_EXTS:
            files.append(full)
    return files


def score_video(video_path: str, label: int, sample_count: int, positive_class: str) -> VideoSample:
    raw_frames, frame_indices, fps, _duration, _had_face_flags = preprocess_video_frames_around_second(
        video_path,
        sample_count=sample_count,
    )

    frame_probs: List[float] = []
    for frame_rgb in raw_frames:
        raw = run_image_inference(preprocess_rgb_array(frame_rgb))
        frame_probs.append(float(deepfake_probability(raw, positive_class=positive_class)))

    prob = float(np.mean(np.array(frame_probs, dtype=np.float64)))
    sampled_seconds = [float(round(idx / fps, 3)) for idx in frame_indices]

    return VideoSample(
        path=video_path,
        label=label,
        deepfake_probability=float(prob),
        sampled_seconds=sampled_seconds,
        sampled_frame_indices=[int(idx) for idx in frame_indices],
        frames_analyzed=int(len(frame_indices)),
        per_frame_deepfake_probabilities=[float(round(v, 6)) for v in frame_probs],
    )


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
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def pick_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    # Use dense grid for stable threshold estimation on small datasets.
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
    parser = argparse.ArgumentParser(description="Calibrate video-as-image threshold using real/ai folders")
    parser.add_argument("--real-dir", required=True, help="Folder with real videos")
    parser.add_argument("--ai-dir", required=True, help="Folder with AI/deepfake videos")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=VIDEO_NUM_SAMPLES,
        help="Number of uniformly sampled frames per video (default: current server config)",
    )
    parser.add_argument(
        "--positive-class",
        default=VIDEO_POSITIVE_CLASS,
        choices=["real", "deepfake"],
        help="Class treated as model-positive before deepfake conversion",
    )
    parser.add_argument(
        "--output-json",
        default="video_as_image_calibration.json",
        help="Where to write calibration summary JSON",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.real_dir):
        raise SystemExit(f"real dir not found: {args.real_dir}")
    if not os.path.isdir(args.ai_dir):
        raise SystemExit(f"ai dir not found: {args.ai_dir}")

    real_files = list_videos(args.real_dir)
    ai_files = list_videos(args.ai_dir)

    if not real_files:
        raise SystemExit("No real videos found")
    if not ai_files:
        raise SystemExit("No AI videos found")

    samples: List[VideoSample] = []
    sample_count = max(1, int(args.sample_count))

    print(f"Scoring {len(real_files)} real videos...")
    for path in real_files:
        try:
            samples.append(score_video(path, label=0, sample_count=sample_count, positive_class=args.positive_class))
        except Exception as exc:
            print(f"[WARN] real skip: {os.path.basename(path)} -> {exc}")

    print(f"Scoring {len(ai_files)} AI videos...")
    for path in ai_files:
        try:
            samples.append(score_video(path, label=1, sample_count=sample_count, positive_class=args.positive_class))
        except Exception as exc:
            print(f"[WARN] ai skip: {os.path.basename(path)} -> {exc}")

    if not samples:
        raise SystemExit("No samples were scored")

    y_true = np.array([s.label for s in samples], dtype=np.int32)
    y_prob = np.array([s.deepfake_probability for s in samples], dtype=np.float64)

    best = pick_best_threshold(y_true, y_prob)

    recommended_env = {
        "DEEPSCAN_VIDEO_AS_IMAGE_NUM_SAMPLES": int(sample_count),
        "DEEPSCAN_VIDEO_AS_IMAGE_POSITIVE_CLASS": args.positive_class,
        "DEEPSCAN_VIDEO_AS_IMAGE_DECISION_THRESHOLD": float(round(best["threshold"], 6)),
    }

    report = {
        "dataset": {
            "real_dir": os.path.abspath(args.real_dir),
            "ai_dir": os.path.abspath(args.ai_dir),
            "real_count": int(np.sum(y_true == 0)),
            "ai_count": int(np.sum(y_true == 1)),
            "total_scored": int(y_true.size),
        },
        "pipeline": {
            "sample_count": int(sample_count),
            "sampling": "uniform_t_i=(i*T)/(N+1)",
            "positive_class": args.positive_class,
            "model": "image_model_on_video_frames_mean_aggregation",
            "current_decision_threshold": float(VIDEO_DECISION_THRESHOLD),
        },
        "best_threshold": best,
        "recommended_env": recommended_env,
        "scores": [asdict(s) for s in samples],
    }

    out_path = os.path.abspath(args.output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nCalibration complete")
    print(f"Total scored: {y_true.size}")
    print(f"Sample count: {sample_count}")
    print(f"Best threshold: {best['threshold']:.6f}")
    print(f"Balanced accuracy: {best['balanced_accuracy']:.4f}")
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"F1: {best['f1']:.4f}")
    print(f"TP={best['tp']} TN={best['tn']} FP={best['fp']} FN={best['fn']}")
    print("Recommended env:")
    for k, v in recommended_env.items():
        print(f"  {k}={v}")
    print(f"Saved report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
