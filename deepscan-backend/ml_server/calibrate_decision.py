"""
Derive the decision engine's four thresholds from balanced validation data and
write calibration.json. Nothing here is hand-picked.

  ai_high / deepfake_high : chosen to hit a target FALSE-REAL rate ceiling while
                            maximising synthetic recall (the project's stated
                            objective), not to maximise raw accuracy.
  ai_low / deepfake_low   : chosen so that scoring below it is strong evidence of
                            authenticity (low synthetic contamination).

Everything between low and high is deliberately UNCERTAIN.
"""

import argparse
import glob
import json
import os
import random
import sys

import numpy as np

from image_server import (  # noqa: E402
    detect_largest_face,
    faceswap_model,
    hf_vit_model,
    preprocess_for_hf_vit,
    run_hf_vit_inference,
    score_face_manipulation,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def listing(pattern, limit):
    files = [p for p in glob.glob(pattern) if os.path.splitext(p)[1].lower() in IMAGE_EXTS]
    random.Random(7).shuffle(files)
    return files[:limit] if limit else files


def score_ai(paths, tag):
    out = []
    for i, p in enumerate(paths):
        try:
            out.append(run_hf_vit_inference(preprocess_for_hf_vit(p)))
        except Exception as exc:
            print(f"  [{tag}] skip {os.path.basename(p)}: {exc}")
        if (i + 1) % 50 == 0:
            print(f"  [{tag}] {i+1}/{len(paths)}", flush=True)
    return np.array(out)


def score_face(paths, tag):
    out = []
    for i, p in enumerate(paths):
        try:
            found, crop = detect_largest_face(p)
            if not found:
                continue
            out.append(score_face_manipulation(crop))
        except Exception as exc:
            print(f"  [{tag}] skip {os.path.basename(p)}: {exc}")
        if (i + 1) % 50 == 0:
            print(f"  [{tag}] {i+1}/{len(paths)}", flush=True)
    return np.array(out)


def pick_high(real_scores, fake_scores, max_false_real_of_reals=0.02):
    """
    Lowest threshold whose false-positive rate on REAL data stays under the cap.

    Also returns the measured PRECISION at that operating point (assuming the
    balanced validation prior). At a permissive operating point a raw score just
    above the threshold badly understates how often such a detection is right,
    so the decision engine uses this precision as the confidence floor for
    verdicts triggered here instead of reporting the bare probability.
    """
    grid = np.linspace(0.01, 0.999, 400)
    best = None
    for t in grid:
        fpr = float((real_scores >= t).mean()) if real_scores.size else 0.0
        if fpr <= max_false_real_of_reals:
            recall = float((fake_scores >= t).mean()) if fake_scores.size else 0.0
            if best is None or recall > best[1]:
                tp = float((fake_scores >= t).sum())
                fp = float((real_scores >= t).sum())
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                best = (float(t), recall, fpr, precision)
    if best is None:
        return float(np.quantile(real_scores, 0.99)), 0.0, 1.0, 0.0
    return best


def pick_low(real_scores, fake_scores, upper_bound, max_fake_below=0.02,
             real_quantile=0.95, min_reals_cleared=0.5):
    """
    The "confidently authentic" ceiling.

    Anchored on the observed REAL distribution: the q-th percentile of genuine
    scores, so ~q% of real images land in the confident-REAL zone and the
    upper tail falls through to UNCERTAIN instead of being asserted REAL with
    feeble confidence. Clamped below `upper_bound` and rejected if too many
    FAKES would leak under it.

    Maximising "reals cleared" instead (the earlier approach) pushed this up to
    meet ai_high whenever the detector separated well, which silently deleted
    the UNCERTAIN band and produced verdicts like "REAL, 52% synthetic".

    Returns (threshold, reals_cleared, fake_leak, valid). valid=False means no
    threshold achieves the required purity — the detector cannot certify
    authenticity at all, and must then be treated as positive-evidence-only
    rather than being allowed to block a REAL verdict (a weak detector that can
    never clear an image would otherwise force UNCERTAIN on everything).
    """
    if real_scores.size == 0:
        return (None, 0.0, 1.0, False)

    candidate = float(np.quantile(real_scores, real_quantile))
    candidate = min(candidate, float(upper_bound) * 0.999)
    leak = float((fake_scores <= candidate).mean()) if fake_scores.size else 0.0

    if leak > max_fake_below:
        # Too many fakes would be cleared; retreat to the strictest threshold
        # that still satisfies the purity requirement.
        grid = np.linspace(0.0005, candidate, 400)[::-1]
        for t in grid:
            l2 = float((fake_scores <= t).mean()) if fake_scores.size else 0.0
            if l2 <= max_fake_below:
                candidate, leak = float(t), l2
                break
        else:
            return (None, 0.0, leak, False)

    kept = float((real_scores <= candidate).mean())
    # A gate that clears only a small minority of genuine images is not a gate,
    # it is a blocker: it would force almost every authentic image to UNCERTAIN.
    # Such a detector is demoted to positive-evidence-only.
    if kept < min_reals_cleared:
        return (None, kept, leak, False)
    return (candidate, kept, leak, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--max-false-real", type=float, default=0.02,
                    help="Max fraction of REAL validation images allowed to cross ai_high")
    ap.add_argument("--max-false-real-deepfake", type=float, default=None,
                    help="Separate cap for deepfake_high (defaults to --max-false-real). "
                         "Raising it trades more false alarms on genuine faces for higher "
                         "face-swap recall; pick it from the measured trade-off, not by feel.")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "calibration.json"))
    args = ap.parse_args()

    if hf_vit_model is None:
        sys.exit("Community Forensics ViT not loaded; cannot calibrate ai_* thresholds.")

    print("=== AI detector calibration (synthetic vs real, whole image) ===")
    real_ai = np.concatenate([
        score_ai(listing("realistic_data/real_dataset/*/*", args.limit), "real_photos"),
        score_ai(listing("calibration_data2/real/*", args.limit // 2), "real_faces"),
    ])
    fake_ai = np.concatenate([
        score_ai(listing("realistic_data/Ai_generated_dataset/*/*", args.limit), "ai_photos"),
        score_ai(listing("calibration_data2/fake/*", args.limit // 2), "gan_faces"),
    ])
    ai_high, ai_rec, ai_fpr, ai_prec = pick_high(real_ai, fake_ai, args.max_false_real)
    ai_low, ai_kept, ai_leak, ai_low_valid = pick_low(real_ai, fake_ai, upper_bound=ai_high)

    result = {
        "source": "calibrate_decision.py on balanced validation data",
        "thresholds": {
            "ai_high": round(ai_high, 4),
            "ai_low": None if ai_low is None else round(ai_low, 4),
            "ai_precision_at_high": round(ai_prec, 4),
        },
        "metrics": {
            "ai_detector": {
                "n_real": int(real_ai.size), "n_fake": int(fake_ai.size),
                "ai_high": round(ai_high, 4),
                "recall_at_high": round(ai_rec, 4),
                "false_real_rate_at_high": round(ai_fpr, 4),
                "precision_at_high": round(ai_prec, 4),
                "ai_low": None if ai_low is None else round(ai_low, 4),
                "ai_low_can_certify_authenticity": bool(ai_low_valid),
                "reals_below_low": round(ai_kept, 4),
                "fakes_leaking_below_low": round(ai_leak, 4),
            }
        },
    }

    if faceswap_model is not None and os.path.isdir("ff_splits/val/real"):
        print("\n=== Face-manipulation detector calibration (FF++ val) ===")
        real_df = score_face(listing("ff_splits/val/real/*", args.limit), "ff_real")
        fake_df = score_face(listing("ff_splits/val/fake/*", args.limit), "ff_fake")
        df_cap = args.max_false_real_deepfake if args.max_false_real_deepfake is not None else args.max_false_real
        df_high, df_rec, df_fpr, df_prec = pick_high(real_df, fake_df, df_cap)
        df_low, df_kept, df_leak, df_low_valid = pick_low(real_df, fake_df, upper_bound=df_high)
        result["thresholds"]["deepfake_high"] = round(df_high, 4)
        result["thresholds"]["deepfake_precision_at_high"] = round(df_prec, 4)
        result["thresholds"]["deepfake_low"] = None if df_low is None else round(df_low, 4)
        result["metrics"]["faceswap_detector"] = {
            "n_real": int(real_df.size), "n_fake": int(fake_df.size),
            "deepfake_high": round(df_high, 4),
            "recall_at_high": round(df_rec, 4),
            "false_real_rate_at_high": round(df_fpr, 4),
            "precision_at_high": round(df_prec, 4),
            "false_real_cap_requested": df_cap,
            "deepfake_low": None if df_low is None else round(df_low, 4),
            "deepfake_low_can_certify_authenticity": bool(df_low_valid),
            "reals_below_low": round(df_kept, 4),
            "fakes_leaking_below_low": round(df_leak, 4),
        }
    else:
        print("\n[warn] face-manipulation model or FF++ val split unavailable; "
              "leaving deepfake_* at conservative defaults.")
        result["thresholds"]["deepfake_high"] = 0.9
        result["thresholds"]["deepfake_low"] = None

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print("\n" + json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
