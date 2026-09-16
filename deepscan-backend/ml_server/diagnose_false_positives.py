"""
Why do genuine images pick up non-trivial synthetic probability?

Prints the full score distribution per real-image category (plus degraded
variants), so ai_low can be set from the observed REAL distribution rather than
guessed, and so we can see WHICH kinds of genuine images drift upward.
"""
import glob
import io
import json
import os
import random
import sys

import numpy as np
from PIL import Image

from image_server import preprocess_for_hf_vit, run_hf_vit_inference

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def listing(pattern, limit, seed=23):
    f = [p for p in glob.glob(pattern) if os.path.splitext(p)[1].lower() in IMAGE_EXTS]
    random.Random(seed).shuffle(f)
    return f[:limit]


def degrade(path, mode):
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    if mode == "jpeg30":
        img.save(buf, format="JPEG", quality=30); ext = ".jpg"
    elif mode == "jpeg60":
        img.save(buf, format="JPEG", quality=60); ext = ".jpg"
    elif mode == "resize_small":
        s = 0.35
        img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BICUBIC).save(buf, format="JPEG", quality=85); ext = ".jpg"
    elif mode == "screenshot_png":
        s = min(1.0, 1000 / max(img.width, img.height))
        img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BICUBIC).save(buf, format="PNG"); ext = ".png"
    elif mode == "social_media":
        # typical social pipeline: downscale to 1080 long edge + moderate JPEG
        s = min(1.0, 1080 / max(img.width, img.height))
        img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BICUBIC).save(buf, format="JPEG", quality=72); ext = ".jpg"
    else:
        raise ValueError(mode)
    out = os.path.join("/tmp", f"fp_{mode}_{abs(hash(path)) % 10**8}{ext}")
    with open(out, "wb") as fh:
        fh.write(buf.getvalue())
    return out


def scores(paths, label):
    out = []
    for i, p in enumerate(paths):
        try:
            out.append(run_hf_vit_inference(preprocess_for_hf_vit(p)))
        except Exception as exc:
            print(f"   skip {os.path.basename(p)}: {exc}")
        if (i + 1) % 40 == 0:
            print(f"   {label} {i+1}/{len(paths)}", flush=True)
    return np.array(out)


def summarize(name, arr):
    if arr.size == 0:
        return None
    q = {f"p{p}": round(float(np.percentile(arr, p)), 4) for p in (50, 75, 90, 95, 99)}
    row = {"name": name, "n": int(arr.size), "mean": round(float(arr.mean()), 4), **q,
           "max": round(float(arr.max()), 4),
           "frac_above_0.1": round(float((arr > 0.1).mean()), 4),
           "frac_above_0.3": round(float((arr > 0.3).mean()), 4),
           "frac_above_0.5": round(float((arr > 0.5).mean()), 4)}
    return row


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    real_sets = {
        "camera_nature":  listing("realistic_data/real_dataset/nature/*", N),
        "camera_city":    listing("realistic_data/real_dataset/city/*", N),
        "camera_food":    listing("realistic_data/real_dataset/food/*", N),
        "camera_animals": listing("realistic_data/real_dataset/animals/*", N),
        "real_people":    listing("realistic_data/real_dataset/people/*", N),
        "real_faces_hq":  listing("calibration_data2/real/*", N),
        "ff_real_frames": listing("ff_splits/test/real/*", N),
    }
    rows, raw = [], {}
    for name, paths in real_sets.items():
        print(f"[{name}] n={len(paths)}", flush=True)
        s = scores(paths, name)
        raw[name] = s.tolist()
        r = summarize(name, s)
        if r:
            rows.append(r)

    # degraded variants of genuine photos
    base = listing("realistic_data/real_dataset/people/*", max(20, N // 2), seed=31) + \
           listing("realistic_data/real_dataset/city/*", max(20, N // 2), seed=31)
    for mode in ("jpeg60", "jpeg30", "resize_small", "screenshot_png", "social_media"):
        tmp = []
        for p in base:
            try:
                tmp.append(degrade(p, mode))
            except Exception:
                pass
        print(f"[real_{mode}] n={len(tmp)}", flush=True)
        s = scores(tmp, mode)
        raw[f"real_{mode}"] = s.tolist()
        r = summarize(f"real_{mode}", s)
        if r:
            rows.append(r)
        for t in tmp:
            try:
                os.unlink(t)
            except OSError:
                pass

    allreal = np.array([v for k, vals in raw.items() for v in vals])
    pooled = summarize("ALL_REAL_POOLED", allreal)

    print("\n" + "=" * 118)
    print(f"{'category':22s} {'n':>4s} {'mean':>7s} {'p50':>7s} {'p90':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s} {'>0.1':>6s} {'>0.3':>6s} {'>0.5':>6s}")
    print("=" * 118)
    for r in rows + [pooled]:
        print(f"{r['name']:22s} {r['n']:4d} {r['mean']:7.3f} {r['p50']:7.3f} {r['p90']:7.3f} "
              f"{r['p95']:7.3f} {r['p99']:7.3f} {r['max']:7.3f} {r['frac_above_0.1']:6.2f} "
              f"{r['frac_above_0.3']:6.2f} {r['frac_above_0.5']:6.2f}")

    with open("false_positive_diagnosis.json", "w") as fh:
        json.dump({"per_category": rows, "pooled": pooled, "raw_scores": raw}, fh, indent=2)
    print("\nwrote false_positive_diagnosis.json")


if __name__ == "__main__":
    main()
