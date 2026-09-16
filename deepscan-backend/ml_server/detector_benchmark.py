"""
Standalone detector benchmark (no web server involved).

    python detector_benchmark.py --real <dir> --fake <dir> --detector effort_face
    python detector_benchmark.py --real <dir> --fake <dir> --detector effort_aigi --out results/effort_aigi
    python detector_benchmark.py --real <dir> --fake <dir> --detector old_deepscan   # live ML server on :7070
    python detector_benchmark.py --polarity --real <dir> --fake <dir> --detector effort_face

Detectors
  effort_face   Effort CLIP ViT-L/14 trained on FaceForensics++ (dlib 5-point aligned face)
  effort_aigi   Effort CLIP ViT-L/14 trained on Chameleon (whole image, 224x224)
  gend          GenD CLIP ViT-L/14 (yermandy/deepfake-detection), Haar face crop
  cf_vit        Community Forensics ViT-S/16 (whole image)
  old_deepscan  The previous DeepScan decision (HTTP /predict/image); fake = AI-GENERATED or DEEPFAKE

Per image the script writes: filename, expected_label, raw_fake_probability,
predicted_label, confidence (probability of the predicted label). Images a face
detector cannot use (no face found) are reported as predicted_label=no_face and
excluded from threshold metrics, but counted and listed separately.

Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
false-positive rate, false-negative rate. Positive class = fake.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def list_images(d):
    out = []
    for root, _, files in os.walk(d):
        for f in files:
            if f.lower().endswith(EXT) and not f.startswith("."):
                out.append(os.path.join(root, f))
    return sorted(out)


# ------------------------------------------------------------------ detectors
class Detector:
    name = ""
    needs_face = False

    def score(self, paths):  # -> list[(prob or None, raw)]
        raise NotImplementedError


class EffortFace(Detector):
    name, needs_face = "effort_face", True

    def __init__(self):
        import effort_model as em
        self.em = em
        self.model = em.load_effort(em.FACE_CKPT)
        self.aligner = em.FaceAligner()

    def score(self, paths):
        import cv2, torch
        out = []
        for p in paths:
            img = cv2.imread(p)
            face = self.aligner.align(img) if img is not None else None
            if face is None:
                out.append((None, "no_face"))
                continue
            probs, logits = self.em.fake_probability(self.model, self.em.to_tensor_bgr(face)[None])
            out.append((probs[0], logits[0]))
        return out


class EffortAIGI(Detector):
    name = "effort_aigi"

    def __init__(self):
        import effort_model as em
        self.em = em
        self.model = em.load_effort(em.AIGI_CKPT)

    def score(self, paths):
        import cv2, torch
        out = []
        for i in range(0, len(paths), 8):
            chunk = paths[i:i + 8]
            tensors = [self.em.to_tensor_bgr(cv2.imread(p)) for p in chunk]
            probs, logits = self.em.fake_probability(self.model, torch.stack(tensors))
            out += list(zip(probs, logits))
        return out


class GenD(Detector):
    name, needs_face = "gend", True

    def __init__(self):
        import glob, cv2, torch
        p = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--yermandy--deepfake-detection/snapshots/*/model.torchscript"))[0]
        self.model = torch.jit.load(p, map_location="cpu").eval()
        self.det = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

    def score(self, paths):
        import cv2, torch
        from PIL import Image
        mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
        out = []
        for p in paths:
            img = cv2.imread(p)
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ms = max(30, int(0.10 * min(g.shape)))
            fs = self.det.detectMultiScale(g, 1.1, 8, minSize=(ms, ms))
            if len(fs) == 0:
                out.append((None, "no_face"))
                continue
            x, y, w, h = max(fs, key=lambda r: r[2] * r[3])
            H, W = g.shape
            cx, cy, half = x + w / 2, y + h / 2, max(w, h) * 1.3 / 2
            crop = img[int(max(0, cy - half)):int(min(H, cy + half)), int(max(0, cx - half)):int(min(W, cx + half))]
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            s = 224 / min(pil.size)
            pil = pil.resize((max(224, round(pil.width * s)), max(224, round(pil.height * s))), Image.BICUBIC)
            l, t = (pil.width - 224) // 2, (pil.height - 224) // 2
            arr = (np.asarray(pil.crop((l, t, l + 224, t + 224)), np.float32) / 255 - mean) / std
            with torch.no_grad():
                logits = self.model(torch.from_numpy(arr.transpose(2, 0, 1).copy())[None]).float()
            out.append((float(torch.softmax(logits, 1)[0, 1]), logits[0].tolist()))
        return out


class CFViT(Detector):
    name = "cf_vit"

    def __init__(self):
        from transformers import ViTForImageClassification
        self.model = ViTForImageClassification.from_pretrained("buildborderless/CommunityForensics-DeepfakeDet-ViT").eval()

    def score(self, paths):
        import torch
        from PIL import Image
        mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
        out = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            s = 440 / min(img.size)
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
            l, t = round((img.width - 384) / 2), round((img.height - 384) / 2)
            arr = (np.asarray(img.crop((l, t, l + 384, t + 384)), np.float32) / 255 - mean) / std
            with torch.no_grad():
                z = self.model(pixel_values=torch.from_numpy(arr.transpose(2, 0, 1).copy())[None]).logits.flatten()[0]
            out.append((float(torch.sigmoid(z)), [float(z)]))
        return out


class OldDeepScan(Detector):
    name = "old_deepscan"

    def __init__(self, url="http://127.0.0.1:7070/predict/image"):
        self.url = url

    def score(self, paths):
        import requests
        out = []
        for p in paths:
            mt = "image/png" if p.lower().endswith(".png") else "image/jpeg"
            with open(p, "rb") as fh:
                d = requests.post(self.url, files={"file": (os.path.basename(p), fh, mt)}, timeout=300).json()
            fake = d["result"] in ("AI-GENERATED", "DEEPFAKE")
            if d["result"] == "UNCERTAIN":
                prob = None
            else:
                prob = 1.0 if fake else 0.0
            out.append((prob, d["result"]))
        return out


DETECTORS = {c.name: c for c in (EffortFace, EffortAIGI, GenD, CFViT, OldDeepScan)}


# ------------------------------------------------------------------ metrics
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def metrics(rows, threshold):
    scored = [r for r in rows if r["raw_fake_probability"] is not None]
    y = np.array([1 if r["expected_label"] == "fake" else 0 for r in scored])
    p = np.array([r["raw_fake_probability"] for r in scored], float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    unusable = [r for r in rows if r["raw_fake_probability"] is None]
    return {
        "threshold": threshold, "n_total": len(rows), "n_scored": len(scored),
        "n_unscored": len(unusable),
        "unscored_by_class": {"real": sum(r["expected_label"] == "real" for r in unusable),
                              "fake": sum(r["expected_label"] == "fake" for r in unusable)},
        "accuracy": round((tp + tn) / max(1, len(scored)), 4),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "roc_auc": None if auc(p[y == 1], p[y == 0]) is None else round(auc(p[y == 1], p[y == 0]), 4),
        "confusion": {"tn_real_as_real": tn, "fp_real_as_fake": fp, "fn_fake_as_real": fn, "tp_fake_as_fake": tp},
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None,
        "false_negative_rate": round(fn / (fn + tp), 4) if fn + tp else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--fake", required=True)
    ap.add_argument("--detector", required=True, choices=sorted(DETECTORS))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="output prefix (writes .csv and .json)")
    ap.add_argument("--polarity", action="store_true", help="only score 5 real + 5 fake and print raw outputs")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    real, fake = list_images(args.real), list_images(args.fake)
    if args.polarity:
        real, fake = real[:5], fake[:5]
    elif args.limit:
        real, fake = real[:args.limit], fake[:args.limit]
    print(f"[{args.detector}] real={len(real)} fake={len(fake)}", flush=True)

    t0 = time.time()
    det = DETECTORS[args.detector]()
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    scores = det.score(real + fake)
    per_img = (time.time() - t0) / max(1, len(real) + len(fake))

    rows = []
    for path, (prob, raw) in zip(real + fake, scores):
        exp = "real" if path in real else "fake"
        if prob is None:
            pred, conf = ("no_face" if raw == "no_face" else "uncertain"), None
        else:
            pred = "fake" if prob >= args.threshold else "real"
            conf = prob if pred == "fake" else 1 - prob
        rows.append({"filename": path, "expected_label": exp,
                     "raw_fake_probability": None if prob is None else round(float(prob), 6),
                     "predicted_label": pred, "confidence": None if conf is None else round(float(conf), 6),
                     "raw_output": raw if isinstance(raw, str) else [round(float(v), 4) for v in raw]})

    if args.polarity:
        for r in rows:
            print(f"  {r['expected_label']:4s} raw={r['raw_output']} P(fake)={r['raw_fake_probability']}  "
                  f"{os.path.basename(r['filename'])}")
        rp = [r["raw_fake_probability"] for r in rows if r["expected_label"] == "real" and r["raw_fake_probability"] is not None]
        fp_ = [r["raw_fake_probability"] for r in rows if r["expected_label"] == "fake" and r["raw_fake_probability"] is not None]
        print(f"mean P(fake): real={np.mean(rp) if rp else float('nan'):.3f} fake={np.mean(fp_) if fp_ else float('nan'):.3f} "
              f"-> polarity {'OK' if rp and fp_ and np.mean(fp_) > np.mean(rp) else 'CHECK/INVERTED'}")
        return

    m = metrics(rows, args.threshold)
    m["detector"] = args.detector
    m["seconds_per_image"] = round(per_img, 3)
    print(json.dumps(m, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out + ".csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["filename", "expected_label", "raw_fake_probability", "predicted_label", "confidence"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in w.fieldnames})
        with open(args.out + ".json", "w") as fh:
            json.dump({"metrics": m, "rows": rows}, fh, indent=1)
        print("wrote", args.out + ".csv/.json")


if __name__ == "__main__":
    sys.exit(main())
