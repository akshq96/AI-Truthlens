"""
System-level acceptance test through the live ML server (exactly what the web app gets).

    python acceptance_test.py --label new --out test_results/acceptance_new
    python acceptance_test.py --label old --out test_results/acceptance_old   # server started with DEEPSCAN_FACE_DISABLED=1
    python acceptance_test.py --compare test_results/acceptance_old.json test_results/acceptance_new.json

Test set: test_sets/acceptance/{real, deepfake, ai_generated, user_ai_screenshots}
Expected label per folder: real -> REAL; deepfake -> DEEPFAKE; ai_generated and user_ai_screenshots -> AI-GENERATED.
A prediction counts as FAKE when the verdict is AI-GENERATED or DEEPFAKE.

Reports per image: expected, predicted, fake probability (max of the detector probabilities),
and per run: REAL accuracy, FAKE recall, F1, ROC-AUC, REAL->FAKE, DEEPFAKE->REAL, AI->REAL,
UNCERTAIN counts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDERS = {"real": "REAL", "deepfake": "DEEPFAKE", "ai_generated": "AI-GENERATED", "user_ai_screenshots": "AI-GENERATED"}
EXT = (".jpg", ".jpeg", ".png", ".webp")


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def summarize(rows):
    fake_verdicts = ("AI-GENERATED", "DEEPFAKE")
    real = [r for r in rows if r["expected"] == "REAL"]
    fakes = [r for r in rows if r["expected"] != "REAL"]
    tp = sum(r["predicted"] in fake_verdicts for r in fakes)
    fp = sum(r["predicted"] in fake_verdicts for r in real)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / len(fakes) if fakes else 0.0
    scored_real = [r["fake_probability"] for r in real if r["fake_probability"] is not None]
    scored_fake = [r["fake_probability"] for r in fakes if r["fake_probability"] is not None]
    by = {}
    for folder in FOLDERS:
        sub = [r for r in rows if r["folder"] == folder]
        by[folder] = {v: sum(r["predicted"] == v for r in sub) for v in ("REAL", "AI-GENERATED", "DEEPFAKE", "UNCERTAIN")}
        by[folder]["n"] = len(sub)
    return {
        "n": len(rows),
        "real_accuracy": round(sum(r["predicted"] == "REAL" for r in real) / max(1, len(real)), 4),
        "fake_recall": round(rec, 4),
        "precision": round(prec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "roc_auc": None if auc(scored_fake, scored_real) is None else round(auc(scored_fake, scored_real), 4),
        "real_to_fake": fp,
        "deepfake_to_real": sum(r["folder"] == "deepfake" and r["predicted"] == "REAL" for r in rows),
        "ai_to_real": sum(r["folder"] in ("ai_generated", "user_ai_screenshots") and r["predicted"] == "REAL" for r in rows),
        "uncertain": sum(r["predicted"] == "UNCERTAIN" for r in rows),
        "exact_label_accuracy": round(sum(r["predicted"] == r["expected"] for r in rows) / max(1, len(rows)), 4),
        "by_folder": by,
    }


def run(label, url, out, test_set="acceptance"):
    health = requests.get(url + "/health", timeout=30).json()
    rows = []
    t0 = time.time()
    for folder, expected in FOLDERS.items():
        d = os.path.join(HERE, "test_sets", test_set, folder)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(EXT):
                continue
            path = os.path.join(d, fn)
            mt = "image/png" if fn.lower().endswith(".png") else "image/jpeg"
            with open(path, "rb") as fh:
                r = requests.post(url + "/predict/image", files={"file": (fn, fh, mt)}, timeout=300).json()
            probs = [v for v in (r.get("ai_probability"), r.get("deepfake_probability")) if v is not None]
            rows.append({"folder": folder, "file": fn, "expected": expected, "predicted": r.get("result"),
                         "fake_probability": max(probs) if probs else None,
                         "ai_probability": r.get("ai_probability"), "deepfake_probability": r.get("deepfake_probability"),
                         "face_detected": r.get("face_detected"), "confidence": r.get("confidence"),
                         "reason": r.get("decision_reason")})
            print(f"{expected:12s} {r.get('result'):12s} fake_p={rows[-1]['fake_probability']}  {folder}/{fn}", flush=True)
    summary = summarize(rows)
    summary.update({"label": label, "test_set": test_set, "face_detector": health.get("face_manipulation_detector"),
                    "seconds": round(time.time() - t0, 1)})
    print(json.dumps({k: v for k, v in summary.items() if k != "by_folder"}, indent=2))
    print(json.dumps(summary["by_folder"], indent=2))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(out + ".json", "w"), indent=1)
    with open(out + ".md", "w") as fh:
        fh.write("| Expected | Predicted | Fake probability | File |\n|---|---|---|---|\n")
        for r in rows:
            fp_ = "—" if r["fake_probability"] is None else f"{r['fake_probability']:.3f}"
            fh.write(f"| {r['expected']} | {r['predicted']} | {fp_} | {r['folder']}/{r['file']} |\n")
    print("wrote", out + ".json/.md")


def compare(a, b):
    A, Bs = json.load(open(a))["summary"], json.load(open(b))["summary"]
    keys = ["real_accuracy", "fake_recall", "precision", "f1", "roc_auc", "real_to_fake", "deepfake_to_real",
            "ai_to_real", "uncertain", "exact_label_accuracy"]
    print(f"{'metric':22s} {A['label']:>10s} {Bs['label']:>10s}")
    for k in keys:
        print(f"{k:22s} {str(A[k]):>10s} {str(Bs[k]):>10s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--url", default="http://127.0.0.1:7070")
    ap.add_argument("--out", default="test_results/acceptance")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--set", default="acceptance", help="folder under test_sets/")
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        run(a.label, a.url, a.out, a.set)
    sys.exit(0)
