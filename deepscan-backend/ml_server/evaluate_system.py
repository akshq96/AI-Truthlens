"""
End-to-end evaluation of the decision engine, per content category.

Reports accuracy / precision / recall / F1 / ROC-AUC / confusion matrix and —
most importantly for this project — the FALSE-REAL RATE: the fraction of
synthetic or manipulated media confidently labelled REAL.

UNCERTAIN is not scored as a binary mistake; it is reported separately as an
abstention rate, because an honest "I don't know" is the intended behaviour for
weak evidence and must not be rewarded or punished as if it were a guess.
"""

import argparse
import glob
import io
import json
import os
import random

import numpy as np
from PIL import Image

import decision_engine
import provenance
import synthid
from image_server import (
    CALIBRATION,
    detect_largest_face,
    faceswap_model,
    hf_vit_model,
    preprocess_for_hf_vit,
    run_hf_vit_inference,
    score_face_manipulation,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SYNTH = {decision_engine.RESULT_AI, decision_engine.RESULT_DEEPFAKE}


def listing(pattern, limit, seed=11):
    files = [p for p in glob.glob(pattern) if os.path.splitext(p)[1].lower() in IMAGE_EXTS]
    random.Random(seed).shuffle(files)
    return files[:limit] if limit else files


def degrade(path, mode):
    """Return a temp file path with a robustness transform applied."""
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    if mode == "jpeg40":
        img.save(buf, format="JPEG", quality=40)
        suffix = ".jpg"
    elif mode == "resize50":
        img.resize((max(1, img.width // 2), max(1, img.height // 2)), Image.BICUBIC).save(buf, format="JPEG", quality=88)
        suffix = ".jpg"
    elif mode == "screenshot":
        # emulate a screen capture: downscale to a typical window size, PNG re-encode
        scale = min(1.0, 1000 / max(img.width, img.height))
        img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BICUBIC).save(buf, format="PNG")
        suffix = ".png"
    else:
        raise ValueError(mode)
    out = os.path.join("/tmp", f"deg_{mode}_{abs(hash(path)) % 10**8}{suffix}")
    with open(out, "wb") as fh:
        fh.write(buf.getvalue())
    return out


def run_one(path):
    """Mirror the server's /predict/image evidence gathering + fusion."""
    errors, scores = [], {}
    p_ai = None
    if hf_vit_model is not None:
        try:
            p_ai = run_hf_vit_inference(preprocess_for_hf_vit(path))
            scores["ai"] = p_ai
        except Exception as exc:
            errors.append(f"ai:{exc}")
    else:
        errors.append("ai:not loaded")

    try:
        has_face, crop = detect_largest_face(path)
    except Exception as exc:
        has_face, crop = False, None
        errors.append(f"face:{exc}")

    p_df = None
    if has_face and faceswap_model is not None and crop is not None:
        try:
            p_df = score_face_manipulation(crop)
            scores["df"] = p_df
        except Exception as exc:
            errors.append(f"df:{exc}")

    try:
        md, c2 = provenance.analyze_with_c2pa(path)
    except Exception as exc:
        md, c2 = {}, {}
        errors.append(f"provenance:{exc}")
    d = decision_engine.decide(p_ai, p_df, has_face, {"status": synthid.STATUS_UNAVAILABLE},
                               CALIBRATION, errors, metadata=md, c2pa=c2)
    # continuous synthetic score for ROC-AUC
    cont = max([v for v in (p_ai, p_df if has_face else None) if v is not None], default=0.0)
    return d, cont, has_face


def roc_auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out", default="evaluation_report.json")
    args = ap.parse_args()
    L = args.limit

    # label 1 = synthetic/manipulated, 0 = authentic
    categories = {
        "camera_photos_no_face":  (listing("realistic_data/real_dataset/nature/*", L) +
                                   listing("realistic_data/real_dataset/city/*", L // 2), 0),
        "camera_photos_faces":    (listing("realistic_data/real_dataset/people/*", L), 0),
        "real_faces_dataset":     (listing("calibration_data2/real/*", L), 0),
        "ff_real_faces":          (listing("ff_splits/test/real/*", L), 0),
        "ai_landscapes_objects":  (listing("realistic_data/Ai_generated_dataset/nature/*", L) +
                                   listing("realistic_data/Ai_generated_dataset/city/*", L // 2), 1),
        "ai_portraits":           (listing("realistic_data/Ai_generated_dataset/people/*", L), 1),
        "gan_faces":              (listing("calibration_data2/fake/*", L), 1),
        "face_swaps":             (listing("ff_splits/test/fake/*", L), 1),
        "unseen_manipulation_faceshifter": (listing("ff_splits/unseen/fake/*", L), 1),
        "real_with_camera_exif": (listing("/tmp/evalset_exif/*", L), 0),
    }

    rows = []
    for cat, (paths, label) in categories.items():
        print(f"[{cat}] n={len(paths)}", flush=True)
        for i, p in enumerate(paths):
            d, cont, face = run_one(p)
            rows.append({"cat": cat, "label": label, "result": d["result"],
                         "cont": cont, "face": face, "conf": d["confidence"]})
            if (i + 1) % 40 == 0:
                print(f"   {i+1}/{len(paths)}", flush=True)

    # robustness: degrade a sample of AI images and real photos
    for mode in ("jpeg40", "resize50", "screenshot"):
        for src, label, tag in ((listing("realistic_data/Ai_generated_dataset/people/*", L // 2, 3), 1, "ai"),
                                (listing("realistic_data/real_dataset/people/*", L // 2, 3), 0, "real")):
            cat = f"{mode}_{tag}"
            print(f"[{cat}] n={len(src)}", flush=True)
            for p in src:
                try:
                    dp = degrade(p, mode)
                except Exception:
                    continue
                d, cont, face = run_one(dp)
                rows.append({"cat": cat, "label": label, "result": d["result"],
                             "cont": cont, "face": face, "conf": d["confidence"]})
                try:
                    os.unlink(dp)
                except OSError:
                    pass

    # ---- metrics -------------------------------------------------------
    def block(subset, name):
        if not subset:
            return None
        y = np.array([r["label"] for r in subset])
        res = [r["result"] for r in subset]
        synth_pred = np.array([r in SYNTH for r in res])
        real_pred = np.array([r == decision_engine.RESULT_REAL for r in res])
        unc = np.array([r == decision_engine.RESULT_UNCERTAIN for r in res])

        tp = int(((y == 1) & synth_pred).sum()); fp = int(((y == 0) & synth_pred).sum())
        tn = int(((y == 0) & real_pred).sum());  fn = int(((y == 1) & real_pred).sum())
        n_unc = int(unc.sum())
        decided = tp + fp + tn + fn
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / max(1, int((y == 1).sum()))          # recall over ALL synthetic, abstentions count against
        f1 = 2 * prec * rec / (prec + rec) if (prec == prec and prec + rec) else float("nan")
        false_real = fn / max(1, int((y == 1).sum()))    # THE key metric
        false_synth = fp / max(1, int((y == 0).sum()))
        cont_pos = np.array([r["cont"] for r in subset if r["label"] == 1])
        cont_neg = np.array([r["cont"] for r in subset if r["label"] == 0])
        return {
            "name": name, "n": len(subset), "n_synthetic": int((y == 1).sum()), "n_authentic": int((y == 0).sum()),
            "accuracy_on_decided": round((tp + tn) / decided, 4) if decided else None,
            "precision": None if prec != prec else round(prec, 4),
            "recall_synthetic": round(rec, 4),
            "f1": None if f1 != f1 else round(f1, 4),
            "roc_auc": None if len(cont_pos) == 0 or len(cont_neg) == 0 else round(roc_auc(cont_pos, cont_neg), 4),
            "FALSE_REAL_RATE": round(false_real, 4),
            "false_synthetic_rate": round(false_synth, 4),
            "uncertain_rate": round(n_unc / len(subset), 4),
            "confusion": {"tp_synth": tp, "fp_synth": fp, "tn_real": tn, "fn_called_real": fn, "uncertain": n_unc},
        }

    report = {"calibration": CALIBRATION, "overall": block(rows, "OVERALL"), "per_category": []}
    for cat in dict.fromkeys(r["cat"] for r in rows):
        b = block([r for r in rows if r["cat"] == cat], cat)
        if b:
            report["per_category"].append(b)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    o = report["overall"]
    print("\n================ OVERALL ================")
    print(json.dumps(o, indent=2))
    print("\n================ PER CATEGORY ================")
    print(f"{'category':34s} {'n':>4s} {'recall':>7s} {'FALSE-REAL':>11s} {'uncert':>7s} {'auc':>6s}")
    for b in report["per_category"]:
        print(f"{b['name']:34s} {b['n']:4d} {b['recall_synthetic']:7.3f} "
              f"{b['FALSE_REAL_RATE']:11.3f} {b['uncertain_rate']:7.3f} "
              f"{(b['roc_auc'] if b['roc_auc'] is not None else float('nan')):6.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
