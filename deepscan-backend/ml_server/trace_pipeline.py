"""
Full stage-by-stage trace of every detector, for label-polarity verification and
root-cause work. Prints raw logits before any probability conversion, so a
mis-signed or mis-mapped class cannot hide behind a calibrated number.

Usage:
  python trace_pipeline.py --real <glob> --deepfake <glob> --ai <glob> [-n 5]
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch

import decision_engine
import provenance
import synthid
from image_server import (
    CALIBRATION, FACESWAP_IMAGE_SIZE, FACESWAP_MODEL_PATH, FACESWAP_POSITIVE_CLASS,
    HF_VIT_MODEL_ID, IMAGE_MODEL_PATH, GENERAL_MODEL_PATH,
    deepfake_probability, detect_largest_face, faceswap_model, general_model,
    hf_vit_model, image_model, preprocess_for_hf_vit, preprocess_rgb_array,
    _positive_probability,
)


def raw_hf_vit(path):
    """Raw logit from the Community Forensics ViT, before sigmoid."""
    if hf_vit_model is None:
        return None
    pv = preprocess_for_hf_vit(path)
    with torch.no_grad():
        logits = hf_vit_model(pixel_values=pv).logits
    return logits.flatten().float().tolist()


def raw_faceswap(face_bgr):
    """Raw logit from the FF++ Xception, before sigmoid."""
    if faceswap_model is None or face_bgr is None:
        return None
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (FACESWAP_IMAGE_SIZE, FACESWAP_IMAGE_SIZE))
    t = preprocess_rgb_array(resized.astype(np.float32))
    with torch.no_grad():
        out = faceswap_model(torch.from_numpy(t).float())
    return out.flatten().float().tolist()


def trace(path, expected):
    row = {"file": os.path.basename(path), "expected": expected}

    # ---- stage 1: face detection -------------------------------------
    try:
        has_face, crop = detect_largest_face(path)
    except Exception as exc:
        has_face, crop = False, None
        row["face_error"] = str(exc)
    row["face_detected"] = has_face
    row["face_crop_shape"] = None if crop is None else list(crop.shape)

    # ---- stage 2: AI detector (raw -> prob) --------------------------
    row["ai_model"] = HF_VIT_MODEL_ID if hf_vit_model is not None else None
    lg = raw_hf_vit(path)
    row["ai_raw_logits"] = lg
    if lg is not None:
        # this model emits a single logit = log-odds of FAKE
        row["ai_sigmoid_of_logit"] = float(torch.sigmoid(torch.tensor(lg[0])))
        row["ai_probability_used"] = row["ai_sigmoid_of_logit"]
        row["ai_semantics"] = "sigmoid(logit) = P(synthetic); positive_class=deepfake"

    # ---- stage 3: face-manipulation detector (raw -> prob) -----------
    row["df_model_checkpoint"] = os.path.basename(FACESWAP_MODEL_PATH) if faceswap_model is not None else None
    row["df_input_size"] = FACESWAP_IMAGE_SIZE
    row["df_positive_class"] = FACESWAP_POSITIVE_CLASS
    if has_face and faceswap_model is not None:
        dlg = raw_faceswap(crop)
        row["df_raw_logits"] = dlg
        if dlg is not None:
            sig = float(torch.sigmoid(torch.tensor(dlg[0])))
            row["df_sigmoid_of_logit"] = sig
            row["df_positive_probability_fn"] = _positive_probability(np.array(dlg))
            row["df_probability_used"] = deepfake_probability(
                np.array([dlg]), positive_class=FACESWAP_POSITIVE_CLASS)
            row["df_semantics"] = (f"positive_class={FACESWAP_POSITIVE_CLASS}; "
                                   f"P(deepfake) = 1 - sigmoid(logit)")
    else:
        row["df_raw_logits"] = None
        row["df_probability_used"] = None
        row["df_skip_reason"] = "no face" if not has_face else "model not loaded"

    # ---- stage 4: provenance ----------------------------------------
    md, c2 = provenance.analyze_with_c2pa(path)
    row["metadata_available"] = md.get("available")
    row["metadata_consistency"] = md.get("consistency_score")
    row["metadata_reliability"] = md.get("reliability")
    row["c2pa_status"] = c2.get("status")
    sid = synthid.check_image(path)
    row["synthid_status"] = sid.get("status")

    # ---- stage 5: thresholds + decision ------------------------------
    row["thresholds"] = {k: CALIBRATION.get(k) for k in
                         ("ai_high", "ai_low", "deepfake_high", "deepfake_low")}
    d = decision_engine.decide(
        p_ai=row.get("ai_probability_used"),
        p_deepfake=row.get("df_probability_used"),
        face_detected=has_face,
        synthid=sid,
        calibration=CALIBRATION,
        detector_errors=[],
        metadata=md,
        c2pa=c2,
    )
    row["final_result"] = d["result"]
    row["final_confidence"] = d["confidence"]
    row["decision_reason"] = d["decision_reason"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="realistic_data/real_dataset/people/*")
    ap.add_argument("--deepfake", default="ff_splits/test/fake/*")
    ap.add_argument("--ai", default="realistic_data/Ai_generated_dataset/people/*")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--out", default="trace_report.json")
    args = ap.parse_args()

    print("=" * 100)
    print("LOADED MODELS")
    print(f"  AI detector      : {HF_VIT_MODEL_ID if hf_vit_model is not None else 'NOT LOADED'}")
    print(f"  face-manip ckpt  : {FACESWAP_MODEL_PATH if faceswap_model is not None else 'NOT LOADED'}")
    print(f"  legacy face ckpt : {IMAGE_MODEL_PATH if image_model is not None else 'NOT LOADED'}")
    print(f"  legacy general   : {GENERAL_MODEL_PATH if general_model is not None else 'NOT LOADED'}")
    print(f"  thresholds       : {json.dumps({k: CALIBRATION.get(k) for k in ('ai_high','ai_low','deepfake_high','deepfake_low')})}")
    print("=" * 100)

    rows = []
    for label, pattern in (("REAL", args.real), ("DEEPFAKE", args.deepfake), ("AI-GENERATED", args.ai)):
        files = sorted(glob.glob(pattern))[:args.n]
        print(f"\n######## {label}  (n={len(files)}) ########")
        for p in files:
            r = trace(p, label)
            rows.append(r)
            print(f"\n  {r['file']}   expected={label}")
            print(f"    face_detected={r['face_detected']} crop={r['face_crop_shape']}")
            print(f"    AI  raw_logits={r.get('ai_raw_logits')}  -> P(synthetic)={r.get('ai_probability_used')}")
            print(f"    DF  raw_logits={r.get('df_raw_logits')}  -> P(deepfake)={r.get('df_probability_used')}"
                  + (f"  [skipped: {r.get('df_skip_reason')}]" if r.get('df_skip_reason') else ""))
            print(f"    meta consistency={r.get('metadata_consistency')} ({r.get('metadata_reliability')})"
                  f"  c2pa={r.get('c2pa_status')}  synthid={r.get('synthid_status')}")
            print(f"    => {r['final_result']}  conf={r['final_confidence']}")

    # --- polarity verification ---------------------------------------
    print("\n" + "=" * 100)
    print("LABEL POLARITY CHECK (mean raw logit per class)")
    print("=" * 100)
    for det, key in (("AI detector (ViT)", "ai_raw_logits"), ("face-manip (Xception)", "df_raw_logits")):
        print(f"\n{det}:")
        for label in ("REAL", "DEEPFAKE", "AI-GENERATED"):
            vals = [r[key][0] for r in rows if r["expected"] == label and r.get(key)]
            if vals:
                print(f"   {label:14s} n={len(vals):2d}  mean_logit={np.mean(vals):+8.3f}  "
                      f"mean_sigmoid={float(np.mean([1/(1+np.exp(-v)) for v in vals])):.4f}")
        print("   Expectation: FAKE classes should sit on the opposite side of REAL.")

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
