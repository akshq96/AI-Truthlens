"""
Mandated sanity test: 5 known REAL, 5 known DEEPFAKE, 5 known AI-GENERATED,
driven through the LIVE HTTP endpoint (not by importing the library), so what is
printed is exactly what the API returns to the frontend.

Prints, per image: expected label, which detectors actually ran, raw logits,
raw and calibrated probabilities, face detection, threshold, final result and
confidence.
"""
import argparse
import glob
import json
import sys

import requests

ENDPOINT = "http://127.0.0.1:7070/predict/image"


def run(path):
    with open(path, "rb") as fh:
        r = requests.post(ENDPOINT, files={"file": fh}, timeout=180)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="realistic_data/real_dataset/people/*")
    ap.add_argument("--deepfake", default="ff_splits/test/fake/*")
    ap.add_argument("--ai", default="realistic_data/Ai_generated_dataset/people/*")
    ap.add_argument("-n", type=int, default=5)
    args = ap.parse_args()

    health = requests.get("http://127.0.0.1:7070/health", timeout=20).json()
    print("LOADED MODELS / THRESHOLDS")
    print(f"  ai detector      : {health.get('hf_vit_model_id')} (loaded={health.get('hf_vit_model_loaded')})")
    print(f"  face-manip ckpt  : {health.get('faceswap_model_path','').split('/')[-1]} (loaded={health.get('faceswap_model_loaded')})")
    print(f"  thresholds       : {json.dumps(health.get('thresholds'))}")
    print(f"  calibration      : {health.get('calibration_source')}")
    print(f"  synthid          : configured={health.get('synthid_configured')}")

    groups = [("REAL", args.real), ("DEEPFAKE", args.deepfake), ("AI-GENERATED", args.ai)]
    tally = {}
    for expected, pattern in groups:
        files = sorted(glob.glob(pattern))[:args.n]
        print("\n" + "=" * 118)
        print(f"EXPECTED: {expected}   (n={len(files)})")
        print("=" * 118)
        for p in files:
            try:
                d = run(p)
            except Exception as exc:
                print(f"  {p.split('/')[-1]:32s} REQUEST FAILED: {exc}")
                continue
            tr = d.get("trace", {})
            ai = tr.get("ai_detector", {})
            df = tr.get("deepfake_detector", {})
            print(f"\n  IMAGE            : {p.split('/')[-1]}")
            print(f"  EXPECTED         : {expected}")
            print(f"  AI MODEL         : {ai.get('model')}  ran={ai.get('ran')}")
            print(f"    raw logits     : {ai.get('raw_logits')}")
            print(f"    raw prob       : {ai.get('raw_probability')}   calibrated: {ai.get('calibrated_probability')}")
            print(f"    thresholds     : low={ai.get('threshold_low')} high={ai.get('threshold_high')}")
            print(f"  DF MODEL         : {df.get('checkpoint')}  ran={df.get('ran')}"
                  + (f"  [{df.get('skipped_reason') or df.get('error')}]" if not df.get("ran") else ""))
            if df.get("ran"):
                print(f"    raw logits     : {df.get('raw_logits')}")
                print(f"    raw prob       : {df.get('raw_probability')}   calibrated P(deepfake): {df.get('calibrated_probability')}")
                print(f"    thresholds     : low={df.get('threshold_low')} high={df.get('threshold_high')}"
                      f"  can_certify={df.get('can_certify_authenticity')}")
            print(f"  FACE             : {d.get('face_detected')}")
            print(f"  METADATA         : available={(d.get('metadata') or {}).get('available')} "
                  f"coherence={(d.get('metadata') or {}).get('consistency_score')} "
                  f"({(d.get('metadata') or {}).get('reliability')})")
            print(f"  C2PA / SYNTHID   : {(d.get('c2pa') or {}).get('status')} / {(d.get('synthid') or {}).get('status')}")
            print(f"  FINAL RESULT     : {d.get('result')}")
            print(f"  CONFIDENCE       : {d.get('confidence')}")
            print(f"  LEGACY (frontend): verdict={d.get('verdict')} final_score={d.get('final_score')}")
            tally.setdefault(expected, []).append(d.get("result"))

    print("\n" + "=" * 118)
    print("SUMMARY (counts of final result per expected class)")
    print("=" * 118)
    ok = True
    for expected, results in tally.items():
        counts = {r: results.count(r) for r in sorted(set(results))}
        print(f"  {expected:14s} n={len(results):2d}  {counts}")
        # acceptance: known fakes must not be systematically called REAL
        if expected in ("DEEPFAKE", "AI-GENERATED") and counts.get("REAL", 0) > len(results) / 2:
            ok = False
            print(f"     ^^ FAIL: majority of {expected} classified REAL")
    print("\nACCEPTANCE (no fake class majority-REAL):", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
