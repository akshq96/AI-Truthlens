"""
Evidence fusion for image authenticity.

Keeps three things strictly separate, which the previous single-model pipeline
conflated:

    raw model output  ->  calibrated probability  ->  final decision

Two independent probabilities go in:

  p_ai        probability the whole image is synthetic (AI-generated), from a
              general synthetic-image detector.
  p_deepfake  probability the FACE has been manipulated (swap/reenactment),
              from a face-forensics detector. Only meaningful when a face is
              actually present — a face-manipulation score on a landscape is
              noise, not evidence.

Thresholds are loaded from calibration.json (produced by calibrate_decision.py
from balanced validation data), never hardcoded guesses.
"""

import json
import os
from typing import Optional

RESULT_REAL = "REAL"
RESULT_AI = "AI-GENERATED"
RESULT_DEEPFAKE = "DEEPFAKE"
RESULT_UNCERTAIN = "UNCERTAIN"

CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), "calibration.json")

# Used only if calibration.json is missing. Deliberately conservative: wide
# uncertainty band rather than confident guesses.
_FALLBACK = {
    "ai_high": 0.9,
    "ai_low": 0.1,
    "deepfake_high": 0.9,
    "deepfake_low": None,
    "source": "fallback-defaults (calibration.json missing)",
}


def load_calibration(path: str = CALIBRATION_PATH) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return dict(_FALLBACK)

    thr = data.get("thresholds", {})

    def _opt(key):
        """A null *_low means the detector cannot certify authenticity at the
        required purity, so it must not be able to block a REAL verdict."""
        v = thr.get(key, _FALLBACK[key])
        return None if v is None else float(v)

    out = {
        "ai_high": float(thr.get("ai_high", _FALLBACK["ai_high"])),
        "ai_low": _opt("ai_low"),
        "deepfake_high": float(thr.get("deepfake_high", _FALLBACK["deepfake_high"])),
        "deepfake_low": _opt("deepfake_low"),
        # Measured precision at each alarm threshold. Used as the confidence
        # floor for verdicts triggered there: at a permissive operating point a
        # score barely over the line badly understates how often such a
        # detection is actually correct, so the bare probability is the wrong
        # thing to report as confidence.
        "ai_precision_at_high": thr.get("ai_precision_at_high"),
        "deepfake_precision_at_high": thr.get("deepfake_precision_at_high"),
        "source": data.get("source", os.path.basename(path)),
    }
    out["metrics"] = data.get("metrics", {})
    return out


def _provenance_adjust(decision: dict, metadata: Optional[dict], c2pa: Optional[dict]) -> dict:
    """
    Apply metadata/C2PA as SUPPORTING evidence.

    Allowed:
      * a verified C2PA claim of AI generation, or an explicit AI declaration in
        metadata, can raise/confirm a synthetic verdict (positive provenance);
      * coherent camera metadata can firm up an already-ML-supported REAL;
      * a flat contradiction between strong metadata and strong forensics is
        surfaced, not silently resolved.

    Forbidden, and enforced here:
      * missing metadata may never push toward synthetic;
      * present camera metadata may never overturn strong forensic evidence of
        manipulation, nor by itself create a REAL verdict.
    """
    metadata = metadata or {}
    c2pa = c2pa or {}
    notes = []
    result = decision["result"]
    conf = decision["confidence"]

    ai_decl = metadata.get("ai_declaration")
    c2pa_status = c2pa.get("status")

    # --- positive provenance of AI origin -------------------------------
    if c2pa_status == "verified" and c2pa.get("ai_generated") is True:
        notes.append("C2PA-verified manifest declares AI generation — strong provenance evidence.")
        if result != RESULT_DEEPFAKE:
            result = RESULT_AI
        conf = max(conf, 0.95)
    elif ai_decl:
        notes.append(f"Metadata declares AI tooling ('{ai_decl}'): supporting evidence of synthetic origin.")
        if result == RESULT_UNCERTAIN:
            # Weak forensics + an explicit generator tag is enough to call it.
            result = RESULT_AI
            conf = max(conf, 0.7)
        elif result in (RESULT_AI, RESULT_DEEPFAKE):
            conf = min(0.99, conf + 0.05)
        elif result == RESULT_REAL:
            notes.append("Forensic evidence indicates authenticity while metadata names an AI tool — "
                         "possible re-export of a real photo through AI software; treating as inconclusive.")
            result = RESULT_UNCERTAIN
            conf = min(conf, 0.5)

    # --- corroboration of authenticity ----------------------------------
    elif result == RESULT_REAL and metadata.get("reliability") == "strong":
        conf = min(0.99, conf + 0.05)
        notes.append("Coherent camera metadata (%s) corroborates authenticity." % metadata.get("camera"))
    elif result == RESULT_UNCERTAIN and decision.get("uncertain_cause") == "face_risk_uncovered":
        # The forensic side found low synthetic evidence but could not clear the
        # face. A verified C2PA capture claim, or a coherent camera record, is a
        # genuine second independent signal — enough to reach REAL together with
        # the low synthetic reading. Metadata alone never gets here.
        if c2pa.get("status") == "verified" and c2pa.get("ai_generated") is False:
            result = RESULT_REAL
            conf = max(conf, 0.9)
            notes.append("C2PA-verified capture provenance independently corroborates authenticity.")
        elif metadata.get("reliability") == "strong":
            result = RESULT_REAL
            conf = max(conf, 0.75)
            notes.append(f"Coherent camera metadata ({metadata.get('camera')}) provides independent "
                         "corroboration of authenticity alongside the low synthetic score.")
        else:
            notes.append("No independent provenance available to corroborate authenticity, so the "
                         "result stays UNCERTAIN rather than asserting REAL.")
    elif result == RESULT_REAL and not metadata.get("available"):
        notes.append("No metadata to corroborate, which is normal for screenshots and "
                     "social-media images and is not counted against authenticity.")

    # --- disagreement surfacing -----------------------------------------
    if result in (RESULT_AI, RESULT_DEEPFAKE) and metadata.get("reliability") == "strong" and not ai_decl:
        notes.append("Note: coherent camera metadata is present but forensic evidence indicates "
                     "manipulation. Metadata is editable and does not override the forensic signal.")

    if c2pa_status == "present_invalid":
        notes.append("A C2PA manifest is present but its signature failed validation — the provenance "
                     "claim cannot be trusted.")
    elif c2pa_status == "present_unverified":
        notes.append("A C2PA manifest is present but was not cryptographically verified here.")

    out = dict(decision)
    out["result"] = result
    out["confidence"] = round(float(max(0.0, min(1.0, conf))), 4)
    if notes:
        out["decision_reason"] = out["decision_reason"] + " " + " ".join(notes)
    out["provenance_notes"] = notes
    return out


def decide(
    p_ai: Optional[float],
    p_deepfake: Optional[float],
    face_detected: bool,
    synthid: dict,
    calibration: dict,
    detector_errors: Optional[list] = None,
    metadata: Optional[dict] = None,
    c2pa: Optional[dict] = None,
    p_modern: Optional[float] = None,
) -> dict:
    """Forensic decision first, then provenance as supporting evidence."""
    base = _decide_forensic(p_ai, p_deepfake, face_detected, synthid, calibration, detector_errors, p_modern)
    out = _provenance_adjust(base, metadata, c2pa)
    # #region agent log
    try:
        import json as _dj, time as _dt
        with open("/Users/akshitraj/.cursor/debug-logs/debug-b72871.log", "a", encoding="utf-8") as _f:
            _f.write(_dj.dumps({"sessionId": "b72871", "hypothesisId": "B", "location": "decision_engine.py:decide", "message": "fused decision", "data": {"p_ai": None if p_ai is None else round(float(p_ai), 4), "p_modern": None if p_modern is None else round(float(p_modern), 4), "p_deepfake": None if p_deepfake is None else round(float(p_deepfake), 4), "face_detected": bool(face_detected), "ai_high": calibration.get("ai_high"), "ai_low": calibration.get("ai_low"), "df_high": calibration.get("deepfake_high"), "df_low": calibration.get("deepfake_low"), "modern_high": calibration.get("modern_high"), "abstain": _abstain_mode(), "forensic_result": base.get("result"), "final_result": out.get("result"), "uncertain_cause": out.get("uncertain_cause")}, "timestamp": int(_dt.time() * 1000)}) + "\n")
    except Exception:
        pass
    # #endregion
    return out


def _decide_forensic(
    p_ai: Optional[float],
    p_deepfake: Optional[float],
    face_detected: bool,
    synthid: dict,
    calibration: dict,
    detector_errors: Optional[list] = None,
    p_modern: Optional[float] = None,
) -> dict:
    """
    Fuse the available evidence. Any of p_ai / p_deepfake may be None when that
    detector is unavailable or not applicable.

    Rules, in order:
      1. A positive SynthID hit is decisive provenance evidence of AI origin.
      2. No usable detector at all -> UNCERTAIN (never REAL).
      3. Strong synthetic evidence -> AI-GENERATED.
      4. Strong face-manipulation evidence, with a face present -> DEEPFAKE.
      5. Everything that is applicable looks clean -> REAL.
      6. Otherwise (middling or conflicting) -> UNCERTAIN.
    """
    errors = list(detector_errors or [])
    ai_high = calibration["ai_high"]
    ai_low = calibration["ai_low"]
    df_high = calibration["deepfake_high"]
    df_low = calibration["deepfake_low"]

    # Face-manipulation evidence only counts when there is a face to manipulate.
    df_applicable = face_detected and p_deepfake is not None

    # 1. Provenance beats statistics: a watermark is a positive claim of origin.
    if synthid.get("status") == "detected":
        return _out(RESULT_AI, 0.99, p_ai, p_deepfake, face_detected,
                    "SynthID watermark detected — direct provenance evidence of AI origin.", errors)

    # 2. Nothing usable to reason from.
    if p_ai is None and not df_applicable:
        return _out(RESULT_UNCERTAIN, 0.0, p_ai, p_deepfake, face_detected,
                    "No detector produced a usable score; refusing to guess.", errors)

    # 3. Strong whole-image synthetic evidence.
    if p_ai is not None and p_ai >= ai_high:
        conf = _triggered_confidence(p_ai, calibration.get("ai_precision_at_high"))
        return _out(RESULT_AI, conf, p_ai, p_deepfake, face_detected,
                    f"Synthetic-image probability {p_ai:.3f} >= calibrated threshold {ai_high:.3f} "
                    f"(measured precision at this operating point: "
                    f"{_fmt(calibration.get('ai_precision_at_high'))}).", errors)

    # 3b. Newer generators (Gemini, GPT-4o) that the base detector under-scores:
    #     a probe on the same ViT embedding, thresholded on held-out real photos.
    modern_high = calibration.get("modern_high")
    if p_modern is not None and modern_high is not None and p_modern >= modern_high:
        conf = _triggered_confidence(p_modern, calibration.get("modern_precision_at_high"))
        return _out(RESULT_AI, conf, max(p_ai or 0.0, p_modern), p_deepfake, face_detected,
                    f"New-generator probe probability {p_modern:.3f} >= calibrated threshold {modern_high:.3f} "
                    f"(measured precision at this operating point: "
                    f"{_fmt(calibration.get('modern_precision_at_high'))}).", errors)

    # 4. Strong face-manipulation evidence, face present.
    if df_applicable and p_deepfake >= df_high:
        conf = _triggered_confidence(p_deepfake, calibration.get("deepfake_precision_at_high"))
        return _out(RESULT_DEEPFAKE, conf, p_ai, p_deepfake, face_detected,
                    f"Face detected and face-manipulation probability {p_deepfake:.3f} "
                    f">= calibrated threshold {df_high:.3f} (measured precision at this "
                    f"operating point: {_fmt(calibration.get('deepfake_precision_at_high'))}).", errors)

    # 5. Binary mode (default): every image gets REAL or a fake verdict. No detector
    #    crossed its calibrated alarm threshold, so the answer is REAL. UNCERTAIN is
    #    kept only for failures (rule 2 above, or a detector error below).
    #    Set DEEPSCAN_ABSTAIN=1 to restore the conservative three-way behaviour.
    if not _abstain_mode():
        if errors and not df_applicable and face_detected:
            return _out(RESULT_UNCERTAIN, 0.0, p_ai, p_deepfake, face_detected,
                        "A face was found but the face-manipulation detector failed, so no verdict "
                        "can be given: " + "; ".join(errors), errors)
        contrary = [v for v in (p_ai, p_deepfake if df_applicable else None, p_modern) if v is not None]
        worst = max(contrary) if contrary else 0.0
        parts = []
        if p_ai is not None:
            parts.append(f"AI-image probability {p_ai:.3f} < threshold {ai_high:.3f}")
        if df_applicable:
            parts.append(f"face-manipulation probability {p_deepfake:.3f} < threshold {df_high:.3f}")
        elif not face_detected:
            parts.append("no face found, so face manipulation does not apply")
        return _out(RESULT_REAL, max(0.5, 1.0 - worst), p_ai, p_deepfake, face_detected,
                    "No detector found manipulation: " + "; ".join(parts) + ".", errors)

    #    Conservative three-way mode (DEEPSCAN_ABSTAIN=1): REAL requires POSITIVE
    #    evidence of authenticity, not merely the absence of an alarm.
    #    If a face is present, whole-image synthetic detection does NOT cover the
    #    face-swap threat model, so an AI-detector-clean reading alone cannot
    #    certify REAL by itself.
    ai_certifies = p_ai is not None and ai_low is not None and p_ai <= ai_low
    df_certifies = df_applicable and df_low is not None and p_deepfake <= df_low

    # Is there a face-manipulation risk that no loaded detector can rule out?
    face_risk_uncovered = face_detected and not df_certifies

    if ai_certifies and not face_risk_uncovered and not errors:
        contrary = max([v for v in (p_ai, p_deepfake if df_applicable else None) if v is not None])
        reason = f"Synthetic probability {p_ai:.3f} <= authenticity threshold {ai_low:.3f}"
        reason += (f" and face-manipulation probability {p_deepfake:.3f} <= {df_low:.3f}."
                   if df_certifies else
                   "; no face present, so the face-manipulation threat model does not apply.")
        return _out(RESULT_REAL, 1.0 - contrary, p_ai, p_deepfake, face_detected, reason, errors)

    # 5b. Face present, synthetic evidence low, but nothing can clear the face.
    #     Corroborating provenance can still carry it to REAL; otherwise this is
    #     genuinely undetermined and must be reported as such.
    if ai_certifies and face_risk_uncovered and not errors:
        why = ("the face-manipulation detector is not accurate enough to certify authenticity"
               if p_deepfake is not None else
               "no face-manipulation detector is available")
        out = _out(RESULT_UNCERTAIN, min(0.5, 1.0 - p_ai), p_ai, p_deepfake, face_detected,
                   f"Whole-image synthetic probability is low ({p_ai:.3f}), but a face is present and "
                   f"{why}, so authenticity cannot be established. A low face-manipulation score is "
                   f"not evidence of authenticity.", errors)
        out["uncertain_cause"] = "face_risk_uncovered"
        return out

    # 6. Middling or conflicting evidence — say so instead of forcing a side.
    bits = []
    if p_ai is not None and ai_low is not None and ai_low < p_ai < ai_high:
        bits.append(f"synthetic probability {p_ai:.3f} falls between {ai_low:.3f} and {ai_high:.3f}")
    elif p_ai is not None:
        bits.append(f"synthetic probability {p_ai:.3f} alone is not conclusive")
    if df_applicable and df_low is not None and df_low < p_deepfake < df_high:
        bits.append(f"face-manipulation probability {p_deepfake:.3f} is not decisive")
    elif df_applicable:
        bits.append(f"face-manipulation probability {p_deepfake:.3f} alone is not conclusive")
    elif p_deepfake is not None and not face_detected:
        bits.append("face-manipulation score ignored (no face detected)")
    if errors:
        bits.append("a detector failed, so authenticity cannot be confirmed")
    # Distance from the nearest decisive boundary, as a weak confidence proxy.
    margin_conf = 0.0
    if p_ai is not None and ai_low is not None:
        span = max(1e-9, ai_high - ai_low)
        margin_conf = max(margin_conf, 1.0 - min(abs(p_ai - ai_low), abs(p_ai - ai_high)) / span)
    return _out(RESULT_UNCERTAIN, min(0.5, margin_conf), p_ai, p_deepfake, face_detected,
                "Evidence inconclusive: " + "; ".join(bits) + ".", errors)


def _abstain_mode() -> bool:
    return os.environ.get("DEEPSCAN_ABSTAIN", "").strip().lower() in ("1", "true", "yes", "on")


def _fmt(v):
    return "unknown" if v is None else f"{float(v):.2f}"


def _triggered_confidence(prob, precision_at_threshold):
    """
    Confidence for a verdict produced by crossing an alarm threshold.

    Uses the empirically measured precision at that operating point as a floor,
    and never claims certainty. Without this, lowering a threshold to gain
    recall would make every fresh detection report a near-zero confidence.
    """
    if precision_at_threshold is None:
        return min(0.99, float(prob))
    return min(0.99, max(float(prob), float(precision_at_threshold)))


def _out(result, confidence, p_ai, p_df, face, reason, errors):
    # detector_disagreement: the two detectors point opposite ways, which is a
    # signal in itself (e.g. face swap = low synthetic, high face-manipulation).
    disagreement = None
    if p_ai is not None and p_df is not None:
        disagreement = round(abs(float(p_ai) - float(p_df)), 4)
    return {
        "result": result,
        "detector_disagreement": disagreement,
        "confidence": round(float(max(0.0, min(1.0, confidence))), 4),
        "ai_probability": None if p_ai is None else round(float(p_ai), 4),
        "deepfake_probability": None if p_df is None else round(float(p_df), 4),
        "face_detected": bool(face),
        "decision_reason": reason,
        "detector_errors": errors,
    }


def legacy_view(decision: dict) -> dict:
    """
    Backward-compatible fields for the existing frontend.

    `final_score` is the SYNTHETIC probability on a 0-100 scale. The frontend's
    ConfidenceMeter renders it on an Authentic(0) -> Synthetic(100) axis, so it
    must be synthetic-ness, NOT confidence-in-the-label. The old pipeline put
    inflated label-confidence here, which made a REAL verdict render as a
    nearly-full bar labelled CRITICAL_RISK.
    """
    result = decision["result"]
    p_ai = decision.get("ai_probability")
    p_df = decision.get("deepfake_probability") if decision.get("face_detected") else None
    synthetic = max([v for v in (p_ai, p_df) if v is not None], default=None)

    if result == RESULT_AI or result == RESULT_DEEPFAKE:
        verdict, prediction = "SYNTHETIC", "deepfake"
        # A SynthID hit can decide SYNTHETIC while the ML probabilities stay
        # low; the meter must not then read as "almost certainly authentic".
        synthetic = max(synthetic or 0.0, decision["confidence"])
    elif result == RESULT_REAL:
        verdict, prediction = "REAL", "real"
    else:
        verdict, prediction = "UNCERTAIN", "uncertain"
        if synthetic is None:
            synthetic = 0.5

    return {
        "prediction": prediction,
        "verdict": verdict,
        "final_score": round(float(synthetic) * 100.0, 2),
        "confidence": decision["confidence"],
    }
