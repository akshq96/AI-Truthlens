"""
Multi-factor fusion: measured signals -> REAL score, FAKE score, uncertainty -> verdict.

Two evidence tracks are scored separately because they are different problems:

  deepfake track (only when a face is found): face-manipulation models
      (DeepScan face head, GenD, Xception) + face / eyes / nose-mouth / skin /
      lighting / frequency measurements
  AI-generated track (always): Community Forensics ViT + background / texture /
      lighting / frequency measurements (+ face measurements when present)

For each track, calibration_fusion.json (written by calibrate_fusion.py from
labelled images) holds, per measurement, a monotone mapping to P(fake) and a
weight derived from how well that measurement separated real from fake
(weight = max(0, AUC - 0.55)^2). Measurements are averaged inside their group,
groups are averaged with their own validated weights, and any unavailable
measurement or group is skipped with the remaining weights renormalised.

FAKE_SCORE: a non-negative logistic regression (fitted per context, face / no face)
over the logit of every group score of both tracks. An unavailable group contributes
0 in logit space, i.e. it is neutral, so the verdict rests on the remaining evidence.
Signals that only repeat what the ML detectors already say get little weight, instead
of diluting a strong detector toward 0.5 as a plain average would. The verdict boundaries
(REAL <= low, FAKE >= high, UNCERTAIN in between) are chosen on held-out
validation images per context (face / no face).

Metadata, C2PA and SynthID are supporting evidence only: they move the fake
score by at most a few points and never decide the verdict on their own.
"""
from __future__ import annotations

import json
import math
import os

import forensic_signals

HERE = os.path.dirname(os.path.abspath(__file__))
CAL_PATH = os.path.join(HERE, "calibration_fusion.json")
ML_MEMBERS = {"deepfake": ["face_head", "gend", "xception"], "ai": ["cf_vit", "modern_probe"]}
DISPLAY_GROUPS = ["ml", "face", "eyes", "nose_mouth", "skin_texture", "background", "lighting", "frequency"]
GROUP_LABELS = {"ml": "ML detectors", "face": "face geometry/blending", "eyes": "eyes", "nose_mouth": "nose/mouth",
                "skin_texture": "skin texture", "background": "background", "lighting": "lighting/colour",
                "frequency": "frequency/compression"}

_cache = {"mtime": None, "cal": None}


def load():
    try:
        mtime = os.path.getmtime(CAL_PATH)
    except OSError:
        return None
    if _cache["mtime"] != mtime:
        with open(CAL_PATH, "r", encoding="utf-8") as fh:
            _cache["cal"] = json.load(fh)
        _cache["mtime"] = mtime
    return _cache["cal"]


def group_members(track: str) -> dict:
    groups = {g: list(m) for g, m in forensic_signals.GROUPS.items()}
    groups["ml"] = list(ML_MEMBERS[track])
    return groups


def _sigmoid(t):
    t = max(-40.0, min(40.0, t))
    return 1.0 / (1.0 + math.exp(-t))


def feature_prob(spec: dict, x: float) -> float:
    if spec["transform"] == "abs_dev":
        z = abs(x - spec["center"]) / spec["scale"]
    else:
        z = (x - spec["center"]) / spec["scale"]
    return _sigmoid(spec["a"] * z + spec["b"])


def track_score(track_cal: dict, values: dict):
    """Returns (score or None, {group: score}, {group: weight used}, {feature: prob})."""
    group_scores, feature_probs = {}, {}
    for g, gspec in track_cal["groups"].items():
        num = den = 0.0
        for m in gspec["members"]:
            fs = track_cal["features"].get(m)
            x = values.get(m)
            if fs is None or x is None or fs["weight"] <= 0:
                continue
            p = feature_prob(fs, float(x))
            feature_probs[m] = round(p, 4)
            num += fs["weight"] * p
            den += fs["weight"]
        if den > 0:
            group_scores[g] = num / den
    num = den = 0.0
    used = {}
    for g, s in group_scores.items():
        w = track_cal["groups"][g]["weight"]
        if w > 0:
            num += w * s
            den += w
            used[g] = w
    return (num / den if den > 0 else None), group_scores, used, feature_probs


def score_tracks(cal: dict, values: dict, face_found: bool) -> dict:
    ai = track_score(cal["tracks"]["ai"], values)
    df = track_score(cal["tracks"]["deepfake"], values) if face_found else (None, {}, {}, {})
    return {"ai": ai, "deepfake": df}


def pick_kind(tracks: dict, face_found: bool) -> str:
    """DEEPFAKE vs AI-GENERATED: compare the tracks; if the whole-image AI model is
    itself the stronger validated detector, prefer AI-GENERATED."""
    ai_s, ai_g = tracks["ai"][0], tracks["ai"][1]
    df_s, df_g = tracks["deepfake"][0], tracks["deepfake"][1]
    if not face_found or df_s is None:
        return "AI-GENERATED"
    if ai_s is None:
        return "DEEPFAKE"
    ai_ml, df_ml = ai_g.get("ml"), df_g.get("ml")
    if ai_ml is not None and ai_ml >= 0.5 and (df_ml is None or ai_ml >= df_ml):
        return "AI-GENERATED"
    return "DEEPFAKE" if df_s >= ai_s else "AI-GENERATED"


def _logit(p):
    p = min(1 - 1e-4, max(1e-4, float(p)))
    return math.log(p / (1 - p))


def stack_inputs(tracks: dict, values: dict | None = None) -> dict:
    """Logit of every measured group score, plus each ML detector's own score as a separate
    input ("raw:<detector>"), so a strong detector is not averaged down by a weaker one."""
    x = {f"{t}:{g}": _logit(s) for t in ("deepfake", "ai") for g, s in tracks[t][1].items() if g != "ml"}
    for m in {m for ms in ML_MEMBERS.values() for m in ms}:
        v = (values or {}).get(m)
        if v is not None:
            x[f"raw:{m}"] = _logit(v)
    return x


def fake_score(tracks: dict, cal: dict | None = None, face_found: bool = False, values: dict | None = None):
    stack = (cal or {}).get("stack", {}).get("face" if face_found else "noface")
    if stack:
        x = stack_inputs(tracks, values)
        if not any(k in x for k in stack["coef"]):
            return None
        return _sigmoid(stack["intercept"] + sum(c * x.get(k, 0.0) for k, c in stack["coef"].items()))
    scores = [s for s in (tracks["ai"][0], tracks["deepfake"][0]) if s is not None]
    return max(scores) if scores else None


def metadata_signal(metadata: dict | None, c2pa: dict | None, synthid: dict | None):
    """Rule-based supporting evidence: (value in [0,1] or None, fake-score adjustment, note)."""
    metadata, c2pa, synthid = metadata or {}, c2pa or {}, synthid or {}
    if synthid.get("status") == "detected":
        return 0.95, 0.10, "SynthID watermark detected (supporting evidence of AI origin)."
    if c2pa.get("status") == "verified" and c2pa.get("ai_generated") is True:
        return 0.95, 0.08, "C2PA-verified manifest declares AI generation."
    if metadata.get("ai_declaration"):
        return 0.8, 0.05, f"Metadata names an AI tool ('{metadata['ai_declaration']}')."
    if c2pa.get("status") == "verified" and c2pa.get("ai_generated") is False:
        return 0.2, -0.05, "C2PA-verified capture provenance."
    if metadata.get("reliability") == "strong":
        return 0.35, -0.03, f"Coherent camera metadata ({metadata.get('camera')})."
    if metadata.get("available"):
        return 0.5, 0.0, "Metadata present but not conclusive."
    return None, 0.0, "No metadata (normal for screenshots and social media; not counted)."


def fuse(values: dict, face_found: bool, metadata=None, c2pa=None, synthid=None) -> dict:
    cal = load()
    if cal is None:
        return {"calibrated": False}
    tracks = score_tracks(cal, values, face_found)
    fake = fake_score(tracks, cal, face_found, values)
    ctx = "face" if face_found else "noface"
    stack = cal.get("stack", {}).get(ctx)
    md_value, md_adj, md_note = metadata_signal(metadata, c2pa, synthid)
    unavailable = dict(forensic_signals.NOT_MEASURED)
    if fake is None:
        return {"calibrated": True, "result": "UNCERTAIN", "confidence": 0.0, "real_score": None, "fake_score": None,
                "uncertainty": 1.0, "signals": {g: None for g in DISPLAY_GROUPS} | {"metadata": md_value},
                "decision_reason": "No detector or measurement produced usable evidence.",
                "unavailable_signals": unavailable}
    kind = pick_kind(tracks, face_found)
    win = tracks["deepfake"] if kind == "DEEPFAKE" else tracks["ai"]
    _, win_groups, win_weights, win_features = win
    if stack:
        wt = "deepfake" if kind == "DEEPFAKE" else "ai"
        win_weights = {g: stack["coef"].get(f"{wt}:{g}", 0.0) for g in win_groups if g != "ml"}
        win_weights["ml"] = sum(stack["coef"].get(f"raw:{m}", 0.0) for m in ML_MEMBERS[wt] if values.get(m) is not None)
        win_weights = {g: w for g, w in win_weights.items() if w > 0 and g in win_groups}
    fake_adj = min(1.0, max(0.0, fake + md_adj))
    real = 1.0 - fake_adj
    th = cal["thresholds"][ctx]
    low, high = float(th["low"]), float(th["high"])
    if fake_adj >= high:
        result = kind
    elif fake_adj <= low:
        result = "REAL"
    else:
        result = "UNCERTAIN"
    uncertainty = 1.0 - abs(2.0 * fake_adj - 1.0)
    confidence = fake_adj if result in ("AI-GENERATED", "DEEPFAKE") else real if result == "REAL" else max(real, fake_adj)

    total_w = sum(win_weights.values()) or 1.0
    signals = {g: (round(win_groups[g], 4) if g in win_groups else None) for g in DISPLAY_GROUPS}
    signals["metadata"] = md_value
    weights = {g: round(w / total_w, 4) for g, w in win_weights.items()}
    for g in DISPLAY_GROUPS:
        if g not in win_groups:
            unavailable.setdefault(g, "not measurable for this image" if g != "ml" else "detector unavailable")

    ranked = sorted(win_weights or win_groups, key=lambda g: win_weights.get(g, 1.0) * abs(win_groups[g] - 0.5),
                    reverse=True)[:3]
    top = ", ".join(f"{GROUP_LABELS[g]} {win_groups[g]:.2f}" for g in ranked)
    if result == "REAL":
        reason = f"Fake score {fake_adj:.2f} is at or below the REAL boundary {low:.2f}. Main signals: {top}."
    elif result == "DEEPFAKE":
        reason = (f"Face detected and fake score {fake_adj:.2f} is at or above {high:.2f}; face-manipulation "
                  f"evidence is stronger than whole-image AI evidence. Strongest signals: {top}.")
    elif result == "AI-GENERATED":
        reason = (f"Fake score {fake_adj:.2f} is at or above {high:.2f}; whole-image AI-generation evidence "
                  f"is the strongest. Strongest signals: {top}.")
    else:
        reason = (f"Fake score {fake_adj:.2f} lies between {low:.2f} and {high:.2f}: the signals conflict or are "
                  f"too weak for a definite answer. Main signals: {top}.")
    reason += " " + md_note

    return {
        "calibrated": True,
        "result": result,
        "confidence": round(confidence, 4),
        "real_score": round(real, 4),
        "fake_score": round(fake_adj, 4),
        "uncertainty": round(uncertainty, 4),
        "signals": signals,
        "signal_weights": weights,
        "track_scores": {"ai_generated": None if tracks["ai"][0] is None else round(tracks["ai"][0], 4),
                         "deepfake": None if tracks["deepfake"][0] is None else round(tracks["deepfake"][0], 4)},
        "feature_scores": win_features,
        "thresholds": {"context": ctx, "real_at_or_below": low, "fake_at_or_above": high},
        "metadata_adjustment": md_adj,
        "unavailable_signals": unavailable,
        "decision_reason": reason,
    }
