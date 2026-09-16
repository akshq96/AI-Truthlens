"""
Calibrate DeepScan's multi-factor fusion from labelled images, through the live ML server
(so the exact production pipeline produces every measurement).

    python calibrate_fusion.py            # collects (resumable) + fits + writes calibration_fusion.json
    python calibrate_fusion.py --fit-only # refit from the cached rows

Steps
 1. Send labelled images (real / deepfake / AI-generated; test_sets/acceptance excluded)
    to POST /predict/image and store the raw measurements and raw ML scores it returns.
 2. Split 60 / 40 per category (fixed seed).
 3. On the 60% split, per track and per measurement: choose the direction
    (linear, or distance from the typical real value), fit a 1-D logistic mapping to
    P(fake), and give it weight max(0, AUC - 0.55)^2. Groups get weights the same way.
 4. On the 40% split, pick the verdict boundaries per context (face / no face):
    FAKE boundary = lowest score with at most 5% of real images above it,
    REAL boundary = highest score with at most 5% of fakes below it
    (if those cross, one balanced-accuracy boundary is used instead).
 5. Report validation metrics and print a full signal breakdown for one real,
    one deepfake and one AI-generated validation image.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time

import numpy as np
import requests

import forensic_signals
import fusion_engine

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
URL = "http://127.0.0.1:7070/predict/image"
ROWS = "feature_cache/fusion_rows.jsonl"
SEED = 2026
EXT = (".jpg", ".jpeg", ".png", ".webp")
MAX_FPR = 0.05
MAX_FAKE_LEAK = 0.10

CATEGORIES = [  # name, glob patterns, label, count, request GenD
    ("real_face_hq", ["calibration_data2/real/*"], "real", 80, True),
    ("real_people_photo", ["realistic_data/real_dataset/people/*"], "real", 60, True),
    ("real_ff_frame", ["ff_splits/val/real/*"], "real", 80, True),
    ("real_scene", ["realistic_data/real_dataset/nature/*", "realistic_data/real_dataset/city/*",
                    "realistic_data/real_dataset/food/*", "realistic_data/real_dataset/animals/*"], "real", 120, False),
    ("real_openfake", ["openfake_data/real/**/*"], "real", 100, False),
    ("deepfake_ffpp", ["ff_splits/val/fake/*"], "deepfake", 120, True),
    ("deepfake_faceshifter", ["ff_splits/unseen/fake/*"], "deepfake", 50, True),
    ("ai_gan_face", ["calibration_data2/fake/*"], "ai", 80, True),
    ("ai_portrait", ["realistic_data/Ai_generated_dataset/people/*"], "ai", 60, True),
    ("ai_scene", ["realistic_data/Ai_generated_dataset/nature/*", "realistic_data/Ai_generated_dataset/city/*",
                  "realistic_data/Ai_generated_dataset/food/*", "realistic_data/Ai_generated_dataset/animals/*"], "ai", 100, False),
    ("ai_openfake", ["openfake_data/fake/**/*"], "ai", 120, False),
]
EXTRA_CATEGORIES = [  # name, glob patterns, label, count
    ("ai_openfake_extra", ["openfake_data/fake/**/*"], "ai", 250),
    ("real_openfake_extra", ["openfake_data/real/**/*"], "real", 150),
    # public face-swap datasets (validation splits only; external_download.py)
    ("deepfake_dff_insightface", ["external_data/dff/val/fake/*"], "deepfake", 80),
    ("real_dff_wiki", ["external_data/dff/val/real/*"], "real", 60),
    ("deepfake_bitmind_faceswap", ["external_data/bitmind_faceswap/val/fake/*"], "deepfake", 60),
]


def log(*a):
    print(*a, flush=True)


def acceptance_names():
    names = set()
    for p in glob.glob("test_sets/acceptance/*/*"):
        b = os.path.basename(p)
        names.add(b)
        parts = b.split("_", 1)
        if len(parts) == 2:
            names.add(parts[1])
            sub = parts[1].split("_", 1)
            if len(sub) == 2:
                names.add(sub[1])
    return names


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([neg, pos])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ------------------------------------------------------------------ collection
def jobs_list():
    excl = acceptance_names()
    jobs = []
    for name, patterns, label, count, want_gend in CATEGORIES:
        files = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)
                        if p.lower().endswith(EXT) and os.path.basename(p) not in excl})
        random.Random(SEED).shuffle(files)
        for i, p in enumerate(files[:count]):
            jobs.append((name, label, p, want_gend and i % 3 == 0))
    # Extra modern-generator images (many contain people) so the face context sees
    # recent AI faces too. Never images the AI probe was trained on.
    used = {j[2] for j in jobs}
    probe_train = set()
    for n in ("fake_tr", "real_tr"):
        f = f"feature_cache/ai_{n}.npz"
        if os.path.exists(f):
            probe_train.update(np.load(f, allow_pickle=True)["paths"].tolist())
    for name, patterns, label, count in EXTRA_CATEGORIES:
        files = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)
                        if p.lower().endswith(EXT) and os.path.basename(p) not in excl
                        and p not in used and p not in probe_train})
        random.Random(SEED).shuffle(files)
        jobs += [(name, label, p, False) for p in files[:count]]
    return jobs


def job_paths():
    return [j[2] for j in jobs_list()]


def collect():
    done = {}
    if os.path.exists(ROWS):
        for line in open(ROWS):
            r = json.loads(line)
            done[r["path"]] = r
    os.makedirs(os.path.dirname(ROWS), exist_ok=True)
    jobs = jobs_list()
    todo = [j for j in jobs if j[2] not in done]
    log(f"images: {len(jobs)} total, {len(todo)} still to analyse")
    t0 = time.time()
    with open(ROWS, "a") as out:
        for k, (name, label, path, gend) in enumerate(todo):
            mt = "image/png" if path.lower().endswith(".png") else "image/jpeg"
            try:
                with open(path, "rb") as fh:
                    r = requests.post(URL, files={"file": (os.path.basename(path), fh, mt)},
                                      data={"gend": "1" if gend else "0"}, timeout=300)
                d = r.json()
                raw = d.get("fusion_raw")
                if not raw:
                    log("  no fusion_raw in response for", path, str(d)[:200])
                    continue
                row = {"path": path, "category": name, "label": label, "face_found": raw["face_found"],
                       "values": raw["values"]}
                out.write(json.dumps(row) + "\n")
                out.flush()
                done[path] = row
            except Exception as exc:
                log("  failed", path, exc)
            if (k + 1) % 50 == 0:
                el = time.time() - t0
                log(f"  {k + 1}/{len(todo)} ({el / (k + 1):.2f}s/img, ~{el / (k + 1) * (len(todo) - k - 1) / 60:.1f} min left)")
    return [done[j[2]] for j in jobs if j[2] in done]


def backfill_modern(rows):
    """modern_probe score (train_ai_detector.py head on the CF ViT CLS token) for every row,
    computed offline with the probe's own preprocessing. Left unavailable if not validated."""
    cm, pp = "calibration_modern.json", "trained_models/modern_probe/probe.pt"
    ok = os.path.exists(cm) and os.path.exists(pp) and json.load(open(cm)).get("validated")
    if not ok:
        log("modern probe not validated: modern_probe signal stays unavailable")
        for r in rows:
            r["values"]["modern_probe"] = None
        return rows
    import torch
    from PIL import Image
    from transformers import ViTForImageClassification
    import train_ai_detector as tad
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ViTForImageClassification.from_pretrained(tad.MODEL_ID).eval().to(dev)
    probe = torch.load(pp, map_location="cpu")
    t0 = time.time()
    for i in range(0, len(rows), 16):
        chunk = rows[i:i + 16]
        x = torch.stack([tad.vit_input(Image.open(r["path"])) for r in chunk]).to(dev)
        with torch.no_grad():
            cls = model.vit(pixel_values=x).last_hidden_state[:, 0].float().cpu()
        z = ((cls - probe["mu"]) / probe["sd"]) @ probe["w"] + probe["b"]
        for r, p in zip(chunk, torch.sigmoid(z).tolist()):
            r["values"]["modern_probe"] = float(p)
    log(f"modern_probe scores for {len(rows)} rows in {time.time() - t0:.0f}s")
    return rows


def backfill_face_head(rows):
    """face_head score from the current trained_models/face_head/face_head.pt for every row that had a
    face crop, with the server's exact crop (Effort dlib alignment, else Haar minNeighbors 8 /
    min 10% of the short side, x1.3 margin, resized to 224). Cached per head file version."""
    hp = "trained_models/face_head/face_head.pt"
    need = [r for r in rows if r["values"].get("face_head") is not None]
    if not os.path.exists(hp) or not need:
        return rows
    cache_path = f"feature_cache/fusion_face_head_{int(os.path.getmtime(hp))}.json"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [r for r in need if r["path"] not in cache]
    if todo:
        import cv2
        import torch
        import effort_model as em
        dev, dt = ("mps", torch.float16) if torch.backends.mps.is_available() else ("cpu", torch.float32)
        ck = torch.load(hp, map_location="cpu")
        hidden = int(ck.get("hidden", 256))
        head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(1024, hidden), torch.nn.GELU(),
                                   torch.nn.Dropout(0.2), torch.nn.Linear(hidden, 1))
        head.load_state_dict(ck["state_dict"])
        head.eval()
        mu, sd = ck["mu"].float(), ck["sd"].float()
        bb = em.load_effort(em.FACE_CKPT, device=dev, dtype=dt)
        aligner = em.FaceAligner()
        haar = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

        def crop(img):
            face = aligner.align(img)
            if face is not None:
                return face
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ms = max(30, int(0.10 * min(g.shape[:2])))
            fs = haar.detectMultiScale(g, scaleFactor=1.1, minNeighbors=8, minSize=(ms, ms))
            if len(fs) == 0:
                return None
            x, y, w, h = max(fs, key=lambda r: int(r[2]) * int(r[3]))
            cx, cy, half = x + w / 2.0, y + h / 2.0, max(w, h) * 1.3 / 2.0
            H, W = img.shape[:2]
            c = img[int(max(0, cy - half)):int(min(H, cy + half)), int(max(0, cx - half)):int(min(W, cx + half))]
            return cv2.resize(img if c.size == 0 else c, (224, 224))

        t0 = time.time()
        for i in range(0, len(todo), 16):
            chunk, tens = [], []
            for r in todo[i:i + 16]:
                img = cv2.imread(r["path"])
                face = crop(img) if img is not None else None
                if face is None:
                    cache[r["path"]] = None
                    continue
                chunk.append(r)
                tens.append(em.to_tensor_bgr(face))
            if tens:
                with torch.no_grad():
                    f = bb.features(torch.stack(tens).to(device=dev, dtype=dt)).float().cpu()
                    z = head((f - mu) / sd).squeeze(1)
                for r, p in zip(chunk, torch.sigmoid(z).tolist()):
                    cache[r["path"]] = float(p)
        json.dump(cache, open(cache_path, "w"))
        log(f"face_head scores recomputed for {len(todo)} rows in {time.time() - t0:.0f}s")
    for r in need:
        r["values"]["face_head"] = cache.get(r["path"])
    return rows


# ------------------------------------------------------------------ fitting
def fit_logistic_1d(z, y, l2=1e-2, iters=60):
    z, y = np.asarray(z, float), np.asarray(y, float)
    wpos = 0.5 / max(1, y.sum())
    wneg = 0.5 / max(1, (1 - y).sum())
    sw = np.where(y == 1, wpos, wneg) * len(y)
    a = b = 0.0
    for _ in range(iters):
        t = np.clip(a * z + b, -40, 40)
        p = 1 / (1 + np.exp(-t))
        g = sw * (p - y)
        W = sw * p * (1 - p)
        ga, gb = float((g * z).sum() + l2 * a), float(g.sum())
        haa, hab, hbb = float((W * z * z).sum() + l2), float((W * z).sum()), float(W.sum() + 1e-6)
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-6:
            break
    return a, b


def fit_track(rows, pos_label, track, face_only):
    sel = [r for r in rows if r["label"] in ("real", pos_label) and (r["face_found"] or not face_only)]
    members = fusion_engine.group_members(track)
    features = {}
    for g, names in members.items():
        for f in names:
            xs, ys = [], []
            for r in sel:
                v = r["values"].get(f)
                if v is not None and np.isfinite(v):
                    xs.append(float(v)); ys.append(1 if r["label"] == pos_label else 0)
            x, y = np.array(xs), np.array(ys)
            if (y == 1).sum() < 15 or (y == 0).sum() < 15:
                continue
            real = x[y == 0]
            mean, std = float(x.mean()), float(x.std() + 1e-9)
            med = float(np.median(real))
            mad = float(np.median(np.abs(real - med)) * 1.4826 + 1e-9)
            a_lin = auc(x[y == 1], x[y == 0])
            a_lin2 = max(a_lin, 1 - a_lin)
            zabs = np.abs(x - med) / mad
            a_abs = auc(zabs[y == 1], zabs[y == 0])
            if a_abs > a_lin2 + 0.02:
                spec = {"transform": "abs_dev", "center": med, "scale": mad}
                z = zabs
            else:
                spec = {"transform": "linear", "center": mean, "scale": std}
                z = (x - mean) / std
            a, b = fit_logistic_1d(z, y)
            spec.update({"a": float(a), "b": float(b)})
            p = np.array([fusion_engine.feature_prob(spec, float(v)) for v in x])
            a_p = auc(p[y == 1], p[y == 0])
            spec.update({"auc": round(a_p, 4), "n": int(len(x)), "weight": round(max(0.0, a_p - 0.55) ** 2, 6)})
            features[f] = spec
    track_cal = {"features": features, "groups": {g: {"members": names, "weight": 0.0} for g, names in members.items()}}
    for g, names in members.items():
        scores, ys = [], []
        for r in sel:
            num = den = 0.0
            for m in names:
                fs, v = features.get(m), r["values"].get(m)
                if fs is None or v is None or fs["weight"] <= 0:
                    continue
                num += fs["weight"] * fusion_engine.feature_prob(fs, float(v)); den += fs["weight"]
            if den > 0:
                scores.append(num / den); ys.append(1 if r["label"] == pos_label else 0)
        s, y = np.array(scores), np.array(ys)
        if len(s) and (y == 1).sum() >= 15 and (y == 0).sum() >= 15:
            a_g = auc(s[y == 1], s[y == 0])
            track_cal["groups"][g].update({"auc": round(a_g, 4), "n": int(len(s)), "weight": round(max(0.0, a_g - 0.55) ** 2, 6)})
    return track_cal


def fit_stack(cal, rows, face_ctx, lam=0.01, iters=6000, lr=0.2):
    """Non-negative L2 logistic regression over group scores (logit space), class-balanced.
    Unavailable groups enter as 0 (neutral). Weights >= 0 because every group score is
    already oriented towards fake by its own fitted mapping."""
    tracks = ("deepfake", "ai") if face_ctx else ("ai",)
    keys = [f"{t}:{g}" for t in tracks for g, gs in cal["tracks"][t]["groups"].items() if gs["weight"] > 0 and g != "ml"]
    keys += sorted({f"raw:{m}" for t in tracks for m in fusion_engine.ML_MEMBERS[t]})
    X, y = [], []
    for r in rows:
        if bool(r["face_found"]) != face_ctx:
            continue
        x = fusion_engine.stack_inputs(fusion_engine.score_tracks(cal, r["values"], r["face_found"]), r["values"])
        X.append([x.get(k, 0.0) for k in keys])
        y.append(0.0 if r["label"] == "real" else 1.0)
    X, y = np.array(X, float), np.array(y, float)
    if len(y) < 30 or y.min() == y.max():
        return None
    sw = np.where(y == 1, 0.5 / y.sum(), 0.5 / (len(y) - y.sum()))
    w, b = np.full(X.shape[1], 0.1), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w + b, -40, 40)))
        g = sw * (p - y)
        w = np.maximum(0.0, w - lr * (X.T @ g + lam * w))
        b -= lr * g.sum()
    return {"coef": {k: round(float(c), 5) for k, c in zip(keys, w)}, "intercept": round(float(b), 5), "n": int(len(y))}


def choose_thresholds(scores_real, scores_fake):
    r, f = np.sort(np.asarray(scores_real)), np.sort(np.asarray(scores_fake))
    cands = np.unique(np.concatenate([r, f, [0.0, 1.0]]))
    high = next((float(t) for t in cands if np.mean(r >= t) <= MAX_FPR), 1.0)
    low = max((float(t) for t in cands if np.mean(f <= t) <= MAX_FAKE_LEAK), default=0.0)
    rule = "error-capped band"
    if low >= high:
        best = max(cands, key=lambda t: 0.5 * np.mean(r < t) + 0.5 * np.mean(f >= t))
        high, low, rule = float(best), float(best) - 1e-6, "single balanced-accuracy boundary"
    return round(low, 6), round(high, 6), rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()
    if args.fit_only:
        rows = [json.loads(l) for l in open(ROWS)]
    else:
        rows = collect()
        if args.collect_only:
            log(f"collected {len(rows)} rows")
            return
    rows = backfill_modern(rows)
    rows = backfill_face_head(rows)
    log(f"rows: {len(rows)} | faces found: {sum(r['face_found'] for r in rows)}")

    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    train, val = [], []
    for cat, rs in by_cat.items():
        rs = sorted(rs, key=lambda r: r["path"])
        random.Random(SEED + 1).shuffle(rs)
        k = int(round(0.6 * len(rs)))
        train += rs[:k]; val += rs[k:]

    cal = {"source": "calibrate_fusion.py (labelled images through the live pipeline)",
           "tracks": {"deepfake": fit_track(train, "deepfake", "deepfake", face_only=True),
                      "ai": fit_track(train, "ai", "ai", face_only=False)},
           "thresholds": {}}
    cal["stack"] = {"face": fit_stack(cal, train, True), "noface": fit_stack(cal, train, False)}

    ctx_scores = {"face": {"real": [], "fake": []}, "noface": {"real": [], "fake": []}}
    for r in val:
        tr = fusion_engine.score_tracks(cal, r["values"], r["face_found"])
        s = fusion_engine.fake_score(tr, cal, r["face_found"], r["values"])
        if s is None:
            continue
        ctx = "face" if r["face_found"] else "noface"
        ctx_scores[ctx]["real" if r["label"] == "real" else "fake"].append(s)
    for ctx in ("face", "noface"):
        c = ctx_scores[ctx]
        if len(c["real"]) >= 15 and len(c["fake"]) >= 15:
            low, high, rule = choose_thresholds(c["real"], c["fake"])
            cal["thresholds"][ctx] = {"low": low, "high": high, "rule": rule,
                                      "val_auc": round(auc(c["fake"], c["real"]), 4),
                                      "n_real": len(c["real"]), "n_fake": len(c["fake"])}
    for ctx, other in (("face", "noface"), ("noface", "face")):
        if ctx not in cal["thresholds"] and other in cal["thresholds"]:
            cal["thresholds"][ctx] = dict(cal["thresholds"][other], rule="copied from " + other)

    # validation report with the final fuse() logic
    tmp = fusion_engine.CAL_PATH + ".tmp"
    json.dump(cal, open(tmp, "w"), indent=1)
    real_path = fusion_engine.CAL_PATH
    fusion_engine.CAL_PATH = tmp
    fusion_engine._cache["mtime"] = None
    per_cat, confusion = {}, {}
    examples = {}
    for r in val:
        out = fusion_engine.fuse(r["values"], r["face_found"])
        res = out["result"]
        per_cat.setdefault(r["category"], {}).setdefault(res, 0)
        per_cat[r["category"]][res] += 1
        key = (r["label"], res)
        confusion[key] = confusion.get(key, 0) + 1
        expect = {"real": "REAL", "deepfake": "DEEPFAKE", "ai": "AI-GENERATED"}[r["label"]]
        if r["label"] not in examples and res == expect:
            examples[r["label"]] = (r, out)
    fusion_engine.CAL_PATH = real_path

    def rate(label, pred):
        n = sum(v for (l, _), v in confusion.items() if l == label)
        return round(sum(v for (l, p), v in confusion.items() if l == label and p in pred) / max(1, n), 4)

    fake_preds = ("AI-GENERATED", "DEEPFAKE")
    report = {
        "val_n": len(val),
        "real_accuracy": rate("real", ("REAL",)),
        "real_to_fake": rate("real", fake_preds),
        "real_uncertain": rate("real", ("UNCERTAIN",)),
        "deepfake_recall_any_fake": rate("deepfake", fake_preds),
        "deepfake_exact": rate("deepfake", ("DEEPFAKE",)),
        "deepfake_to_real": rate("deepfake", ("REAL",)),
        "deepfake_uncertain": rate("deepfake", ("UNCERTAIN",)),
        "ai_recall_any_fake": rate("ai", fake_preds),
        "ai_exact": rate("ai", ("AI-GENERATED",)),
        "ai_to_real": rate("ai", ("REAL",)),
        "ai_uncertain": rate("ai", ("UNCERTAIN",)),
        "per_category": per_cat,
    }
    cal["validation"] = report
    cal["weights_summary"] = {
        t: {g: {"weight": v["weight"], "auc": v.get("auc")} for g, v in cal["tracks"][t]["groups"].items()}
        for t in cal["tracks"]}
    cal["top_features"] = {
        t: sorted(((f, s["auc"], s["weight"]) for f, s in cal["tracks"][t]["features"].items()), key=lambda x: -x[1])[:12]
        for t in cal["tracks"]}
    json.dump(cal, open(real_path, "w"), indent=1)
    os.remove(tmp)
    fusion_engine._cache["mtime"] = None

    log("\nSTACK", json.dumps(cal["stack"], indent=1))
    log("THRESHOLDS", json.dumps(cal["thresholds"], indent=1))
    log("GROUP WEIGHTS / AUC", json.dumps(cal["weights_summary"], indent=1))
    log("TOP FEATURES (name, auc, weight)", json.dumps(cal["top_features"], indent=1))
    log("VALIDATION", json.dumps(report, indent=1))

    for label, (r, out) in examples.items():
        log(f"\n=== DEBUG {label.upper()} example: {r['path']}")
        ml = {k: r["values"].get(k) for k in ("cf_vit", "face_head", "gend", "xception")}
        log("  ML scores       :", ml)
        log("  face detected   :", r["face_found"])
        for g in ("face", "eyes", "nose_mouth", "skin_texture", "background", "lighting", "frequency", "ml", "metadata"):
            log(f"  {g:16s}: {out['signals'].get(g)}")
        log("  track scores    :", out["track_scores"])
        log(f"  FAKE_SCORE {out['fake_score']}  REAL_SCORE {out['real_score']}  -> {out['result']}")
        log("  reason          :", out["decision_reason"])
    log("\nwrote", real_path)


if __name__ == "__main__":
    main()
