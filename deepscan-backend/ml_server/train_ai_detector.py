"""
Train DeepScan's AI-generated-image head for recent generators (GPT-image, Nano-Banana,
Imagen, Midjourney, Flux, ...) on top of the frozen Community Forensics ViT.

The base ViT misses many images from current commercial generators (on the held-out
user screenshots it flagged 3 of 11 at its threshold). A logistic head on the ViT's
CLS embedding is trained here with data that matches those images:

  fake : OpenFake (ComplexDataLab/OpenFake, CC BY-NC 4.0) images from recent generators,
         realistic AI photos (realistic_data), GAN faces (calibration_data2)
  real : OpenFake real photos (LAION, Pexels, ImageNet, DOCCI), realistic camera photos,
         real faces, FF++ real frames

Controls
  * identical random degradations for both classes (resize, JPEG, screenshot borders),
    so image quality or UI chrome carries no label information
  * generators gpt-image-2, nano-banana-pro and sora are held out entirely (test only),
    plus 20% of real photos -> honest out-of-distribution numbers
  * nothing in test_sets/acceptance is used for training or thresholds
  * threshold: lowest score with <= 3% false positives on validation reals (clean + degraded)
  * the head is switched on in the app only if held-out real photos stay <= 5% false positives

Outputs: trained_models/modern_probe/probe.pt, calibration_modern.json,
         test_results/ai_detector_training.json, feature_cache/ai_*.npz
"""
from __future__ import annotations

import glob
import io
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# Images reserved for calibrate_fusion.py are never used to train the probe, so the
# fusion weights are fitted on probe scores for unseen images.
CALIB_PATHS: set = set()
SEED = 11
EXT = (".png", ".jpg", ".jpeg", ".webp")
MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
HOLDOUT_GENERATORS = ("gpt-image-2", "nano-banana-pro", "sora")
MAX_FPR_VAL = 0.03
MAX_FPR_HELDOUT_REAL = 0.05
CACHE = "feature_cache"


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


def files(pattern, excl, limit=None, seed=SEED):
    f = sorted(p for p in glob.glob(pattern, recursive=True) if p.lower().endswith(EXT) and os.path.basename(p) not in excl and p not in CALIB_PATHS)
    random.Random(seed).shuffle(f)
    return f[:limit] if limit else f


def vit_input(img):
    img = img.convert("RGB")
    w, h = img.size
    s = 440 / min(w, h)
    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    l, t = round((img.width - 384) / 2), round((img.height - 384) / 2)
    arr = (np.asarray(img.crop((l, t, l + 384, t + 384)), np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def degrade(img, rng):
    img = img.convert("RGB")
    if rng.random() < 0.5:
        s = rng.uniform(0.45, 0.95)
        img = img.resize((max(96, int(img.width * s)), max(96, int(img.height * s))), Image.BICUBIC)
    if rng.random() < 0.4:  # screenshot: dark sidebar and/or top/bottom bars
        side = int(img.width * rng.uniform(0.0, 0.08))
        bar = int(img.height * rng.uniform(0.0, 0.05))
        shade = tuple(int(rng.uniform(10, 50)) for _ in range(3))
        c = Image.new("RGB", (img.width + side, img.height + 2 * bar), shade)
        c.paste(img, (side if rng.random() < 0.5 else 0, bar))
        img = c
    if rng.random() < 0.6:
        buf = io.BytesIO()
        if rng.random() < 0.7:
            img.save(buf, "JPEG", quality=int(rng.uniform(55, 95)))
        else:
            img.save(buf, "PNG")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


@torch.no_grad()
def features(model, paths, aug_seed=None, tag="", batch=16):
    rng = random.Random(aug_seed) if aug_seed is not None else None
    out, kept, buf, bp = [], [], [], []
    t0 = time.time()
    for i, p in enumerate(paths):
        try:
            img = Image.open(p)
            if rng is not None:
                img = degrade(img, rng)
            buf.append(vit_input(img)); bp.append(p)
        except Exception as exc:
            log("  skip", os.path.basename(p), exc)
        if len(buf) == batch or (i == len(paths) - 1 and buf):
            out.append(model.vit(pixel_values=torch.stack(buf).to(DEVICE)).last_hidden_state[:, 0].float().cpu())
            kept += bp
            buf, bp = [], []
        if i == 63:
            per = (time.time() - t0) / 64
            log(f"  [{tag}] {per:.3f}s/img -> ~{per * len(paths) / 60:.1f} min for {len(paths)}")
    X = torch.cat(out).numpy() if out else np.zeros((0, 384), np.float32)
    log(f"  [{tag}] {len(kept)} features in {(time.time()-t0)/60:.1f} min")
    return X, kept


def cached(name, fn):
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/ai_{name}.npz"
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return d["X"], list(d["paths"])
    X, p = fn()
    np.savez(path, X=X, paths=np.array(p, dtype=object))
    return X, p


def train_logreg(X, y, wd=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = torch.from_numpy((X - mu) / sd).float(); yt = torch.from_numpy(y).float()
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    pw = torch.tensor((len(y) - y.sum()) / max(1.0, y.sum()))
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xn @ w + b, yt, pos_weight=pw) + wd * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return {"w": w.detach(), "b": b.detach(), "mu": torch.from_numpy(mu), "sd": torch.from_numpy(sd)}


def predict(probe, X):
    return torch.sigmoid(((torch.from_numpy(X) - probe["mu"]) / probe["sd"]) @ probe["w"] + probe["b"]).numpy()


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def split(lst, frac=0.8):
    k = int(len(lst) * frac)
    return lst[:k], lst[k:]


def main():
    t_start = time.time()
    excl = acceptance_names()
    import calibrate_fusion
    CALIB_PATHS.update(calibrate_fusion.job_paths())
    log(f"excluding {len(CALIB_PATHS)} fusion-calibration images from probe training")
    of_fake_all = files("openfake_data/fake/**/*", excl)
    of_fake_hold = [p for p in of_fake_all if any(g in p for g in HOLDOUT_GENERATORS)]
    of_fake = [p for p in of_fake_all if p not in set(of_fake_hold)]
    of_real = files("openfake_data/real/**/*", excl)
    of_real_tr, of_real_hold = split(of_real)
    local_ai = files("realistic_data/Ai_generated_dataset/*/*", excl) + files("calibration_data2/fake/*", excl, 300)
    local_real = files("realistic_data/real_dataset/*/*", excl) + files("calibration_data2/real/*", excl, 300) + files("ff_splits/train/real/*", excl, 300)
    local_real_tr, local_real_hold = split(local_real)
    fake_tr, fake_val = split(of_fake + local_ai)
    real_tr, real_val = split(of_real_tr + local_real_tr)
    log(f"fake train {len(fake_tr)} val {len(fake_val)} | real train {len(real_tr)} val {len(real_val)} | "
        f"held-out generators {len(of_fake_hold)} | held-out reals {len(of_real_hold) + len(local_real_hold)}")
    if len(of_fake) < 200:
        raise SystemExit("not enough OpenFake images downloaded yet")

    model = ViTForImageClassification.from_pretrained(MODEL_ID).eval().to(DEVICE)
    F = {}
    for name, paths, aug in (("fake_tr", fake_tr, SEED + 1), ("real_tr", real_tr, SEED + 2),
                             ("fake_val", fake_val, None), ("real_val", real_val, None),
                             ("real_val_aug", real_val, SEED + 3),
                             ("fake_hold", of_fake_hold, None), ("real_hold", of_real_hold + local_real_hold, None),
                             ("real_hold_aug", of_real_hold + local_real_hold, SEED + 4),
                             ("acc_real", files("test_sets/acceptance/real/*", set()), None),
                             ("acc_ai", files("test_sets/acceptance/ai_generated/*", set()), None),
                             ("acc_user", files("test_sets/acceptance/user_ai_screenshots/*", set()), None)):
        F[name] = cached(name, lambda paths=paths, aug=aug, name=name: features(model, paths, aug, name))[0]

    X = np.concatenate([F["fake_tr"], F["real_tr"]])
    y = np.concatenate([np.ones(len(F["fake_tr"])), np.zeros(len(F["real_tr"]))]).astype(np.float32)
    probe = train_logreg(X, y)

    val_real = np.concatenate([predict(probe, F["real_val"]), predict(probe, F["real_val_aug"])])
    val_fake = predict(probe, F["fake_val"])
    cands = np.unique(np.concatenate([val_real, [1.0]]))
    thr = float(max(0.5, next(t for t in cands if np.mean(val_real >= t) <= MAX_FPR_VAL)))
    tp, fp = int((val_fake >= thr).sum()), int((val_real >= thr).sum())

    hold_real = np.concatenate([predict(probe, F["real_hold"]), predict(probe, F["real_hold_aug"])])
    rep = {
        "threshold": round(thr, 4),
        "val_auc": round(auc(val_fake, val_real), 4),
        "val_recall": round(float(np.mean(val_fake >= thr)), 4),
        "val_fpr": round(float(np.mean(val_real >= thr)), 4),
        "val_precision": round(tp / max(1, tp + fp), 4),
        "heldout_generators_recall": round(float(np.mean(predict(probe, F["fake_hold"]) >= thr)), 4) if len(F["fake_hold"]) else None,
        "heldout_generators_n": int(len(F["fake_hold"])),
        "heldout_real_fpr": round(float(np.mean(hold_real >= thr)), 4),
        "heldout_real_n": int(len(F["real_hold"])),
        "acceptance_report_only": {
            "real_fpr": round(float(np.mean(predict(probe, F["acc_real"]) >= thr)), 4),
            "ai_generated_recall": round(float(np.mean(predict(probe, F["acc_ai"]) >= thr)), 4),
            "user_screenshots_recall": round(float(np.mean(predict(probe, F["acc_user"]) >= thr)), 4),
            "user_screenshots_scores": [round(float(v), 4) for v in predict(probe, F["acc_user"])],
        },
        "counts": {k: int(len(v)) for k, v in F.items()},
    }
    validated = rep["heldout_real_fpr"] <= MAX_FPR_HELDOUT_REAL
    rep["validated"] = bool(validated)
    log(json.dumps(rep, indent=2))

    os.makedirs("trained_models/modern_probe", exist_ok=True)
    os.makedirs("test_results", exist_ok=True)
    torch.save({**probe, "model_id": MODEL_ID, "feature": "vit.last_hidden_state[:,0]"}, "trained_models/modern_probe/probe.pt")
    json.dump({"source": "train_ai_detector.py (OpenFake + local data, frozen Community Forensics ViT)",
               "validated": bool(validated),
               "thresholds": {"modern_high": round(thr, 4)},
               "metrics": {"val_precision_new_generators": rep["val_precision"], **rep}},
              open("calibration_modern.json", "w"), indent=2)
    json.dump(rep, open("test_results/ai_detector_training.json", "w"), indent=2)
    log(f"saved (validated={validated}) in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
