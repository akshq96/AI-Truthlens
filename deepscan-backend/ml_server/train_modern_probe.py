"""
Linear probe for newer image generators (Gemini 2.5 Flash Image "Nano Banana",
GPT-4o / ChatGPT) on top of the frozen Community Forensics ViT.

Why a probe and not a new model: the ViT already computes a 384-d CLS embedding
for every upload; a logistic head on that embedding costs nothing extra at
inference and trains in seconds on a CPU.

Leakage and shortcut controls:
  * Every training image, real or fake, goes through the same random
    augmentations (resize, JPEG re-encode, a dark screenshot-style side strip),
    so file format or screenshot borders carry no label information.
  * Files are split 75/25 per source; thresholds are chosen on the 25% split only.
  * The user's test screenshots (~/Desktop/testing image) are scored at the end
    and are never used for training or threshold selection.

Outputs:
  trained_models/modern_probe/probe.pt
  calibration_modern.json
"""
import glob
import io
import json
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
SEED = 7
MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
USER_TEST_DIR = os.path.expanduser("~/Desktop/testing image")
MAX_FP_REAL = 0.02
EXT = (".png", ".jpg", ".jpeg", ".webp")


def log(*a):
    print(*a, flush=True)


def listing(pattern, limit=None, seed=SEED):
    files = sorted(f for f in glob.glob(pattern) if f.lower().endswith(EXT))
    random.Random(seed).shuffle(files)
    return files[:limit] if limit else files


def split(files, frac=0.75):
    k = int(round(len(files) * frac))
    return files[:k], files[k:]


# ---------------------------------------------------------------- preprocessing (identical to image_server)
def vit_preprocess(img):
    img = img.convert("RGB")
    w, h = img.size
    scale = 440 / min(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - 384) / 2, (nh - 384) / 2
    img = img.crop((round(left), round(top), round(left) + 384, round(top) + 384))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)).copy())


def augment(img, rng):
    """Same distribution for both classes."""
    img = img.convert("RGB")
    if rng.random() < 0.5:  # screenshot-style resize
        s = rng.uniform(0.5, 1.0)
        img = img.resize((max(64, int(img.width * s)), max(64, int(img.height * s))), Image.BICUBIC)
    if rng.random() < 0.35:  # dark UI strip on one side, like a gallery sidebar
        strip = int(img.width * rng.uniform(0.03, 0.09))
        canvas = Image.new("RGB", (img.width + strip, img.height), tuple(int(rng.uniform(15, 45)) for _ in range(3)))
        canvas.paste(img, (strip, 0) if rng.random() < 0.5 else (0, 0))
        img = canvas
    if rng.random() < 0.6:  # re-encode
        buf = io.BytesIO()
        if rng.random() < 0.7:
            img.save(buf, "JPEG", quality=int(rng.uniform(60, 95)))
        else:
            img.save(buf, "PNG")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


# ---------------------------------------------------------------- features
@torch.no_grad()
def features(model, paths, aug_rng=None, batch=16, tag=""):
    feats, keep = [], []
    t0 = time.time()
    for i in range(0, len(paths), batch):
        xs, ok = [], []
        for p in paths[i:i + batch]:
            try:
                img = Image.open(p)
                if aug_rng is not None:
                    img = augment(img, aug_rng)
                xs.append(vit_preprocess(img))
                ok.append(p)
            except Exception as exc:
                log("  skip", os.path.basename(p), exc)
        if not xs:
            continue
        hs = model.vit(pixel_values=torch.stack(xs)).last_hidden_state[:, 0]
        feats.append(hs.float())
        keep += ok
        if i == 0:
            per = (time.time() - t0) / len(xs)
            log(f"  [{tag}] {per:.3f}s/img, ~{per * len(paths) / 60:.1f} min for {len(paths)} images")
    return (torch.cat(feats) if feats else torch.zeros(0, 384)), keep


def train_logreg(X, y, weight_decay=1e-3, iters=200):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pos = y.sum().item()
    pw = torch.tensor((len(y) - pos) / max(pos, 1.0))
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xn @ w + b, y, pos_weight=pw) + weight_decay * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return {"w": w.detach(), "b": b.detach(), "mu": mu, "sd": sd}


def predict(probe, X):
    return torch.sigmoid(((X - probe["mu"]) / probe["sd"]) @ probe["w"] + probe["b"]).numpy()


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def main():
    t_start = time.time()
    gemini = listing("modern_data/gemini/*", 300)
    gpt4o = listing("modern_data/gpt4o_raw/*", 300)
    old_ai = listing("realistic_data/Ai_generated_dataset/*/*", 150)
    reals = listing("realistic_data/real_dataset/*/*") + listing("calibration_data2/real/*", 300)
    log(f"sources: gemini={len(gemini)} gpt4o={len(gpt4o)} old_ai={len(old_ai)} reals={len(reals)}")
    if len(gemini) + len(gpt4o) < 100:
        sys.exit("not enough new-generator images downloaded")

    groups = {"gemini": gemini, "gpt4o": gpt4o, "old_ai": old_ai, "real": reals}
    tr, va = {}, {}
    for k, v in groups.items():
        tr[k], va[k] = split(v)

    model = ViTForImageClassification.from_pretrained(MODEL_ID).eval()

    # sanity: CLS -> classifier reproduces the served logit
    probe_img = (gemini or gpt4o)[0]
    with torch.no_grad():
        x = vit_preprocess(Image.open(probe_img)).unsqueeze(0)
        full = model(pixel_values=x).logits.flatten()[0].item()
        via_cls = model.classifier(model.vit(pixel_values=x).last_hidden_state[:, 0]).flatten()[0].item()
    log(f"sanity: served logit {full:.4f} vs classifier(CLS) {via_cls:.4f}")
    assert abs(full - via_cls) < 1e-3

    rng = random.Random(SEED)
    Xtr, ytr = [], []
    for k in tr:
        X, _ = features(model, tr[k], aug_rng=rng, tag=f"train/{k}")
        Xtr.append(X)
        ytr += [0.0 if k == "real" else 1.0] * len(X)
    Xtr, ytr = torch.cat(Xtr), torch.tensor(ytr)

    val_scores = {}
    probe = train_logreg(Xtr, ytr)
    for k in va:
        Xc, _ = features(model, va[k], aug_rng=None, tag=f"val/{k}/clean")
        Xa, _ = features(model, va[k], aug_rng=random.Random(SEED + 1), tag=f"val/{k}/aug")
        val_scores[k] = np.concatenate([predict(probe, Xc), predict(probe, Xa)])

    real_v = np.sort(val_scores["real"])
    cands = np.unique(np.concatenate([real_v, [1.0]]))
    high = next(t for t in cands if np.mean(real_v >= t) <= MAX_FP_REAL)
    high = float(min(0.999, max(high, 0.5)))
    fp = float(np.mean(real_v >= high))
    fake_v = np.concatenate([val_scores["gemini"], val_scores["gpt4o"]])
    tp = int((fake_v >= high).sum())
    fpn = int((real_v >= high).sum())
    metrics = {
        "threshold_rule": f"lowest threshold with <= {MAX_FP_REAL:.0%} false positives on held-out real views (clean + augmented), floor 0.5",
        "modern_high": round(high, 4),
        "val_false_positive_rate_real": round(fp, 4),
        "val_recall_gemini": round(float(np.mean(val_scores["gemini"] >= high)), 4) if len(val_scores["gemini"]) else None,
        "val_recall_gpt4o": round(float(np.mean(val_scores["gpt4o"] >= high)), 4) if len(val_scores["gpt4o"]) else None,
        "val_recall_old_ai": round(float(np.mean(val_scores["old_ai"] >= high)), 4),
        "val_precision_new_generators": round(tp / max(1, tp + fpn), 4),
        "val_auc_new_vs_real": round(auc(fake_v, real_v), 4),
        "counts_train": {k: len(v) for k, v in tr.items()},
        "counts_val_views": {k: int(len(v)) for k, v in val_scores.items()},
    }
    log(json.dumps(metrics, indent=2))

    # user test screenshots: scored only, never trained on
    user_files = sorted(glob.glob(os.path.join(USER_TEST_DIR, "*.png")))
    if user_files:
        Xu, kept = features(model, user_files, tag="user-test")
        pu = predict(probe, Xu)
        with torch.no_grad():
            p_ai = torch.sigmoid(model.classifier(Xu)).flatten().numpy()
        log("\nuser test screenshots (not used in training):")
        for f, a, m in zip(kept, p_ai, pu):
            log(f"  {os.path.basename(f)[-15:]:15s} p_ai={a:.4f}  p_modern={m:.4f}  {'FLAG' if m >= high else ''}")
        metrics["user_test"] = {os.path.basename(f): {"p_ai": round(float(a), 4), "p_modern": round(float(m), 4)} for f, a, m in zip(kept, p_ai, pu)}

    os.makedirs("trained_models/modern_probe", exist_ok=True)
    torch.save({k: v for k, v in probe.items()} | {"model_id": MODEL_ID, "feature": "vit.last_hidden_state[:,0]"},
               "trained_models/modern_probe/probe.pt")
    with open("calibration_modern.json", "w") as fh:
        json.dump({
            "source": "train_modern_probe.py: logistic probe on Community Forensics ViT CLS features",
            "data_sources": {
                "gemini": "bitmind/nano-banana (Hugging Face, MIT), shard 0",
                "gpt4o": "Yejy53/GPT-ImgEval GenEval.zip (Hugging Face, MIT)",
                "old_ai": "realistic_data/Ai_generated_dataset",
                "real": "realistic_data/real_dataset + calibration_data2/real",
            },
            "thresholds": {"modern_high": round(high, 4)},
            "metrics": metrics,
        }, fh, indent=2)
    log(f"\nsaved probe + calibration_modern.json in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
