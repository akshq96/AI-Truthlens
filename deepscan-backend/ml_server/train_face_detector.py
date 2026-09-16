"""
Train DeepScan's face-manipulation detector WITH SYNTHETIC DATA AUGMENTATION,
and measure whether the augmentation helps.

Backbone: Effort CLIP ViT-L/14 (FaceForensics++ checkpoint), kept frozen.
Head: small MLP on the 1024-d pooled face embedding.

Training data
  fake : FaceForensics++ C32 train split (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
  real : FF++ real frames + real portraits / people photos (plus degraded copies)
  synthetic fake (augmented condition only): real training faces passed through the
        five synthetic manipulations in ./synthetic (blend_warp, freq_perturb,
        compression_artifact, color_perturb, autoencoder_swap), ratio 0.3 of the
        real training set, each applied to the aligned face crop with another real
        training face as donor. Synthetic samples come ONLY from training images.

Study (answers the project's research question on real data)
  * baseline vs augmented head, identical data otherwise
  * evaluated on FF++ test, on FaceShifter (never used in training), and in a
    leave-one-method-out setting: for each FF++ method H, heads are retrained
    without H and tested on H.

The head used by the app is the augmented one unless it is worse on validation.
Nothing in test_sets/acceptance is used. Thresholds come from validation data.

Face preprocessing = Effort's own dlib 5-point alignment (224 px); Haar crop fallback.
Outputs: trained_models/face_head/face_head.pt, calibration_face.json,
         test_results/augmentation_study.json, feature_cache/face_*.npz
"""
from __future__ import annotations

import glob
import json
import os
import random
import time

import cv2
import numpy as np
import torch

import effort_model as em
import synthetic

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
SEED = 7
CACHE = "feature_cache"
EXT = (".jpg", ".jpeg", ".png", ".webp")
# Operating point: at most 15% of validation real faces flagged. Measured trade-off on the
# previous run: false alarms at this level come from low-quality FF++ video frames, while
# real portraits/photos stayed at 0%; a stricter cap missed about 1 in 3 deepfakes.
MAX_FPR_REAL = 0.15
SYNTHETIC_RATIO = 0.3
N_TRAIN_FAKE = 3500          # all FF++ C32 training fakes (875 per method)
MAX_FAKE_BELOW_LOW = 0.05    # authenticity threshold: at most 5% of val fakes may score below it
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


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


def listing(pattern, exclude, limit=None, seed=SEED):
    f = sorted(p for p in glob.glob(pattern) if p.lower().endswith(EXT) and os.path.basename(p) not in exclude)
    random.Random(seed).shuffle(f)
    return f[:limit] if limit else f


def degrade(img_bgr, rng):
    img = img_bgr
    if rng.random() < 0.4:
        s = rng.uniform(0.45, 0.9)
        img = cv2.resize(img, (max(48, int(img.shape[1] * s)), max(48, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.25:
        strip = max(8, int(img.shape[1] * rng.uniform(0.03, 0.08)))
        c = np.full((img.shape[0], img.shape[1] + strip, 3), int(rng.uniform(15, 45)), np.uint8)
        c[:, strip:] = img
        img = c
    if rng.random() < 0.5:
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(45, 95))])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


class FaceCropper:
    def __init__(self):
        self.aligner = em.FaceAligner()
        self.haar = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

    def __call__(self, img_bgr):
        face = self.aligner.align(img_bgr)
        if face is not None:
            return face, "dlib"
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        ms = max(24, int(0.10 * min(g.shape)))
        fs = self.haar.detectMultiScale(g, 1.1, 6, minSize=(ms, ms))
        if len(fs):
            x, y, w, h = max(fs, key=lambda r: r[2] * r[3])
            H, W = g.shape
            cx, cy, half = x + w / 2, y + h / 2, max(w, h) * 1.3 / 2
            crop = img_bgr[int(max(0, cy - half)):int(min(H, cy + half)), int(max(0, cx - half)):int(min(W, cx + half))]
            if crop.size:
                return cv2.resize(crop, (224, 224)), "haar"
        return None, "none"


@torch.no_grad()
def extract(model, cropper, paths, augment_seed=None, tag="", face_transform=None, batch=16):
    rng = random.Random(augment_seed) if augment_seed is not None else None
    p0 = next(model.parameters())
    feats, keep, how = [], [], {"dlib": 0, "haar": 0, "none": 0, "unreadable": 0}
    buf, buf_paths = [], []
    t0 = time.time()

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(device=p0.device, dtype=p0.dtype)
        feats.append(model.features(x).float().cpu())
        keep.extend(buf_paths)
        buf.clear(); buf_paths.clear()

    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            how["unreadable"] += 1
            continue
        if rng is not None:
            img = degrade(img, rng)
        face, method = cropper(img)
        how[method] += 1
        if face is None:
            continue
        if face_transform is not None:
            face = face_transform(face, i)
        buf.append(em.to_tensor_bgr(face)); buf_paths.append(p)
        if len(buf) == batch:
            flush()
        if i == 63:
            per = (time.time() - t0) / 64
            log(f"  [{tag}] {per:.3f}s/img -> ~{per * len(paths) / 60:.1f} min for {len(paths)}")
    flush()
    X = torch.cat(feats).numpy() if feats else np.zeros((0, 1024), np.float32)
    log(f"  [{tag}] kept {len(keep)}/{len(paths)} faces {how} in {(time.time()-t0)/60:.1f} min")
    return X, keep


def cached(name, fn):
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/face_{name}.npz"
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return d["X"], list(d["paths"])
    X, paths = fn()
    np.savez(path, X=X, paths=np.array(paths, dtype=object))
    return X, paths


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def train_head(X, y, Xv, yv, hidden=256, epochs=60, wd=1e-3):
    torch.manual_seed(SEED)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.from_numpy((X - mu) / sd).float(); yt = torch.from_numpy(y).float()
    Xvt = torch.from_numpy((Xv - mu) / sd).float()
    net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Dropout(0.2), torch.nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    pw = torch.tensor((len(y) - y.sum()) / max(1.0, y.sum()))
    best, best_state = -1.0, None
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[idx]).squeeze(1), yt[idx], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xvt).squeeze(1)).numpy()
        a = auc(pv[yv == 1], pv[yv == 0])
        if a > best:
            best, best_state = a, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, mu, sd, best


def predict(net, mu, sd, X):
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy((X - mu) / sd).float()).squeeze(1)).numpy()


def fpr_threshold(real_scores):
    r = np.sort(real_scores)
    return max(0.5, float(r[min(len(r) - 1, int(np.ceil((1 - MAX_FPR_REAL) * len(r))))]))


def method_of(path):
    b = os.path.basename(path)
    for m in METHODS:
        if b.startswith(m + "_"):
            return m
    return None


def main():
    t_start = time.time()
    excl = acceptance_names()
    device, dtype = ("mps", torch.float16) if torch.backends.mps.is_available() else ("cpu", torch.float32)
    log("device", device, dtype)
    model = em.load_effort(em.FACE_CKPT, device=device, dtype=dtype)
    cropper = FaceCropper()

    portraits = listing("calibration_data2/real/*", excl) + listing("realistic_data/real_dataset/people/*", excl)
    n_pv = len(portraits) // 4
    port_val, port_train = portraits[:n_pv], portraits[n_pv:]
    ff_train_real = listing("ff_splits/train/real/*", excl)  # features already cached from an earlier run

    # synthetic fakes: from real TRAINING faces only, cycling the five techniques
    synth_sources = ff_train_real + port_train
    # ratio 0.3 relative to the training fakes actually used (bounded for an 8 GB laptop)
    n_synth = int(round(SYNTHETIC_RATIO * N_TRAIN_FAKE / (1 - SYNTHETIC_RATIO)))
    rng = random.Random(SEED + 9)
    synth_paths = [synth_sources[rng.randrange(len(synth_sources))] for _ in range(n_synth)]
    donor_paths = [synth_sources[rng.randrange(len(synth_sources))] for _ in range(n_synth)]
    names = list(synthetic.TECHNIQUES)
    techs = {n: synthetic.build(n) for n in names}
    synth_tech_of = [names[i % len(names)] for i in range(n_synth)]

    def synth_transform(face, i):
        donor_img = cv2.imread(donor_paths[i])
        donor_face = cropper(donor_img)[0] if donor_img is not None else None
        # extract() runs under no_grad; autoencoder_swap trains a small network per call
        with torch.enable_grad():
            return techs[synth_tech_of[i]].apply(face, donor_img=donor_face, seed=SEED * 1000 + i)

    sets = {
        "train_ff_real": (ff_train_real, 0, SEED + 1, None),
        f"train_ff_fake_{N_TRAIN_FAKE}": (listing("ff_splits/train/fake/*", excl, N_TRAIN_FAKE), 1, SEED + 2, None),
        "train_portraits": (port_train, 0, SEED + 3, None),
        f"train_synthetic_{n_synth}": (synth_paths, 1, None, synth_transform),
        "val_ff_real": (listing("ff_splits/val/real/*", excl, 300), 0, None, None),
        "val_ff_fake": (listing("ff_splits/val/fake/*", excl, 300), 1, None, None),
        "val_portraits": (port_val, 0, None, None),
        "test_ff_real": (listing("ff_splits/test/real/*", excl, 250), 0, None, None),
        "test_ff_fake": (listing("ff_splits/test/fake/*", excl, 250), 1, None, None),
        "test_faceshifter": (listing("ff_splits/unseen/fake/*", excl, 200), 1, None, None),
    }
    feats, paths_of = {}, {}
    for name, (paths, label, aug, tf) in sets.items():
        X, kept = cached(name, lambda paths=paths, aug=aug, tf=tf, name=name: extract(model, cropper, paths, aug, name, tf))
        feats[name] = (X, np.full(len(X), label, np.float32))
        paths_of[name] = kept
        log(f"{name}: {len(X)} features")

    def stack(names_, mask=None):
        Xs, ys = [], []
        for n in names_:
            X, y = feats[n]
            if mask is not None and n in mask:
                keep = mask[n]
                X, y = X[keep], y[keep]
            Xs.append(X); ys.append(y)
        return np.concatenate(Xs), np.concatenate(ys)

    fake_key, synth_key = f"train_ff_fake_{N_TRAIN_FAKE}", f"train_synthetic_{n_synth}"
    base_names = ["train_ff_real", fake_key, "train_portraits"]
    Xv, yv = stack(["val_ff_real", "val_ff_fake", "val_portraits"])
    study = {"synthetic_ratio": SYNTHETIC_RATIO, "n_synthetic": int(len(feats[synth_key][0])),
             "techniques": names, "max_fpr_real_val": MAX_FPR_REAL, "n_train_fake": N_TRAIN_FAKE, "conditions": {}}
    heads = {}
    for cond, train_names in (("baseline", base_names), ("augmented", base_names + [synth_key])):
        X, y = stack(train_names)
        net, mu, sd, val_auc = train_head(X, y, Xv, yv)
        val_real = np.concatenate([feats["val_ff_real"][0], feats["val_portraits"][0]])
        thr = fpr_threshold(predict(net, mu, sd, val_real))
        res = {"val_auc": round(val_auc, 4), "threshold": round(thr, 4)}
        Xr = feats["test_ff_real"][0]
        for tname, fk in (("ffpp_test", "test_ff_fake"), ("faceshifter_unseen", "test_faceshifter")):
            pr, pf = predict(net, mu, sd, Xr), predict(net, mu, sd, feats[fk][0])
            res[tname] = {"auc": round(auc(pf, pr), 4), "recall": round(float(np.mean(pf >= thr)), 4),
                          "fpr": round(float(np.mean(pr >= thr)), 4)}
        res["val_portraits_fpr"] = round(float(np.mean(predict(net, mu, sd, feats["val_portraits"][0]) >= thr)), 4)

        # leave-one-method-out: retrain without method H, test on H (test split)
        loo = {}
        fake_train_methods = np.array([method_of(p) for p in paths_of[fake_key]])
        test_methods = np.array([method_of(p) for p in paths_of["test_ff_fake"]])
        for H in METHODS:
            mask = {fake_key: fake_train_methods != H}
            Xh, yh = stack(train_names, mask)
            neth, muh, sdh, _ = train_head(Xh, yh, Xv, yv, epochs=40)
            pf = predict(neth, muh, sdh, feats["test_ff_fake"][0][test_methods == H])
            pr = predict(neth, muh, sdh, Xr)
            loo[H] = round(auc(pf, pr), 4)
        res["leave_one_method_out_auc"] = loo
        res["leave_one_method_out_mean_auc"] = round(float(np.mean(list(loo.values()))), 4)
        study["conditions"][cond] = res
        heads[cond] = (net, mu, sd, thr, val_auc)
        log(cond, json.dumps(res))

    # pretrained Effort head on the same features, for reference
    with torch.no_grad():
        def pre(Z):
            logits = model.head(torch.from_numpy(Z).to(device=device, dtype=dtype)).float().cpu()
            return torch.softmax(logits, 1)[:, 1].numpy()
        pr = pre(feats["test_ff_real"][0])
        study["pretrained_effort_head"] = {
            "ffpp_test_auc": round(auc(pre(feats["test_ff_fake"][0]), pr), 4),
            "faceshifter_unseen_auc": round(auc(pre(feats["test_faceshifter"][0]), pr), 4)}

    chosen = "augmented" if heads["augmented"][4] >= heads["baseline"][4] - 0.005 else "baseline"
    study["chosen_for_app"] = chosen
    net, mu, sd, thr, _ = heads[chosen]

    # authenticity threshold (lets the app say REAL for a face): highest score below which
    # at most MAX_FAKE_BELOW_LOW of validation fakes fall, kept only if it clears >= 50% of val reals
    pv_fake = predict(net, mu, sd, np.concatenate([feats["val_ff_fake"][0]]))
    pv_real = predict(net, mu, sd, np.concatenate([feats["val_ff_real"][0], feats["val_portraits"][0]]))
    low = float(np.quantile(pv_fake, MAX_FAKE_BELOW_LOW))
    low = min(low, thr - 1e-3)
    real_cleared = float(np.mean(pv_real <= low))
    face_low = round(low, 4) if real_cleared >= 0.5 else None
    tp = int((pv_fake >= thr).sum()); fp = int((pv_real >= thr).sum())
    study["face_low"] = face_low
    study["val_reals_below_low"] = round(real_cleared, 4)
    study["val_precision_at_high"] = round(tp / max(1, tp + fp), 4)
    log(json.dumps(study, indent=2))

    os.makedirs("trained_models/face_head", exist_ok=True)
    os.makedirs("test_results", exist_ok=True)
    for cond, (n_, m_, s_, t_, _) in heads.items():  # keep both heads for the study / manual switching
        torch.save({"state_dict": n_.state_dict(), "mu": torch.from_numpy(m_), "sd": torch.from_numpy(s_), "hidden": 256,
                    "condition": cond, "threshold": round(t_, 4)}, f"trained_models/face_head/face_head_{cond}.pt")
    torch.save({"state_dict": net.state_dict(), "mu": torch.from_numpy(mu), "sd": torch.from_numpy(sd), "hidden": 256,
                "condition": chosen, "backbone": "effort_clip_L14_trainOn_FaceForensic.pth (frozen)"},
               "trained_models/face_head/face_head.pt")
    json.dump({"source": "train_face_detector.py", "condition": chosen,
               "thresholds": {"face_high": round(thr, 4), "face_low": face_low,
                              "face_precision_at_high": study["val_precision_at_high"]}, "study": study,
               "counts": {k: int(len(v[0])) for k, v in feats.items()}}, open("calibration_face.json", "w"), indent=2)
    json.dump(study, open("test_results/augmentation_study.json", "w"), indent=2)
    log(f"saved ({chosen}) in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
