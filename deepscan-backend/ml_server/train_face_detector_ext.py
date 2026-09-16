"""
Retrain DeepScan's face-manipulation head with public face-swap datasets added to
FaceForensics++, and measure what each data source contributes.

Backbone: Effort CLIP ViT-L/14 (FF++ checkpoint), frozen; head: the same MLP as
train_face_detector.py, on the 1024-d face embedding, with the same face cropping.

Added data (external_download.py; identity-safe splits):
  DeepFakeFace (OpenRL): InsightFace face swaps + the paired real IMDB-WIKI photos
  Identity-Isolated (ThinothW, DF40 subsets): Celeb-DF face swaps + Celeb-DF reals,
      diffusion-generated faces (entire face synthesis). The DF40 FF++-identity subsets
      are NOT used: they share people with our FF++ test and acceptance images.
  bitmind/face-swap: face-swap images (fakes only; used in moderation)

Conditions (identical head/training otherwise)
  ff_baseline         FF++ fakes, FF++ real frames, real portraits
  ff_synthetic        + the five synthetic manipulations (cached from train_face_detector.py)
  external            ff_baseline + the external datasets
  external_synthetic  external + synthetic
Chosen by AUC on the combined validation set (FF++ val, portraits, DeepFakeFace val,
bitmind val). Tests are never used for choosing. Nothing from test_sets/acceptance is used.

Outputs: trained_models/face_head/face_head.pt (+ per-condition heads), calibration_face.json,
         test_results/face_detector_external.json. Previous files are kept as *.before_external.*
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import time

import numpy as np
import torch

import effort_model as em
import train_face_detector as tfd

EXT_DIR = "external_data"
CAP = {  # images per set (bounded for an 8 GB laptop)
    "train_dff_fake": 1200, "train_dff_real": 1200, "train_cdf_fake": 1200, "train_cdf_real": 1500,
    "train_efs_fake": 800, "train_bitmind_fake": 1000,
    "val_dff_fake": 300, "val_dff_real": 300, "val_bitmind_fake": 300,
    "test_dff_fake": 300, "test_dff_real": 300, "test_cdf_fake": 400, "test_cdf_real": 400, "test_bitmind_fake": 300,
}
PATTERNS = {
    "train_dff_fake": "dff/train/fake/*", "train_dff_real": "dff/train/real/*",
    "train_cdf_fake": "df40/train/fake/*DF40_CDF_FS*", "train_cdf_real": "df40/train/real/*DF40_CDF_FS*",
    "train_efs_fake": "df40/train/fake/*DF40_FF_EFS*", "train_bitmind_fake": "bitmind_faceswap/train/fake/*",
    "val_dff_fake": "dff/val/fake/*", "val_dff_real": "dff/val/real/*", "val_bitmind_fake": "bitmind_faceswap/val/fake/*",
    "test_dff_fake": "dff/test/fake/*", "test_dff_real": "dff/test/real/*",
    "test_cdf_fake": "df40/test/fake/*DF40_CDF_FS*", "test_cdf_real": "df40/test/real/*DF40_CDF_FS*",
    "test_bitmind_fake": "bitmind_faceswap/test/fake/*",
}
CACHED = ["train_ff_real", "train_ff_fake", "train_portraits", "train_synthetic",
          "val_ff_real", "val_ff_fake", "val_portraits", "test_ff_real", "test_ff_fake", "test_faceshifter"]
log = tfd.log


def main():
    t_start = time.time()
    excl = tfd.acceptance_names()
    device, dtype = ("mps", torch.float16) if torch.backends.mps.is_available() else ("cpu", torch.float32)
    model = em.load_effort(em.FACE_CKPT, device=device, dtype=dtype)
    cropper = tfd.FaceCropper()

    feats = {}
    for name in CACHED:
        path = f"{tfd.CACHE}/face_{name}.npz"
        if not os.path.exists(path):
            raise SystemExit(f"missing cached features {path} (run train_face_detector.py once)")
        feats[name] = np.load(path, allow_pickle=True)["X"]
    for i, (name, pat) in enumerate(PATTERNS.items()):
        paths = tfd.listing(os.path.join(EXT_DIR, pat), excl, CAP[name])
        if not paths:
            raise SystemExit(f"no images for {name} ({pat}); run external_download.py first")
        aug = tfd.SEED + 100 + i if name.startswith("train_") else None
        X, _ = tfd.cached(f"ext_{name}", lambda paths=paths, aug=aug, name=name:
                          tfd.extract(model, cropper, paths, aug, name))
        feats[name] = X
        log(f"{name}: {len(X)} features")
    lab = {n: (1.0 if "fake" in n or n in ("train_synthetic", "test_faceshifter") else 0.0) for n in feats}

    def stack(names):
        X = np.concatenate([feats[n] for n in names])
        y = np.concatenate([np.full(len(feats[n]), lab[n], np.float32) for n in names])
        return X, y

    ff = ["train_ff_real", "train_ff_fake", "train_portraits"]
    external = ["train_dff_fake", "train_dff_real", "train_cdf_fake", "train_cdf_real", "train_efs_fake", "train_bitmind_fake"]
    conditions = {"ff_baseline": ff, "ff_synthetic": ff + ["train_synthetic"],
                  "external": ff + external, "external_synthetic": ff + external + ["train_synthetic"]}
    val_names = ["val_ff_real", "val_ff_fake", "val_portraits", "val_dff_fake", "val_dff_real", "val_bitmind_fake"]
    Xv, yv = stack(val_names)
    val_real_names = ["val_ff_real", "val_portraits", "val_dff_real"]
    all_test_real = np.concatenate([feats[n] for n in ("test_ff_real", "test_dff_real", "test_cdf_real")])
    tests = {  # name: (fake set, real feature matrix)
        "ffpp_test": ("test_ff_fake", feats["test_ff_real"]),
        "faceshifter_unseen": ("test_faceshifter", feats["test_ff_real"]),
        "deepfakeface_insightface": ("test_dff_fake", feats["test_dff_real"]),
        "celebdf_df40": ("test_cdf_fake", feats["test_cdf_real"]),
        "bitmind_faceswap": ("test_bitmind_fake", all_test_real),
    }

    report, heads = {"counts": {k: int(len(v)) for k, v in feats.items()}, "conditions": {}}, {}
    for cond, names in conditions.items():
        X, y = stack(names)
        net, mu, sd, val_auc = tfd.train_head(X, y, Xv, yv)
        val_real = np.concatenate([feats[n] for n in val_real_names])
        thr = tfd.fpr_threshold(tfd.predict(net, mu, sd, val_real))
        res = {"n_train": int(len(y)), "val_auc": round(val_auc, 4), "threshold": round(thr, 4), "tests": {}}
        all_fake, all_real = [], tfd.predict(net, mu, sd, all_test_real)
        for tname, (fk, Xr) in tests.items():
            pf, pr = tfd.predict(net, mu, sd, feats[fk]), tfd.predict(net, mu, sd, Xr)
            all_fake.append(pf)
            res["tests"][tname] = {"auc": round(tfd.auc(pf, pr), 4), "recall": round(float(np.mean(pf >= thr)), 4),
                                   "fpr": round(float(np.mean(pr >= thr)), 4)}
        pf = np.concatenate(all_fake)
        res["tests"]["all_sources"] = {"auc": round(tfd.auc(pf, all_real), 4), "recall": round(float(np.mean(pf >= thr)), 4),
                                       "fpr": round(float(np.mean(all_real >= thr)), 4)}
        report["conditions"][cond] = res
        heads[cond] = (net, mu, sd, thr, val_auc)
        log(cond, json.dumps(res))

    chosen = max(heads, key=lambda c: heads[c][4])
    report["chosen_for_app"] = chosen
    report["chosen_by"] = "highest combined validation AUC"
    net, mu, sd, thr, _ = heads[chosen]
    pv_fake = tfd.predict(net, mu, sd, np.concatenate([feats[n] for n in ("val_ff_fake", "val_dff_fake", "val_bitmind_fake")]))
    pv_real = tfd.predict(net, mu, sd, np.concatenate([feats[n] for n in val_real_names]))
    low = min(float(np.quantile(pv_fake, tfd.MAX_FAKE_BELOW_LOW)), thr - 1e-3)
    real_cleared = float(np.mean(pv_real <= low))
    face_low = round(low, 4) if real_cleared >= 0.5 else None
    tp, fp = int((pv_fake >= thr).sum()), int((pv_real >= thr).sum())
    report.update({"face_low": face_low, "val_reals_below_low": round(real_cleared, 4),
                   "val_precision_at_high": round(tp / max(1, tp + fp), 4)})
    log(json.dumps(report, indent=2))

    os.makedirs("trained_models/face_head", exist_ok=True)
    os.makedirs("test_results", exist_ok=True)
    for src, dst in (("trained_models/face_head/face_head.pt", "trained_models/face_head/face_head.before_external.pt"),
                     ("calibration_face.json", "calibration_face.before_external.json")):
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    for cond, (n_, m_, s_, t_, _) in heads.items():
        torch.save({"state_dict": n_.state_dict(), "mu": torch.from_numpy(m_), "sd": torch.from_numpy(s_), "hidden": 256,
                    "condition": cond, "threshold": round(t_, 4)}, f"trained_models/face_head/face_head_{cond}.pt")
    torch.save({"state_dict": net.state_dict(), "mu": torch.from_numpy(mu), "sd": torch.from_numpy(sd), "hidden": 256,
                "condition": chosen, "backbone": "effort_clip_L14_trainOn_FaceForensic.pth (frozen)"},
               "trained_models/face_head/face_head.pt")
    json.dump({"source": "train_face_detector_ext.py (FF++ + DeepFakeFace + Celeb-DF/DF40 + bitmind face-swap)",
               "condition": chosen,
               "thresholds": {"face_high": round(thr, 4), "face_low": face_low,
                              "face_precision_at_high": report["val_precision_at_high"]},
               "study": report}, open("calibration_face.json", "w"), indent=2)
    json.dump(report, open("test_results/face_detector_external.json", "w"), indent=2)
    log(f"saved ({chosen}) in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
