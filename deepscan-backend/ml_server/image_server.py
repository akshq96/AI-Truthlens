"""
DeepScan — Image Prediction Server (FastAPI)
=============================================
Loads the trained image model from trained_models/image_prediction_model
and serves predictions directly from that model.

Endpoint:
    POST /predict/image   – accepts an image file, returns prediction + confidence
    GET  /health          – health check

Default port: 7000
"""

import os
import traceback
import tempfile

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchcv.model_provider import get_model as ptcv_get_model

try:
    from transformers import ViTForImageClassification
except Exception:
    ViTForImageClassification = None

import decision_engine
import provenance
import synthid
import forensic_signals
import fusion_engine

# ─── Config ──────────────────────────────────────────────────────────────────
# In Docker, the volumes are mounted at these absolute paths
# Relative path is safer for Hugging Face/Docker relative uploads
TRAINED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
_IMAGE_SUBDIR = "image_prediction_model"
IMAGE_MODELS_DIR = os.path.join(TRAINED_MODELS_DIR, _IMAGE_SUBDIR)
_GENERAL_SUBDIR = "general_ai_image_model"
GENERAL_MODELS_DIR = os.path.join(TRAINED_MODELS_DIR, _GENERAL_SUBDIR)
_FACESWAP_SUBDIR = "faceswap_model"
FACESWAP_MODELS_DIR = os.path.join(TRAINED_MODELS_DIR, _FACESWAP_SUBDIR)


def _env_truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "")
    if value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip())
    except Exception:
        return default


_size = _env_int("DEEPSCAN_IMAGE_SIZE", 64)
IMAGE_SIZE = (_size, _size)
USE_IMAGENET_NORM = _env_truthy("DEEPSCAN_IMAGE_IMAGENET_NORM", default=True)
# This checkpoint in this project is calibrated with positive class as real.
POSITIVE_CLASS = os.environ.get("DEEPSCAN_IMAGE_POSITIVE_CLASS", "real").strip().lower()
# model_v3.pth was fine-tuned from ImageNet-pretrained Xception on ~10k labeled
# real/fake face images (Kaggle: rvf10k + hardfakevsrealfaces) after the prior
# checkpoint was found to have ~58% balanced accuracy (near chance) on held-out
# data. Threshold re-calibrated on pooled held-out data (n=3000, two independent
# sources): best balanced-accuracy threshold is 0.25 (balanced_accuracy=0.873,
# vs. 0.834 at the old 0.07 threshold carried over from the previous model).
DEEPFAKE_THRESHOLD = _env_float("DEEPSCAN_IMAGE_DEEPFAKE_THRESHOLD", 0.25)
VIDEO_POSITIVE_CLASS = os.environ.get("DEEPSCAN_VIDEO_AS_IMAGE_POSITIVE_CLASS", POSITIVE_CLASS).strip().lower()
VIDEO_DEEPFAKE_THRESHOLD = _env_float("DEEPSCAN_VIDEO_AS_IMAGE_DEEPFAKE_THRESHOLD", DEEPFAKE_THRESHOLD)
VIDEO_NUM_SAMPLES = 8
# Not yet calibrated against labeled video data (only labeled images were
# available) — uses the image threshold as a reasoned default since this
# endpoint scores video frames with the same image model and averages.
VIDEO_DECISION_THRESHOLD = _env_float("DEEPSCAN_VIDEO_AS_IMAGE_DECISION_THRESHOLD", DEEPFAKE_THRESHOLD)
VIDEO_FACE_SIZE = (_env_int("DEEPSCAN_VIDEO_FACE_SIZE", 440), _env_int("DEEPSCAN_VIDEO_FACE_SIZE", 440))

# model_general.pth: Xception fine-tuned on CIFAKE. It reaches 95% balanced
# accuracy on held-out CIFAKE, but CIFAKE is 32x32 native, so it has only ever
# seen tiny images — on realistic-resolution photos (n=500, 512px-1080px) it
# collapses to 50% balanced accuracy by calling EVERY input fake (mean score
# 0.9997 on real photos). Kept only as a last-resort fallback for non-face
# uploads when the ViT below is unavailable, and responses from it are marked
# reliable=false. Do not treat its verdicts as trustworthy.
GENERAL_POSITIVE_CLASS = os.environ.get("DEEPSCAN_GENERAL_POSITIVE_CLASS", "real").strip().lower()
GENERAL_DEEPFAKE_THRESHOLD = _env_float("DEEPSCAN_GENERAL_DEEPFAKE_THRESHOLD", 0.95)

# PRIMARY DETECTOR for all content, faces and non-faces alike.
# "Community Forensics" ViT-S/16 (Park & Owens, CVPR 2025, arXiv:2411.04125),
# trained on 2.7M images from 4803 generators; the paper reports 92.5% accuracy
# / 0.991 mAP on generators unseen during training. Measured here on held-out
# data never used to pick this threshold:
#   - 800 real/fake face images:        96.4% balanced accuracy
#   - 500 realistic AI-vs-real photos:  99.6% balanced accuracy (250/250 fakes
#     caught, 248/250 real photos kept real)
# Both beat our own checkpoints on their own turf (87.2% faces / 50% general),
# so face detection no longer selects the model — it is reported as metadata
# only. Needs network access on first run to fetch weights from the HF Hub;
# the local Xception checkpoints remain as offline fallbacks.
HF_VIT_MODEL_ID = os.environ.get("DEEPSCAN_HF_VIT_MODEL_ID", "buildborderless/CommunityForensics-DeepfakeDet-ViT")

# Face-manipulation (swap/reenactment) detector: Xception fine-tuned on
# FaceForensics++ C32 cropped faces (Deepfakes / Face2Face / FaceSwap /
# NeuralTextures), with FaceShifter held out entirely to measure generalization
# to an unseen manipulation method. This is a DIFFERENT task from synthetic-image
# detection: in a face swap most pixels are authentic camera output and only the
# face region is manipulated, which is why a whole-image synthetic detector
# scores such images as real. Runs on the cropped face, at its training size.
# MUST match the size the checkpoint was trained at (train_image_model.py
# --image-size), otherwise inference preprocessing silently diverges from
# training and the scores are meaningless.
FACESWAP_IMAGE_SIZE = _env_int("DEEPSCAN_FACESWAP_IMAGE_SIZE", 128)
FACESWAP_POSITIVE_CLASS = os.environ.get("DEEPSCAN_FACESWAP_POSITIVE_CLASS", "real").strip().lower()
FACESWAP_FACE_MARGIN = _env_float("DEEPSCAN_FACESWAP_FACE_MARGIN", 1.3)
HF_VIT_DEEPFAKE_THRESHOLD = _env_float("DEEPSCAN_HF_VIT_DEEPFAKE_THRESHOLD", 0.719)
HF_VIT_DISABLED = _env_truthy("DEEPSCAN_HF_VIT_DISABLED", default=False)
HF_VIT_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
HF_VIT_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _pick_model_path(env_var: str, models_dir: str, preferred_filename: str) -> str:
    """
    Pick a model file in priority order:
      1) env_var (if set)
      2) preferred_filename
      3) first .pth/.pt file in models_dir (sorted)
    """
    env_file = os.environ.get(env_var, "").strip()
    if env_file:
        candidate = env_file
        if not os.path.isabs(candidate):
            candidate = os.path.join(models_dir, candidate)
        return candidate

    preferred = os.path.join(models_dir, preferred_filename)
    if os.path.isfile(preferred):
        return preferred

    if not os.path.isdir(models_dir):
        return preferred

    supported_exts = (".pth", ".pt")
    candidates = sorted(
        [
            os.path.join(models_dir, name)
            for name in os.listdir(models_dir)
            if name.lower().endswith(supported_exts)
        ]
    )
    return candidates[0] if candidates else preferred


def _pick_image_model_path() -> str:
    return _pick_model_path("DEEPSCAN_IMAGE_MODEL_FILE", IMAGE_MODELS_DIR, "model_v3.pth")


def _pick_general_model_path() -> str:
    return _pick_model_path("DEEPSCAN_GENERAL_MODEL_FILE", GENERAL_MODELS_DIR, "model_general.pth")


def _pick_faceswap_model_path() -> str:
    return _pick_model_path("DEEPSCAN_FACESWAP_MODEL_FILE", FACESWAP_MODELS_DIR, "model_faceswap.pth")


IMAGE_MODEL_PATH = _pick_image_model_path()
GENERAL_MODEL_PATH = _pick_general_model_path()
FACESWAP_MODEL_PATH = _pick_faceswap_model_path()


class _XceptionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.BatchNorm1d(2048)
        self.l = nn.Linear(2048, 512)
        self.b2 = nn.BatchNorm1d(512)
        self.o = nn.Linear(512, 1)

    def forward(self, x):
        x = self.b1(x)
        x = self.l(x)
        x = F.relu(x, inplace=False)
        x = self.b2(x)
        x = self.o(x)
        return x


class _ImageTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = ptcv_get_model("xception", pretrained=False)
        # The default pool is AvgPool2d(kernel_size=10), which assumes large inputs.
        # Replace it so 64x64 inference remains valid.
        base_model.features.final_block.pool = nn.AdaptiveAvgPool2d(1)
        self.base = nn.Sequential(base_model.features)
        self.h1 = _XceptionHead()

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)
        x = self.h1(x)
        return x


def _load_torch_image_model(weights_path: str):
    if not os.path.isfile(weights_path):
        raise RuntimeError(f"Image model file not found: {weights_path}")
    if not weights_path.lower().endswith((".pt", ".pth")):
        raise RuntimeError(
            "Unsupported image model format. Only .pt/.pth are allowed for image inference."
        )

    state = torch.load(weights_path, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError("Unsupported .pth format: expected state_dict dictionary")

    model = _ImageTorchModel().eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"State dict mismatch. missing={len(missing)} unexpected={len(unexpected)}"
        )
    return model


# ─── Load models at module level (sync, before FastAPI starts) ───────────────
image_model = None
print("[ImageModel] Loading face deepfake model ...")
try:
    image_model = _load_torch_image_model(IMAGE_MODEL_PATH)
    print(f"[ImageModel] Loaded successfully from: {IMAGE_MODEL_PATH}")
    print(f"[ImageModel] Input shape: [N, 3, {IMAGE_SIZE[0]}, {IMAGE_SIZE[1]}]")
    print("[ImageModel] Output shape: [N, 1]")
    print(
        "[ImageModel] Calibration: "
        f"positive_class={POSITIVE_CLASS}, deepfake_threshold={DEEPFAKE_THRESHOLD}, "
        f"imagenet_norm={USE_IMAGENET_NORM}"
    )
    print(
        "[VideoAsImage] Calibration: "
        f"positive_class={VIDEO_POSITIVE_CLASS}, deepfake_threshold={VIDEO_DEEPFAKE_THRESHOLD}, "
        f"samples={VIDEO_NUM_SAMPLES}, decision_threshold={VIDEO_DECISION_THRESHOLD}, "
        f"face_size={VIDEO_FACE_SIZE[0]}"
    )
except Exception as exc:
    print(f"[ImageModel] Failed to load model: {exc}")
    traceback.print_exc()

general_model = None
print("[GeneralModel] Loading general AI-image model ...")
try:
    general_model = _load_torch_image_model(GENERAL_MODEL_PATH)
    print(f"[GeneralModel] Loaded successfully from: {GENERAL_MODEL_PATH}")
    print(
        "[GeneralModel] Calibration: "
        f"positive_class={GENERAL_POSITIVE_CLASS}, deepfake_threshold={GENERAL_DEEPFAKE_THRESHOLD}"
    )
except Exception as exc:
    print(f"[GeneralModel] Failed to load model: {exc}")
    traceback.print_exc()

faceswap_model = None
print("[FaceSwapModel] Loading face-manipulation model ...")
try:
    faceswap_model = _load_torch_image_model(FACESWAP_MODEL_PATH)
    print(f"[FaceSwapModel] Loaded successfully from: {FACESWAP_MODEL_PATH}")
except Exception as exc:
    print(f"[FaceSwapModel] Not available ({exc}); DEEPFAKE verdicts will be unavailable.")

# DeepScan face-manipulation detector: Effort CLIP ViT-L/14 backbone (FF++
# checkpoint, frozen) + a head trained WITH SYNTHETIC DATA AUGMENTATION by
# train_face_detector.py (five techniques in ./synthetic). Threshold and the
# baseline-vs-augmented study are in calibration_face.json. Preferred detector;
# GenD and the 128 px Xception remain as fallbacks only when it is unavailable.
_HERE = os.path.dirname(os.path.abspath(__file__))
DEEPSCAN_FACE_HEAD_PATH = os.path.join(_HERE, "trained_models", "face_head", "face_head.pt")
DEEPSCAN_FACE_CAL_PATH = os.path.join(_HERE, "calibration_face.json")
DEEPSCAN_FACE = None  # dict(backbone, head, mu, sd, aligner, haar, device, dtype)
if not _env_truthy("DEEPSCAN_FACE_DISABLED") and os.path.exists(DEEPSCAN_FACE_HEAD_PATH) and os.path.exists(DEEPSCAN_FACE_CAL_PATH):
    print("[DeepScanFace] Loading Effort backbone + synthetic-augmentation head ...")
    try:
        import effort_model as _em
        _dev, _dt = ("mps", torch.float16) if torch.backends.mps.is_available() else ("cpu", torch.float32)
        _ckpt = torch.load(DEEPSCAN_FACE_HEAD_PATH, map_location="cpu")
        _hidden = int(_ckpt.get("hidden", 256))
        _head = nn.Sequential(nn.Dropout(0.2), nn.Linear(1024, _hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(_hidden, 1))
        _head.load_state_dict(_ckpt["state_dict"])
        DEEPSCAN_FACE = {
            "backbone": _em.load_effort(_em.FACE_CKPT, device=_dev, dtype=_dt),
            "head": _head.eval(), "mu": _ckpt["mu"].float(), "sd": _ckpt["sd"].float(),
            "aligner": _em.FaceAligner(), "em": _em, "device": _dev, "dtype": _dt,
            "condition": _ckpt.get("condition", "unknown"),
        }
        print(f"[DeepScanFace] Loaded ({DEEPSCAN_FACE['condition']} head) on {_dev}")
    except Exception as exc:
        print(f"[DeepScanFace] Not available ({exc}); falling back to GenD.")
        traceback.print_exc()
        DEEPSCAN_FACE = None

# GenD (Yermakov et al., arXiv:2503.19683): CLIP ViT-L/14 LN-tuned on
# FaceForensics++. Fallback face detector when the DeepScan head is unavailable.
GEND_MODEL_ID = "yermandy/deepfake-detection"
GEND_CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_gend.json")
VIDEO_MAX_FACE_FRAMES = 4
gend_model = None
# It is also loaded next to the DeepScan face detector: the multi-factor fusion
# uses both as independent face-manipulation evidence.
if _env_truthy("DEEPSCAN_GEND_DISABLED"):
    print("[GenD] Disabled via DEEPSCAN_GEND_DISABLED; using Xception face model.")
else:
    print(f"[GenD] Loading {GEND_MODEL_ID} ...")
    try:
        from huggingface_hub import hf_hub_download
        _gend_path = hf_hub_download(GEND_MODEL_ID, "model.torchscript", local_files_only=True)
        # The traced graph casts its input to bfloat16 internally, so weights stay bf16.
        gend_model = torch.jit.load(_gend_path, map_location="cpu").eval()
        print(f"[GenD] Loaded successfully from: {_gend_path}")
    except Exception as exc:
        print(f"[GenD] Not available ({exc}); falling back to Xception face model.")
        gend_model = None

CALIBRATION = decision_engine.load_calibration()
FACE_DETECTOR_NAME = "faceswap_xception"
if gend_model is not None:
    try:
        import json as _json
        with open(GEND_CALIBRATION_PATH, "r", encoding="utf-8") as _fh:
            _gcal = _json.load(_fh)
        _gthr = _gcal["thresholds"]
        CALIBRATION["deepfake_high"] = float(_gthr["deepfake_high"])
        CALIBRATION["deepfake_low"] = None if _gthr.get("deepfake_low") is None else float(_gthr["deepfake_low"])
        CALIBRATION["deepfake_precision_at_high"] = _gthr.get("deepfake_precision_at_high")
        CALIBRATION["source"] = f"{CALIBRATION.get('source')}; face thresholds: {_gcal.get('source')}"
        CALIBRATION.setdefault("metrics", {})["gend_detector"] = _gcal.get("metrics", {})
        FACE_DETECTOR_NAME = "gend_clip_vitl14"
    except Exception as exc:
        print(f"[GenD] calibration_gend.json unusable ({exc}); GenD disabled, using Xception.")
        gend_model = None
if DEEPSCAN_FACE is not None:
    try:
        import json as _json_face
        with open(DEEPSCAN_FACE_CAL_PATH, "r", encoding="utf-8") as _fh_face:
            _fcal = _json_face.load(_fh_face)
        CALIBRATION["deepfake_high"] = float(_fcal["thresholds"]["face_high"])
        _flow = _fcal["thresholds"].get("face_low")
        CALIBRATION["deepfake_low"] = None if _flow is None else float(_flow)
        CALIBRATION["deepfake_precision_at_high"] = _fcal["thresholds"].get("face_precision_at_high")
        CALIBRATION["source"] = f"{CALIBRATION.get('source')}; face thresholds: {_fcal.get('source')}"
        CALIBRATION.setdefault("metrics", {})["deepscan_face_detector"] = _fcal.get("study", {})
        FACE_DETECTOR_NAME = "deepscan_face_effort_augmented"
    except Exception as exc:
        print(f"[DeepScanFace] calibration_face.json unusable ({exc}); disabling.")
        DEEPSCAN_FACE = None
# dlib 81-point landmarks for the measured face signals in forensic_signals.py.
LANDMARKER = None
try:
    LANDMARKER = forensic_signals.Landmarker()
    print("[Forensics] dlib 81-point landmarker loaded")
except Exception as exc:
    print(f"[Forensics] landmarker not available ({exc}); face measurements will be reported unavailable")
print(f"[Fusion] calibration_fusion.json {'loaded' if fusion_engine.load() else 'missing: legacy decision rules in use'}")

# Probe for newer generators (Gemini 2.5 Flash Image, GPT-4o) on the ViT's CLS
# embedding, which the base ViT under-scores. See train_modern_probe.py and
# calibration_modern.json for data sources and the held-out threshold.
MODERN_PROBE = None
_modern_probe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models", "modern_probe", "probe.pt")
_modern_cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_modern.json")
# Enabled when train_ai_detector.py marked it validated (held-out real photos kept
# within the false-positive limit), or when DEEPSCAN_MODERN_PROBE_ENABLED=1.
# DEEPSCAN_MODERN_PROBE_DISABLED=1 always turns it off.
if (not _env_truthy("DEEPSCAN_MODERN_PROBE_DISABLED") and os.path.exists(_modern_probe_path)
        and os.path.exists(_modern_cal_path)):
    try:
        import json as _json_modern
        with open(_modern_cal_path, "r", encoding="utf-8") as _fh_modern:
            _mcal = _json_modern.load(_fh_modern)
        if not (_mcal.get("validated") or _env_truthy("DEEPSCAN_MODERN_PROBE_ENABLED")):
            raise RuntimeError("not validated (set DEEPSCAN_MODERN_PROBE_ENABLED=1 to force)")
        MODERN_PROBE = torch.load(_modern_probe_path, map_location="cpu")
        CALIBRATION["modern_high"] = float(_mcal["thresholds"]["modern_high"])
        CALIBRATION["modern_precision_at_high"] = _mcal["metrics"].get("val_precision_new_generators")
        print(f"[ModernProbe] loaded; modern_high={CALIBRATION['modern_high']}")
    except Exception as exc:
        print(f"[ModernProbe] not available ({exc})")
        MODERN_PROBE = None
print(f"[Calibration] thresholds from {CALIBRATION.get('source')}: "
      f"ai_high={CALIBRATION['ai_high']} ai_low={CALIBRATION['ai_low']} "
      f"deepfake_high={CALIBRATION['deepfake_high']} deepfake_low={CALIBRATION['deepfake_low']}")
print(f"[SynthID] configured={synthid.is_configured()}")

hf_vit_model = None
if HF_VIT_DISABLED:
    print("[HFViT] Disabled via DEEPSCAN_HF_VIT_DISABLED, using Xception face model only.")
elif ViTForImageClassification is None:
    print("[HFViT] transformers not installed, using Xception face model only.")
else:
    print(f"[HFViT] Loading {HF_VIT_MODEL_ID} ...")
    try:
        hf_vit_model = ViTForImageClassification.from_pretrained(HF_VIT_MODEL_ID).eval()
        print(f"[HFViT] Loaded successfully. Calibration: deepfake_threshold={HF_VIT_DEEPFAKE_THRESHOLD}")
    except Exception as exc:
        print(f"[HFViT] Failed to load (will fall back to Xception face model): {exc}")

# ─── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(title="Deepfake Image Prediction Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def preprocess_rgb_array(arr_rgb: np.ndarray) -> np.ndarray:
    arr = arr_rgb.astype(np.float32) / 255.0

    # Xception backbones are typically trained with ImageNet normalization.
    if USE_IMAGENET_NORM:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    arr_rgb = np.array(img, dtype=np.float32)
    return preprocess_rgb_array(arr_rgb)


def _hf_vit_preprocess_pil(img: Image.Image) -> torch.Tensor:
    """Shortest-edge resize to 440, center-crop to 384, CLIP-normalize (per model card)."""
    img = img.convert("RGB")
    w, h = img.size
    short = min(w, h)
    scale = 440 / short
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - 384) / 2
    top = (new_h - 384) / 2
    img = img.crop((round(left), round(top), round(left) + 384, round(top) + 384))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - HF_VIT_CLIP_MEAN) / HF_VIT_CLIP_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def preprocess_for_hf_vit(image_path: str) -> torch.Tensor:
    return _hf_vit_preprocess_pil(Image.open(image_path))


def preprocess_rgb_array_for_hf_vit(arr_rgb_uint8: np.ndarray) -> torch.Tensor:
    return _hf_vit_preprocess_pil(Image.fromarray(arr_rgb_uint8))


def run_hf_vit_inference(pixel_values: torch.Tensor) -> float:
    """
    Returns the deepfake probability directly.

    This model emits a single raw logit, so sigmoid is applied here explicitly
    rather than going through _positive_probability(), whose "value already in
    [0,1] must already be a probability" heuristic would silently skip the
    sigmoid for any logit between 0 and 1 — exactly the uncertain range around
    the decision threshold, where it would flip verdicts.
    """
    with torch.no_grad():
        logits = hf_vit_model(pixel_values=pixel_values).logits
    return float(torch.sigmoid(logits.flatten()[0]).item())


def vit_scores(pixel_values: torch.Tensor):
    """One ViT pass -> (p_ai, p_modern or None, raw logits). The probe reads the
    same CLS embedding the classifier uses, so it adds no extra forward pass."""
    with torch.no_grad():
        cls = hf_vit_model.vit(pixel_values=pixel_values).last_hidden_state[:, 0]
        logits = hf_vit_model.classifier(cls)
        p_modern = None
        if MODERN_PROBE is not None:
            z = ((cls.float() - MODERN_PROBE["mu"]) / MODERN_PROBE["sd"]) @ MODERN_PROBE["w"] + MODERN_PROBE["b"]
            p_modern = float(torch.sigmoid(z).flatten()[0].item())
    return float(torch.sigmoid(logits.flatten()[0]).item()), p_modern, logits.flatten().float().tolist()


def _load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Could not load face detector at: {cascade_path}")
    return detector


FACE_DETECTOR = _load_face_detector()


def _uniform_sample_indices(total_frames: int, fps: float, sample_count: int) -> tuple[list[int], float]:
    if total_frames <= 0:
        raise RuntimeError("Video has no frames")

    duration_seconds = float(total_frames / max(fps, 1e-6))
    n = max(1, int(sample_count))

    # Uniform timestamps: t_i = (i * T) / (N + 1), for i = 1..N
    timestamps = [(i * duration_seconds) / (n + 1) for i in range(1, n + 1)]
    indices: list[int] = []
    for ts in timestamps:
        idx = int(round(ts * fps))
        idx = max(0, min(total_frames - 1, idx))
        indices.append(idx)

    if not indices:
        indices = [total_frames // 2]
    return indices, duration_seconds


# Haar with minNeighbors=5 / minSize=30px reported "faces" in 39 of 112 real
# photos that contain no people, which sent scenery to the face-manipulation
# detector. Requiring more neighbours and a face covering >=10% of the shorter
# side cut that to 8/112 while keeping 118/120 FF++ face crops.
FACE_MIN_NEIGHBORS = 8
FACE_MIN_FRACTION = 0.10


def _detect_faces(gray: np.ndarray):
    min_side = max(30, int(FACE_MIN_FRACTION * min(gray.shape[:2])))
    return FACE_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=(min_side, min_side),
    )


def _extract_face_or_full_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _detect_faces(gray)

    had_face = len(faces) > 0
    if not had_face:
        roi = frame_bgr
    else:
        x, y, w, h = max(faces, key=lambda r: int(r[2]) * int(r[3]))
        roi = frame_bgr[y : y + h, x : x + w]

    if roi.size == 0:
        roi = frame_bgr

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    return cv2.resize(roi_rgb, VIDEO_FACE_SIZE), had_face


def preprocess_video_frames_around_second(
    video_path: str,
    sample_count: int = VIDEO_NUM_SAMPLES,
) -> tuple[list[np.ndarray], list[int], float, float, list[bool]]:
    """Returns raw uint8 RGB frames (not yet model-normalized) so the caller can
    pick per-frame preprocessing based on which model ends up handling the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video has no frames")

    indices, duration_seconds = _uniform_sample_indices(
        total_frames=total_frames,
        fps=fps,
        sample_count=max(1, int(sample_count)),
    )

    raw_frames: list[np.ndarray] = []
    used_indices: list[int] = []
    had_face_flags: list[bool] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb, had_face = _extract_face_or_full_frame(frame)
        raw_frames.append(frame_rgb)
        used_indices.append(int(idx))
        had_face_flags.append(had_face)

    cap.release()

    if not raw_frames:
        raise RuntimeError("Could not extract frames from video")

    return raw_frames, used_indices, fps, duration_seconds, had_face_flags


def run_image_inference(tensor: np.ndarray, model: nn.Module = None) -> np.ndarray:
    net = model if model is not None else image_model
    with torch.no_grad():
        t = torch.from_numpy(tensor).to(dtype=torch.float32)
        pred = net(t)
        return pred.detach().cpu().numpy()


def _largest_face_crop(img_bgr: np.ndarray):
    """Largest detected face expanded by FACESWAP_FACE_MARGIN (context around the
    face carries much of the blending evidence), or None when no face is found."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _detect_faces(gray)
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda r: int(r[2]) * int(r[3]))
    cx, cy = x + w / 2.0, y + h / 2.0
    half = max(w, h) * FACESWAP_FACE_MARGIN / 2.0
    H, W = img_bgr.shape[:2]
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
    crop = img_bgr[y0:y1, x0:x1]
    return img_bgr if crop.size == 0 else crop


def detect_largest_face(image_path: str):
    """Returns (found, bgr_crop)."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return False, None
    crop = _largest_face_crop(img_bgr)
    return crop is not None, crop


_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _gend_preprocess(face_bgr: np.ndarray) -> torch.Tensor:
    """Equivalent of CLIPProcessor(openai/clip-vit-large-patch14), as used by the
    model's inference_torchscript.py: shortest edge 224 (bicubic), center-crop
    224, CLIP mean/std, RGB."""
    img = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    w, h = img.size
    scale = 224.0 / min(w, h)
    img = img.resize((max(224, round(w * scale)), max(224, round(h * scale))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    arr = np.asarray(img.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255.0
    arr = (arr - _CLIP_MEAN) / _CLIP_STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def run_gend(face_crops: list) -> tuple[list[float], list[list[float]]]:
    """Per face crop: P(deepfake) = softmax(logits)[1], and the raw [real, fake] logits."""
    batch = torch.stack([_gend_preprocess(c) for c in face_crops])
    with torch.no_grad():
        logits = gend_model(batch).float()
    probs = torch.softmax(logits, dim=1)[:, 1]
    return [float(p) for p in probs], logits.tolist()


def deepscan_face_crop(img_bgr: np.ndarray):
    """Face crop exactly as used in train_face_detector.py: Effort's dlib 5-point
    alignment (224 px); if dlib finds no face, the Haar margin crop resized to 224.
    Returns (crop or None, method)."""
    face = DEEPSCAN_FACE["aligner"].align(img_bgr)
    if face is not None:
        return face, "dlib_aligned"
    crop = _largest_face_crop(img_bgr)
    if crop is not None:
        return cv2.resize(crop, (224, 224)), "haar_crop"
    return None, "none"


def run_deepscan_face(face_crops: list) -> tuple[list[float], list[float]]:
    """P(face manipulated) per aligned crop, and the head's raw logit."""
    em_ = DEEPSCAN_FACE["em"]
    bb = DEEPSCAN_FACE["backbone"]
    x = torch.stack([em_.to_tensor_bgr(c) for c in face_crops]).to(device=DEEPSCAN_FACE["device"], dtype=DEEPSCAN_FACE["dtype"])
    with torch.no_grad():
        feats = bb.features(x).float().cpu()
        z = DEEPSCAN_FACE["head"]((feats - DEEPSCAN_FACE["mu"]) / DEEPSCAN_FACE["sd"]).squeeze(1)
    return [float(v) for v in torch.sigmoid(z)], [float(v) for v in z]


def fusion_evidence(img_bgr: np.ndarray, p_ai, p_face_head, run_gend_now: bool = True, p_modern=None):
    """Raw inputs for fusion_engine: every ML score that ran plus the measured
    forensic signals. Anything that could not run stays None (unavailable).
    Returns (values, landmarks_found)."""
    values = {"cf_vit": p_ai, "modern_probe": p_modern, "face_head": p_face_head, "gend": None, "xception": None}
    haar = _largest_face_crop(img_bgr)
    if haar is not None:
        if faceswap_model is not None:
            try:
                values["xception"] = float(score_face_manipulation(haar))
            except Exception:
                traceback.print_exc()
        if gend_model is not None and run_gend_now:
            try:
                values["gend"] = float(run_gend([haar])[0][0])
            except Exception:
                traceback.print_exc()
    measured = forensic_signals.analyze(img_bgr, LANDMARKER)
    values.update(measured["features"])
    return values, measured["landmarks_found"]


FUSION_FIELDS = ("real_score", "fake_score", "uncertainty", "signals", "signal_weights", "track_scores",
                 "feature_scores", "metadata_adjustment", "unavailable_signals")


def apply_fusion(decision: dict, fusion: dict, face_found: bool) -> dict:
    if not fusion.get("calibrated"):
        return dict(decision, decision_method="legacy_rules")
    return dict(decision, result=fusion["result"], confidence=fusion["confidence"],
                decision_reason=fusion["decision_reason"], face_detected=bool(face_found),
                decision_method="multi_factor_fusion")


def fusion_response_fields(fusion: dict) -> dict:
    if not fusion.get("calibrated"):
        return {}
    out = {k: fusion.get(k) for k in FUSION_FIELDS}
    out["fusion_thresholds"] = fusion.get("thresholds")
    if fusion.get("fake_score") is not None:
        out["final_score"] = round(float(fusion["fake_score"]) * 100.0, 2)
    return out


def _sample_video_frames(video_path: str, sample_count: int):
    """Evenly spaced frames as (full-frame RGB uint8, margin face crop BGR or None)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video has no frames")
    indices, duration_seconds = _uniform_sample_indices(
        total_frames=total_frames, fps=fps, sample_count=max(1, int(sample_count))
    )
    frames, used_indices = [], []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frames.append((cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), _largest_face_crop(frame_bgr)))
        used_indices.append(int(idx))
    cap.release()
    if not frames:
        raise RuntimeError("Could not extract frames from video")
    return frames, used_indices, fps, duration_seconds


def image_has_face(image_path: str) -> bool:
    """Cheap content router: does this upload contain a detectable human face?"""
    found, _ = detect_largest_face(image_path)
    return found


def _band(value, low, high, labels=("LOW", "MODERATE", "HIGH")):
    if value is None:
        return "NOT MEASURED"
    if value >= high:
        return labels[2]
    if low is not None:
        return labels[0] if value <= low else labels[1]
    # No calibrated authenticity threshold for this detector: it can raise an
    # alarm but cannot certify a clean result, so don't imply a clean reading.
    return f"{labels[0]} ({value:.2f}) — below alarm threshold, not certifiable"


def _evidence_summary(decision, p_ai, p_df, has_face, metadata, c2pa, synthid_result) -> dict:
    """
    Human-readable breakdown for the UI. Every line reflects something actually
    measured; anything not measured says so rather than inventing a value.
    """
    cal = CALIBRATION
    md = metadata or {}
    if not has_face:
        df_line = "NOT APPLICABLE (no face detected)"
    elif p_df is None:
        df_line = "NOT MEASURED (face-manipulation model unavailable)"
    else:
        df_line = _band(p_df, cal.get("deepfake_low"), cal["deepfake_high"])

    if not md.get("available"):
        md_line = "ABSENT (normal for screenshots / social media; not counted against authenticity)"
    elif md.get("ai_declaration"):
        md_line = f"DECLARES AI TOOL ({md['ai_declaration']})"
    else:
        md_line = f"{md.get('reliability', 'none').upper()} (consistency {md.get('consistency_score', 0):.2f})"

    return {
        "ai_forensic_evidence": _band(p_ai, cal.get("ai_low"), cal["ai_high"]),
        "deepfake_evidence": df_line,
        "face_detected": bool(has_face),
        "camera_metadata": md_line,
        "c2pa": (c2pa or {}).get("status", "not_found").upper(),
        "synthid": (synthid_result or {}).get("status", "unavailable").upper(),
        "reason": decision.get("decision_reason"),
    }


def score_face_manipulation(face_bgr: np.ndarray) -> float:
    """Probability the FACE has been manipulated (swap/reenactment)."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (FACESWAP_IMAGE_SIZE, FACESWAP_IMAGE_SIZE))
    raw = run_image_inference(preprocess_rgb_array(resized.astype(np.float32)), model=faceswap_model)
    return deepfake_probability(raw, positive_class=FACESWAP_POSITIVE_CLASS)


def _positive_probability(raw_output: np.ndarray) -> float:
    raw = np.asarray(raw_output).flatten().astype(np.float64)
    n = raw.size

    if n == 1:
        v = float(raw[0])
        if 0.0 <= v <= 1.0:
            return v
        return float(1.0 / (1.0 + np.exp(-np.clip(v, -40.0, 40.0))))

    if n == 2:
        e = np.exp(raw - np.max(raw))
        p = e / np.sum(e)
        return float(p[1])

    e = np.exp(raw - np.max(raw))
    p = e / np.sum(e)
    return float(np.max(p))


def deepfake_probability(raw_output: np.ndarray, positive_class: str = POSITIVE_CLASS) -> float:
    if positive_class not in ("real", "deepfake"):
        raise RuntimeError("positive_class must be either 'real' or 'deepfake'")

    positive_prob = _positive_probability(raw_output)
    deepfake_prob = positive_prob if positive_class == "deepfake" else (1.0 - positive_prob)
    return float(np.clip(deepfake_prob, 0.0, 1.0))


def interpret_deepfake_probability(
    deepfake_prob: float,
    threshold: float = DEEPFAKE_THRESHOLD,
) -> tuple[str, float, float]:
    """
    Turn a deepfake probability into (prediction, confidence, deepfake_probability).

    Confidence is the model-supported probability of the reported label — i.e.
    p for 'deepfake', 1-p for 'real'. It is NOT stretched into a flattering
    85-99% band the way this function used to do; that inflation made every
    verdict look near-certain and destroyed the link between the number shown
    and the model's actual belief.
    """
    prediction = "deepfake" if deepfake_prob >= threshold else "real"
    confidence = deepfake_prob if prediction == "deepfake" else (1.0 - deepfake_prob)
    return prediction, float(round(confidence, 6)), float(round(deepfake_prob, 6))


def interpret_prediction(
    raw_output: np.ndarray,
    threshold: float = DEEPFAKE_THRESHOLD,
    positive_class: str = POSITIVE_CLASS,
) -> tuple[str, float, float]:
    """Same as interpret_deepfake_probability, starting from a model's raw output."""
    deepfake_prob = deepfake_probability(raw_output, positive_class=positive_class)
    return interpret_deepfake_probability(deepfake_prob, threshold=threshold)


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "image_model_loaded": image_model is not None,
        "image_model_path": IMAGE_MODEL_PATH,
        "general_model_loaded": general_model is not None,
        "general_model_path": GENERAL_MODEL_PATH,
        "hf_vit_model_loaded": hf_vit_model is not None,
        "hf_vit_model_id": HF_VIT_MODEL_ID if hf_vit_model is not None else None,
        "faceswap_model_loaded": faceswap_model is not None,
        "faceswap_model_path": FACESWAP_MODEL_PATH,
        "deepscan_face_loaded": DEEPSCAN_FACE is not None,
        "deepscan_face_condition": DEEPSCAN_FACE["condition"] if DEEPSCAN_FACE is not None else None,
        "gend_model_loaded": gend_model is not None,
        "modern_probe_loaded": MODERN_PROBE is not None,
        "modern_high": CALIBRATION.get("modern_high"),
        "face_manipulation_detector": FACE_DETECTOR_NAME,
        "synthid_configured": synthid.is_configured(),
        "calibration_source": CALIBRATION.get("source"),
        "thresholds": {
            "ai_high": CALIBRATION["ai_high"],
            "ai_low": CALIBRATION["ai_low"],
            "deepfake_high": CALIBRATION["deepfake_high"],
            "deepfake_low": CALIBRATION["deepfake_low"],
        },
        "runtime": "torch",
    }


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...), gend: str = Form("1")):
    if image_model is None and general_model is None and hf_vit_model is None:
        raise HTTPException(status_code=503, detail="No image model loaded")

    suffix = os.path.splitext(file.filename or "upload.jpg")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.close()

        errors: list = []
        model_scores: dict = {}

        # Per-detector trace: raw logits before any probability conversion, so a
        # mis-signed or mis-mapped class cannot hide behind a calibrated number.
        trace: dict = {}

        # --- Evidence 1: whole-image synthetic detection -------------------
        p_ai = None
        p_modern = None
        if hf_vit_model is not None:
            try:
                pv = preprocess_for_hf_vit(tmp.name)
                p_ai, p_modern, raw = vit_scores(pv)
                # #region agent log
                try:
                    import json as _dj, time as _dt
                    with open("/Users/akshitraj/.cursor/debug-logs/debug-b72871.log", "a", encoding="utf-8") as _f:
                        _f.write(_dj.dumps({"sessionId": "b72871", "hypothesisId": "A", "location": "image_server.py:predict_image", "message": "vit scores", "data": {"filename": file.filename, "p_ai": round(float(p_ai), 4), "p_modern": None if p_modern is None else round(float(p_modern), 4), "modern_probe_loaded": MODERN_PROBE is not None, "modern_high": CALIBRATION.get("modern_high"), "ai_high": CALIBRATION.get("ai_high")}, "timestamp": int(_dt.time() * 1000)}) + "\n")
                except Exception:
                    pass
                # #endregion
                model_scores["community_forensics_vit"] = round(p_ai, 6)
                if p_modern is not None:
                    model_scores["modern_generator_probe"] = round(p_modern, 6)
                trace["ai_detector"] = {
                    "model": HF_VIT_MODEL_ID,
                    "checkpoint": "hf-hub",
                    "ran": True,
                    "input": {"size": 384, "resize": "shortest-edge 440 -> center-crop 384",
                              "normalization": "CLIP mean/std", "channel_order": "RGB"},
                    "raw_logits": raw,
                    "output_semantics": "single logit; sigmoid(logit) = P(synthetic)",
                    "positive_class": "deepfake/synthetic",
                    "raw_probability": p_ai,
                    "calibrated_probability": p_ai,
                    "threshold_high": CALIBRATION["ai_high"],
                    "threshold_low": CALIBRATION["ai_low"],
                    "modern_probe_probability": p_modern,
                    "modern_threshold_high": CALIBRATION.get("modern_high"),
                }
            except Exception as exc:
                errors.append(f"community_forensics_vit: {type(exc).__name__}: {exc}")
                trace["ai_detector"] = {"model": HF_VIT_MODEL_ID, "ran": False, "error": str(exc)}
                traceback.print_exc()
        else:
            errors.append("community_forensics_vit: not loaded")
            trace["ai_detector"] = {"model": HF_VIT_MODEL_ID, "ran": False, "error": "not loaded"}

        # --- Evidence 2: face presence + face-manipulation detection -------
        face_method = "haar"
        try:
            if DEEPSCAN_FACE is not None:
                _img = cv2.imread(tmp.name)
                face_crop, face_method = deepscan_face_crop(_img) if _img is not None else (None, "none")
                has_face = face_crop is not None
            else:
                has_face, face_crop = detect_largest_face(tmp.name)
        except Exception as exc:
            has_face, face_crop = False, None
            errors.append(f"face_detector: {type(exc).__name__}: {exc}")

        p_deepfake = None
        if has_face and DEEPSCAN_FACE is not None and face_crop is not None:
            try:
                df_probs, df_logits = run_deepscan_face([face_crop])
                p_deepfake = df_probs[0]
                model_scores[FACE_DETECTOR_NAME] = round(p_deepfake, 6)
                trace["deepfake_detector"] = {
                    "model": f"DeepScan face detector: Effort CLIP ViT-L/14 (FF++) + {DEEPSCAN_FACE['condition']} head "
                             f"(synthetic-augmentation study in calibration_face.json)",
                    "checkpoint": "effort_clip_L14_trainOn_FaceForensic.pth + face_head.pt",
                    "training_condition": DEEPSCAN_FACE["condition"],
                    "ran": True,
                    "input": {"size": 224, "face_extraction": face_method,
                              "resize": "dlib 5-point similarity alignment to 224 px (Haar crop fallback)",
                              "normalization": "CLIP mean/std", "channel_order": "RGB"},
                    "raw_logits": df_logits,
                    "output_semantics": "single logit; sigmoid(logit) = P(face manipulated)",
                    "positive_class": "deepfake",
                    "raw_probability": p_deepfake,
                    "calibrated_probability": p_deepfake,
                    "threshold_high": CALIBRATION["deepfake_high"],
                    "threshold_low": CALIBRATION["deepfake_low"],
                    "can_certify_authenticity": CALIBRATION["deepfake_low"] is not None,
                }
            except Exception as exc:
                errors.append(f"{FACE_DETECTOR_NAME}: {type(exc).__name__}: {exc}")
                trace["deepfake_detector"] = {"ran": False, "error": str(exc)}
                traceback.print_exc()
        elif has_face and gend_model is not None and face_crop is not None:
            try:
                gend_probs, gend_logits = run_gend([face_crop])
                p_deepfake = gend_probs[0]
                model_scores["gend_clip_vitl14"] = round(p_deepfake, 6)
                trace["deepfake_detector"] = {
                    "model": f"GenD CLIP ViT-L/14 ({GEND_MODEL_ID})",
                    "checkpoint": "hf-hub model.torchscript",
                    "ran": True,
                    "input": {"size": 224,
                              "resize": f"face crop (margin {FACESWAP_FACE_MARGIN}x) -> shortest-edge 224 -> center-crop 224",
                              "normalization": "CLIP mean/std", "channel_order": "RGB",
                              "crop_shape": list(face_crop.shape)},
                    "raw_logits": gend_logits[0],
                    "output_semantics": "two logits [real, fake]; P(deepfake) = softmax[1]",
                    "positive_class": "deepfake",
                    "raw_probability": p_deepfake,
                    "calibrated_probability": p_deepfake,
                    "threshold_high": CALIBRATION["deepfake_high"],
                    "threshold_low": CALIBRATION["deepfake_low"],
                    "can_certify_authenticity": CALIBRATION["deepfake_low"] is not None,
                }
            except Exception as exc:
                errors.append(f"gend_clip_vitl14: {type(exc).__name__}: {exc}")
                trace["deepfake_detector"] = {"ran": False, "error": str(exc)}
                traceback.print_exc()
        elif has_face and faceswap_model is not None and face_crop is not None:
            try:
                rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (FACESWAP_IMAGE_SIZE, FACESWAP_IMAGE_SIZE))
                tensor = preprocess_rgb_array(resized.astype(np.float32))
                with torch.no_grad():
                    out = faceswap_model(torch.from_numpy(tensor).float())
                raw_df = out.flatten().float().tolist()
                p_deepfake = deepfake_probability(np.array([raw_df]), positive_class=FACESWAP_POSITIVE_CLASS)
                model_scores["faceswap_xception"] = round(p_deepfake, 6)
                trace["deepfake_detector"] = {
                    "model": "Xception (FaceForensics++ fine-tune)",
                    "checkpoint": os.path.basename(FACESWAP_MODEL_PATH),
                    "ran": True,
                    "input": {"size": FACESWAP_IMAGE_SIZE,
                              "resize": f"face crop (margin {FACESWAP_FACE_MARGIN}x) -> {FACESWAP_IMAGE_SIZE}px",
                              "normalization": "ImageNet mean/std", "channel_order": "RGB",
                              "crop_shape": list(face_crop.shape)},
                    "raw_logits": raw_df,
                    "output_semantics": f"single logit; positive_class={FACESWAP_POSITIVE_CLASS}; "
                                        f"P(deepfake) = 1 - sigmoid(logit)",
                    "positive_class": FACESWAP_POSITIVE_CLASS,
                    "raw_probability": float(torch.sigmoid(torch.tensor(raw_df[0])).item()),
                    "calibrated_probability": p_deepfake,
                    "threshold_high": CALIBRATION["deepfake_high"],
                    "threshold_low": CALIBRATION["deepfake_low"],
                    "can_certify_authenticity": CALIBRATION["deepfake_low"] is not None,
                }
            except Exception as exc:
                errors.append(f"faceswap_xception: {type(exc).__name__}: {exc}")
                trace["deepfake_detector"] = {"ran": False, "error": str(exc)}
                traceback.print_exc()
        elif has_face and faceswap_model is None:
            errors.append("faceswap_xception: not loaded (cannot assess face manipulation)")
            trace["deepfake_detector"] = {"ran": False, "error": "not loaded"}
        else:
            trace["deepfake_detector"] = {"ran": False,
                                          "skipped_reason": "no face detected — face-manipulation "
                                                            "threat model not applicable"}
        trace["face_detector"] = {
            "model": "OpenCV Haar cascade (haarcascade_frontalface_default)",
            "ran": True, "face_detected": bool(has_face),
            "crop_margin": FACESWAP_FACE_MARGIN,
        }

        # --- Evidence 3: provenance (metadata, C2PA, SynthID) ---------------
        synthid_result = synthid.check_image(tmp.name)
        try:
            metadata_result, c2pa_result = provenance.analyze_with_c2pa(tmp.name)
        except Exception as exc:
            metadata_result, c2pa_result = {}, {}
            errors.append(f"provenance: {type(exc).__name__}: {exc}")
            traceback.print_exc()

        # --- Evidence 4: measured forensic signals + multi-factor fusion ---
        fusion, fusion_raw, face_found = {"calibrated": False}, None, bool(has_face)
        try:
            img_full = cv2.imread(tmp.name)
            if img_full is None:
                img_full = cv2.cvtColor(np.asarray(Image.open(tmp.name).convert("RGB")), cv2.COLOR_RGB2BGR)
            fvals, lm_found = fusion_evidence(img_full, p_ai, p_deepfake if DEEPSCAN_FACE is not None else None,
                                              run_gend_now=gend != "0", p_modern=p_modern)
            face_found = bool(has_face or lm_found)
            fusion_raw = {"values": fvals, "face_found": face_found}
            if fvals.get("gend") is not None:
                model_scores["gend_clip_vitl14"] = round(fvals["gend"], 6)
            if fvals.get("xception") is not None:
                model_scores["faceswap_xception"] = round(fvals["xception"], 6)
            fusion = fusion_engine.fuse(fvals, face_found, metadata_result, c2pa_result, synthid_result)
        except Exception as exc:
            errors.append(f"fusion: {type(exc).__name__}: {exc}")
            traceback.print_exc()

        # --- Fuse -----------------------------------------------------------
        # #region agent log
        try:
            import json as _dj, time as _dt
            with open("/Users/akshitraj/.cursor/debug-logs/debug-b72871.log", "a", encoding="utf-8") as _f:
                _f.write(_dj.dumps({"sessionId": "b72871", "hypothesisId": "C", "location": "image_server.py:predict_image", "message": "face detector scores", "data": {"filename": file.filename, "has_face": bool(has_face), "p_deepfake": None if p_deepfake is None else round(float(p_deepfake), 4), "face_detector": FACE_DETECTOR_NAME, "face_condition": None if DEEPSCAN_FACE is None else DEEPSCAN_FACE.get("condition"), "gend_loaded": gend_model is not None, "df_high": CALIBRATION.get("deepfake_high"), "df_low": CALIBRATION.get("deepfake_low")}, "timestamp": int(_dt.time() * 1000)}) + "\n")
        except Exception:
            pass
        # #endregion
        decision = decision_engine.decide(
            p_ai=p_ai,
            p_modern=p_modern,
            p_deepfake=p_deepfake,
            face_detected=has_face,
            synthid=synthid_result,
            calibration=CALIBRATION,
            detector_errors=errors,
            metadata=metadata_result,
            c2pa=c2pa_result,
        )
        decision = apply_fusion(decision, fusion, face_found)

        print(
            f"[Decision] {file.filename or 'upload'} -> {decision['result']} "
            f"(conf={decision['confidence']}, p_ai={decision['ai_probability']}, "
            f"p_df={decision['deepfake_probability']}, face={has_face}, errors={len(errors)})"
        )

        response = dict(decision)
        response["synthid"] = {"available": synthid_result["available"], "status": synthid_result["status"]}
        if synthid_result.get("detail"):
            response["synthid"]["detail"] = synthid_result["detail"]
        response["metadata"] = {
            "available": metadata_result.get("available", False),
            "camera": metadata_result.get("camera"),
            "lens": metadata_result.get("lens"),
            "software": metadata_result.get("software"),
            "capture_date": metadata_result.get("capture_date"),
            "gps_present": metadata_result.get("gps_present", False),
            "has_exif": metadata_result.get("has_exif", False),
            "has_xmp": metadata_result.get("has_xmp", False),
            "has_iptc": metadata_result.get("has_iptc", False),
            "has_icc": metadata_result.get("has_icc", False),
            "ai_declaration": metadata_result.get("ai_declaration"),
            "editor_software": metadata_result.get("editor_software"),
            "consistency_score": metadata_result.get("consistency_score", 0.0),
            "reliability": metadata_result.get("reliability", "none"),
            "signals": metadata_result.get("signals", []),
            "notes": metadata_result.get("notes", []),
        }
        response["c2pa"] = {
            "available": c2pa_result.get("available", False),
            "verified": c2pa_result.get("verified", False),
            "status": c2pa_result.get("status", "not_found"),
            "detail": c2pa_result.get("detail"),
            "verifier": c2pa_result.get("verifier"),
        }
        response["evidence_summary"] = _evidence_summary(
            decision, p_ai, p_deepfake, has_face, metadata_result, c2pa_result, synthid_result
        )
        response["model_scores"] = model_scores
        response["trace"] = trace
        response["detector_disagreement"] = decision.get("detector_disagreement")
        response["calibration"] = {
            "source": CALIBRATION.get("source"),
            "ai_high": CALIBRATION["ai_high"],
            "ai_low": CALIBRATION["ai_low"],
            "deepfake_high": CALIBRATION["deepfake_high"],
            "deepfake_low": CALIBRATION["deepfake_low"],
        }
        # Legacy fields so the existing frontend keeps working unchanged.
        response.update(decision_engine.legacy_view(decision))
        response.update(fusion_response_fields(fusion))
        response["fusion_raw"] = fusion_raw
        return response
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.post("/predict/video-as-image")
async def predict_video_as_image(file: UploadFile = File(...)):
    if image_model is None and general_model is None and hf_vit_model is None:
        raise HTTPException(status_code=503, detail="No image model loaded")

    suffix = os.path.splitext(file.filename or "upload.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.close()

        frames, frame_indices, fps, duration_seconds = _sample_video_frames(tmp.name, VIDEO_NUM_SAMPLES)
        errors: list = []
        model_scores: dict = {}

        # Evidence 1: whole-frame synthetic probability, averaged over sampled frames.
        p_ai = None
        p_modern = None
        frame_ai_probs: list = []
        if hf_vit_model is not None:
            try:
                scored = [vit_scores(preprocess_rgb_array_for_hf_vit(rgb)) for rgb, _ in frames]
                frame_ai_probs = [sc[0] for sc in scored]
                p_ai = float(np.mean(frame_ai_probs))
                if MODERN_PROBE is not None:
                    p_modern = float(np.mean([sc[1] for sc in scored]))
                    model_scores["modern_generator_probe"] = round(p_modern, 6)
                model_scores["community_forensics_vit"] = round(p_ai, 6)
            except Exception as exc:
                errors.append(f"community_forensics_vit: {type(exc).__name__}: {exc}")
                traceback.print_exc()
        else:
            errors.append("community_forensics_vit: not loaded")

        # Evidence 2: face manipulation, averaged over face frames. The old video
        # path only ran the whole-image AI detector on tight face crops; that
        # detector reads ~0.01 on face swaps, so face-swap videos were never caught.
        # Any sampled frame with a face brings the face-manipulation threat model
        # into play: phone videos often show the face in only some frames, and
        # requiring a majority let such videos be judged on the AI detector alone.
        if DEEPSCAN_FACE is not None:
            face_crops = []
            for rgb, _ in frames:
                crop, _m = deepscan_face_crop(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if crop is not None:
                    face_crops.append(crop)
        else:
            face_crops = [crop for _, crop in frames if crop is not None]
        has_face = len(face_crops) > 0
        p_deepfake = None
        frame_df_probs: list = []
        if has_face:
            step = max(1, len(face_crops) // VIDEO_MAX_FACE_FRAMES)
            chosen = face_crops[::step][:VIDEO_MAX_FACE_FRAMES]
            try:
                if DEEPSCAN_FACE is not None:
                    frame_df_probs, _ = run_deepscan_face(chosen)
                elif gend_model is not None:
                    frame_df_probs, _ = run_gend(chosen)
                elif faceswap_model is not None:
                    frame_df_probs = [score_face_manipulation(c) for c in chosen]
                else:
                    errors.append("face-manipulation detector not loaded (cannot assess face manipulation)")
                if frame_df_probs:
                    p_deepfake = float(np.mean(frame_df_probs))
                    model_scores[FACE_DETECTOR_NAME] = round(p_deepfake, 6)
            except Exception as exc:
                errors.append(f"{FACE_DETECTOR_NAME}: {type(exc).__name__}: {exc}")
                traceback.print_exc()

        synthid_result = {"available": False, "status": "unavailable"}
        decision = decision_engine.decide(
            p_ai=p_ai,
            p_modern=p_modern,
            p_deepfake=p_deepfake,
            face_detected=has_face,
            synthid=synthid_result,
            calibration=CALIBRATION,
            detector_errors=errors,
        )
        # Multi-factor fusion over up to 4 sampled frames (measurements averaged per signal).
        fusion, v_face = {"calibrated": False}, bool(has_face)
        try:
            if fusion_engine.load() is not None and frames:
                step = max(1, len(frames) // 4)
                per_frame, lm_any = [], False
                for rgb, _ in frames[::step][:4]:
                    v, lm = fusion_evidence(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None, None, run_gend_now=False)
                    per_frame.append(v)
                    lm_any = lm_any or lm
                vals = {}
                for k in set().union(*per_frame):
                    xs = [pf[k] for pf in per_frame if pf.get(k) is not None]
                    vals[k] = float(np.mean(xs)) if xs else None
                vals["cf_vit"] = p_ai
                vals["modern_probe"] = p_modern
                vals["face_head"] = p_deepfake if DEEPSCAN_FACE is not None else None
                v_face = bool(has_face or lm_any)
                fusion = fusion_engine.fuse(vals, v_face)
        except Exception as exc:
            errors.append(f"fusion: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        decision = apply_fusion(decision, fusion, v_face)
        print(
            f"[Decision/video] {file.filename or 'upload'} -> {decision['result']} "
            f"(conf={decision['confidence']}, p_ai={decision['ai_probability']}, "
            f"p_df={decision['deepfake_probability']}, face_frames={len(face_crops)}/{len(frames)})"
        )

        center_idx = frame_indices[len(frame_indices) // 2]
        response = dict(decision)
        response.update({
            "synthid": synthid_result,
            "evidence_summary": _evidence_summary(decision, p_ai, p_deepfake, has_face, {}, {}, synthid_result),
            "model_scores": model_scores,
            "detector": FACE_DETECTOR_NAME if has_face else "community_forensics_vit",
            "face_frames_detected": int(len(face_crops)),
            "sampled_second": float(round(center_idx / fps, 3)),
            "sampled_frame_index": int(center_idx),
            "sampled_frame_indices": [int(i) for i in frame_indices],
            "frames_analyzed": int(len(frames)),
            "mode": "video_multi_frame",
            "video_duration_seconds": float(round(duration_seconds, 3)),
            "aggregation": "mean over sampled frames",
            "frame_ai_probabilities": [round(float(v), 6) for v in frame_ai_probs],
            "frame_deepfake_probabilities": [round(float(v), 6) for v in frame_df_probs],
            "calibration": {
                "source": CALIBRATION.get("source"),
                "ai_high": CALIBRATION["ai_high"],
                "ai_low": CALIBRATION["ai_low"],
                "deepfake_high": CALIBRATION["deepfake_high"],
                "deepfake_low": CALIBRATION["deepfake_low"],
            },
        })
        response.update(decision_engine.legacy_view(decision))
        response.update(fusion_response_fields(fusion))
        return response
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Image Prediction Server starting on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
