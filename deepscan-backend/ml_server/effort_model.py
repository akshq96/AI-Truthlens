"""
Effort detectors (Yan et al., "Orthogonal Subspace Decomposition for Generalizable
AI-Generated Image Detection", ICML 2025) loaded without the DeepfakeBench
training stack.

Official implementation: DeepfakeBench/training/detectors/effort_detector.py and
training/demo.py in https://github.com/YZY-stack/Effort-AIGI-Detection.

Model: CLIP ViT-L/14 vision tower (224 px) whose self-attention q/k/v/out
projections were replaced by SVDResidualLinear(weight_main + U_res diag(S_res) V_res),
followed by head = Linear(1024, 2) on pooler_output.
Output: prob = softmax(logits)[:, 1]; DeepfakeBench label convention 0 = real, 1 = fake.

At load time each SVD residual layer is merged into a plain nn.Linear weight:
    W = weight_main + U_residual @ diag(S_residual) @ V_residual
which is exactly what SVDResidualLinear.forward computes, so inference is
numerically identical while using a standard CLIP vision transformer.

Preprocessing (demo.py), reproduced exactly:
  * face checkpoint: dlib frontal face detector (upsample 1) on RGB, largest face,
    5 key points from the 81-point predictor (parts 37, 44, 30, 49, 55),
    similarity transform to the ArcFace-style template scaled to 224 px with a
    1.3 margin, cv2.warpAffine; then RGB, resize 224 (bilinear), ToTensor,
    CLIP mean/std.
  * general AI-image checkpoint: no face crop; the whole image is resized to
    224x224 (bilinear, aspect ratio not kept), then ToTensor and CLIP mean/std.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
EFFORT_DIR = os.path.join(HERE, "trained_models", "effort")
FACE_CKPT = os.path.join(EFFORT_DIR, "effort_clip_L14_trainOn_FaceForensic.pth")
AIGI_CKPT = os.path.join(EFFORT_DIR, "effort_clip_L14_trainOn_chameleon.pth")
LANDMARK_MODEL = os.path.join(EFFORT_DIR, "shape_predictor_81_face_landmarks.dat")

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
FAKE_CLASS_INDEX = 1  # verified empirically in detector_benchmark.py --polarity


class EffortNet(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import CLIPVisionConfig
        from transformers.models.clip.modeling_clip import CLIPVisionTransformer

        cfg = CLIPVisionConfig(
            hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16,
            image_size=224, patch_size=14, projection_dim=768, hidden_act="quick_gelu", layer_norm_eps=1e-5,
        )
        cfg._attn_implementation = "eager"  # not set when building from a bare config
        self.backbone = CLIPVisionTransformer(cfg)
        self.head = nn.Linear(1024, 2)

    def features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values=pixel_values).pooler_output

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(pixel_values))


def _merge_svd_state(state: dict) -> dict:
    """Convert SVDResidualLinear parameters into plain Linear weights."""
    state = {k.replace("module.", ""): v for k, v in state.items()}
    merged, prefixes = {}, set()
    for k in state:
        if k.endswith(".weight_main"):
            prefixes.add(k[: -len(".weight_main")])
    for k, v in state.items():
        if any(k.startswith(p + ".") for p in prefixes):
            continue
        merged[k] = v
    for p in prefixes:
        w = state[p + ".weight_main"].float()
        s_res = state.get(p + ".S_residual")
        if s_res is not None:
            w = w + state[p + ".U_residual"].float() @ torch.diag(s_res.float()) @ state[p + ".V_residual"].float()
        merged[p + ".weight"] = w
        if p + ".bias" in state:
            merged[p + ".bias"] = state[p + ".bias"]
    return merged


def load_effort(ckpt_path: str, device: str = "cpu", dtype: torch.dtype = torch.float32) -> EffortNet:
    raw = torch.load(ckpt_path, map_location="cpu")
    raw = raw.get("state_dict", raw)
    state = _merge_svd_state(raw)
    del raw
    model = EffortNet()
    missing, unexpected = model.load_state_dict(state, strict=False)
    # position_ids is a non-persistent buffer in newer transformers
    missing = [m for m in missing if not m.endswith("position_ids")]
    if missing or unexpected:
        raise RuntimeError(f"Effort checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.to(device=device, dtype=dtype).eval()


# ---------------------------------------------------------------- preprocessing
class FaceAligner:
    """dlib detection + 5-point similarity alignment, as in Effort's demo.py."""

    def __init__(self, landmark_model: str = LANDMARK_MODEL):
        import dlib

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(landmark_model)

    def align(self, img_bgr: np.ndarray, res: int = 224, scale: float = 1.3) -> Optional[np.ndarray]:
        from skimage import transform as trans

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces = self.detector(rgb, 1)
        if not len(faces):
            return None
        face = max(faces, key=lambda r: r.width() * r.height())
        shape = self.predictor(rgb, face)
        pts = np.array([[shape.part(i).x, shape.part(i).y] for i in (37, 44, 30, 49, 55)], dtype=np.float32)

        dst = np.array([[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
                        [33.5493, 92.3655], [62.7299, 92.2041]], dtype=np.float32)
        dst[:, 0] += 8.0
        dst[:, 0] = dst[:, 0] * res / 112
        dst[:, 1] = dst[:, 1] * res / 112
        margin = scale - 1
        x_margin, y_margin = res * margin / 2.0, res * margin / 2.0
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin
        dst[:, 0] *= res / (res + 2 * x_margin)
        dst[:, 1] *= res / (res + 2 * y_margin)

        tform = trans.SimilarityTransform()
        tform.estimate(pts, dst)
        aligned_rgb = cv2.warpAffine(rgb, tform.params[0:2, :], (res, res))
        return cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)


def to_tensor_bgr(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    arr = (rgb.astype(np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


@torch.no_grad()
def fake_probability(model: EffortNet, batch: torch.Tensor):
    """Returns (P(fake) per row, raw logits per row)."""
    p = next(model.parameters())
    logits = model(batch.to(device=p.device, dtype=p.dtype)).float().cpu()
    probs = torch.softmax(logits, dim=1)[:, FAKE_CLASS_INDEX]
    return probs.tolist(), logits.tolist()
