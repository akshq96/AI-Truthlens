"""Shared interface for every synthetic-manipulation technique.

Each technique operates on an already face-cropped image (the output of
ml/data_pipeline/face_crop.py), NOT on a raw frame — it does not re-run face
detection. This is a deliberate scope boundary: Phase 1 already solved "find
and crop the face"; Phase 3's job is purely "apply a manipulation-like pixel
transform to a face crop," which keeps every technique testable on any image
(including non-photographic fixtures) without depending on a real face being
detectable in it.

Where a technique needs to know "which part of the crop is the face" (e.g.
blend_warp, color_perturb), it approximates that as a centered ellipse
covering ~70% of the crop — a reasonable assumption given Phase 1's crop
convention centers the detected face with a fixed margin (see
ml/data_pipeline/face_crop.py), documented here rather than re-deriving
landmarks.

Determinism contract (enforced by unit tests): `apply(img, seed=42)` called
twice on byte-identical input with the same seed must return byte-identical
output. This matters for reproducibility — a synthetic-augmentation ablation
run must be exactly repeatable from its config + seed alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def face_region_mask(shape: tuple[int, int]) -> np.ndarray:
    """A centered-ellipse mask approximating 'the face' within a face crop,
    covering ~70% of the crop's shorter dimension. Shared across techniques
    that need a face-vs-background distinction, so they agree on what "the
    face region" means.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.35), int(h * 0.42))
    import cv2

    cv2.ellipse(mask, center, axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)
    return mask


class SyntheticTechnique(ABC):
    name: str

    @abstractmethod
    def apply(
        self, face_img: np.ndarray, *, donor_img: np.ndarray | None = None, seed: int = 0
    ) -> np.ndarray:
        """Return a new image (same shape/dtype as `face_img`, uint8 HxWx3
        BGR) with the technique's manipulation artifact applied. Must not
        mutate `face_img` in place. Must be deterministic given `seed`.
        """
        raise NotImplementedError
