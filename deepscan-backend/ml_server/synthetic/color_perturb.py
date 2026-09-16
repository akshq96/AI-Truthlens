"""Color/illumination-domain perturbation: shift the face region's color
balance relative to the surrounding background, mimicking the illumination/
white-balance mismatch cue between a spliced face and its background that
real detectors are known to exploit.
"""

from __future__ import annotations

import numpy as np

from synthetic.base import SyntheticTechnique, face_region_mask


class ColorPerturbTechnique(SyntheticTechnique):
    name = "color_perturb"

    def __init__(self, max_channel_shift: float = 25.0, max_gamma_delta: float = 0.3):
        self._max_channel_shift = max_channel_shift
        self._max_gamma_delta = max_gamma_delta

    def apply(
        self, face_img: np.ndarray, *, donor_img: np.ndarray | None = None, seed: int = 0
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        h, w = face_img.shape[:2]
        mask = face_region_mask((h, w)).astype(np.float64) / 255.0  # soft [0,1] blend weight
        mask = mask[..., None]

        img = face_img.astype(np.float64)

        channel_shift = rng.uniform(-self._max_channel_shift, self._max_channel_shift, size=(1, 1, 3))
        gamma = 1.0 + rng.uniform(-self._max_gamma_delta, self._max_gamma_delta)
        gamma = max(gamma, 0.1)

        shifted = img + channel_shift
        normalized = np.clip(shifted, 0, 255) / 255.0
        gamma_corrected = np.power(normalized, gamma) * 255.0

        result = mask * gamma_corrected + (1 - mask) * img
        return np.clip(result, 0, 255).astype(np.uint8)
