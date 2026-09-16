"""JPEG re-compression artifact injection: re-encode at a random low quality
(optionally twice, simulating double-compression) then decode. Simulates the
recompression traces manipulated videos often carry after re-encoding.

Also reused (deliberately) by the robustness-evaluation perturbations
(ml/evaluation/experiments/robustness.py, Phase 6) — see the caveat in the
project plan / docs/methodology.md once written: a model trained with this
technique in its synthetic mix must not have its "robustness to compression"
reported without flagging that as a controlled, expected result rather than
general robustness.
"""

from __future__ import annotations

import numpy as np

from synthetic.base import SyntheticTechnique


class CompressionArtifactTechnique(SyntheticTechnique):
    name = "compression_artifact"

    def __init__(
        self, quality_range: tuple[int, int] = (20, 50), double_compress_prob: float = 0.5
    ):
        self._quality_range = quality_range
        self._double_compress_prob = double_compress_prob

    def apply(
        self, face_img: np.ndarray, *, donor_img: np.ndarray | None = None, seed: int = 0
    ) -> np.ndarray:
        import cv2

        rng = np.random.default_rng(seed)
        img = face_img
        n_passes = 2 if rng.random() < self._double_compress_prob else 1
        low, high = self._quality_range
        for _ in range(n_passes):
            quality = int(rng.integers(low, high + 1))
            ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError("JPEG encoding failed in CompressionArtifactTechnique")
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return img.astype(np.uint8)
