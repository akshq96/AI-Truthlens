"""Frequency-domain perturbation: perturb the high-frequency magnitude
spectrum of each color channel while preserving phase, then invert. Targets
the same "unnatural frequency spectrum" fingerprint the deepfake-detection
literature reports for GAN-generated/manipulated faces — an easy-to-classify
synthetic proxy for that artifact family, not a claim of replicating any
specific generator's exact spectral signature.
"""

from __future__ import annotations

import numpy as np

from synthetic.base import SyntheticTechnique


class FreqPerturbTechnique(SyntheticTechnique):
    name = "freq_perturb"

    def __init__(self, perturb_strength: float = 0.5):
        self._perturb_strength = perturb_strength

    def apply(
        self, face_img: np.ndarray, *, donor_img: np.ndarray | None = None, seed: int = 0
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        img = face_img.astype(np.float64)
        h, w = img.shape[:2]

        cy, cx = h / 2.0, w / 2.0
        y_idx, x_idx = np.ogrid[:h, :w]
        dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
        max_dist = np.sqrt(cy**2 + cx**2) + 1e-8
        highpass_weight = (dist / max_dist) ** 2  # 0 at DC, 1 at corners

        out_channels = []
        for c in range(img.shape[2]):
            channel = img[..., c]
            spectrum = np.fft.fftshift(np.fft.fft2(channel))
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            noise = rng.normal(loc=0.0, scale=self._perturb_strength, size=spectrum.shape)
            perturbed_magnitude = np.clip(magnitude * (1.0 + highpass_weight * noise), 0, None)

            perturbed_spectrum = perturbed_magnitude * np.exp(1j * phase)
            channel_out = np.real(np.fft.ifft2(np.fft.ifftshift(perturbed_spectrum)))
            out_channels.append(channel_out)

        result = np.stack(out_channels, axis=-1)
        return np.clip(result, 0, 255).astype(np.uint8)
