"""The one *learned* (not hand-crafted) synthetic technique: a small
convolutional autoencoder, deliberately bottlenecked and undertrained — the
same structure as the original toy "Deepfakes" autoencoder-swap method,
downscaled — trained for a handful of steps to reconstruct a target image,
then applied to the input face crop.

Trained on a target image (`donor_img` if given, else `face_img` itself) for
a small, fixed number of gradient steps at a fixed low resolution, purely
because that is what fits a single `apply()` call at interactive/test speed.
This intentionally produces the blur/color-smoothing reconstruction artifacts
characteristic of real autoencoder-based face-swap outputs (a genuinely
undertrained bottleneck, not a simulation of one). For a real research run
(Phase 11), train per-identity on more images/steps via a dedicated offline
script reusing `TinyAutoencoder` — this class's fast single-call path is for
the synthetic-augmentation pipeline's per-sample use, not for producing
publication-quality reenactment.
"""

from __future__ import annotations

import numpy as np

from synthetic.base import SyntheticTechnique

_WORK_SIZE = 64


class TinyAutoencoder:
    """Lazily builds the torch nn.Module on first use, so importing this
    module doesn't require torch unless the technique is actually invoked
    (keeps Phase 1/early-Phase-3 tests independent of the torch install).
    """

    def __new__(cls, latent_dim: int = 16):
        import torch.nn as nn

        class _Impl(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 16, 4, stride=2, padding=1),  # 64 -> 32
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, 4, stride=2, padding=1),  # 32 -> 16
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, latent_dim, 4, stride=2, padding=1),  # 16 -> 8
                    nn.ReLU(inplace=True),
                )
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(latent_dim, 32, 4, stride=2, padding=1),  # 8 -> 16
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 16 -> 32
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),  # 32 -> 64
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))

        return _Impl()


class AutoencoderSwapTechnique(SyntheticTechnique):
    name = "autoencoder_swap"

    def __init__(self, latent_dim: int = 16, train_steps: int = 40, lr: float = 5e-3):
        self._latent_dim = latent_dim
        self._train_steps = train_steps
        self._lr = lr

    def apply(
        self, face_img: np.ndarray, *, donor_img: np.ndarray | None = None, seed: int = 0
    ) -> np.ndarray:
        import cv2
        import torch

        h, w = face_img.shape[:2]
        target = donor_img if donor_img is not None else face_img

        torch.manual_seed(seed)

        target_small = cv2.resize(target, (_WORK_SIZE, _WORK_SIZE), interpolation=cv2.INTER_AREA)
        input_small = cv2.resize(face_img, (_WORK_SIZE, _WORK_SIZE), interpolation=cv2.INTER_AREA)

        target_t = self._to_tensor(target_small)
        input_t = self._to_tensor(input_small)

        model = TinyAutoencoder(self._latent_dim)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        criterion = torch.nn.MSELoss()

        for _ in range(self._train_steps):
            optimizer.zero_grad()
            recon = model(target_t)
            loss = criterion(recon, target_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out_t = model(input_t)

        out_small = self._to_image(out_t)
        return cv2.resize(out_small, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _to_tensor(img_bgr: np.ndarray):
        import torch

        arr = img_bgr.astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def _to_image(tensor) -> np.ndarray:
        arr = tensor.squeeze(0).permute(1, 2, 0).numpy()
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
