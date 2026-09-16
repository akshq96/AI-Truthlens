"""Landmark-free approximation of classic face-swap splicing artifacts:
smoothly warp a donor image (or the source image itself, if no donor is
given) via a random smooth displacement field, then Poisson-blend it into
the face region of the target — the same warp-then-blend structure that
produces the boundary/warping artifacts real face-swap tools leave behind,
without depending on landmark detection succeeding (see base.py).
"""

from __future__ import annotations

import numpy as np

from synthetic.base import SyntheticTechnique, face_region_mask


class BlendWarpTechnique(SyntheticTechnique):
    name = "blend_warp"

    def __init__(self, max_displacement_frac: float = 0.06, grid_size: int = 4):
        self._max_displacement_frac = max_displacement_frac
        self._grid_size = grid_size

    def apply(
        self, face_img: np.ndarray, *, donor_img: np.ndarray | None = None, seed: int = 0
    ) -> np.ndarray:
        import cv2

        h, w = face_img.shape[:2]
        donor = donor_img if donor_img is not None else face_img
        if donor.shape[:2] != (h, w):
            donor = cv2.resize(donor, (w, h), interpolation=cv2.INTER_AREA)

        rng = np.random.default_rng(seed)
        max_disp = self._max_displacement_frac * min(h, w)
        coarse_dx = rng.uniform(-max_disp, max_disp, size=(self._grid_size, self._grid_size)).astype(
            np.float32
        )
        coarse_dy = rng.uniform(-max_disp, max_disp, size=(self._grid_size, self._grid_size)).astype(
            np.float32
        )
        dx = cv2.resize(coarse_dx, (w, h), interpolation=cv2.INTER_CUBIC)
        dy = cv2.resize(coarse_dy, (w, h), interpolation=cv2.INTER_CUBIC)

        x_coords, y_coords = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = (x_coords + dx).astype(np.float32)
        map_y = (y_coords + dy).astype(np.float32)

        warped_donor = cv2.remap(
            donor, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

        mask = face_region_mask((h, w))
        center = (w // 2, h // 2)
        try:
            result = cv2.seamlessClone(warped_donor, face_img, mask, center, cv2.NORMAL_CLONE)
        except cv2.error:
            # seamlessClone can fail on degenerate (e.g. near-uniform, tiny)
            # inputs; fall back to a plain masked composite, which still
            # produces the intended warped-region artifact, just without
            # Poisson gradient blending at the seam.
            mask_bool = mask.astype(bool)[..., None]
            result = np.where(mask_bool, warped_donor, face_img)
        return result.astype(np.uint8)
