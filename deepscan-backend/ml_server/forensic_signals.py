"""
Measured forensic signals for DeepScan's multi-factor fusion.

Every value is computed from the image itself. Nothing is estimated or filled in:
when a measurement cannot be taken (no face landmarks, face too small, region too
small, no background visible around the face), it is returned as None and the
fusion engine treats it as unavailable and renormalises the remaining weights.

A raw measurement is NOT a probability. calibrate_fusion.py learns from labelled
real / deepfake / AI-generated images how each measurement relates to "fake" and
how much weight it deserves; measurements that do not separate the classes get
(close to) zero weight.

Face landmarks: dlib 81-point predictor (68 standard points + 13 forehead points).
Not measured, because there is no reliable method for them here: ears and cast
shadows. They are reported as unavailable.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANDMARK_MODEL = os.path.join(HERE, "trained_models", "effort", "shape_predictor_81_face_landmarks.dat")
MAX_SIDE = 800

GROUPS = {
    "face": ["prop_eye_face_width", "prop_nose_len", "prop_mouth_width", "prop_face_height", "jaw_symmetry",
             "boundary_grad_ratio", "boundary_color_jump", "hair_boundary_grad_ratio"],
    "eyes": ["eye_ear_asym", "eye_size_asym", "iris_color_diff", "eye_sharpness_asym", "catchlight_offset_diff",
             "brow_asym"],
    "nose_mouth": ["nose_midline_dev", "mouth_corner_asym", "lip_sharpness_rel", "teeth_detail"],
    "skin_texture": ["skin_hf_energy", "skin_noise", "skin_fft_hf", "face_bg_sharpness_ratio", "face_bg_noise_ratio"],
    "background": ["bg_noise_cv", "bg_texture_entropy_cv", "bg_sharpness_cv", "edge_density", "edge_contrast"],
    "lighting": ["face_bg_light_diff", "illumination_cv", "saturation_cv"],
    "frequency": ["fft_hf_ratio", "fft_radial_peak", "blockiness", "blockiness_face_bg_diff", "channel_noise_corr",
                  "residual_kurtosis"],
}
ALL_FEATURES = [f for fs in GROUPS.values() for f in fs]
NOT_MEASURED = {"ears": "no reliable ear landmarks in the 81-point model",
                "shadows": "no reliable cast-shadow estimation implemented"}


class Landmarker:
    def __init__(self, path: str = LANDMARK_MODEL):
        import dlib

        self.dlib = dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(path)

    def __call__(self, rgb: np.ndarray):
        # Detect on a <=640 px copy (HOG cost grows with area); upsample only small images.
        h, w = rgb.shape[:2]
        s = min(1.0, 640.0 / max(h, w))
        small = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
        faces = self.detector(small, 0)
        if not len(faces) and max(small.shape[:2]) <= 400:
            faces = self.detector(small, 1)
        if not len(faces):
            return None
        face = max(faces, key=lambda r: r.width() * r.height())
        if s < 1:
            face = self.dlib.rectangle(int(face.left() / s), int(face.top() / s),
                                       int(face.right() / s), int(face.bottom() / s))
        shape = self.predictor(rgb, face)
        return np.array([[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)], dtype=np.float32)


# ---------------------------------------------------------------- helpers
def _kernel(k):
    k = max(1, int(k))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))


def _poly_mask(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 255)
    return m


def _hull_mask(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(pts).astype(np.int32)), 255)
    return m


def _vals(arr, mask, min_px=30):
    if mask is None:
        return None
    v = arr[mask > 0]
    return v if v.shape[0] >= min_px else None


def _lapvar(lap, mask):
    v = _vals(lap, mask)
    return None if v is None else float(v.var())


def _mad_noise(resid, mask):
    v = _vals(resid, mask)
    return None if v is None else float(np.median(np.abs(v - np.median(v))) * 1.4826)


def _hf_ratio(patch):
    if patch is None or min(patch.shape[:2]) < 16:
        return None
    p = patch.astype(np.float32) - patch.mean()
    win = np.outer(np.hanning(p.shape[0]), np.hanning(p.shape[1]))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(p * win))) ** 2
    h, w = spec.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    total = spec[r > 0.02].sum()
    return float(spec[r > 0.5].sum() / (total + 1e-9))


def _blockiness(gray):
    if gray is None or gray.shape[0] < 32 or gray.shape[1] < 32:
        return None
    g = gray.astype(np.float32)
    dh = np.abs(np.diff(g, axis=1))
    cols = np.arange(dh.shape[1])
    dv = np.abs(np.diff(g, axis=0))
    rows = np.arange(dv.shape[0])
    at = dh[:, cols % 8 == 7].mean() + dv[rows % 8 == 7, :].mean()
    off = dh[:, cols % 8 != 7].mean() + dv[rows % 8 != 7, :].mean()
    return float(at / (off + 1e-6))


def _radial_peak(grayf):
    h, w = grayf.shape
    if min(h, w) >= 256:
        y0, x0 = (h - 256) // 2, (w - 256) // 2
        g = grayf[y0:y0 + 256, x0:x0 + 256]
    else:
        g = cv2.resize(grayf, (256, 256), interpolation=cv2.INTER_AREA)
    g = g - g.mean()
    win = np.outer(np.hanning(256), np.hanning(256))
    spec = np.log(np.abs(np.fft.fftshift(np.fft.fft2(g * win))) + 1.0)
    yy, xx = np.mgrid[:256, :256]
    r = np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2).astype(np.int32)
    prof = (np.bincount(r.ravel(), spec.ravel()) / np.maximum(np.bincount(r.ravel()), 1))[:128]
    smooth = np.convolve(prof, np.ones(9) / 9, mode="same")
    return float((prof - smooth)[30:120].max())


def _channel_noise_corr(img):
    res = [c.astype(np.float32) - cv2.GaussianBlur(c.astype(np.float32), (0, 0), 1.0) for c in cv2.split(img)]
    sub = [r[::2, ::2].ravel() for r in res]
    cs = []
    for a, b in ((0, 1), (1, 2), (0, 2)):
        if sub[a].std() > 1e-6 and sub[b].std() > 1e-6:
            cs.append(float(np.corrcoef(sub[a], sub[b])[0, 1]))
    return float(np.mean(cs)) if cs else None


def _kurtosis(x):
    x = x.astype(np.float64)
    x = x - x.mean()
    v = (x ** 2).mean()
    return float((x ** 4).mean() / (v ** 2 + 1e-12)) if v > 1e-9 else None


def _lateral_light(L, mask, M, u):
    ys, xs = np.nonzero(mask)
    if xs.size < 60:
        return None
    lat = (xs - M[0]) * u[0] + (ys - M[1]) * u[1]
    vals = L[ys, xs]
    left, right = vals[lat < 0], vals[lat > 0]
    if left.size < 20 or right.size < 20:
        return None
    return float((right.mean() - left.mean()) / (right.mean() + left.mean() + 1e-6))


def _catchlight(gray, mask, center, width, u, v):
    ys, xs = np.nonzero(mask)
    if xs.size < 20 or width < 8:
        return None
    vals = gray[ys, xs].astype(np.float32)
    if vals.max() - np.median(vals) < 20:  # no visible specular highlight
        return None
    sel = vals >= np.percentile(vals, 95)
    if sel.sum() < 2:
        return None
    c = np.array([xs[sel].mean(), ys[sel].mean()], np.float32)
    d = (c - center) / width
    return np.array([np.dot(d, u), np.dot(d, v)], np.float32)


# ---------------------------------------------------------------- face measurements
def _face_features(img, gray, grayf, lap, resid, lab, pts):
    out, masks = {}, {}
    eR, eL = pts[36:42], pts[42:48]
    cR, cL = eR.mean(0), eL.mean(0)
    D = float(np.linalg.norm(cL - cR))
    if D < 18:  # face too small for landmark-level measurements
        return out, masks
    u = (cL - cR) / D
    v = np.array([-u[1], u[0]], np.float32)
    M = (cR + cL) / 2
    lat = lambda p: float(np.dot(p - M, u))
    shape = img.shape

    def ear(e):
        return (np.linalg.norm(e[1] - e[5]) + np.linalg.norm(e[2] - e[4])) / (2 * np.linalg.norm(e[0] - e[3]) + 1e-6)

    earR, earL = ear(eR), ear(eL)
    out["eye_ear_asym"] = float(abs(earR - earL) / ((earR + earL) / 2 + 1e-6))
    aR, aL = abs(cv2.contourArea(eR)), abs(cv2.contourArea(eL))
    out["eye_size_asym"] = float(abs(aR - aL) / ((aR + aL) / 2 + 1e-6))
    mR, mL = _poly_mask(shape, eR), _poly_mask(shape, eL)
    labR, labL = _vals(lab, mR, 15), _vals(lab, mL, 15)
    if labR is not None and labL is not None:
        out["iris_color_diff"] = float(np.linalg.norm(labR.mean(0) - labL.mean(0)) / 100.0)
    lvR, lvL = _lapvar(lap, mR), _lapvar(lap, mL)
    if lvR is not None and lvL is not None:
        out["eye_sharpness_asym"] = float(abs(np.log(lvR + 1) - np.log(lvL + 1)))
    wR, wL = float(np.linalg.norm(eR[0] - eR[3])), float(np.linalg.norm(eL[0] - eL[3]))
    clR, clL = _catchlight(gray, mR, cR, wR, u, v), _catchlight(gray, mL, cL, wL, u, v)
    if clR is not None and clL is not None:
        out["catchlight_offset_diff"] = float(np.linalg.norm(clR - clL))
    bR, bL = pts[17:22].mean(0), pts[22:27].mean(0)
    hR, hL = float(np.dot(cR - bR, v)), float(np.dot(cL - bL, v))
    out["brow_asym"] = float(abs(hR - hL) / D)

    out["nose_midline_dev"] = float(abs(lat(pts[30])) / D)
    out["mouth_corner_asym"] = float(abs(lat(pts[48]) + lat(pts[54])) / D)
    mouth = _poly_mask(shape, pts[48:60])
    cheekR = _hull_mask(shape, pts[[1, 2, 3, 4, 48, 31, 41]])
    cheekL = _hull_mask(shape, pts[[15, 14, 13, 12, 54, 35, 46]])
    skin = cv2.bitwise_or(cheekR, cheekL)
    for m in (mR, mL, mouth):
        skin = cv2.bitwise_and(skin, cv2.bitwise_not(cv2.dilate(m, _kernel(0.05 * D))))
    lv_skin, lv_mouth = _lapvar(lap, skin), _lapvar(lap, mouth)
    if lv_skin is not None and lv_mouth is not None:
        out["lip_sharpness_rel"] = float(np.log(lv_mouth + 1) - np.log(lv_skin + 1))
    if np.linalg.norm(pts[62] - pts[66]) / D > 0.08:
        lv_inner = _lapvar(lap, _poly_mask(shape, pts[60:68]))
        if lv_inner is not None and lv_skin is not None:
            out["teeth_detail"] = float(np.log(lv_inner + 1) - np.log(lv_skin + 1))

    fw = float(np.linalg.norm(pts[16] - pts[0]))
    out["prop_eye_face_width"] = D / (fw + 1e-6)
    out["prop_nose_len"] = float(np.linalg.norm(pts[27] - pts[33]) / D)
    out["prop_mouth_width"] = float(np.linalg.norm(pts[48] - pts[54]) / D)
    out["prop_face_height"] = float(np.linalg.norm(pts[27] - pts[8]) / D)
    out["jaw_symmetry"] = float(np.mean([abs(lat(pts[i]) + lat(pts[16 - i])) for i in range(8)]) / D)

    if lv_skin is not None:
        out["skin_hf_energy"] = float(np.log(lv_skin + 1))
    ns = _mad_noise(resid, skin)
    if ns is not None:
        out["skin_noise"] = float(np.log(ns + 1e-3))
    ys, xs = np.nonzero(cheekR)
    if xs.size:
        half = max(8, int(0.18 * D))
        cx, cy = int(xs.mean()), int(ys.mean())
        patch = grayf[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
        out_hf = _hf_ratio(patch)
        if out_hf is not None:
            out["skin_fft_hf"] = out_hf

    hull_pts = np.concatenate([pts[0:17], pts[68:81]]) if pts.shape[0] >= 81 else pts[0:27]
    H = _hull_mask(shape, hull_pts)
    masks["face"] = H
    k_ring, k_band, k_in = 0.25 * D, 0.06 * D, 0.2 * D
    ring = cv2.bitwise_and(cv2.dilate(H, _kernel(k_ring)), cv2.bitwise_not(H))
    ring_ok = np.count_nonzero(ring) >= 0.15 * np.count_nonzero(H)
    interior = cv2.erode(H, _kernel(k_in))
    if ring_ok:
        a, b = _lapvar(lap, interior), _lapvar(lap, ring)
        if a is not None and b is not None:
            out["face_bg_sharpness_ratio"] = float(np.log(a + 1) - np.log(b + 1))
        na, nb = _mad_noise(resid, interior), _mad_noise(resid, ring)
        if na is not None and nb is not None:
            out["face_bg_noise_ratio"] = float(np.log(na + 1e-3) - np.log(nb + 1e-3))
    gx = cv2.Sobel(grayf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grayf, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)
    band = cv2.bitwise_and(cv2.dilate(H, _kernel(k_band)), cv2.bitwise_not(cv2.erode(H, _kernel(k_band))))
    inner_band = cv2.bitwise_and(cv2.erode(H, _kernel(k_band)), cv2.bitwise_not(cv2.erode(H, _kernel(k_in))))
    gb, gi = _vals(gmag, band), _vals(gmag, inner_band)
    if gb is not None and gi is not None:
        out["boundary_grad_ratio"] = float(gb.mean() / (gi.mean() + 1e-6))
        top = band.copy()
        top[int(max(0, pts[17:27, 1].min())):, :] = 0
        gt = _vals(gmag, top)
        if gt is not None:
            out["hair_boundary_grad_ratio"] = float(gt.mean() / (gi.mean() + 1e-6))
    outer_thin = cv2.bitwise_and(cv2.dilate(H, _kernel(k_band)), cv2.bitwise_not(H))
    inner_thin = cv2.bitwise_and(H, cv2.bitwise_not(cv2.erode(H, _kernel(k_band))))
    lo, li = _vals(lab, outer_thin), _vals(lab, inner_thin)
    if lo is not None and li is not None:
        out["boundary_color_jump"] = float(np.linalg.norm(lo.mean(0) - li.mean(0)) / 100.0)
    x, y, wb, hb = cv2.boundingRect(H)
    bf, bgb = _blockiness(gray[y:y + hb, x:x + wb]), _blockiness(gray)
    if bf is not None and bgb is not None:
        out["blockiness_face_bg_diff"] = float(bf - bgb)
    if ring_ok:
        gF, gB = _lateral_light(lab[:, :, 0], skin, M, u), _lateral_light(lab[:, :, 0], ring, M, u)
        if gF is not None and gB is not None:
            out["face_bg_light_diff"] = float(abs(gF - gB))
    masks["ring"] = cv2.dilate(H, _kernel(k_ring))
    return out, masks


# ---------------------------------------------------------------- whole-image measurements
def _global_features(img, gray, grayf, lap, resid, lab, exclude):
    out = {}
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    L = lab[:, :, 0]
    gh, gw = gray.shape
    n = 8
    bh, bw = gh // n, gw // n
    if bh >= 12 and bw >= 12:
        noise, ent, sharp, lum, satm = [], [], [], [], []
        for i in range(n):
            for j in range(n):
                ys, xs = slice(i * bh, (i + 1) * bh), slice(j * bw, (j + 1) * bw)
                if exclude is not None and (exclude[ys, xs] > 0).mean() > 0.3:
                    continue
                r = resid[ys, xs]
                noise.append(float(np.median(np.abs(r - np.median(r))) * 1.4826))
                hist = np.histogram(gray[ys, xs], bins=32, range=(0, 256))[0].astype(np.float64)
                p = hist / max(hist.sum(), 1)
                p = p[p > 0]
                ent.append(float(-(p * np.log2(p)).sum()))
                sharp.append(float(np.log(lap[ys, xs].var() + 1)))
                lum.append(float(L[ys, xs].mean()))
                satm.append(float(sat[ys, xs].mean()))
        if len(noise) >= 8:
            cv = lambda a: float(np.std(a) / (np.mean(a) + 1e-6))
            out.update({"bg_noise_cv": cv(noise), "bg_texture_entropy_cv": cv(ent), "bg_sharpness_cv": float(np.std(sharp)),
                        "illumination_cv": cv(lum), "saturation_cv": cv(satm)})
    edges = cv2.Canny(gray, 50, 150)
    gx = cv2.Sobel(grayf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grayf, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)
    out["edge_density"] = float(np.count_nonzero(edges) / edges.size)
    if np.count_nonzero(edges) > 50:
        out["edge_contrast"] = float(gmag[edges > 0].mean() / (gmag.mean() + 1e-6))
    h, w = grayf.shape
    s = min(h, w)
    out["fft_hf_ratio"] = _hf_ratio(grayf[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s])
    out["fft_radial_peak"] = _radial_peak(grayf)
    out["blockiness"] = _blockiness(gray)
    out["channel_noise_corr"] = _channel_noise_corr(img)
    out["residual_kurtosis"] = _kurtosis(resid[::2, ::2].ravel())
    return out


def analyze(img_bgr: np.ndarray, landmarker: "Landmarker | None") -> dict:
    """Returns {"features": {name: float|None}, "landmarks_found": bool, "analysis_size": [w, h]}."""
    feats = {k: None for k in ALL_FEATURES}
    h, w = img_bgr.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    img = cv2.resize(img_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayf = gray.astype(np.float32)
    lap = cv2.Laplacian(grayf, cv2.CV_32F, ksize=3)
    resid = grayf - cv2.GaussianBlur(grayf, (0, 0), 1.0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    pts = None
    if landmarker is not None:
        try:
            pts = landmarker(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception:
            pts = None
    exclude = None
    if pts is not None and pts.shape[0] >= 68:
        face_out, masks = _face_features(img, gray, grayf, lap, resid, lab, pts)
        feats.update(face_out)
        exclude = masks.get("ring")
    feats.update(_global_features(img, gray, grayf, lap, resid, lab, exclude))
    clean = {}
    for k, v in feats.items():
        clean[k] = None if v is None or not np.isfinite(v) else round(float(v), 6)
    return {"features": clean, "landmarks_found": pts is not None, "analysis_size": [img.shape[1], img.shape[0]]}
