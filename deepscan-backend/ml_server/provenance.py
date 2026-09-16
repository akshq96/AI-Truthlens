"""
Metadata / provenance analysis: SUPPORTING evidence only.

Hard rules encoded here, because getting them wrong is worse than having no
metadata analysis at all:

  * Missing metadata is NOT evidence of fakery. Screenshots, social-media
    re-uploads, messaging apps and ordinary re-encoding all strip EXIF from
    perfectly genuine photos.
  * Present camera metadata is NOT proof of authenticity. EXIF is trivially
    editable and can be copied wholesale from a real photo onto a synthetic one.

So this module produces a *reliability* score describing how coherent and
plausible the metadata is, plus any explicit AI-generation declarations. The
decision engine may use it to firm up or soften a verdict, never to originate
one.

C2PA (Content Credentials) is different in kind: it is a cryptographically
signed provenance manifest, so a verified claim is much stronger evidence than
ordinary EXIF. Detection here follows the C2PA technical specification's
container rules — the manifest store is a JUMBF superbox labelled "c2pa",
carried in JPEG APP11 (0xFFEB) segments, or in a PNG 'caBX' ancillary chunk.

IMPORTANT HONESTY CONSTRAINT: cryptographic verification requires the official
c2pa library, which does not import on this interpreter (Python 3.9). We
therefore report presence but NEVER claim `verified: true` on our own. The
verify hook is wired so enabling the real library later upgrades the result
without touching the pipeline.
"""


import re
from datetime import datetime, timezone

from PIL import Image, ExifTags

try:
    import exifread
except Exception:
    exifread = None

try:  # official verifier; absent on py3.9
    import c2pa
except Exception:
    c2pa = None

# Substrings that indicate generative-AI or heavy editing provenance when they
# appear in software/tool fields. Presence is a positive signal of AI origin;
# absence means nothing.
AI_SOFTWARE_MARKERS = (
    "midjourney", "stable diffusion", "stablediffusion", "dall-e", "dalle",
    "openai", "gpt-image", "firefly", "adobe firefly", "imagen", "gemini",
    "flux", "leonardo.ai", "playground", "nightcafe", "novelai",
    "automatic1111", "comfyui", "invokeai", "craiyon", "runway", "sora",
    "generative", "ai-generated", "ai generated", "seedream", "ideogram",
)
EDITOR_SOFTWARE_MARKERS = (
    "photoshop", "lightroom", "gimp", "affinity", "capture one", "luminar",
    "snapseed", "vsco", "facetune", "picsart", "canva",
)
CAMERA_EXIF_KEYS = ("Make", "Model", "LensModel", "LensMake", "FNumber",
                    "ExposureTime", "ISOSpeedRatings", "FocalLength")


def _decode_exif(img):
    out = {}
    try:
        raw = img.getexif()
    except Exception:
        return out
    if not raw:
        return out
    for tag_id, value in raw.items():
        name = ExifTags.TAGS.get(tag_id, str(tag_id))
        try:
            out[name] = value.decode("utf-8", "replace").strip("\x00").strip() if isinstance(value, bytes) else value
        except Exception:
            out[name] = str(value)
    # IFD blocks (Exif/GPS) carry the interesting capture fields
    for ifd_name, ifd_id in (("Exif", 0x8769), ("GPS", 0x8825)):
        try:
            sub = raw.get_ifd(ifd_id)
        except Exception:
            continue
        table = ExifTags.GPSTAGS if ifd_name == "GPS" else ExifTags.TAGS
        for tag_id, value in (sub or {}).items():
            name = table.get(tag_id, f"{ifd_name}:{tag_id}")
            try:
                out[name] = value.decode("utf-8", "replace").strip("\x00").strip() if isinstance(value, bytes) else value
            except Exception:
                out[name] = str(value)
    return out


def _read_xmp(img, path):
    """XMP packet: PIL exposes getxmp() for some formats; fall back to raw scan."""
    xmp = {}
    try:
        got = img.getxmp()
        if got:
            xmp = got
    except Exception:
        pass
    if xmp:
        return xmp, None
    try:
        with open(path, "rb") as fh:
            blob = fh.read(4_000_000)
        m = re.search(br"<x:xmpmeta.*?</x:xmpmeta>", blob, re.S)
        if m:
            return {}, m.group(0).decode("utf-8", "replace")
    except Exception:
        pass
    return {}, None


def detect_c2pa(path: str) -> dict:
    """
    Presence detection per the C2PA spec container rules, plus verification via
    the official library when it is importable.

    status: verified | present_unverified | not_found | error
    """
    result = {"available": True, "verified": False, "status": "not_found",
              "detail": None, "verifier": "presence-scan (c2pa library unavailable)"}

    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
            fh.seek(0)
            blob = fh.read(20_000_000)
    except Exception as exc:
        return {"available": False, "verified": False, "status": "error",
                "detail": f"{type(exc).__name__}: {exc}", "verifier": None}

    found = False
    where = []
    # JPEG: APP11 (0xFFEB) segments carrying JUMBF ('jumb' box) with 'c2pa' label
    if head[:2] == b"\xff\xd8":
        for m in re.finditer(b"\xff\xeb", blob):
            seg = blob[m.start():m.start() + 4096]
            if b"jumb" in seg or b"c2pa" in seg:
                found = True
                where.append("jpeg:APP11/JUMBF")
                break
    # PNG: 'caBX' (manifest store), 'caMs'/'caSt' also indicate C2PA data
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        for chunk in (b"caBX", b"caMs", b"caSt"):
            if chunk in blob:
                found = True
                where.append(f"png:{chunk.decode()}")
                break
    # Generic fallback: a JUMBF superbox labelled c2pa (covers WebP/HEIF/MP4 boxes)
    if not found and (b"jumb" in blob and b"c2pa" in blob):
        found = True
        where.append("jumbf:c2pa-label")

    if found:
        result["status"] = "present_unverified"
        result["detail"] = ("C2PA manifest container detected (" + ", ".join(where) +
                            "). Signature NOT verified: the official c2pa library is "
                            "unavailable on this interpreter, so authenticity of the "
                            "claim itself is unknown.")

    # Real verification path, used automatically once the library is importable.
    if c2pa is not None:
        try:
            reader = c2pa.Reader.from_file(path) if hasattr(c2pa, "Reader") else None
            manifest_json = reader.json() if reader is not None else None
            if manifest_json:
                result.update({"verified": True, "status": "verified",
                               "detail": "C2PA manifest cryptographically verified.",
                               "verifier": f"c2pa-python {getattr(c2pa, '__version__', '?')}",
                               "manifest_excerpt": str(manifest_json)[:2000]})
        except Exception as exc:
            # A signature that fails to validate is meaningful — report it.
            if found:
                result["status"] = "present_invalid"
                result["detail"] = f"C2PA manifest present but verification failed: {exc}"
    return result


def analyze(path: str) -> dict:
    """Collect metadata evidence and score its internal coherence."""
    info = {
        "available": False,
        "camera": None, "software": None, "capture_date": None, "lens": None,
        "gps_present": False,
        "has_exif": False, "has_xmp": False, "has_iptc": False, "has_icc": False,
        "ai_declaration": None,
        "editor_software": None,
        "consistency_score": 0.0,
        "reliability": "none",
        "signals": [],
        "notes": [],
        "format": None, "dimensions": None,
    }

    try:
        img = Image.open(path)
    except Exception as exc:
        info["notes"].append(f"Unreadable image: {type(exc).__name__}: {exc}")
        return info

    info["format"] = img.format
    info["dimensions"] = list(img.size)

    exif = _decode_exif(img)
    info["has_exif"] = bool(exif)
    xmp_dict, xmp_raw = _read_xmp(img, path)
    info["has_xmp"] = bool(xmp_dict or xmp_raw)
    info["has_icc"] = bool(img.info.get("icc_profile"))
    info["has_iptc"] = bool(img.info.get("photoshop") or img.info.get("iptc"))

    make = str(exif.get("Make") or "").strip()
    model = str(exif.get("Model") or "").strip()
    if make or model:
        info["camera"] = " ".join(x for x in (make, model) if x)
    info["lens"] = str(exif.get("LensModel") or "").strip() or None
    software = str(exif.get("Software") or "").strip()
    info["software"] = software or None
    info["capture_date"] = str(exif.get("DateTimeOriginal") or exif.get("DateTime") or "").strip() or None
    info["gps_present"] = any(k.startswith("GPS") for k in exif) or bool(exif.get("GPSInfo"))

    haystack = " ".join(str(v).lower() for v in exif.values())
    if xmp_raw:
        haystack += " " + xmp_raw.lower()
    if xmp_dict:
        haystack += " " + str(xmp_dict).lower()

    ai_hit = next((m for m in AI_SOFTWARE_MARKERS if m in haystack), None)
    if ai_hit:
        info["ai_declaration"] = ai_hit
        info["signals"].append(f"metadata declares AI tooling: '{ai_hit}'")
    editor_hit = next((m for m in EDITOR_SOFTWARE_MARKERS if m in haystack), None)
    if editor_hit:
        info["editor_software"] = editor_hit
        info["signals"].append(f"editing software present: '{editor_hit}'")

    info["available"] = bool(exif or info["has_xmp"] or info["has_icc"] or info["has_iptc"])

    # ---- coherence scoring ------------------------------------------------
    # Rewards a self-consistent capture record. A low score means "no useful
    # corroboration", NOT "fake".
    score, checks = 0.0, []
    cam_fields = sum(1 for k in CAMERA_EXIF_KEYS if exif.get(k) not in (None, ""))
    if cam_fields:
        score += min(0.45, 0.09 * cam_fields)
        checks.append(f"{cam_fields}/{len(CAMERA_EXIF_KEYS)} camera capture fields present")
    if info["capture_date"]:
        score += 0.15
        checks.append("capture timestamp present")
        try:
            dt = datetime.strptime(info["capture_date"][:19], "%Y:%m:%d %H:%M:%S")
            if dt > datetime.now() + (datetime.now() - datetime.now()):
                pass
            if dt.year < 1990 or dt > datetime.now(timezone.utc).replace(tzinfo=None):
                score -= 0.20
                info["notes"].append("capture timestamp implausible (future or pre-1990)")
        except Exception:
            score -= 0.05
            info["notes"].append("capture timestamp unparseable")
    if info["gps_present"]:
        score += 0.12
        checks.append("GPS present")
    if info["has_icc"]:
        score += 0.06
        checks.append("ICC profile present")
    if info["has_xmp"]:
        score += 0.06
        checks.append("XMP present")
    if info["camera"] and not info["capture_date"]:
        score -= 0.08
        info["notes"].append("camera identified but no capture timestamp — internally inconsistent")
    if ai_hit:
        # An explicit AI declaration is coherent provenance, just not of a camera.
        checks.append("explicit AI-generation declaration")

    info["consistency_score"] = round(max(0.0, min(1.0, score)), 3)
    info["signals"].extend(checks)
    if not info["available"]:
        info["reliability"] = "none"
        info["notes"].append("No metadata present. This is common for screenshots, "
                             "social-media downloads and messaging apps, and is NOT "
                             "evidence that the image is synthetic.")
    elif info["consistency_score"] >= 0.55:
        info["reliability"] = "strong"
    elif info["consistency_score"] >= 0.25:
        info["reliability"] = "moderate"
    else:
        info["reliability"] = "weak"

    return info


def analyze_with_c2pa(path: str) -> tuple:
    return analyze(path), detect_c2pa(path)
