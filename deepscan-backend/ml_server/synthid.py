"""
SynthID / provenance signal.

Deliberately a thin, honest interface. Semantics that must not be broken:

  * detected      -> STRONG evidence the content carries a Google SynthID
                     watermark (i.e. Google-AI origin). Only ever returned by a
                     real verifier.
  * not_detected  -> NOT evidence of authenticity. Most AI images carry no
                     watermark at all, and watermarks do not survive heavy
                     re-encoding. Must never push a decision toward REAL.
  * unavailable   -> no verifier configured/reachable; the ML pipeline simply
                     continues unaffected.
  * error         -> verifier was configured but failed.

There is no local/offline SynthID verifier: detection requires Google's
official SynthID Detector. So unless a verifier is explicitly configured this
module reports `unavailable` and never guesses. Do not "approximate" SynthID by
inspecting pixels or metadata — that would fabricate a provenance claim.
"""

import os

STATUS_DETECTED = "detected"
STATUS_NOT_DETECTED = "not_detected"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ERROR = "error"

# Set DEEPSCAN_SYNTHID_ENDPOINT (+ DEEPSCAN_SYNTHID_API_KEY) to a real verifier
# to turn this on. Left unset, the pipeline behaves exactly as if SynthID did
# not exist.
SYNTHID_ENDPOINT = os.environ.get("DEEPSCAN_SYNTHID_ENDPOINT", "").strip()
SYNTHID_API_KEY = os.environ.get("DEEPSCAN_SYNTHID_API_KEY", "").strip()
SYNTHID_TIMEOUT_S = float(os.environ.get("DEEPSCAN_SYNTHID_TIMEOUT_S", "10"))


def is_configured() -> bool:
    return bool(SYNTHID_ENDPOINT)


def check_image(image_path: str) -> dict:
    """
    Returns {"available": bool, "status": str, "detail": Optional[str]}.

    `available` reports whether a verifier answered — not whether a watermark
    was found.
    """
    if not is_configured():
        return {
            "available": False,
            "status": STATUS_UNAVAILABLE,
            "detail": "No SynthID verifier configured (set DEEPSCAN_SYNTHID_ENDPOINT).",
        }

    try:
        import requests  # imported lazily so the dep is optional
    except Exception:
        return {
            "available": False,
            "status": STATUS_ERROR,
            "detail": "requests not installed; cannot reach SynthID verifier.",
        }

    try:
        headers = {"Authorization": f"Bearer {SYNTHID_API_KEY}"} if SYNTHID_API_KEY else {}
        with open(image_path, "rb") as fh:
            resp = requests.post(
                SYNTHID_ENDPOINT,
                files={"file": fh},
                headers=headers,
                timeout=SYNTHID_TIMEOUT_S,
            )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return {
            "available": False,
            "status": STATUS_ERROR,
            "detail": f"SynthID verifier call failed: {type(exc).__name__}: {exc}",
        }

    # Only an explicit positive from the verifier counts as detected.
    detected = bool(payload.get("watermark_detected", payload.get("detected", False)))
    return {
        "available": True,
        "status": STATUS_DETECTED if detected else STATUS_NOT_DETECTED,
        "detail": payload.get("detail"),
    }
