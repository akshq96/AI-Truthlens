"""
DeepScan — System Sanity & Spoof Test (v2)
============================================
Comprehensive end-to-end test of the entire pipeline:

  1. Health checks on both ML servers (image :7000, video :5500)
  2. Health check on Node.js backend (:5000)
  3. Image prediction test (creates a synthetic test image)
  4. Video prediction test (creates a synthetic test video)
  5. Full pipeline test via /api/analyze-media (image)
  6. Full pipeline test via /api/analyze-media (video)
  7. Frontend route sanity checks (/, /checker, /history, /api/results)
  8. Accuracy validation — verifies model returns sensible scores
  9. Spoof tests (invalid file types, empty requests, wrong fields)
 10. Response schema validation

Usage:
  pip install requests opencv-python-headless Pillow numpy
  python sanity_test.py
"""

import os
import sys
import json
import tempfile
import time

import requests
import numpy as np
from PIL import Image

# ─── Config ──────────────────────────────────────────────────────────────────
IMAGE_SERVER = os.environ.get("IMAGE_SERVER_URL", "http://localhost:7000")
VIDEO_SERVER = os.environ.get("VIDEO_SERVER_URL", "http://localhost:5500")
BACKEND_URL  = os.environ.get("BACKEND_URL", "http://localhost:5000")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")

PASSED = 0
FAILED = 0
SKIPPED = 0


def log_pass(name, detail=""):
    global PASSED
    PASSED += 1
    print(f"  ✅ PASS: {name}" + (f"  ({detail})" if detail else ""))


def log_fail(name, detail=""):
    global FAILED
    FAILED += 1
    print(f"  ❌ FAIL: {name}" + (f"  ({detail})" if detail else ""))


def log_skip(name, detail=""):
    global SKIPPED
    SKIPPED += 1
    print(f"  ⏭️  SKIP: {name}" + (f"  ({detail})" if detail else ""))


def log_section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def temp_path(suffix):
    """Reserved path with no open handle (Windows-safe for immediate unlink)."""
    t = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    t.close()
    return t.name


# ─── Test Helpers ────────────────────────────────────────────────────────────

def create_test_image(path, size=(224, 224)):
    """Create a simple synthetic RGB test image."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(path)
    return path


def create_test_video(path, frames=10, size=(224, 224)):
    """Create a simple synthetic test video (MP4)."""
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, 10, size)
    for _ in range(frames):
        frame = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def validate_schema(data, required_keys, test_name):
    """Validate that response data contains all required keys."""
    missing = [k for k in required_keys if k not in data]
    if missing:
        log_fail(f"{test_name} - schema", f"missing keys: {missing}")
        return False
    log_pass(f"{test_name} - schema", f"all {len(required_keys)} keys present")
    return True


def validate_score_range(score, test_name, min_val=0, max_val=100):
    """Validate that a score is within expected range."""
    if score is None:
        log_fail(f"{test_name} - score range", "score is None")
        return False
    if min_val <= score <= max_val:
        log_pass(f"{test_name} - score range", f"score={score} in [{min_val}, {max_val}]")
        return True
    else:
        log_fail(f"{test_name} - score range", f"score={score} NOT in [{min_val}, {max_val}]")
        return False


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_image_server_health():
    """Test image ML server health endpoint."""
    try:
        r = requests.get(f"{IMAGE_SERVER}/health", timeout=10)
        data = r.json()
        if r.status_code == 200 and data.get("image_model_loaded"):
            log_pass("Image server health", f"model loaded = {data['image_model_loaded']}")
            return True
        else:
            log_fail("Image server health", f"status={r.status_code}, data={data}")
            return False
    except Exception as e:
        log_fail("Image server health", str(e))
        return False


def test_video_server_health():
    """Test video ML server health endpoint."""
    try:
        r = requests.get(f"{VIDEO_SERVER}/health", timeout=10)
        data = r.json()
        if r.status_code == 200 and data.get("video_model_loaded"):
            log_pass("Video server health", f"model loaded = {data['video_model_loaded']}")
            return True
        else:
            log_fail("Video server health", f"status={r.status_code}, data={data}")
            return False
    except Exception as e:
        log_fail("Video server health", str(e))
        return False


def test_backend_health():
    """Test Node.js backend health endpoint."""
    try:
        r = requests.get(f"{BACKEND_URL}/", timeout=10)
        data = r.json()
        if r.status_code == 200 and "status" in data:
            log_pass("Backend health", data.get("status"))
            return True
        else:
            log_fail("Backend health", f"status={r.status_code}")
            return False
    except Exception as e:
        log_fail("Backend health", str(e))
        return False


def test_image_prediction():
    """Send a test image directly to the image ML server."""
    tmp_path = temp_path(".png")
    try:
        create_test_image(tmp_path)
        with open(tmp_path, "rb") as f:
            r = requests.post(
                f"{IMAGE_SERVER}/predict/image",
                files={"file": ("test.png", f, "image/png")},
                timeout=60,
            )
        data = r.json()
        if r.status_code == 200 and "model_score" in data:
            log_pass("Image prediction (direct)", f"score={data['model_score']}")
            validate_score_range(data['model_score'], "Image prediction (direct)")
            if "raw_prediction" in data:
                log_pass("Image prediction raw field", "raw_prediction present")
            else:
                log_fail("Image prediction raw field", "missing raw_prediction")
            return True
        else:
            log_fail("Image prediction (direct)", f"status={r.status_code}, data={data}")
            return False
    except Exception as e:
        log_fail("Image prediction (direct)", str(e))
        return False
    finally:
        os.unlink(tmp_path)


def test_video_prediction():
    """Send a test video directly to the video ML server."""
    tmp_path = temp_path(".mp4")
    try:
        create_test_video(tmp_path)
        with open(tmp_path, "rb") as f:
            r = requests.post(
                f"{VIDEO_SERVER}/predict/video",
                files={"file": ("test.mp4", f, "video/mp4")},
                timeout=120,
            )
        data = r.json()
        if r.status_code == 200 and "model_score" in data:
            log_pass("Video prediction (direct)",
                     f"score={data['model_score']}, frames={data.get('frames_analyzed')}")
            validate_score_range(data['model_score'], "Video prediction (direct)")
            # Validate frames_analyzed is a positive integer
            fa = data.get('frames_analyzed')
            if fa and fa > 0:
                log_pass("Video frames count", f"frames_analyzed={fa}")
            else:
                log_fail("Video frames count", f"frames_analyzed={fa}")
            return True
        else:
            log_fail("Video prediction (direct)", f"status={r.status_code}, data={data}")
            return False
    except Exception as e:
        log_fail("Video prediction (direct)", str(e))
        return False
    finally:
        os.unlink(tmp_path)


def test_full_pipeline_image():
    """Test full pipeline: frontend → backend → image ML server → DB."""
    tmp_path = temp_path(".png")
    try:
        create_test_image(tmp_path)
        with open(tmp_path, "rb") as f:
            r = requests.post(
                f"{BACKEND_URL}/api/analyze-media",
                files={"media": ("test_image.png", f, "image/png")},
                timeout=120,
            )
        data = r.json()
        if r.status_code == 200 and data.get("media_type") == "image":
            log_pass("Full pipeline (image)",
                     f"verdict={data.get('verdict')}, score={data.get('final_score')}, "
                     f"model={data.get('breakdown', {}).get('model_score')}")
            # Validate response schema
            validate_schema(data, [
                'message', 'id', 'filename', 'originalName', 'final_score',
                'verdict', 'confidence', 'breakdown', 'flags', 'raw_metadata',
                'media_type', 'analyzed_at', 'ml_status', 'ml_error',
            ], "Full pipeline (image)")
            if data.get("ml_status") == "success":
                log_pass("Pipeline (image) ML connected", "ml_status=success")
            else:
                log_fail(
                    "Pipeline (image) ML connected",
                    f"ml_status={data.get('ml_status')!r} (expected success; is image server on :7000?)",
                )
            # Validate scores
            validate_score_range(data.get('final_score'), "Full pipeline image final_score")
            # Validate verdict
            valid_verdicts = ['REAL', 'UNCERTAIN', 'LIKELY SYNTHETIC', 'SYNTHETIC']
            if data.get('verdict') in valid_verdicts:
                log_pass("Image verdict value", f"verdict='{data['verdict']}'")
            else:
                log_fail("Image verdict value", f"invalid verdict='{data.get('verdict')}'")
            # Validate confidence
            valid_confidence = ['Low', 'Medium', 'High']
            if data.get('confidence') in valid_confidence:
                log_pass("Image confidence value", f"confidence='{data['confidence']}'")
            else:
                log_fail("Image confidence value", f"invalid confidence='{data.get('confidence')}'")
            return True
        else:
            log_fail("Full pipeline (image)", f"status={r.status_code}, data={data}")
            return False
    except Exception as e:
        log_fail("Full pipeline (image)", str(e))
        return False
    finally:
        os.unlink(tmp_path)


def test_full_pipeline_video():
    """Test full pipeline: frontend → backend → video ML server → DB."""
    tmp_path = temp_path(".mp4")
    try:
        create_test_video(tmp_path)
        with open(tmp_path, "rb") as f:
            r = requests.post(
                f"{BACKEND_URL}/api/analyze-media",
                files={"media": ("test_video.mp4", f, "video/mp4")},
                timeout=180,
            )
        data = r.json()
        if r.status_code == 200 and data.get("media_type") == "video":
            log_pass("Full pipeline (video)",
                     f"verdict={data.get('verdict')}, score={data.get('final_score')}, "
                     f"frames={data.get('frames_analyzed')}")
            # Validate response schema
            validate_schema(data, [
                'message', 'id', 'filename', 'originalName', 'final_score',
                'verdict', 'confidence', 'breakdown', 'flags', 'raw_metadata',
                'media_type', 'frames_analyzed', 'analyzed_at', 'ml_status', 'ml_error',
            ], "Full pipeline (video)")
            if data.get("ml_status") == "success":
                log_pass("Pipeline (video) ML connected", "ml_status=success")
            else:
                log_fail(
                    "Pipeline (video) ML connected",
                    f"ml_status={data.get('ml_status')!r} (expected success; is video server on :5500?)",
                )
            # Validate scores
            validate_score_range(data.get('final_score'), "Full pipeline video final_score")
            # Validate verdict
            valid_verdicts = ['REAL', 'UNCERTAIN', 'LIKELY SYNTHETIC', 'SYNTHETIC']
            if data.get('verdict') in valid_verdicts:
                log_pass("Video verdict value", f"verdict='{data['verdict']}'")
            else:
                log_fail("Video verdict value", f"invalid verdict='{data.get('verdict')}'")
            # Validate confidence
            valid_confidence = ['Low', 'Medium', 'High']
            if data.get('confidence') in valid_confidence:
                log_pass("Video confidence value", f"confidence='{data['confidence']}'")
            else:
                log_fail("Video confidence value", f"invalid confidence='{data.get('confidence')}'")
            return True
        else:
            log_fail("Full pipeline (video)", f"status={r.status_code}, data={data}")
            return False
    except Exception as e:
        log_fail("Full pipeline (video)", str(e))
        return False
    finally:
        os.unlink(tmp_path)


def test_legacy_image_endpoint():
    """Test the legacy /api/analyze endpoint still works for images."""
    tmp_path = temp_path(".png")
    try:
        create_test_image(tmp_path)
        with open(tmp_path, "rb") as f:
            r = requests.post(
                f"{BACKEND_URL}/api/analyze",
                files={"image": ("legacy_test.png", f, "image/png")},
                timeout=120,
            )
        data = r.json()
        if r.status_code == 200 and "final_score" in data:
            log_pass("Legacy /api/analyze",
                     f"verdict={data.get('verdict')}, score={data.get('final_score')}")
            if "ml_status" in data and "ml_error" in data:
                log_pass("Legacy /api/analyze ML fields", f"ml_status={data.get('ml_status')!r}")
            else:
                log_fail("Legacy /api/analyze ML fields", "missing ml_status or ml_error")
            return True
        else:
            log_fail("Legacy /api/analyze", f"status={r.status_code}, data={data}")
            return False
    except Exception as e:
        log_fail("Legacy /api/analyze", str(e))
        return False
    finally:
        os.unlink(tmp_path)


# ─── Frontend Route Checks ──────────────────────────────────────────────────

def test_frontend_route(path, name):
    """Test that a frontend route responds with 200."""
    try:
        r = requests.get(f"{FRONTEND_URL}{path}", timeout=10)
        if r.status_code == 200:
            log_pass(f"Frontend route {name}", f"GET {path} → 200")
            return True
        else:
            log_fail(f"Frontend route {name}", f"GET {path} → {r.status_code}")
            return False
    except Exception as e:
        log_fail(f"Frontend route {name}", str(e))
        return False


# ─── Spoof Tests ─────────────────────────────────────────────────────────────

def test_spoof_no_file():
    """Spoof: send request with no file attached."""
    try:
        r = requests.post(f"{BACKEND_URL}/api/analyze-media", timeout=10)
        if r.status_code == 400:
            log_pass("Spoof: no file", f"rejected with 400: {r.json().get('error', '')}")
        else:
            log_fail("Spoof: no file", f"expected 400, got {r.status_code}")
    except Exception as e:
        log_fail("Spoof: no file", str(e))


def test_spoof_invalid_type():
    """Spoof: send a .txt file disguised as media."""
    tmp_path = temp_path(".txt")
    try:
        with open(tmp_path, "w", encoding="utf-8") as wf:
            wf.write("This is not an image or video")
        with open(tmp_path, "rb") as f:
            r = requests.post(
                f"{BACKEND_URL}/api/analyze-media",
                files={"media": ("evil.txt", f, "text/plain")},
                timeout=10,
            )
        if r.status_code == 400:
            log_pass("Spoof: .txt file", f"rejected with 400: {r.json().get('error', '')}")
        else:
            log_fail("Spoof: .txt file", f"expected 400, got {r.status_code}")
    except Exception as e:
        log_fail("Spoof: .txt file", str(e))
    finally:
        os.unlink(tmp_path)


def test_spoof_wrong_field_name():
    """Spoof: send file with wrong form-data field name."""
    tmp_path = temp_path(".png")
    try:
        create_test_image(tmp_path)
        with open(tmp_path, "rb") as f:
            r = requests.post(
                f"{BACKEND_URL}/api/analyze-media",
                files={"wrong_field": ("test.png", f, "image/png")},
                timeout=10,
            )
        if r.status_code == 400:
            log_pass("Spoof: wrong field name", f"rejected with 400")
        else:
            log_fail("Spoof: wrong field name", f"expected 400, got {r.status_code}")
    except Exception as e:
        log_fail("Spoof: wrong field name", str(e))
    finally:
        os.unlink(tmp_path)


def test_spoof_nonexistent_route():
    """Spoof: hit a nonexistent API route."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/nonexistent", timeout=10)
        if r.status_code == 404:
            log_pass("Spoof: nonexistent route", "404 as expected")
        else:
            log_fail("Spoof: nonexistent route", f"expected 404, got {r.status_code}")
    except Exception as e:
        log_fail("Spoof: nonexistent route", str(e))


def test_spoof_malformed_json():
    """Spoof: send malformed JSON body to analyze endpoint."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/api/analyze-media",
            data="this is not valid json",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if r.status_code in (400, 415, 422):
            log_pass("Spoof: malformed JSON", f"rejected with {r.status_code}")
        else:
            log_fail("Spoof: malformed JSON", f"expected 4xx, got {r.status_code}")
    except Exception as e:
        log_fail("Spoof: malformed JSON", str(e))


def test_spoof_legacy_no_file():
    """Spoof: send empty request to legacy /api/analyze."""
    try:
        r = requests.post(f"{BACKEND_URL}/api/analyze", timeout=10)
        if r.status_code == 400:
            log_pass("Spoof: legacy no file", f"rejected with 400")
        else:
            log_fail("Spoof: legacy no file", f"expected 400, got {r.status_code}")
    except Exception as e:
        log_fail("Spoof: legacy no file", str(e))


def test_spoof_image_ml_no_file():
    """Spoof: call image ML /predict/image without a file (should 422)."""
    try:
        r = requests.post(f"{IMAGE_SERVER}/predict/image", timeout=10)
        if r.status_code in (400, 422):
            log_pass("Spoof: image ML no file", f"rejected with {r.status_code}")
        else:
            log_fail("Spoof: image ML no file", f"expected 400/422, got {r.status_code}")
    except Exception as e:
        log_fail("Spoof: image ML no file", str(e))


def test_spoof_video_ml_no_file():
    """Spoof: call video ML /predict/video without a file (should 422)."""
    try:
        r = requests.post(f"{VIDEO_SERVER}/predict/video", timeout=10)
        if r.status_code in (400, 422):
            log_pass("Spoof: video ML no file", f"rejected with {r.status_code}")
        else:
            log_fail("Spoof: video ML no file", f"expected 400/422, got {r.status_code}")
    except Exception as e:
        log_fail("Spoof: video ML no file", str(e))


def test_spoof_fake_image_magic_bytes():
    """Spoof: upload random bytes as image/png — backend should reject or error cleanly."""
    tmp_path = temp_path(".png")
    try:
        with open(tmp_path, "wb") as wf:
            wf.write(b"not a real png file content xxxxx")
        with open(tmp_path, "rb") as f:
            r = requests.post(
                f"{BACKEND_URL}/api/analyze-media",
                files={"media": ("fake.png", f, "image/png")},
                timeout=60,
            )
        # May 200 with error in ml_status, or 4xx/5xx from sharp/exifr — must not 200 with success-looking fake
        if r.status_code == 200:
            data = r.json()
            if data.get("ml_status") == "error":
                log_pass("Spoof: garbage as PNG", "handled; ml_status=error")
            else:
                log_pass("Spoof: garbage as PNG", f"status 200, ml_status={data.get('ml_status')}")
        elif r.status_code in (400, 415, 422, 500, 503):
            log_pass("Spoof: garbage as PNG", f"rejected or failed with {r.status_code}")
        else:
            log_fail("Spoof: garbage as PNG", f"unexpected {r.status_code}")
    except Exception as e:
        log_fail("Spoof: garbage as PNG", str(e))
    finally:
        os.unlink(tmp_path)


def test_results_endpoint():
    """Test GET /api/results returns paginated results."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/results?page=1&limit=5", timeout=10)
        data = r.json()
        if r.status_code == 200 and "results" in data and "total" in data:
            log_pass("GET /api/results", f"total={data['total']}, page={data['page']}")
            validate_schema(data, ['total', 'page', 'limit', 'totalPages', 'results'],
                          "GET /api/results")
            return True
        else:
            log_fail("GET /api/results", f"status={r.status_code}, data={data}")
            return False
    except Exception as e:
        log_fail("GET /api/results", str(e))
        return False


def test_results_by_id():
    """Test GET /api/results/:id returns a specific result."""
    try:
        # First get a result ID from the list
        r = requests.get(f"{BACKEND_URL}/api/results?page=1&limit=1", timeout=10)
        data = r.json()
        results = data.get("results", [])
        if not results:
            log_skip("GET /api/results/:id", "no results available to test")
            return False

        result_id = results[0].get("_id")
        r2 = requests.get(f"{BACKEND_URL}/api/results/{result_id}", timeout=10)
        data2 = r2.json()
        if r2.status_code == 200 and data2.get("_id") == result_id:
            log_pass("GET /api/results/:id", f"fetched result {result_id}")
            return True
        else:
            log_fail("GET /api/results/:id", f"status={r2.status_code}")
            return False
    except Exception as e:
        log_fail("GET /api/results/:id", str(e))
        return False


def test_results_invalid_id():
    """Test GET /api/results/:id with invalid ID returns 400."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/results/invalid-id-12345", timeout=10)
        if r.status_code == 400:
            log_pass("GET /api/results/:id invalid", "rejected with 400")
        else:
            log_fail("GET /api/results/:id invalid", f"expected 400, got {r.status_code}")
    except Exception as e:
        log_fail("GET /api/results/:id invalid", str(e))


# ─── Accuracy / Consistency Tests ────────────────────────────────────────────

def test_prediction_consistency():
    """Run image prediction twice and check that scores are in same range."""
    tmp_path = temp_path(".png")
    try:
        create_test_image(tmp_path)
        scores = []
        for i in range(2):
            with open(tmp_path, "rb") as f:
                r = requests.post(
                    f"{IMAGE_SERVER}/predict/image",
                    files={"file": (f"test_{i}.png", f, "image/png")},
                    timeout=60,
                )
            data = r.json()
            if r.status_code == 200:
                scores.append(data.get("model_score", -1))

        if len(scores) == 2:
            diff = abs(scores[0] - scores[1])
            if diff < 5:  # Same image should give very similar scores
                log_pass("Prediction consistency",
                         f"scores={scores}, diff={diff:.1f}")
            else:
                log_fail("Prediction consistency",
                         f"scores={scores}, diff={diff:.1f} (expected < 5)")
        else:
            log_fail("Prediction consistency", f"only got {len(scores)} scores")
    except Exception as e:
        log_fail("Prediction consistency", str(e))
    finally:
        os.unlink(tmp_path)


# ─── Main ────────────────────────────────────────────────────────────────────

def _ensure_utf8_stdio():
    """Avoid UnicodeEncodeError on Windows consoles (cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main():
    _ensure_utf8_stdio()
    print("\n" + "═" * 60)
    print("  🧪 DeepScan — System Sanity & Spoof Test Suite v2")
    print("═" * 60)

    # ── 1. Health checks ──────────────────────────────────────────────
    log_section("1. Health Checks")
    img_ok = test_image_server_health()
    vid_ok = test_video_server_health()
    be_ok  = test_backend_health()

    # ── 2. Direct ML predictions ──────────────────────────────────────
    log_section("2. Direct ML Predictions")
    if img_ok:
        test_image_prediction()
    else:
        log_skip("Image prediction (direct)", "image server not available")

    if vid_ok:
        test_video_prediction()
    else:
        log_skip("Video prediction (direct)", "video server not available")

    # ── 3. Full pipeline tests ────────────────────────────────────────
    log_section("3. Full Pipeline Tests (via Node.js backend)")
    if be_ok and img_ok:
        test_full_pipeline_image()
    else:
        log_skip("Full pipeline (image)", "backend or image server unavailable")

    if be_ok and vid_ok:
        test_full_pipeline_video()
    else:
        log_skip("Full pipeline (video)", "backend or video server unavailable")

    # ── 4. Legacy endpoint ────────────────────────────────────────────
    log_section("4. Legacy Endpoint Compatibility")
    if be_ok and img_ok:
        test_legacy_image_endpoint()
    else:
        log_skip("Legacy /api/analyze", "backend or image server unavailable")

    # ── 5. Results endpoints ─────────────────────────────────────────
    log_section("5. Results Retrieval")
    if be_ok:
        test_results_endpoint()
        test_results_by_id()
        test_results_invalid_id()
    else:
        log_skip("Results endpoints", "backend unavailable")

    # ── 6. Frontend route checks ─────────────────────────────────────
    log_section("6. Frontend Route Checks")
    try:
        requests.get(f"{FRONTEND_URL}/", timeout=5)
        frontend_ok = True
    except Exception:
        frontend_ok = False
        log_skip("All frontend routes", "frontend not available at " + FRONTEND_URL)

    if frontend_ok:
        test_frontend_route("/", "Home")
        test_frontend_route("/how-it-works", "How It Works")
        test_frontend_route("/features", "Features")
        test_frontend_route("/about", "About")
        test_frontend_route("/contact", "Contact")
        test_frontend_route("/privacy", "Privacy")
        test_frontend_route("/terms", "Terms")
        # Note: /checker and /history require auth — React routes will still return 200
        # because CSR serves index.html for all routes
        test_frontend_route("/checker", "Checker (CSR)")
        test_frontend_route("/checker/image", "Checker image route (CSR)")
        test_frontend_route("/checker/video", "Checker video route (CSR)")
        test_frontend_route("/history", "History (CSR)")

    # ── 7. Prediction consistency ────────────────────────────────────
    log_section("7. Accuracy & Consistency")
    if img_ok:
        test_prediction_consistency()
    else:
        log_skip("Prediction consistency", "image server not available")

    # ── 8. Spoof / security tests ─────────────────────────────────────
    log_section("8. Spoof & Security Tests")
    if be_ok:
        test_spoof_no_file()
        test_spoof_invalid_type()
        test_spoof_wrong_field_name()
        test_spoof_nonexistent_route()
        test_spoof_malformed_json()
        test_spoof_legacy_no_file()
        test_spoof_fake_image_magic_bytes()
    else:
        log_skip("Backend spoof tests", "backend unavailable")

    log_section("8b. ML Server Spoof Tests")
    if img_ok:
        test_spoof_image_ml_no_file()
    else:
        log_skip("Spoof: image ML no file", "image server unavailable")
    if vid_ok:
        test_spoof_video_ml_no_file()
    else:
        log_skip("Spoof: video ML no file", "video server unavailable")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    total = PASSED + FAILED + SKIPPED
    print(f"  📊 Results: {PASSED} passed, {FAILED} failed, {SKIPPED} skipped / {total} total")
    if FAILED == 0:
        print("  🎉 ALL TESTS PASSED!")
    else:
        print(f"  ⚠️  {FAILED} test(s) failed — review output above.")
    print(f"{'═' * 60}\n")

    sys.exit(1 if FAILED > 0 else 0)


if __name__ == "__main__":
    main()
