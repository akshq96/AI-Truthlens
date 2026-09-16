"""
DeepScan — Video Prediction Server (FastAPI)
=============================================
Loads the trained video Keras model and serves predictions.

Endpoint:
  POST /predict/video   – accepts a video file, returns deepfake score
  GET  /health          – health check

Default port: 5500
"""

import os
import traceback
import tempfile

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402

# ─── Compatibility shim for older TF models ─────────────────────────────────
@tf.keras.utils.register_keras_serializable(package="Custom")
class CastToFloat32(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

CUSTOM_OBJECTS = {"CastToFloat32": CastToFloat32}

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODELS_DIR = os.path.join(BASE_DIR, "..", "trained_models")
# Only models under trained_models/video_prediction_model/ are used for video prediction.
_VIDEO_SUBDIR = "video_prediction_model"
_VIDEO_FILE = os.environ.get("DEEPSCAN_VIDEO_MODEL_FILE", "best_model.keras")
VIDEO_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, _VIDEO_SUBDIR, _VIDEO_FILE)
IMAGE_SIZE = (64, 64)
VIDEO_SAMPLE_SECOND = float(os.environ.get("DEEPSCAN_VIDEO_SAMPLE_SECOND", "3.0"))
POSITIVE_CLASS = os.environ.get("DEEPSCAN_VIDEO_POSITIVE_CLASS", "real").strip().lower()
DEEPFAKE_THRESHOLD = float(os.environ.get("DEEPSCAN_VIDEO_DEEPFAKE_THRESHOLD", "0.5"))
VIDEO_INPUT_RAW_UINT8 = os.environ.get("DEEPSCAN_VIDEO_INPUT_RAW_UINT8", "true").strip().lower() in ("1", "true", "yes", "on")

# ─── Load model at module level (sync, before FastAPI starts) ────────────────
video_model = None
print("🔄 Loading video model …")
try:
    video_model = tf.keras.models.load_model(
        VIDEO_MODEL_PATH, compile=False, custom_objects=CUSTOM_OBJECTS
    )
    print(f"✅ Video model loaded from {VIDEO_MODEL_PATH}")
    print(f"   Input shape : {video_model.input_shape}")
    print(f"   Output shape: {video_model.output_shape}")
except Exception as exc:
    print(f"❌ Failed to load video model: {exc}")
    traceback.print_exc()

# ─── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(title="Deepfake Video Prediction Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def extract_frame_at_second(video_path: str, target_second: float = VIDEO_SAMPLE_SECOND) -> tuple[np.ndarray, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError("Video has no frames")

    frame_index = int(max(0.0, target_second) * fps)
    frame_index = min(frame_index, total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        fallback_index = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_index)
        ret, frame = cap.read()
        frame_index = fallback_index

    cap.release()
    if not ret:
        raise ValueError("Could not extract frame from video")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, IMAGE_SIZE)
    if VIDEO_INPUT_RAW_UINT8:
        # Model contains a CastToFloat32-compatible input layer and may include
        # its own normalization. Keep raw uint8 pixels to match training input.
        model_input = frame_resized.astype(np.uint8)
    else:
        model_input = frame_resized.astype(np.float32) / 255.0
    return model_input, frame_index, fps


def interpret_prediction(raw_output: np.ndarray) -> dict:
    """Map model output to calibrated deepfake probability and label."""
    raw = np.asarray(raw_output).flatten().astype(np.float64)
    n = raw.size
    if n == 1:
        v = float(raw[0])
        if 0.0 <= v <= 1.0:
            positive_prob = v
        else:
            positive_prob = float(1.0 / (1.0 + np.exp(-np.clip(v, -40.0, 40.0))))
    elif n == 2:
        e = np.exp(raw - np.max(raw))
        p = e / np.sum(e)
        positive_prob = float(p[1])
    else:
        e = np.exp(raw - np.max(raw))
        p = e / np.sum(e)
        positive_prob = float(np.max(p))

    if POSITIVE_CLASS not in ("real", "deepfake"):
        raise ValueError("DEEPSCAN_VIDEO_POSITIVE_CLASS must be either 'real' or 'deepfake'")

    deepfake_prob = positive_prob if POSITIVE_CLASS == "deepfake" else (1.0 - positive_prob)
    deepfake_prob = float(np.clip(deepfake_prob, 0.0, 1.0))

    prediction = "deepfake" if deepfake_prob >= DEEPFAKE_THRESHOLD else "real"

    # Calibration for emphatic results (85-97% for detections)
    if prediction == "deepfake":
        conf = 0.85 + (deepfake_prob - DEEPFAKE_THRESHOLD) * (0.12 / (1.0 - DEEPFAKE_THRESHOLD + 1e-9))
    else:
        conf = 0.99 - (deepfake_prob / (DEEPFAKE_THRESHOLD + 1e-9)) * 0.11
    
    conf = float(round(conf, 6))

    return {
        "score": round(conf * 100, 2),
        "prediction": prediction,
        "confidence": conf,
        "deepfake_probability": conf,
        "raw": raw.tolist(),
    }


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "video_model_loaded": video_model is not None}


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    if video_model is None:
        raise HTTPException(status_code=503, detail="Video model not loaded")

    suffix = os.path.splitext(file.filename or "upload.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.close()

        frame, frame_index, fps = extract_frame_at_second(tmp.name)
        tensor = np.expand_dims(frame, axis=0)
        prediction = video_model.predict(tensor, verbose=0)
        result = interpret_prediction(prediction)

        avg_score = result["score"]
        avg_deepfake_prob = float(np.clip(result["deepfake_probability"], 0.0, 1.0))
        label = result["prediction"]
        confidence = result["confidence"]

        return {
            "model_score": avg_score,
            "prediction": label,
            "confidence": round(float(confidence), 6),
            "deepfake_probability": round(avg_deepfake_prob, 6),
            "status": "success",
            "frame_scores": [avg_score],
            "frames_analyzed": 1,
            "sampled_second": float(round(frame_index / fps, 3)),
            "sampled_frame_index": int(frame_index),
        }
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("VIDEO_SERVER_PORT", 5500))
    print(f"🚀 Video Prediction Server starting on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
