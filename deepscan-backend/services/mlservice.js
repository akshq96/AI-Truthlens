const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');
const path = require('path');

// --- Config -----------------------------------------------------------------
const IMAGE_SERVER_URL = process.env.FLASK_ML_URL || 'http://127.0.0.1:7000';

function parseTimeoutMs(value, fallback) {
  const n = Number.parseInt(String(value ?? ''), 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

const IMAGE_PREDICT_TIMEOUT_MS = parseTimeoutMs(process.env.ML_IMAGE_TIMEOUT_MS, 180000);
const VIDEO_PREDICT_TIMEOUT_MS = parseTimeoutMs(process.env.ML_VIDEO_TIMEOUT_MS, 300000);

const IMAGE_MIMES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif', 'image/bmp'];
const VIDEO_MIMES = ['video/mp4', 'video/webm', 'video/avi', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska'];

const detectMediaType = (filePath, mimeType) => {
  if (mimeType) {
    if (IMAGE_MIMES.includes(mimeType)) return 'image';
    if (VIDEO_MIMES.includes(mimeType)) return 'video';
  }

  const ext = path.extname(filePath).toLowerCase();
  const imageExts = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'];
  const videoExts = ['.mp4', '.webm', '.avi', '.mov', '.mkv'];

  if (imageExts.includes(ext)) return 'image';
  if (videoExts.includes(ext)) return 'video';
  return 'image';
};

const runImageModel = async (filePath) => {
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath));

  try {
    const response = await axios.post(
      `${IMAGE_SERVER_URL}/predict/image`,
      form,
      {
        headers: form.getHeaders(),
        timeout: IMAGE_PREDICT_TIMEOUT_MS,
        maxContentLength: 50 * 1024 * 1024,
        maxBodyLength: 50 * 1024 * 1024,
      }
    );

    const data = response.data || {};

    // `final_score` from the ML server is the SYNTHETIC probability on a 0-100
    // scale (0 = authentic, 100 = synthetic), which is what the frontend meter
    // renders. Do not substitute label-confidence here.
    const score = Number.isFinite(data.final_score)
      ? Number(data.final_score)
      : Number((data.ai_probability ?? 0) * 100);

    return {
      prediction: data.prediction,
      result: data.result,
      confidence: data.confidence,
      model_score: score,
      ai_probability: data.ai_probability,
      deepfake_probability: data.deepfake_probability,
      face_detected: data.face_detected,
      synthid: data.synthid,
      metadata_provenance: data.metadata,
      c2pa: data.c2pa,
      evidence_summary: data.evidence_summary,
      trace: data.trace,
      provenance_notes: data.provenance_notes,
      decision_reason: data.decision_reason,
      model_scores: data.model_scores,
      detector_errors: data.detector_errors,
      calibration: data.calibration,
      verdict: data.verdict,
      real_score: data.real_score,
      fake_score: data.fake_score,
      uncertainty: data.uncertainty,
      signals: data.signals,
      signal_weights: data.signal_weights,
      unavailable_signals: data.unavailable_signals,
      decision_method: data.decision_method,
      status: 'success',
      media_type: 'image',
    };
  } catch (err) {
    console.error('Image ML Service Error:', err.message);
    if (err.response) {
      console.error('Server response:', err.response.data);
    }
    throw err;
  }
};

const runVideoModel = async (filePath) => {
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath));

  try {
    const response = await axios.post(
      `${IMAGE_SERVER_URL}/predict/video-as-image`,
      form,
      {
        headers: form.getHeaders(),
        timeout: VIDEO_PREDICT_TIMEOUT_MS,
        maxContentLength: 50 * 1024 * 1024,
        maxBodyLength: 50 * 1024 * 1024,
      }
    );

    const data = response.data || {};
    const score = Number.isFinite(data.final_score)
      ? Number(data.final_score)
      : Number((data.deepfake_probability ?? 0) * 100);

    return {
      model_score: score,
      prediction: data.prediction,
      result: data.result,
      confidence: data.confidence,
      verdict: data.verdict,
      real_score: data.real_score,
      fake_score: data.fake_score,
      uncertainty: data.uncertainty,
      signals: data.signals,
      signal_weights: data.signal_weights,
      unavailable_signals: data.unavailable_signals,
      decision_method: data.decision_method,
      ai_probability: data.ai_probability,
      deepfake_probability: data.deepfake_probability,
      face_detected: data.face_detected ?? data.face_frames_detected > 0,
      decision_reason: data.decision_reason,
      evidence_summary: data.evidence_summary,
      model_scores: data.model_scores,
      detector_errors: data.detector_errors,
      calibration: data.calibration,
      detector: data.detector,
      frames_analyzed: data.frames_analyzed,
      sampled_second: data.sampled_second,
      sampled_frame_index: data.sampled_frame_index,
      status: 'success',
      media_type: 'video',
    };
  } catch (err) {
    console.error('Video ML Service Error:', err.message);
    if (err.response) {
      console.error('Server response:', err.response.data);
    }
    throw err;
  }
};

const runMLModel = async (filePath, mimeType) => {
  const mediaType = detectMediaType(filePath, mimeType);
  console.log(`Running ${mediaType} prediction model for: ${path.basename(filePath)}`);

  if (mediaType === 'video') {
    return runVideoModel(filePath);
  }
  return runImageModel(filePath);
};

module.exports = { runMLModel, detectMediaType };
