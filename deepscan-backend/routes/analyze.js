const express = require('express');
const router = express.Router();
const fs = require('fs').promises;
const mongoose = require('mongoose');
const { upload, uploadVideo, handleMulterError } = require('../middleware/fileValidator');
const { analyzeMetadata } = require('../services/metadataService');
const { runMLModel } = require('../services/mlservice');
const Result = require('../models/Result');

const labelFromScore = (score) => (score >= 50 ? 'deepfake' : 'real');

// Verdict must come from the ML server's decision engine. If it is missing we
// report UNCERTAIN — never silently fall through to REAL, which would turn an
// infrastructure failure into a false clean bill of health.
const verdictFromResult = (result, prediction) => {
  if (result === 'AI-GENERATED' || result === 'DEEPFAKE') return 'SYNTHETIC';
  if (result === 'REAL') return 'REAL';
  if (result === 'UNCERTAIN') return 'UNCERTAIN';
  if (prediction === 'deepfake') return 'SYNTHETIC';
  if (prediction === 'real') return 'REAL';
  return 'UNCERTAIN';
};

const confidenceBand = (confidence) => {
  if (!Number.isFinite(confidence)) return 'Low';
  if (confidence >= 0.85) return 'High';
  if (confidence >= 0.6) return 'Medium';
  return 'Low';
};

const dbReady = () => mongoose.connection.readyState === 1;

// --- POST /api/analyze ------------------------------------------------------
router.post('/analyze', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image file provided.' });
  }

  const filePath = req.file.path;

  try {
    console.log('File received:', req.file.filename);

    const [metadataResult, mlResult] = await Promise.all([
      analyzeMetadata(filePath),
      runMLModel(filePath, req.file.mimetype),
    ]);

    const modelScore = Number(mlResult.model_score);
    if (!Number.isFinite(modelScore)) {
      throw new Error('Invalid model score returned by ML service');
    }

    const prediction = mlResult.prediction || labelFromScore(modelScore);
    const confidence = Number.isFinite(mlResult.confidence)
      ? Number(mlResult.confidence)
      : Number((modelScore / 100).toFixed(6));
    const verdict = verdictFromResult(mlResult.result, mlResult.prediction);

    let savedResult = null;
    if (dbReady()) {
      savedResult = await Result.create({
        filename: req.file.filename,
        originalName: req.file.originalname,
        final_score: modelScore,
        verdict,
        confidence: confidenceBand(confidence),
        breakdown: {
          model_score: modelScore,
          metadata_score: metadataResult.metadata_score,
        },
        flags: metadataResult.flags,
        raw_metadata: metadataResult.raw,
        description: String(req.body.description || '').slice(0, 1000),
      });
    }

    fs.unlink(filePath)
      .then(() => console.log('Temp file deleted'))
      .catch((e) => console.warn('Could not delete temp file:', e.message));

    return res.status(200).json({
      // New decision-engine fields
      result: mlResult.result || (verdict === 'UNCERTAIN' ? 'UNCERTAIN' : undefined),
      ai_probability: mlResult.ai_probability,
      deepfake_probability: mlResult.deepfake_probability,
      face_detected: mlResult.face_detected,
      synthid: mlResult.synthid,
      metadata_provenance: mlResult.metadata_provenance,
      c2pa: mlResult.c2pa,
      evidence_summary: mlResult.evidence_summary,
      trace: mlResult.trace,
      provenance_notes: mlResult.provenance_notes,
      decision_reason: mlResult.decision_reason,
      model_scores: mlResult.model_scores,
      detector_errors: mlResult.detector_errors,
      calibration: mlResult.calibration,
      real_score: mlResult.real_score,
      fake_score: mlResult.fake_score,
      uncertainty: mlResult.uncertainty,
      signals: mlResult.signals,
      signal_weights: mlResult.signal_weights,
      unavailable_signals: mlResult.unavailable_signals,
      decision_method: mlResult.decision_method,
      // Legacy fields (frontend compatibility)
      prediction,
      confidence,
      score: modelScore,
      final_score: modelScore,
      verdict,
      breakdown: {
        model_score: modelScore,
        metadata_score: metadataResult.metadata_score,
      },
      raw_metadata: metadataResult.raw,
      flags: metadataResult.flags,
      id: savedResult ? savedResult._id : null,
      analyzed_at: savedResult ? savedResult.analyzed_at : new Date().toISOString(),
      metadata_flags: metadataResult.flags,
    });
  } catch (err) {
    fs.unlink(filePath).catch(() => {});
    console.error('Analysis error:', err.message);
    return res.status(503).json({ error: 'Analysis service failed. Please try again.' });
  }
});

// --- POST /api/analyze-video ------------------------------------------------
router.post('/analyze-video', uploadVideo.single('video'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No video file provided.' });
  }

  const filePath = req.file.path;

  try {
    console.log('Video received:', req.file.filename);

    const mlResult = await runMLModel(filePath, req.file.mimetype);
    const modelScore = Number(mlResult.model_score);
    if (!Number.isFinite(modelScore)) {
      throw new Error('Invalid video model score returned by ML service');
    }

    const prediction = mlResult.prediction || labelFromScore(modelScore);
    const verdict = verdictFromResult(mlResult.result, mlResult.prediction);
    const band = confidenceBand(mlResult.confidence);

    let savedResult = null;
    if (dbReady()) {
      savedResult = await Result.create({
        filename: req.file.filename,
        originalName: req.file.originalname,
        final_score: modelScore,
        verdict,
        confidence: band,
        breakdown: {
          model_score: modelScore,
          metadata_score: 50,
        },
        flags: ['Video metadata check skipped'],
        raw_metadata: null,
        description: String(req.body.description || '').slice(0, 1000),
      });
    }

    fs.unlink(filePath)
      .then(() => console.log('Temp video file deleted'))
      .catch((e) => console.warn('Could not delete temp video:', e.message));

    return res.status(200).json({
      message: 'Video analysis complete',
      // Decision-engine fields, same as the image route
      result: mlResult.result || 'UNCERTAIN',
      ai_probability: mlResult.ai_probability,
      deepfake_probability: mlResult.deepfake_probability,
      face_detected: mlResult.face_detected,
      decision_reason: mlResult.decision_reason,
      evidence_summary: mlResult.evidence_summary,
      model_scores: mlResult.model_scores,
      detector_errors: mlResult.detector_errors,
      calibration: mlResult.calibration,
      real_score: mlResult.real_score,
      fake_score: mlResult.fake_score,
      uncertainty: mlResult.uncertainty,
      signals: mlResult.signals,
      signal_weights: mlResult.signal_weights,
      unavailable_signals: mlResult.unavailable_signals,
      decision_method: mlResult.decision_method,
      detector: mlResult.detector,
      prediction,
      id: savedResult ? savedResult._id : null,
      filename: req.file.filename,
      originalName: req.file.originalname,
      final_score: modelScore,
      verdict,
      confidence: Number.isFinite(mlResult.confidence) ? Number(mlResult.confidence) : null,
      confidence_band: band,
      breakdown: {
        model_score: modelScore,
        metadata_score: 50,
      },
      flags: ['Video metadata check skipped'],
      raw_metadata: null,
      analyzed_at: savedResult ? savedResult.analyzed_at : new Date().toISOString(),
      frames_analyzed: mlResult.frames_analyzed || 0,
      sampled_second: Number.isFinite(mlResult.sampled_second) ? mlResult.sampled_second : null,
      sampled_frame_index: Number.isFinite(mlResult.sampled_frame_index) ? mlResult.sampled_frame_index : null,
      image_model_test: mlResult.image_model_test || null,
    });
  } catch (err) {
    fs.unlink(filePath).catch(() => {});
    console.error('Video Analysis error:', err.message);
    return res.status(503).json({ error: 'Video analysis service failed. Please try again.' });
  }
});

// --- GET /api/results -------------------------------------------------------
router.get('/results', async (req, res) => {
  try {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const limit = Math.min(100, Math.max(1, parseInt(req.query.limit, 10) || 20));
    const skip = (page - 1) * limit;

    const [results, total] = await Promise.all([
      Result.find()
        .sort({ analyzed_at: -1 })
        .skip(skip)
        .limit(limit)
        .select('-raw_metadata'),
      Result.countDocuments(),
    ]);

    return res.json({
      total,
      page,
      limit,
      totalPages: Math.ceil(total / limit),
      results,
    });
  } catch (err) {
    console.error('Fetch results error:', err.message);
    return res.status(500).json({ error: 'Could not fetch results.' });
  }
});

// --- GET /api/results/:id ---------------------------------------------------
router.get('/results/:id', async (req, res) => {
  try {
    const result = await Result.findById(req.params.id);
    if (!result) {
      return res.status(404).json({ error: 'Result not found.' });
    }
    return res.json(result);
  } catch (err) {
    if (err.name === 'CastError') {
      return res.status(400).json({ error: 'Invalid result ID format.' });
    }
    console.error('Fetch result error:', err.message);
    return res.status(500).json({ error: 'Could not fetch result.' });
  }
});

router.use(handleMulterError);

module.exports = router;
