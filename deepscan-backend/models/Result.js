const mongoose = require('mongoose');

const ResultSchema = new mongoose.Schema(
  {
    filename: {
      type: String,
      required: true
    },
    // Preserves the user's original upload filename (before we rename it)
    originalName: {
      type: String,
      required: true
    },
    final_score: {
      type: Number,
      required: true
    },
    verdict: {
      type: String,
      enum: ['REAL', 'UNCERTAIN', 'LIKELY SYNTHETIC', 'SYNTHETIC'],
      required: true
    },
    confidence: {
      type: String,
      enum: ['Low', 'Medium', 'High'],
      required: true
    },
    breakdown: {
      model_score: Number,
      metadata_score: Number
    },
    flags: [String],
    raw_metadata: {
      type: mongoose.Schema.Types.Mixed,
      default: null
    },
    description: {
      type: String,
      default: ''
    },
    analyzed_at: {
      type: Date,
      default: Date.now
    }
  }
);

// Index on analyzed_at for fast sort queries (newest first)
ResultSchema.index({ analyzed_at: -1 });

module.exports = mongoose.model('Result', ResultSchema);