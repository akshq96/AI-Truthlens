// Legacy helper retained for backward compatibility in case it is imported.
// No fallback defaults are allowed: invalid values raise immediately.

const aggregateScores = (metadataScore, modelScore) => {
  const ms = Number(metadataScore);
  const ls = Number(modelScore);

  if (!Number.isFinite(ms) || !Number.isFinite(ls)) {
    throw new Error('aggregateScores requires numeric metadataScore and modelScore');
  }

  const finalScore = ls;
  const verdict = finalScore >= 50 ? 'SYNTHETIC' : 'REAL';
  const confidence = finalScore >= 80 ? 'High' : finalScore >= 60 ? 'Medium' : 'Low';

  return {
    final_score: finalScore,
    verdict,
    confidence,
    breakdown: {
      model_score: ls,
      metadata_score: ms,
    },
  };
};

module.exports = { aggregateScores };