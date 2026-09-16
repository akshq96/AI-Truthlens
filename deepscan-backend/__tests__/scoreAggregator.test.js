/**
 * DeepScan Backend — Score Aggregator Tests
 *
 * Unit tests for the legacy score aggregation helper.
 * It now returns model-driven score only and rejects non-numeric inputs.
 */

const { aggregateScores } = require('../services/scoreAggregator');

// ─── Unit Tests (no server needed) ─────────────────────────────────────────

describe('Score Aggregator', () => {
  test('should return SYNTHETIC for high scores', () => {
    const result = aggregateScores(90, 90);
    expect(result.verdict).toBe('SYNTHETIC');
    expect(result.confidence).toBe('High');
    expect(result.final_score).toBe(90);
  });

  test('should return REAL for low scores', () => {
    const result = aggregateScores(10, 10);
    expect(result.verdict).toBe('REAL');
    expect(result.confidence).toBe('Low');
    expect(result.final_score).toBe(10);
  });

  test('should handle neutral scores', () => {
    const result = aggregateScores(50, 50);
    expect(result.verdict).toBe('SYNTHETIC');
    expect(result.final_score).toBe(50);
  });

  test('should use model score as final score', () => {
    const result = aggregateScores(0, 100);
    expect(result.final_score).toBe(100);
  });

  test('should throw on non-numeric inputs', () => {
    expect(() => aggregateScores(undefined, null)).toThrow(
      'aggregateScores requires numeric metadataScore and modelScore'
    );
  });

  test('should include breakdown in result', () => {
    const result = aggregateScores(60, 70);
    expect(result.breakdown).toEqual({
      model_score: 70,
      metadata_score: 60
    });
  });
});

