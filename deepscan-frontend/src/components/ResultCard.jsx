import React from 'react';
import ConfidenceMeter from './ConfidenceMeter';

function getVerdictMeta(verdict, score) {
  const v = (verdict || '').toUpperCase();
  if (v.includes('FAKE') || v.includes('SYNTHETIC') || v.includes('MANIPULATED')) {
    return { 
      color: 'var(--rose)', 
      bg: 'rgba(244, 63, 94, 0.08)', 
      border: 'rgba(244, 63, 94, 0.25)', 
      icon: '⚠', 
      label: 'DEEPFAKE_DETECTED' 
    };
  }
  if (v.includes('REAL') || v.includes('AUTHENTIC') || v.includes('GENUINE')) {
    return { 
      color: 'var(--emerald)', 
      bg: 'rgba(16, 185, 129, 0.08)', 
      border: 'rgba(16, 185, 129, 0.25)', 
      icon: '✓', 
      label: 'AUTHENTIC_MEDIA' 
    };
  }
  // fallback based on score
  if (score > 60) {
    return { 
      color: 'var(--rose)', 
      bg: 'rgba(244, 63, 94, 0.08)', 
      border: 'rgba(244, 63, 94, 0.25)', 
      icon: '⚠', 
      label: 'SUSPICIOUS_SIGNALS' 
    };
  }
  return { 
    color: 'var(--amber)', 
    bg: 'rgba(245, 158, 11, 0.08)', 
    border: 'rgba(245, 158, 11, 0.25)', 
    icon: '?', 
    label: 'UNVERIFIED_STATUS' 
  };
}

export default function ResultCard({
  score,
  verdict,
  model_score: modelScore,
  metadata_score: metadataScore,
  result,
  confidence,
  evidence_summary: evidence,
  ai_probability: aiProb,
  deepfake_probability: dfProb,
  metadata_provenance: md,
  trace,
  real_score: realScore,
  fake_score: fakeScore,
  signals,
  unavailable_signals: unavailable,
  decision_reason: decisionReason,
}) {
  // `score` is the SYNTHETIC probability (0 = authentic, 100 = synthetic) and
  // drives the Authentic->Synthetic meter. The headline number must instead be
  // confidence in the stated verdict; showing synthetic-ness there produced
  // contradictions like "REAL 52%".
  const safeScore = Number.isFinite(score) ? Number(score) : 0;
  const meta = getVerdictMeta(result || verdict, safeScore);
  const confPct = Number.isFinite(confidence) ? Number(confidence) * 100 : null;

  return (
    <section className="result-card">
      <div className="result-card__kicker" style={{ fontFamily: "'JetBrains Mono', monospace", color: 'var(--primary-light)', fontSize: '0.6rem' }}>
        LOG_INFERENCE_RESULT
      </div>

      {/* Verdict header */}
      <div
        className="result-card__verdict-block"
        style={{ 
          background: meta.bg, 
          border: `1px solid ${meta.border}`, 
          borderRadius: 16, 
          padding: '24px 20px', 
          marginBottom: 20,
          backdropFilter: 'blur(12px)',
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        <div style={{ position: 'absolute', top: 0, left: 0, width: '2px', height: '100%', background: meta.color }} />
        
        <div className="result-card__verdict-icon" style={{ color: meta.color, fontSize: '1.4rem' }}>{meta.icon}</div>
        <div>
          <div className="result-card__verdict-label" style={{ color: meta.color, fontSize: '0.62rem', fontWeight: 800, letterSpacing: '0.15em', textTransform: 'uppercase', fontFamily: "'JetBrains Mono', monospace" }}>
            {meta.label}
          </div>
          <div className="result-card__verdict-raw" style={{ color: '#fff', fontWeight: 700, fontSize: '1.1rem', marginTop: 4, fontFamily: "'Space Grotesk', sans-serif" }}>
            {result || verdict || 'Running...'}
          </div>
        </div>
        <div className="result-card__score-badge" style={{ marginLeft: 'auto', textAlign: 'right' }}>
          <div style={{ fontSize: '2.2rem', fontWeight: 900, color: meta.color, lineHeight: 1, fontFamily: "'Space Grotesk', sans-serif", letterSpacing: '-0.03em' }}>
            {confPct === null ? '—' : `${confPct.toFixed(0)}%`}
          </div>
          <div style={{ fontSize: '0.62rem', color: 'var(--muted)', textTransform: 'uppercase', fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
            Confidence
          </div>
        </div>
      </div>

      <ConfidenceMeter score={safeScore} color={meta.color} />

      {evidence && (
        <div style={{ marginTop: 20, padding: '14px 16px', border: '1px solid var(--border2)', borderRadius: 12 }}>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.65rem', color: 'var(--primary-light)', marginBottom: 10 }}>
            Evidence_Breakdown
          </div>
          {[
            ['AI forensic evidence', evidence.ai_forensic_evidence],
            ['Deepfake evidence', evidence.deepfake_evidence],
            ['Face detected', evidence.face_detected ? 'YES' : 'NO'],
            ['Camera metadata', evidence.camera_metadata],
            ['C2PA', evidence.c2pa],
            ['SynthID', evidence.synthid],
          ].map(([label, value]) => (
            <div key={label} style={{ display: 'flex', justifyContent: 'space-between', gap: 12, fontSize: '0.72rem', padding: '3px 0' }}>
              <span style={{ color: 'var(--muted)' }}>{label}</span>
              <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#fff', textAlign: 'right' }}>{String(value)}</span>
            </div>
          ))}
          {evidence.reason && (
            <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid var(--border2)', fontSize: '0.72rem', color: 'var(--muted)', lineHeight: 1.5 }}>
              {evidence.reason}
            </div>
          )}
        </div>
      )}

      {signals && (
        <div style={{ marginTop: 20, padding: '14px 16px', border: '1px solid var(--border2)', borderRadius: 12 }}>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.65rem', color: 'var(--primary-light)', marginBottom: 10 }}>
            Multi_Factor_Scores (0 = real, 1 = fake)
          </div>
          {[
            ['REAL_SCORE', realScore],
            ['FAKE_SCORE', fakeScore],
            ['ML detectors', signals.ml, 'ml'],
            ['Face geometry / blending', signals.face, 'face'],
            ['Eyes', signals.eyes, 'eyes'],
            ['Nose / mouth', signals.nose_mouth, 'nose_mouth'],
            ['Skin texture', signals.skin_texture, 'skin_texture'],
            ['Background', signals.background, 'background'],
            ['Lighting / colour', signals.lighting, 'lighting'],
            ['Frequency / compression', signals.frequency, 'frequency'],
            ['Metadata (supporting only)', signals.metadata, 'metadata'],
          ].map(([label, value, key]) => (
            <div key={label} style={{ display: 'flex', justifyContent: 'space-between', gap: 12, fontSize: '0.72rem', padding: '3px 0' }}>
              <span style={{ color: 'var(--muted)' }}>{label}</span>
              <span
                title={value == null && key && unavailable ? unavailable[key] || 'not measured for this image' : undefined}
                style={{ fontFamily: "'JetBrains Mono', monospace", color: value == null ? 'var(--muted)' : '#fff', textAlign: 'right' }}
              >
                {value == null ? 'unavailable' : Number(value).toFixed(2)}
              </span>
            </div>
          ))}
          {decisionReason && !evidence?.reason && (
            <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid var(--border2)', fontSize: '0.72rem', color: 'var(--muted)', lineHeight: 1.5 }}>
              {decisionReason}
            </div>
          )}
        </div>
      )}

      <div className="result-card__breakdown" style={{ marginTop: 24 }}>
        <div className="result-card__breakdown-title" style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.65rem' }}>
          Breakdown_Signals
        </div>
        <div className="result-card__breakdown-grid">
          <div className="result-card__breakdown-item">
            <div className="result-card__breakdown-meta">
              <span className="result-card__breakdown-icon">🧠</span>
              <span>AI_SYNTHETIC_PROB</span>
            </div>
            <strong style={{ color: aiProb == null ? 'var(--muted)' : (aiProb > 0.5 ? 'var(--rose)' : 'var(--emerald)'), fontFamily: "'JetBrains Mono', monospace" }}>
              {aiProb != null ? `${(Number(aiProb) * 100).toFixed(1)}%` : '--'}
            </strong>
          </div>
          <div className="result-card__breakdown-item">
            <div className="result-card__breakdown-meta">
              <span className="result-card__breakdown-icon">🎭</span>
              <span>FACE_MANIP_PROB</span>
            </div>
            <strong style={{ color: dfProb == null ? 'var(--muted)' : (dfProb > 0.5 ? 'var(--rose)' : 'var(--emerald)'), fontFamily: "'JetBrains Mono', monospace" }}>
              {dfProb != null ? `${(Number(dfProb) * 100).toFixed(1)}%` : 'N/A'}
            </strong>
          </div>
          <div className="result-card__breakdown-item">
            <div className="result-card__breakdown-meta">
              <span className="result-card__breakdown-icon">📊</span>
              {/* Coherence of the capture record, 0-1. NOT an authenticity score:
                  the old METADATA_EXT value here was the legacy suspicion score,
                  whose polarity is inverted (higher = more suspicious), so it read
                  as "95% good" when it meant "95% suspicious". */}
              <span>METADATA_COHERENCE</span>
            </div>
            <strong style={{ color: 'var(--muted)', fontFamily: "'JetBrains Mono', monospace" }}>
              {md && md.available
                ? `${(Number(md.consistency_score || 0) * 100).toFixed(0)}% (${md.reliability})`
                : 'ABSENT'}
            </strong>
          </div>
        </div>
      </div>

      {trace && (
        <details style={{ marginTop: 18 }}>
          <summary style={{ cursor: 'pointer', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.65rem', color: 'var(--primary-light)' }}>
            Raw_Detector_Trace
          </summary>
          <pre style={{ marginTop: 10, fontSize: '0.62rem', color: 'var(--muted)', whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 320, overflow: 'auto' }}>
{JSON.stringify(trace, null, 2)}
          </pre>
        </details>
      )}
    </section>
  );
}

