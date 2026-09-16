import React, { useEffect, useState } from 'react';

function getRiskLabel(score) {
  if (score >= 80) return { label: 'CRITICAL_RISK', color: 'var(--rose)' };
  if (score >= 60) return { label: 'HIGH_VOLATILITY', color: 'var(--amber)' };
  if (score >= 40) return { label: 'MODERATE_SIG', color: 'var(--amber)' };
  if (score >= 20) return { label: 'LOW_SIGNAL', color: 'var(--emerald)' };
  return { label: 'NO_ARTIFACTS', color: 'var(--emerald)' };
}

export default function ConfidenceMeter({ score, color }) {
  const [animatedWidth, setAnimatedWidth] = useState(0);
  const safeScore = Number.isFinite(score) ? Math.min(100, Math.max(0, score)) : 0;
  const risk = getRiskLabel(safeScore);
  const barColor = color || risk.color;

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedWidth(safeScore), 100);
    return () => clearTimeout(timer);
  }, [safeScore]);

  return (
    <div className="confidence-meter" style={{ marginTop: '12px' }}>
      <div className="confidence-meter__header" style={{ marginBottom: '8px' }}>
        <span className="confidence-meter__label" style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.65rem', textTransform: 'uppercase', opacity: 0.7 }}>
          Confidence_Inference
        </span>
        <span className="confidence-meter__risk" style={{ color: risk.color, fontFamily: "'JetBrains Mono', monospace", fontSize: '0.65rem', fontWeight: 800 }}>
          {risk.label}
        </span>
      </div>
      <div className="confidence-meter__track" style={{ height: '8px', background: 'var(--border2)', borderRadius: '4px', overflow: 'hidden', position: 'relative' }}>
        <div
          className="confidence-meter__bar"
          style={{
            width: `${animatedWidth}%`,
            background: barColor,
            boxShadow: `0 0 15px ${barColor}aa`,
            height: '100%',
            borderRadius: '4px',
            transition: 'width 1.2s cubic-bezier(0.23, 1, 0.32, 1)',
            position: 'relative'
          }}
        >
          <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent)', animation: 'shimmer 2s infinite' }} />
        </div>
      </div>
      <div className="confidence-meter__scale" style={{ marginTop: '10px', display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', fontFamily: "'JetBrains Mono', monospace", opacity: 0.6, textTransform: 'uppercase' }}>
        <span>0.0%</span>
        <span style={{ color: 'var(--emerald)' }}>Authentic</span>
        <span style={{ color: 'var(--amber)' }}>Unverified</span>
        <span style={{ color: 'var(--rose)' }}>Synthetic</span>
        <span>100.0%</span>
      </div>
    </div>
  );
}

