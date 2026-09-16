import React from 'react';
import { Link } from 'react-router-dom';

const STATS = [
  { num: '3-Part', label: 'Evidence', sub: 'AI-image model • face model • metadata' },
  { num: '5', label: 'Synthetic Types', sub: 'tested for augmenting face-model training' },
  { num: '3', label: 'Verdicts', sub: 'REAL • AI-GENERATED • DEEPFAKE' },
];

const PRIORITIES = [
  { title: 'Readable results', body: 'Verdict + confidence meter + breakdown so the score is explainable, not a black box.' },
  { title: 'Practical signals', body: 'We surface what\'s most useful: artifacts and metadata anomalies alongside model signals.' },
  { title: 'Privacy minded', body: 'Your uploads are used for analysis only. We do not sell, share, or retain your data.' },
];

export default function AboutPage() {
  return (
    <div className="about-page">

      {/* Band 1 — Hero split */}
      <div className="about-band about-band--hero">
        <div>
          <p className="about__kicker">✦ DeepScan</p>
          <h1 className="about__title">DeepScan: A Synthetic Data-Augmented Deepfake Detection</h1>
          <p className="about__subtitle">
            DeepScan detects face-swap deepfakes and AI-generated images, and turns the evidence
            from each detector into a readable verdict and breakdown.
          </p>
          <div className="about__hero-actions">
            <Link to="/" className="about__btn about__btn--primary">Try DeepScan</Link>
            <Link to="/contact" className="about__btn about__btn--secondary">Get in touch</Link>
          </div>
        </div>

        <div className="about__hero-right">
          <div className="about__stat-side">
            <div className="about__stat-label">Signal set</div>
            <div className="about__stat-value">3-part</div>
            <div className="about__stat-sub">model · artifacts · metadata</div>
          </div>
          <div className="about__stat-side">
            <div className="about__stat-label">Output</div>
            <div className="about__stat-value">0–100%</div>
            <div className="about__stat-sub">probability-style score</div>
          </div>
        </div>
      </div>

      {/* Band 2 — Stats row */}
      <div className="about-band about-band--stats">
        {STATS.map((s, i) => (
          <div key={i} className="about-stat-card">
            <div className="about-stat-num">{s.num}</div>
            <div className="about-stat-label" style={{ color: 'var(--muted)', fontSize: '0.82rem', marginTop: 6 }}>{s.label}</div>
            <div style={{ fontSize: '0.78rem', color: 'var(--faint)', marginTop: 2 }}>{s.sub}</div>
          </div>
        ))}
      </div>

      {/* Band 3 — Timeline priorities */}
      <div className="about-band about-band--timeline">
        <p style={{ fontSize: '0.72rem', letterSpacing: '0.18em', textTransform: 'uppercase', color: 'var(--red)', marginBottom: 28 }}>
          What we prioritize
        </p>
        <div className="about__timeline-grid">
          {PRIORITIES.map((p, i) => (
            <div key={i} className="about__tl-item">
              <div className="about__tl-num">0{i + 1}</div>
              <h2 className="about__tl-title">{p.title}</h2>
              <p className="about__tl-body">{p.body}</p>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
