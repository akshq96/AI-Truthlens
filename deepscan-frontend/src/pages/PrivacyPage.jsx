import React from 'react';
import { Link } from 'react-router-dom';

export default function PrivacyPage() {
  return (
    <div className="page">
      <div className="page__content">
        <h1 className="page__title">Privacy Policy</h1>
        <p className="page__intro">
          Last updated: 2026. This policy describes how DeepScan (A Synthetic Data-Augmented Deepfake Detection) collects, uses, and protects your information.
        </p>
        <section className="page__section">
          <h2>Information we collect</h2>
          <p>When you use DeepScan, we process images you upload for analysis. We do not store images beyond what is needed to return results.</p>
        </section>
        <section className="page__section">
          <h2>How we use your data</h2>
          <p>Your uploads are used solely to perform AI authenticity analysis and return scores. We do not sell or share your data with third parties.</p>
        </section>
        <div className="page__actions">
          <Link to="/" className="page__btn page__btn--primary">Back to home</Link>
        </div>
      </div>
    </div>
  );
}
