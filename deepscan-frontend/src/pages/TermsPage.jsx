import React from 'react';
import { Link } from 'react-router-dom';

export default function TermsPage() {
  return (
    <div className="page">
      <div className="page__content">
        <h1 className="page__title">Terms of Use</h1>
        <p className="page__intro">
          By using DeepScan (A Synthetic Data-Augmented Deepfake Detection), you agree to these terms of service.
        </p>
        <section className="page__section">
          <h2>Acceptable use</h2>
          <p>You may use DeepScan to analyze images for authenticity. Do not use the service for illegal purposes or to infringe on others' rights.</p>
        </section>
        <section className="page__section">
          <h2>Disclaimer</h2>
          <p>Analysis results are probability-based and not guaranteed. Use results as one factor in your assessment, not as definitive proof.</p>
        </section>
        <div className="page__actions">
          <Link to="/" className="page__btn page__btn--primary">Back to home</Link>
        </div>
      </div>
    </div>
  );
}
