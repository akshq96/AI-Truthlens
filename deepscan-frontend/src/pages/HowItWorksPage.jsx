import React from 'react';
import { Link } from 'react-router-dom';

export default function HowItWorksPage() {
  return (
    <div className="hiw">
      {/* Top header layer */}
      <header className="hiw__header">
        <p className="hiw__kicker">Guide</p>
        <h1 className="hiw__title">How DeepScan Works</h1>
        <p className="hiw__subtitle">
          Upload an image or video, run the analysis, then read the verdict and the evidence
          behind it from each detector.
        </p>
        <div className="hiw__btns">
          <Link to="/" className="hiw__btn hiw__btn--primary">Try it now</Link>
          <Link to="/features" className="hiw__btn hiw__btn--outline">Explore features</Link>
        </div>
      </header>

      {/* Layer 1 */}
      <div className="hiw__layer">
        <div className="hiw__layer-num">01</div>
        <div>
          <h2 className="hiw__layer-title">Upload your image or video</h2>
          <p className="hiw__layer-body">
            Drag &amp; drop or click to browse. Supported formats: JPEG, PNG, WEBP, MP4, MOV and WEBM.
            The uploaded file is deleted from the server after analysis.
          </p>
        </div>
      </div>

      {/* Layer 2 */}
      <div className="hiw__layer">
        <div className="hiw__layer-num">02</div>
        <div>
          <h2 className="hiw__layer-title">AI analysis runs</h2>
          <p className="hiw__layer-body">
            An AI-image detector scores the whole image. If a face is found, a face-manipulation
            detector (studied with synthetic data augmentation) checks it. Metadata and C2PA provenance
            are read as supporting evidence. Videos are checked on evenly sampled frames.
          </p>
        </div>
      </div>

      {/* Layer 3 — wide with columns */}
      <div className="hiw__layer">
        <div className="hiw__layer-num">03</div>
        <div style={{ flex: 1 }}>
          <h2 className="hiw__layer-title">Understand your result</h2>
          <div className="hiw__layer-cols">
            <div>
              <h3>What the score means</h3>
              <p>Every file gets REAL, AI-GENERATED, DEEPFAKE or UNCERTAIN. ML detector scores and measured face and image signals are combined into a REAL score and a FAKE score; UNCERTAIN means the signals conflict or are too weak.</p>
            </div>
            <div>
              <h3>Read the breakdown</h3>
              <p>Review contributing signals for each analyzed image to understand why it was flagged.</p>
            </div>
          </div>

          {/* Summary card inside the layer */}
          <div className="hiw__summary" style={{ marginTop: 24 }}>
            <div className="hiw__summary-label">You'll get</div>
            <ul>
              <li>Verdict: REAL, or FAKE as AI-GENERATED or DEEPFAKE</li>
              <li>Confidence and each detector's probability</li>
              <li>Evidence breakdown and the reason for the verdict</li>
              <li>Metadata and C2PA provenance status</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
