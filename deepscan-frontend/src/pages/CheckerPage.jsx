import React from 'react';
import UploadZone from '../components/UploadZone';

export default function CheckerPage() {
  return (
    <div className="checker-page">
      {/* Hero Header */}
      <div className="checker-page__hero">
        <div className="checker-page__badge">
          <span className="checker-page__badge-dot" />
          SYSTEM_STATUS: NEURAL_ENGINE_ACTIVE
        </div>
        <h1 className="checker-page__title">
          Media <span>Authenticity</span> Audit
        </h1>
        <p className="checker-page__subtitle">
          Upload an image or video. DeepScan checks it for AI generation and face
          manipulation, reads its metadata, and explains the verdict.
        </p>
      </div>

      {/* Two column layout */}
      <div className="checker-page__body">
        {/* Left: Upload zone */}
        <div className="checker-page__main">
          <UploadZone id="analyzer" />
        </div>

        {/* Right: Info sidebar */}
        <aside className="checker-page__sidebar">
          <div className="checker-sidebar__card">
            <div className="checker-sidebar__label">Engine Specifications</div>
            <ul className="checker-sidebar__list">
              <li>
                <span className="checker-sidebar__chip">🧠</span>
                <div>
                  <strong>AI-Image Detector</strong>
                  <p>Community Forensics Vision Transformer scores the whole image</p>
                </div>
              </li>
              <li>
                <span className="checker-sidebar__chip">🎭</span>
                <div>
                  <strong>Face-Manipulation Detector</strong>
                  <p>CLIP ViT-L/14 face model, evaluated with and without synthetic data augmentation</p>
                </div>
              </li>
              <li>
                <span className="checker-sidebar__chip">📊</span>
                <div>
                  <strong>Provenance Check</strong>
                  <p>EXIF, XMP, IPTC metadata and C2PA manifest detection</p>
                </div>
              </li>
            </ul>
          </div>

          <div className="checker-sidebar__card checker-sidebar__card--tips">
            <div className="checker-sidebar__label">Inference Tips</div>
            <div className="checker-sidebar__tip">
              <span>✓</span> High-res samples yield higher confidence
            </div>
            <div className="checker-sidebar__tip">
              <span>✓</span> Supported: JPG, PNG, WEBP, MP4, MOV
            </div>
            <div className="checker-sidebar__tip">
              <span>✓</span> Results show each detector's score and the reason for the verdict
            </div>
            <div className="checker-sidebar__tip">
              <span>✓</span> Every file gets REAL, AI-GENERATED, DEEPFAKE or UNCERTAIN with its evidence
            </div>
          </div>

          <div className="checker-sidebar__stat-row">
            <div className="checker-sidebar__stat">
              <span className="checker-sidebar__stat-val">5</span>
              <span className="checker-sidebar__stat-lbl">Synthetic types</span>
            </div>
            <div className="checker-sidebar__stat">
              <span className="checker-sidebar__stat-val">4</span>
              <span className="checker-sidebar__stat-lbl">Verdicts</span>
            </div>
            <div className="checker-sidebar__stat">
              <span className="checker-sidebar__stat-val">&lt;2s</span>
              <span className="checker-sidebar__stat-lbl">Latency</span>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}