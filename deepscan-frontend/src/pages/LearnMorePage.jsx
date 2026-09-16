import React from 'react';
import { Link } from 'react-router-dom';

const TECH_BLOCKS = [
  {
    id: '01',
    category: 'Core AI',
    title: 'Detection Models',
    desc: 'A Community Forensics Vision Transformer detects fully AI-generated images. A CLIP ViT-L/14 face model with a head trained on FaceForensics++ and real portraits detects face swaps and reenactment.',
    metrics: ['Community Forensics ViT', 'CLIP ViT-L/14 face model', 'Calibrated thresholds']
  },
  {
    id: '02',
    category: 'Training',
    title: 'Synthetic Data Augmentation',
    desc: 'Five synthetic manipulations imitate the traces left by real deepfake tools. The face model is trained with and without them and tested on a manipulation method it never saw, to measure whether augmentation helps.',
    metrics: ['blend_warp · freq_perturb', 'compression_artifact · color_perturb', 'autoencoder_swap']
  },
  {
    id: '03',
    category: 'Registry',
    title: 'Provenance Evidence',
    desc: 'Reads EXIF, XMP and IPTC metadata and looks for C2PA manifests. Metadata only supports a verdict; missing metadata is never treated as proof of a fake.',
    metrics: ['EXIF · XMP · IPTC', 'C2PA detection', 'Supporting evidence only']
  }
];

export default function LearnMorePage() {
  return (
    <div className="learn-v2">
      <header className="learn-v2__header">
        <div className="learn-v2__kicker">Diagnostic / Architecture</div>
        <h1 className="learn-v2__title">System <span>Blueprint</span></h1>
        <p className="learn-v2__subtitle">
          DeepScan: A Synthetic Data-Augmented Deepfake Detection. It combines two detection
          models and provenance evidence in a decision engine that explains every verdict.
        </p>
      </header>

      <section className="learn-v2__architecture">
        {TECH_BLOCKS.map((block) => (
          <div key={block.id} className="tech-module">
            <div className="tech-module__header">
              <span className="tech-module__num">{block.id}</span>
              <span className="tech-module__cat">{block.category}</span>
            </div>
            <h2 className="tech-module__title">{block.title}</h2>
            <p className="tech-module__desc">{block.desc}</p>
            <div className="tech-module__metrics">
              {block.metrics.map((m, i) => (
                <div key={i} className="tech-metric">
                  <span className="tech-metric__dot"></span>
                  {m}
                </div>
              ))}
            </div>
            {/* Visual connector line */}
            <div className="tech-module__connector"></div>
          </div>
        ))}

        {/* Central aggregator visual side-piece */}
        <div className="aggregator-core">
          <div className="aggregator-core__box">
            <div className="aggregator-core__logo">🔥</div>
            <div className="aggregator-core__label">Aggregator Core</div>
            <div className="aggregator-core__pulse"></div>
          </div>
        </div>
      </section>

      <div className="learn-v2__footer">
        <Link to="/" className="landing__btn-v2 primary">Try Detection Vault</Link>
      </div>
    </div>
  );
}
