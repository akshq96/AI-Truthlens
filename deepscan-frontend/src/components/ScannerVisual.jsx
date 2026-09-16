import React, { useState } from 'react';

export default function ScannerVisual() {
  const [rotation, setRotation] = useState({ x: 0, y: 0 });

  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const rotateX = ((y - rect.height / 2) / rect.height) * -8;
    const rotateY = ((x - rect.width / 2) / rect.width) * 8;
    setRotation({ x: rotateX, y: rotateY });
  };

  const handleMouseLeave = () => setRotation({ x: 0, y: 0 });

  return (
    <div 
      className="scanner-vault"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{
        transform: `rotateX(${rotation.x}deg) rotateY(${rotation.y}deg)`,
        transition: 'transform 0.1s ease-out'
      }}
    >
      {/* Top smaller panel */}
      <div className="scanner-vault__top-panel">
        <div className="scanner-vault__header-bar">
          <div className="scanner-vault__logo">
            <span className="scanner-vault__logo-icon"></span>
          </div>
          <div className="scanner-vault__lines">
            <span></span>
            <span></span>
          </div>
          <div className="scanner-vault__close-btn">
            <span></span>
          </div>
        </div>
        
        <div className="scanner-vault__content">
          <div className="scanner-vault__text-block">
            <p className="scanner-vault__label">Neural Engine v2</p>
            <h3 className="scanner-vault__title">
              DeepScan <span>Detection</span>
            </h3>
            <div className="scanner-vault__desc-lines">
              <span></span>
              <span></span>
              <span></span>
            </div>
            <div className="scanner-vault__btn"></div>
          </div>
          <div className="scanner-vault__scan-box-small">
            <div className="scanner-vault__scan-frame">
              <div className="scanner-vault__scan-head-shadow"></div>
              <div className="scanner-vault__scan-reticle"></div>
            </div>
          </div>
        </div>
      </div>

      {/* Main big panel */}
      <div className="scanner-vault__main-panel">
        <div className="scanner-vault__header-bar">
          <div className="scanner-vault__logo">
            <span className="scanner-vault__logo-icon"></span>
          </div>
          <div className="scanner-vault__lines">
            <span className="long"></span>
            <span></span>
            <span></span>
            <span></span>
            <span></span>
          </div>
          <div className="scanner-vault__outline-btn"></div>
        </div>
        
        <div className="scanner-vault__panel-body">
          <div className="scanner-vault__body-left">
            <p className="scanner-vault__label">Deepfake AI</p>
            <h2 className="scanner-vault__main-title">
              Analysis <br /> Vault<span className="red-dot">.</span>
            </h2>
            <div className="scanner-vault__desc-lines">
              <span className="long"></span>
              <span className="med"></span>
              <span className="short"></span>
            </div>
            
            <div className="scanner-vault__actions">
              <div className="scanner-vault__action-btn primary"></div>
              <div className="scanner-vault__action-btn outline"></div>
            </div>
          </div>
          
          <div className="scanner-vault__body-right">
            <div className="scanner-vault__monitor">
              <div className="scanner-vault__monitor-inner">
                {/* Silhouette / Face */}
                <div className="scanner-vault__face-silhouette"></div>
                {/* Crosshairs & grid */}
                <div className="scanner-vault__monitor-grid"></div>
                {/* Red Scanning Laser Line */}
                <div className="scanner-vault__laser-line"></div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Decorative PCB/tech lines */}
        <div className="scanner-vault__tech-lines"></div>
      </div>
    </div>
  );
}
