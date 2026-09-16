import React from 'react';
import { Link } from 'react-router-dom';

export default function Footer() {
  return (
    <footer className="footer-simple">
      <div className="footer-simple__inner">
        <div className="footer-simple__grid">
          <div className="footer-simple__info">
            <h3 className="footer-simple__logo">Deepfake<span>.</span></h3>
            <p className="footer-simple__tagline">DeepScan: A Synthetic Data-Augmented Deepfake Detection</p>
          </div>
          
          <div className="footer-simple__contacts">
            <div className="contact-item">
              <span className="contact-label">Support</span>
              <a href="mailto:support@aideepfake.com">support@aideepfake.com</a>
            </div>
            <div className="contact-item">
              <span className="contact-label">Press</span>
              <a href="mailto:press@aideepfake.com">press@aideepfake.com</a>
            </div>
            <div className="contact-item">
              <span className="contact-label">Partners</span>
              <a href="mailto:partners@aideepfake.com">partners@aideepfake.com</a>
            </div>
            <div className="contact-item">
              <span className="contact-label">General</span>
              <a href="mailto:info@aideepfake.com">info@aideepfake.com</a>
            </div>
          </div>
        </div>
        
        <div className="footer-simple__bottom">
          <div className="footer-simple__links">
            <Link to="/privacy">Privacy</Link>
            <Link to="/terms">Terms</Link>
            <Link to="/contact">Contact</Link>
          </div>
          <p className="footer-simple__copy">© 2026 DeepScan. Academic project.</p>
        </div>
      </div>
    </footer>
  );
}
