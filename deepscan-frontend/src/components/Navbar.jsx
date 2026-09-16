import React, { useState } from 'react';
import { Link } from 'react-router-dom';

export default function Navbar({ isAuthed, userEmail, onAuthNavigate, onLogout, onGoToChecker }) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);
  const closeMenu = () => setIsMenuOpen(false);

  return (
    <header className="navbar">
      <div className="navbar__inner">
        <Link to="/" className="navbar__logo" onClick={closeMenu}>
          <img src="/logo.png" alt="DeepScan logo" className="navbar__logo-img" />
          <div className="navbar__logo-wrapper">
            <span className="navbar__logo-text">DeepScan</span>
            <span className="navbar__logo-subtitle">Synthetic Data-Augmented Deepfake Detection</span>
          </div>
        </Link>

        {/* Hamburger Icon */}
        <div className={`navbar__hamburger ${isMenuOpen ? 'open' : ''}`} onClick={toggleMenu}>
          <span></span>
          <span></span>
          <span></span>
        </div>

        {/* Dark overlay for mobile */}
        {isMenuOpen && <div className="navbar__overlay" onClick={closeMenu}></div>}

        <nav className={`navbar__right ${isMenuOpen ? 'open' : ''}`}>
          <Link to="/how-it-works" className="navbar__link" onClick={closeMenu}>How it works</Link>
          <Link to="/features" className="navbar__link" onClick={closeMenu}>Features</Link>
          <Link to="/about" className="navbar__link" onClick={closeMenu}>About</Link>
          <Link to="/learn-more" className="navbar__btn navbar__btn--secondary" onClick={closeMenu}>Learn more</Link>

          {isAuthed ? (
            <button
              type="button"
              className="navbar__btn navbar__btn--contact"
              onClick={() => { onLogout(); closeMenu(); }}
              title={userEmail || 'Logout'}
            >
              Logout
            </button>
          ) : (
            <button
              type="button"
              className="navbar__btn navbar__btn--contact"
              onClick={() => { onAuthNavigate(); closeMenu(); }}
            >
              Sign in
            </button>
          )}
        </nav>
      </div>
    </header>
  );
}
