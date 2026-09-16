import React, { useCallback, useMemo, useState } from 'react';
import { Navigate, Routes, Route, useNavigate } from 'react-router-dom';
import './App.css';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import HowItWorksPage from './pages/HowItWorksPage';
import LearnMorePage from './pages/LearnMorePage';
import FeaturesPage from './pages/FeaturesPage';
import AboutPage from './pages/AboutPage';
import ContactPage from './pages/ContactPage';
import PrivacyPage from './pages/PrivacyPage';
import TermsPage from './pages/TermsPage';
import Footer from './components/Footer';
import CheckerPage from './pages/CheckerPage';
import AuthPage from './pages/AuthPage';
import { clearAuth, getAuth, setAuth } from './auth/authStorage';
import DevBackground from './components/DevBackground';

function App() {
  const navigate = useNavigate();
  const [auth, setAuthState] = useState(() => getAuth());

  const isAuthed = !!auth?.email;

  const handleAuthSuccess = (payload) => {
    const saved = setAuth(payload);
    setAuthState(saved);
  };

  const handleLogout = useCallback(() => {
    clearAuth();
    setAuthState(null);
    navigate('/');
  }, [navigate]);

  const navbarProps = useMemo(
    () => ({
      isAuthed,
      userEmail: auth?.email || null,
      onAuthNavigate: () => navigate('/auth'),
      onLogout: handleLogout,
      onGoToChecker: () => navigate('/checker'),
    }),
    [isAuthed, auth?.email, navigate, handleLogout]
  );

  return (
    <div className="app">
      <div className="texture-overlay" />
      <DevBackground />
      <Navbar {...navbarProps} />
      <main className="app__content">
        <Routes>
          <Route path="/" element={<HomePage isAuthed={isAuthed} />} />
          <Route path="/how-it-works" element={<HowItWorksPage />} />
          <Route path="/learn-more" element={<LearnMorePage />} />
          <Route path="/features" element={<FeaturesPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/contact" element={<ContactPage />} />
          <Route path="/privacy" element={<PrivacyPage />} />
          <Route path="/terms" element={<TermsPage />} />
          <Route path="/auth" element={<AuthPage onSuccess={handleAuthSuccess} />} />
          <Route path="/login" element={<Navigate to="/auth" replace />} />
          <Route
            path="/checker"
            element={
              isAuthed ? (
                <CheckerPage />
              ) : (
                <Navigate to="/auth" replace state={{ authMode: 'signin', redirectTo: '/checker' }} />
              )
            }
          />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}

export default App;