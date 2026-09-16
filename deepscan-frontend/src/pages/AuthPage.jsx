import React, { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import AuthForm from '../components/AuthForm';

export default function AuthPage({ onSuccess }) {
  const navigate = useNavigate();
  const location = useLocation();

  const initialMode = location.state?.authMode || 'signin';
  const redirectTo = location.state?.redirectTo || '/checker';

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  const handleSuccess = (payload) => {
    onSuccess?.(payload);
    navigate(redirectTo);
  };

  return (
    <section className="auth-page" aria-label="Authentication page">
      <div className="auth-page__inner">
        <AuthForm initialMode={initialMode} onSuccess={handleSuccess} />
      </div>
    </section>
  );
}

