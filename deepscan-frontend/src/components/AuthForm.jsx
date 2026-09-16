import React, { useEffect, useMemo, useState } from 'react';

function isValidEmail(value) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
}

export default function AuthForm({ initialMode = 'signin', showClose = false, onClose, onSuccess }) {
  const normalizedInitialMode = initialMode === 'signup' ? 'signup' : 'signin';
  const [activeMode, setActiveMode] = useState(normalizedInitialMode);
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);

  useEffect(() => {
    setActiveMode(normalizedInitialMode);
    setError(null);
    setPassword('');
  }, [normalizedInitialMode]);

  const title = useMemo(
    () => (activeMode === 'signup' ? 'Create your account' : 'Welcome back'),
    [activeMode]
  );

  const submitLabel = activeMode === 'signup' ? 'Create account' : 'Sign in';

  const handleSubmit = (e) => {
    e.preventDefault();
    setError(null);

    if (activeMode === 'signup') {
      if (!firstName.trim() || !lastName.trim()) {
        setError('Please enter your first and last name.');
        return;
      }
    }

    if (!isValidEmail(email)) {
      setError('Please enter a valid email address.');
      return;
    }
    if (!password || password.length < 6) {
      setError('Password must be at least 6 characters.');
      return;
    }

    onSuccess?.({
      email: email.trim().toLowerCase(),
      name: activeMode === 'signup' ? `${firstName.trim()} ${lastName.trim()}` : null,
      createdAt: new Date().toISOString(),
    });
  };

  return (
    <div className="auth-modal" role="region" aria-label="Authentication">
      <div className="auth-modal__header">
        <div>
          <div className="auth-modal__kicker">DeepScan</div>
          <h2 className="auth-modal__title">{title}</h2>
          <div className="auth-modal__subtitle">DeepScan: A Synthetic Data-Augmented Deepfake Detection</div>
        </div>
        {showClose ? (
          <button type="button" className="auth-modal__close" onClick={onClose} aria-label="Close">
            ×
          </button>
        ) : null}
      </div>

      <div className="auth-modal__tabs" role="tablist" aria-label="Auth mode">
        <button
          type="button"
          className={`auth-modal__tab ${activeMode === 'signin' ? 'is-active' : ''}`}
          onClick={() => setActiveMode('signin')}
          role="tab"
          aria-selected={activeMode === 'signin'}
        >
          Sign in
        </button>
        <button
          type="button"
          className={`auth-modal__tab ${activeMode === 'signup' ? 'is-active' : ''}`}
          onClick={() => setActiveMode('signup')}
          role="tab"
          aria-selected={activeMode === 'signup'}
        >
          Sign up
        </button>
      </div>

      <form className="auth-modal__form" onSubmit={handleSubmit}>


        {activeMode === 'signup' && (
          <div className="auth-modal__row">
            <div className="auth-modal__field">
              <label htmlFor="firstName">First name</label>
              <input
                id="firstName"
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
                autoComplete="given-name"
              />
            </div>
            <div className="auth-modal__field">
              <label htmlFor="lastName">Last name</label>
              <input
                id="lastName"
                value={lastName}
                onChange={(e) => setLastName(e.target.value)}
                autoComplete="family-name"
              />
            </div>
          </div>
        )}

        <div className="auth-modal__row">
          <div className="auth-modal__field">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              type="email"
              autoComplete="email"
            />
          </div>
          <div className="auth-modal__field">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              type="password"
              autoComplete={activeMode === 'signup' ? 'new-password' : 'current-password'}
            />
          </div>
        </div>

        {error && <div className="auth-modal__error">{error}</div>}

        <button className="auth-modal__submit" type="submit">
          {submitLabel}
        </button>

        <div className="auth-modal__fineprint">
          By clicking “{submitLabel}”, you agree that DeepScan may store and process your data to
          deliver the DeepScan service.
        </div>
      </form>
    </div>
  );
}

