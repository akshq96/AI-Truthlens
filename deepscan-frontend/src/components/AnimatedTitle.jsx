import React, { useEffect, useState } from 'react';

export default function AnimatedTitle({
  eyebrow = 'DeepScan',
  firstLine = 'Building',
  secondLine = 'trust online',
  subtitle = 'Upload an image and get a fast authenticity score using artifact and metadata signals.',
}) {
  const [glitch, setGlitch] = useState(false);

  useEffect(() => {
    // trigger glitch animation on mount
    setGlitch(true);
    const timer = setTimeout(() => setGlitch(false), 500);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="animated-hero">
      <div className={`animated-hero__eyebrow glitch-flicker ${glitch ? 'animate' : ''}`}>
        {eyebrow}
      </div>

      <h1 className="animated-hero__title typewriter" aria-label={`${firstLine} ${secondLine}`}>
        {firstLine} <span className="animated-hero__line--muted">{secondLine}</span>
      </h1>

      <p className="animated-hero__subtitle fade-in-delay">{subtitle}</p>
    </div>
  );
}

