import React, { useEffect, useState } from 'react';
import './HeroOrb.css';

export default function HeroOrb() {
  const [rotation, setRotation] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e) => {
      // Calculate tilt based on mouse position relative to screen center
      const centerX = window.innerWidth / 2;
      const centerY = window.innerHeight / 2;
      const moveX = (e.clientX - centerX) / centerX;
      const moveY = (e.clientY - centerY) / centerY;
      
      // subtle tilt up to 15 degrees
      setRotation({
        x: -moveY * 15,
        y: moveX * 15
      });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className="hero-orb-container">
      <div 
        className="hero-orb-wrapper"
        style={{
          transform: `rotateX(${rotation.x}deg) rotateY(${rotation.y}deg)`
        }}
      >
        <div className="hero-orb-beam" />
        <div className="hero-orb-ring" />
        <div className="hero-orb" />
      </div>
    </div>
  );
}
