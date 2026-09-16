import React, { useEffect, useRef } from 'react';

export default function CanvasParticles() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let animationId;
    let particles = [];
    const numParticles = window.innerWidth < 768 ? 40 : 90;

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    let mouse = { x: null, y: null };
    const handleMouseMove = (e) => { mouse.x = e.clientX; mouse.y = e.clientY; };
    const handleMouseLeave = () => { mouse.x = null; mouse.y = null; };
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseleave', handleMouseLeave);

    // Forest Luxe palette
    const colors = ['#0A1C1A', '#D9C8A9', '#10B981', '#0E2623', '#7A8C8A'];

    class Particle {
      constructor() {
        this.reset();
      }

      reset() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 1.8 + 0.5;
        this.opacity = Math.random() * 0.6 + 0.2;
        this.color = colors[Math.floor(Math.random() * colors.length)];
        this.baseVelocity = {
          x: (Math.random() - 0.5) * 0.4,
          y: (Math.random() - 0.5) * 0.4,
        };
        this.twinkleSpeed = Math.random() * 0.02 + 0.005;
        this.twinkleDir = Math.random() > 0.5 ? 1 : -1;
      }

      update() {
        // Drift
        this.x += this.baseVelocity.x;
        this.y += this.baseVelocity.y;

        // Twinkle
        this.opacity += this.twinkleSpeed * this.twinkleDir;
        if (this.opacity > 0.8 || this.opacity < 0.1) this.twinkleDir *= -1;

        // Repel from mouse
        if (mouse.x !== null && mouse.y !== null) {
          const mx = this.x - mouse.x;
          const my = this.y - mouse.y;
          const mouseDist = Math.sqrt(mx * mx + my * my);
          if (mouseDist < 100 && mouseDist > 0) {
            const force = (100 - mouseDist) / 100;
            this.x += (mx / mouseDist) * force * 2.5;
            this.y += (my / mouseDist) * force * 2.5;
          }
        }

        // Wrap
        if (this.x < 0) this.x = canvas.width;
        if (this.x > canvas.width) this.x = 0;
        if (this.y < 0) this.y = canvas.height;
        if (this.y > canvas.height) this.y = 0;
      }

      draw() {
        ctx.globalAlpha = this.opacity;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = this.color;
        ctx.shadowBlur = 8;
        ctx.shadowColor = this.color;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1;
      }
    }

    // Draw connecting lines between nearby particles
    function drawConnections() {
      const maxDist = 120;
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < maxDist) {
            const alpha = (1 - dist / maxDist) * 0.12;
            ctx.globalAlpha = alpha;
            ctx.strokeStyle = particles[i].color;
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
            ctx.globalAlpha = 1;
          }
        }
      }
    }

    for (let i = 0; i < numParticles; i++) {
      particles.push(new Particle());
    }

    const animate = () => {
      if (!document.hidden) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawConnections();
        particles.forEach(p => { p.update(); p.draw(); });
      }
      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseleave', handleMouseLeave);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0, left: 0,
        width: '100%', height: '100%',
        zIndex: 0,
        pointerEvents: 'none',
      }}
    />
  );
}
