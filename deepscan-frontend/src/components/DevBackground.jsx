import React, { useEffect, useRef } from 'react';

/**
 * DevBackground — Neural Mesh & Code Artifacts
 * A high-performance canvas background with interactive nodes, 
 * glowing connections, and drifting dev symbols.
 */
export default function DevBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animationFrameId;

    let width = window.innerWidth;
    let height = window.innerHeight;

    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width;
      canvas.height = height;
    };

    window.addEventListener('resize', resize);
    resize();

    // Configuration
    const NODE_COUNT = 80;
    const CONNECTION_DIST = 200;
    const MOUSE_REPULSE_DIST = 160;
    const SYMBOLS = ['{', '}', '=>', '[]', '<>', '&&', '||', 'null', 'void', 'const', '01', 'export', 'await'];
    
    class Node {
      constructor() {
        this.init();
      }

      init() {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        this.vx = (Math.random() - 0.5) * 0.5;
        this.vy = (Math.random() - 0.5) * 0.5;
        this.radius = Math.random() * 2 + 0.8;
        this.symbol = SYMBOLS[Math.floor(Math.random() * SYMBOLS.length)];
        this.isSymbol = Math.random() > 0.65;
        this.pulse = Math.random() * Math.PI;
      }

      update(mouse) {
        this.x += this.vx;
        this.y += this.vy;

        // Mouse Interactivity
        const dx = this.x - mouse.x;
        const dy = this.y - mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        if (dist < MOUSE_REPULSE_DIST && dist > 0) {
          const force = (MOUSE_REPULSE_DIST - dist) / MOUSE_REPULSE_DIST;
          this.x += (dx / dist) * force * 3;
          this.y += (dy / dist) * force * 3;
        }

        // Screen wrap
        if (this.x < -50) this.x = width + 50;
        if (this.x > width + 50) this.x = -50;
        if (this.y < -50) this.y = height + 50;
        if (this.y > height + 50) this.y = -50;

        this.pulse += 0.025;
      }

      draw() {
        const baseOpacity = 0.25 + (Math.sin(this.pulse) * 0.1);
        
        if (this.isSymbol) {
          ctx.font = '700 11px "JetBrains Mono", monospace';
          ctx.fillStyle = `rgba(16, 185, 129, ${baseOpacity * 1.5})`; // Emerald
          ctx.fillText(this.symbol, this.x, this.y);
        } else {
          ctx.beginPath();
          ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(217, 200, 169, ${baseOpacity})`; // Gold
          ctx.fill();
          
          // Small glow for nodes
          if (baseOpacity > 0.3) {
            ctx.shadowBlur = 10;
            ctx.shadowColor = 'rgba(217, 200, 169, 0.5)';
            ctx.fill();
            ctx.shadowBlur = 0;
          }
        }
      }
    }

    const nodes = Array.from({ length: NODE_COUNT }, () => new Node());
    const mouse = { x: -1000, y: -1000 };

    const handleMouseMove = (e) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    };

    window.addEventListener('mousemove', handleMouseMove);

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Draw Connections
      ctx.lineWidth = 1;
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < CONNECTION_DIST) {
            const alpha = (1 - dist / CONNECTION_DIST) * 0.25;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            
            // Gradient connection with more contrast
            const grad = ctx.createLinearGradient(nodes[i].x, nodes[i].y, nodes[j].x, nodes[j].y);
            grad.addColorStop(0, `rgba(16, 185, 129, ${alpha})`); // Emerald
            grad.addColorStop(1, `rgba(217, 200, 169, ${alpha})`); // Gold
            
            ctx.strokeStyle = grad;
            ctx.stroke();
          }
        }
      }

      // Update & Draw Nodes
      nodes.forEach(node => {
        node.update(mouse);
        node.draw();
      });

      animationFrameId = requestAnimationFrame(render);
    };


    render();

    return () => {
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', handleMouseMove);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      id="dev-background"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 0,
        pointerEvents: 'none',
        background: 'transparent'
      }}
    />
  );
}
