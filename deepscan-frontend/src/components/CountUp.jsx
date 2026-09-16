import React, { useEffect, useState } from 'react';

export default function CountUp({ end, duration = 2000, suffix = '', decimals = 0, isTime = false }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let startTime = null;
    let animationFrame;
    const animate = (currentTime) => {
      if (!startTime) startTime = currentTime;
      const progress = Math.min((currentTime - startTime) / duration, 1);
      
      // easeOutExpo
      const easeProgress = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
      
      setCount(easeProgress * end);
      
      if (progress < 1) {
        animationFrame = requestAnimationFrame(animate);
      }
    };
    animationFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrame);
  }, [end, duration]);

  let formatted = count.toFixed(decimals);
  if (!isTime && decimals === 0) {
    formatted = Math.floor(count).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  return <span>{formatted}{suffix}</span>;
}
