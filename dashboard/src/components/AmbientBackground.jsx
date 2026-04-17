import React from 'react';

export default function AmbientBackground() {
  return (
    <div className="fixed inset-0 z-[-1] pointer-events-none overflow-hidden bg-[#000000]">
      {/* Absolute Obsidian Depth Gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-[#020202] to-[#000000] z-0" />

      {/* Cinematic Fog & Atmospheric Blurs */}
      <div className="absolute -top-[30%] -left-[20%] w-[80%] h-[70%] rounded-full bg-[rgba(255,255,255,0.02)] blur-[120px] animate-morph-slow z-10" />
      <div className="absolute top-[40%] -right-[15%] w-[60%] h-[60%] rounded-full bg-[rgba(255,255,255,0.015)] blur-[140px] animate-morph-medium z-10" style={{ animationDelay: '3s' }} />
      <div className="absolute -bottom-[40%] left-[10%] w-[90%] h-[60%] rounded-full bg-[rgba(255,255,255,0.01)] blur-[180px] animate-morph-slow z-10" style={{ animationDelay: '6s', animationDirection: 'reverse' }} />
      
      {/* High-Contrast Grain Layer */}
      <div className="absolute inset-0 noise-layer pointer-events-none" />
    </div>
  )
}
