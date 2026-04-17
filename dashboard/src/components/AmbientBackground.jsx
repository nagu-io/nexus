import React from 'react';

export default function AmbientBackground() {
  return (
    <div className="fixed inset-0 z-[-1] pointer-events-none overflow-hidden bg-[var(--app-bg)]">
      {/* Intense Background Mesh Gadients */}
      <div className="absolute inset-0 bg-gradient-to-b from-[#080b12] to-[#05070a] z-0"></div>

      {/* Dynamic Morphing Orbs */}
      <div className="absolute -top-[20%] -left-[10%] w-[60%] h-[60%] rounded-full bg-[rgba(0,240,255,0.08)] blur-[100px] mix-blend-screen animate-morph-slow z-10"></div>
      <div className="absolute top-[40%] -right-[10%] w-[50%] h-[50%] rounded-full bg-[rgba(191,0,255,0.06)] blur-[120px] mix-blend-screen animate-morph-medium z-10" style={{ animationDelay: '2s' }}></div>
      <div className="absolute -bottom-[20%] left-[20%] w-[70%] h-[50%] rounded-full bg-[rgba(0,128,255,0.05)] blur-[150px] mix-blend-screen animate-morph-fast z-10" style={{ animationDelay: '4s' }}></div>
      
      {/* Noise Overlay Effect */}
      <div className="absolute inset-0 noise-layer opacity-40 z-20"></div>
    </div>
  );
}
