import React, { useEffect, useState } from 'react';

const tasks = [
  { id: 'auth', label: 'initializing core systems', total: 100, current: 0 },
  { id: 'agent', label: 'connecting neural pathways', total: 100, current: 0 },
  { id: 'reflect', label: 'warming aesthetic engines', total: 100, current: 0 },
];

export default function LoadingScreen({ onReady }) {
  const [progress, setProgress] = useState(tasks);
  const [allReady, setAllReady] = useState(false);

  useEffect(() => {
    let timers = [];
    
    // Smooth progress animation
    progress.forEach((task, idx) => {
      let currentVal = 0;
      const simulate = () => {
        currentVal += Math.random() * 8 + 2; 
        if (currentVal >= 100) currentVal = 100;

        setProgress(prev => {
          const newProg = [...prev];
          newProg[idx] = { ...newProg[idx], current: currentVal };
          
          const allDone = newProg.every(p => p.current >= 100);
          if (allDone && !allReady) {
            setAllReady(true);
            setTimeout(() => onReady(), 800);
          }
          return newProg;
        });

        if (currentVal < 100) {
          timers.push(setTimeout(simulate, Math.random() * 100 + 50));
        }
      };
      
      timers.push(setTimeout(simulate, Math.random() * 400 + idx * 300));
    });

    return () => timers.forEach(clearTimeout);
  }, [allReady, onReady]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-[var(--text)] transition-opacity duration-1000 relative z-10 px-6">
      <div className="w-full max-w-md animate-slide-up">
        <div className="text-center mb-12">
          <h1 className="font-display text-5xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 mb-4 drop-shadow-[0_0_25px_rgba(0,240,255,0.4)]">
            NEXUS
          </h1>
          <p className="text-[#94a3b8] text-[11px] tracking-widest uppercase font-mono bg-black/30 py-1.5 px-4 rounded-full inline-block backdrop-blur-md border border-[var(--border)] shadow-[0_4px_10px_rgba(0,0,0,0.5)]">
            v0.2.0 • Autonomous Core
          </p>
        </div>
        
        <div className="space-y-6">
          {progress.map((t, index) => (
            <div key={t.id} className="w-full animate-slide-up opacity-0" style={{ animationDelay: `${index * 150}ms`, animationFillMode: 'forwards' }}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-mono text-[var(--accent)] uppercase tracking-widest">{t.label}</span>
                <span className="text-[10px] font-mono text-[var(--text-soft)]">{Math.round(t.current)}%</span>
              </div>
              <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden relative backdrop-blur-sm border border-[var(--border)]">
                <div 
                  className="absolute top-0 left-0 h-full bg-gradient-to-r from-[var(--accent)] to-[var(--accent-2)] transition-all duration-300 ease-out shadow-[0_0_15px_rgba(0,240,255,0.6)]" 
                  style={{ width: `${t.current}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-12 text-center h-6 animate-slide-up opacity-0" style={{ animationDelay: `600ms`, animationFillMode: 'forwards' }}>
          {allReady ? (
            <span className="text-[var(--success)] text-[11px] font-mono tracking-widest uppercase animate-pulse drop-shadow-[0_0_10px_rgba(16,185,129,0.5)]">
              Core Systems Online
            </span>
          ) : (
            <span className="text-[var(--text-soft)] text-[11px] font-mono tracking-widest uppercase animate-pulse-slow">
              Initializing...
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
