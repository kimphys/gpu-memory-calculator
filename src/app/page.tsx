'use client';

import InputPanel from '../components/InputPanel';
import ResultsDashboard from '../components/ResultsDashboard';
import InfoModal from '../components/InfoModal';
import { Info } from 'lucide-react';
import { useSimulatorStore } from '../store/useSimulatorStore';

export default function Home() {
  const setIsInfoOpen = useSimulatorStore(s => s.setIsInfoOpen);
  
  return (
    <main className="min-h-screen p-4 md:p-8 flex flex-col items-center">
      <InfoModal />
      <div className="max-w-[1400px] w-full flex flex-col gap-6 h-[calc(100vh-4rem)]">
        
        <header className="flex flex-col gap-2 shrink-0">
          <div className="flex justify-between items-start gap-4">
            <h1 className="text-3xl md:text-4xl font-black bg-gradient-to-r from-primary-500 to-accent-500 bg-clip-text text-transparent drop-shadow-md">
              LLM GPU Memory & Performance Simulator
            </h1>
            <button 
              onClick={() => setIsInfoOpen(true)}
              className="flex items-center gap-2 text-sm font-medium text-gray-300 hover:text-white bg-surface-100 hover:bg-surface-200 border border-white/10 px-4 py-2 flex-shrink-0 rounded-full transition-all shadow-sm"
            >
              <Info className="w-4 h-4 text-primary-500" />
              <span>계산 기준 보기</span>
            </button>
          </div>
          <p className="text-gray-400 max-w-2xl text-sm md:text-base font-medium">
            Quickly estimate VRAM requirements and token generation speeds for popular open-source LLMs across various NVIDIA GPU configurations.
          </p>
        </header>

        <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-0">
          {/* Left: Input Configurations */}
          <div className="col-span-1 lg:col-span-5 h-full overflow-hidden">
            <InputPanel />
          </div>

          {/* Right: Results Dashboard */}
          <div className="col-span-1 lg:col-span-7 h-full overflow-hidden">
            <ResultsDashboard />
          </div>
        </div>
      </div>
    </main>
  );
}
