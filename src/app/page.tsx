'use client';

import InputPanel from '../components/InputPanel';
import ResultsDashboard from '../components/ResultsDashboard';
import InfoModal from '../components/InfoModal';
import { Info, Activity, AlertTriangle } from 'lucide-react';
import { useSimulatorStore } from '../store/useSimulatorStore';

export default function Home() {
  const setIsInfoOpen = useSimulatorStore(s => s.setIsInfoOpen);

  return (
    <main className="min-h-screen p-3 md:p-6 flex flex-col items-center">
      <InfoModal />
      <div className="max-w-[1400px] w-full flex flex-col gap-6">

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
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-4 bg-primary-500/5 border border-primary-500/10 rounded-2xl">
            <div className="flex items-center gap-3">
              <Activity className="w-5 h-5 text-primary-500 shrink-0" />
              <p className="text-sm text-gray-400 leading-relaxed tracking-tight">
                본 시뮬레이션은 이론적 모델에 기반한 이상적인 결과이며, 실제 서버 환경 및 프레임워크 오버헤드에 따라 오차가 발생할 수 있습니다.
                정밀한 계산 로직이 궁금하시다면 우측의 <strong className="text-white">"계산 기준 보기"</strong>를 클릭해 주세요.
              </p>
            </div>
          </div>
          <div className="mt-3 flex items-start gap-3 px-5 py-3 bg-amber-500/5 border border-amber-500/10 rounded-xl">
            {/* 아이콘 크기도 글자 크기에 맞춰 w-4에서 w-5로 살짝 키우는 게 균형이 맞습니다 */}
            <AlertTriangle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
            <p className="text-sm text-amber-200/70 leading-relaxed">
              <strong>주의:</strong> 설정값이 VRAM 한계에 근접할수록 비선형적 병목으로 인해 예측 오차가 커질 수 있습니다. 
              정확한 리소스 설계를 위해 <span className="text-amber-400 font-medium underline underline-offset-4">반드시 실제 환경에서의 벤치마크 측정을 병행</span>하시기 바랍니다.
            </p>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: Input Configurations */}
          <div className="col-span-1 lg:col-span-4">
            <InputPanel />
          </div>

          {/* Right: Results Dashboard */}
          <div className="col-span-1 lg:col-span-8">
            <ResultsDashboard />
          </div>
        </div>
      </div>
    </main>
  );
}
