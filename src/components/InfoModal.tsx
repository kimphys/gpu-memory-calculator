'use client';

import { useSimulatorStore } from '../store/useSimulatorStore';
import { X, Info, HelpCircle } from 'lucide-react';

export default function InfoModal() {
  const store = useSimulatorStore();

  if (!store.isInfoOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm transition-opacity">
      <div className="glass-panel w-full max-w-2xl max-h-[90vh] overflow-y-auto flex flex-col shadow-2xl animate-in fade-in zoom-in-95 duration-200">
        
        <div className="flex items-center justify-between p-6 border-b border-white/10 sticky top-0 bg-[#0f1115]/80 backdrop-blur-xl z-10">
          <h2 className="text-xl font-bold flex items-center gap-2 text-white">
            <Info className="w-5 h-5 text-primary-500" />
            VRAM & Speed Calculation Logic
          </h2>
          <button 
            onClick={() => store.setIsInfoOpen(false)}
            className="p-2 hover:bg-white/10 rounded-full transition-colors"
          >
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        <div className="p-6 flex flex-col gap-8 text-gray-300 text-sm leading-relaxed">
          
          <section className="flex flex-col gap-2">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-primary-500"></span>
              Model Weights (가중치 메모리)
            </h3>
            <p>
              LLM 자체를 GPU 메모리(VRAM)에 올리기 위한 최소한의 고정 비용입니다. 모델의 총 파라미터 수와 정밀도(Quantization Bits)에 의해 결정됩니다.
            </p>
            <div className="bg-surface-100 p-3 rounded-lg border border-white/5 font-mono text-xs text-primary-100">
              Weights (GB) = 파라미터 수(Billion) × (정밀도 Bit ÷ 8)
            </div>
            <p className="text-gray-400 text-xs">
              예시: 8B(80억 개) 파라미터 모델을 기본 FP16(16-bit)으로 로드하면 8 × (16/8) = 약 16 GB 소모.
            </p>
          </section>

          <section className="flex flex-col gap-2">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-accent-500"></span>
              KV Cache (컨텍스트 캐시 메모리)
            </h3>
            <p>
              추론(Inference) 과정에서 토큰을 생성할 때, 이전 문맥의 연산 결과를 저장해두는 공간인 K(Key), V(Value) 텐서의 크기입니다. <strong className="text-white">사용자 입력 길이(Context Length)와 동시 처리 유저 수(Batch Size)가 커질수록 기하급수적으로 증가합니다.</strong> GQA(Grouped-Query Attention) 기술이 적용된 모델은 KV Heads 수가 적어 캐시 소모량이 크게 줄어듭니다.
            </p>
            <div className="bg-surface-100 p-3 rounded-lg border border-white/5 font-mono text-xs text-accent-100 mb-1 overflow-x-auto whitespace-nowrap">
              KV = 2 × Batch Size × Context Length × KV Heads × Head Dim × Layers × 2 bytes
            </div>
          </section>

          <section className="flex flex-col gap-2">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-warning"></span>
              Training Overhead (학습 메모리 비용)
            </h3>
            <p>
              모델을 파인튜닝(Full Fine-tuning) 할 때는 순전파/역전파 과정의 Activations, Gradients, 그리고 AdamW 옵티마이저 등 상태값을 저장해야 하므로 <strong className="text-white">기본 모델 크기의 3~4배에 달하는 거대한 VRAM오버헤드</strong>가 추가로 발생합니다.
            </p>
            <p>
              하지만 <strong>PEFT / LoRA</strong> 기법을 사용하면 모델 가중치의 극히 일부(저랭크 행렬)만 학습하므로 이 막대한 메모리 비용을 기존의 절반 이하 수준으로 드라마틱하게 줄일 수 있습니다.
            </p>
          </section>

          <section className="flex flex-col gap-2 border-t border-white/10 pt-4">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <HelpCircle className="w-4 h-4 text-gray-400" />
              Estimated Speed (예상 초당 토큰 생성 속도)
            </h3>
            <p>
              초당 생성 가능한 토큰 수(Tokens / sec)는 (배치 사이즈 1 기준) 연산 능력(FLOPS)보다 <strong>GPU의 물리적 메모리 대역폭(Memory Bandwidth)</strong>에 100% 병목이 발생합니다. (Memory-Bound 현상)
            </p>
            <div className="bg-surface-100 p-3 rounded-lg border border-white/5 font-mono text-xs text-gray-200">
              Speed (Tokens/sec) = 총 메모리 대역폭(GB/s) ÷ 모델 가중치 용량(GB)
            </div>
            <p className="text-gray-400 text-xs mt-1">
              * 멀티 GPU 사용 시 Tensor Parallelism 통신 지연(Overhead)을 고려해 총 대역폭 효율을 보수적으로 80%로 산정합니다. 본 수치는 하드웨어의 이론상 100% 한계치(Upper Bound)를 의미하며, vLLM 적용 시 실제 기대 속도와 근사합니다.
            </p>
          </section>

        </div>
      </div>
    </div>
  );
}
