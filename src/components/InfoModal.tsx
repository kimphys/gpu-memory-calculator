'use client';

import { useSimulatorStore } from '../store/useSimulatorStore';
import { X, Info, HelpCircle } from 'lucide-react';

export default function InfoModal() {
  const store = useSimulatorStore();

  if (!store.isInfoOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm transition-opacity">
      <div className="glass-panel w-full max-w-4xl max-h-[90vh] overflow-y-auto flex flex-col shadow-2xl animate-in fade-in zoom-in-95 duration-200">
        
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

          <section className="flex flex-col gap-4 border-t border-white/10 pt-6">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <HelpCircle className="w-5 h-5 text-primary-500" />
              Advanced Performance Metrics
            </h3>
            
            <div className="grid grid-cols-1 gap-6 text-sm text-gray-300">
              <div className="bg-surface-50/50 p-4 rounded-xl border border-white/5">
                <p className="font-bold text-white mb-2 flex items-center gap-2">
                  <span className="text-primary-400">01.</span> Total Throughput (Tokens/sec)
                </p>
                <ul className="list-none p-0 m-0 mb-2"> {/* Added ul to contain the li */}
                  <li className="flex gap-2 text-primary-400">
                    <span>•</span>
                    <span><strong>단위 기준</strong>: 본 시뮬레이터는 직관성을 위해 모든 단위에 10진수(1GB = 10⁹ bytes)를 사용합니다.</span>
                  </li>
                </ul>
                <p className="leading-relaxed">
                  시스템이 출력하는 초당 전체 토큰 수입니다. <strong>Roofline Model</strong>을 기반으로 GPU의 메모리 대역폭(Memory-bound)과 연산량(Compute-bound) 중 더 심각한 병목 지점을 자동으로 찾아 계산합니다. 
                  <span className="text-gray-400 block mt-1 text-[12px]">※ 여기에 기업용 운영 환경을 고려한 0.8배의 보정 계수가 적용되어 있습니다.</span>
                </p>
              </div>

              <div className="bg-surface-50/50 p-4 rounded-xl border border-white/5">
                <p className="font-bold text-white mb-2 flex items-center gap-2">
                  <span className="text-accent-400">02.</span> Max Concurrency (OOM Limit)
                </p>
                <p className="leading-relaxed">
                  VRAM이 고갈(OOM)되지 않고 동시에 수용할 수 있는 최대 유저(Batch) 수입니다. 
                  가중치와 시스템 예약을 제외한 나머지 VRAM을 <strong>PagedAttention</strong> 슬롯으로 나누어 계산합니다. 
                  본 시뮬레이터는 모든 슬롯이 꽉 찬 최악의 조건을 가정하여 안전한 장비 도입 가이드를 제공합니다.
                </p>
              </div>

              <div className="bg-surface-50/50 p-4 rounded-xl border border-white/5">
                <p className="font-bold text-white mb-2 flex items-center gap-2">
                  <span className="text-yellow-400">03.</span> Est. TTFT (Time To First Token)
                </p>
                <p className="leading-relaxed">
                  첫 글자가 나오기까지의 지연 시간입니다. 입력 프롬프트(Input)를 한꺼번에 읽는 <strong>Prefill</strong> 단계의 속도를 의미하며, 데이터 읽기보다는 GPU의 순수 연산력(TFLOPS)에 크게 좌우됩니다.
                </p>
              </div>

              <div className="bg-surface-50/50 p-4 rounded-xl border border-white/5">
                <p className="font-bold text-white mb-2 flex items-center gap-2">
                  <span className="text-secondary-400">04.</span> Max RPS (Requests Per Second)
                </p>
                <p className="leading-relaxed">
                  초당 완결 가능한 요청 수입니다. <strong>TTFT + (출력길이 / 생성속도)</strong> 공식을 사용하여, 입력을 빠르게 읽고 출력을 느리게 생성하는 실제 LLM의 라이프사이클을 정확히 모델링합니다.
                </p>
              </div>
            </div>

            <div className="mt-4 p-5 bg-primary-950/20 border border-primary-500/20 rounded-xl">
              <h4 className="text-white font-bold mb-3 flex items-center gap-2">📐 Enterprise Reality Factors (보정 계수)</h4>
              <ul className="space-y-3 text-xs text-gray-300 list-disc pl-4">
                <li><strong className="text-primary-300">CUDA/Static Overhead (5GB)</strong>: GPU 드라이버, CUDA 컨텍스트, 액티베이션 버퍼 등 실제 구동 시 '숨겨진' 고정 메모리 비용을 강제로 차감하여 OOM 예측의 신뢰도를 높였습니다.</li>
                <li><strong className="text-primary-300">Realistic Derating (0.8x)</strong>: 기업용 로깅, 모니터링, 스케줄링 간섭을 고려하여 이론적 최대 속도에서 20%를 감쇄한 '실효 속도'를 보여줍니다.</li>
              </ul>
            </div>

             <div className="bg-surface-100 p-4 rounded-lg border border-white/5 font-mono text-[11px] text-gray-200 mt-2 leading-relaxed">
              <span className="text-gray-500">// 핵심 물리 모델 수식 (HPC Standard)</span><br/>
              Bandwidth (GiB/s) = (GBps × 10⁹) ÷ 1024³<br/>
              T_decode (Step) ≈ MAX(Weights/Bandwidth, Compute/Tflops) + 2ms<br/>
              Activation (GiB) ≈ (2 × Batch × Seq × Hidden × Layers × 2 bytes) ÷ 1024³<br/>
              Optimizer (GiB) ≈ (Params × 16 bytes) ÷ 1024³
            </div>
            <div className="space-y-4">
              <h3 className="text-lg font-bold text-accent-400 flex items-center gap-2">🛡️ 학습 시뮬레이션 (Training Simulation)</h3>
              <div className="space-y-4 text-sm text-gray-300 leading-relaxed bg-white/2 p-4 rounded-xl border border-white/5">
                <div>
                  <strong className="text-white block mb-1">총 학습 토큰 수 (Total Tokens)</strong>
                  <p>데이터셋 샘플 수(Samples) × 에포크(Epochs) × 컨텍스트 길이(Context)로 계산됩니다. 모델이 전체 학습 과정에서 보게 되는 총 데이터의 양입니다.</p>
                </div>
                <div>
                  <strong className="text-white block mb-1">연산량 물리 모델 (Total FLOPs)</strong>
                  <p>업계 표준인 <strong>6P tokens</strong> 공식을 사용합니다. 파라미터 1개당 토큰 1개를 처리하는 데 약 6번의 부동소수점 연산(Forward 3, Backward 3)이 소요된다는 물리적 기반의 추정치입니다.</p>
                  <code className="block mt-2 p-2 bg-black/30 rounded text-accent-300">Total FLOPs = 6 × Parameters × Total Tokens</code>
                </div>

              </div>
            </div>
          </section>

        </div>
      </div>
    </div>
  );
}
