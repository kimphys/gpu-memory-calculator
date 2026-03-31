'use client';

import { useSimulatorStore } from '../store/useSimulatorStore';
import { X, Info, HelpCircle, AlertTriangle, Cpu, Database, Activity, Zap } from 'lucide-react';

export default function InfoModal() {
  const store = useSimulatorStore();

  if (!store.isInfoOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm transition-opacity">
      <div className="glass-panel w-full max-w-4xl max-h-[90vh] overflow-y-auto flex flex-col shadow-2xl animate-in fade-in zoom-in-95 duration-200">
        
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-white/10 sticky top-0 bg-[#0f1115]/80 backdrop-blur-xl z-10">
          <h2 className="text-xl font-bold flex items-center gap-2 text-white">
            <Info className="w-5 h-5 text-primary-500" />
            VRAM & Performance Calculation Logic
          </h2>
          <button 
            onClick={() => store.setIsInfoOpen(false)}
            className="p-2 hover:bg-white/10 rounded-full transition-colors"
          >
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 flex flex-col gap-8 text-gray-300 text-sm leading-relaxed">
          
          {/* Section 1: Memory (VRAM) Models */}
          <section className="flex flex-col gap-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2 border-b border-white/10 pb-2">
              <Database className="w-5 h-5 text-accent-500" />
              1. Memory Capacity (VRAM 점유율 모델)
            </h3>
            
            <div className="space-y-5">
              <div>
                <h4 className="font-semibold text-white flex items-center gap-2 mb-1">
                  <span className="w-2 h-2 rounded-full bg-primary-500"></span>
                  Model Weights (가중치 메모리)
                </h4>
                <p className="mb-2">
                  LLM을 GPU에 로드하기 위한 최소한의 고정 비용입니다. 파라미터 수와 양자화(Quantization) 비트 수에 비례합니다.
                </p>
                <code className="block bg-surface-100 p-3 rounded-lg border border-white/5 font-mono text-xs text-primary-100">
                  Weights (GB) = Parameters (Billion) × (Quantization Bits ÷ 8)
                </code>
              </div>

              <div>
                <h4 className="font-semibold text-white flex items-center gap-2 mb-1">
                  <span className="w-2 h-2 rounded-full bg-accent-500"></span>
                  KV Cache (동적 컨텍스트 캐시)
                </h4>
                <p className="mb-2">
                  토큰을 생성할 때 과거 문맥을 기억해두는 VRAM 공간입니다. <strong>사용자의 입력/출력 길이와 동시 접속자 수(Batch Size)가 늘어날수록 기하급수적으로 팽창</strong>하여 OOM(Out of Memory)의 주원인이 됩니다.
                </p>
                <ul className="list-disc pl-5 space-y-1 text-gray-400 text-xs mb-2">
                  <li><strong className="text-white">FP8 KV 양자화:</strong> 적용 시 메모리 소모를 절반으로 줄입니다.</li>
                  <li><strong className="text-white">GQA (Grouped-Query Attention):</strong> Key/Value 헤드 개수를 줄여 캐시를 최적화한 모델(예: EXAONE, Llama3)은 소모량이 대폭 감소합니다.</li>
                  <li><strong className="text-white">Hybrid Architecture:</strong> Qwen 9B처럼 Linear Layer(Mamba 계열)가 섞인 모델은 전체 레이어가 아닌 Attention 레이어에만 무거운 KV 캐시가 쌓이므로 메모리를 크게 절약합니다.</li>
                </ul>
                <code className="block bg-surface-100 p-3 rounded-lg border border-white/5 font-mono text-xs text-accent-100 overflow-x-auto whitespace-nowrap">
                  KV Cache = 2 × Batch × (In + Out Len) × KV Heads × Head Dim × Attn Layers × (KV_Bits ÷ 8)
                </code>
              </div>
            </div>
          </section>

          {/* Section 2: Throughput & Performance Models */}
          <section className="flex flex-col gap-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2 border-b border-white/10 pb-2">
              <Zap className="w-5 h-5 text-yellow-500" />
              2. Core Performance Metrics (성능 지표 모델)
            </h3>
            
            <div className="grid grid-cols-1 gap-6">
              
              <div className="bg-surface-50/50 p-5 rounded-xl border border-white/5">
                <h4 className="font-bold text-white mb-3 flex items-center gap-2 text-base">
                  <Activity className="w-5 h-5 text-primary-400" /> 
                  Total Throughput & Speed per User (가장 중요한 속도 지표)
                </h4>
                <p className="mb-3 text-gray-300">
                  시스템의 생성 속도를 결정하는 핵심 지표입니다. 본 시뮬레이터는 업계 표준인 <strong>HPC Roofline Model</strong>을 기반으로 성능을 계산합니다. 매 토큰을 생성할 때 GPU는 가중치와 KV 캐시를 메모리에서 퍼와야 하고(Memory-bound), 동시에 행렬 곱 연산을 수행해야 합니다(Compute-bound). 이 둘 중 <strong>더 시간이 오래 걸리는 병목 지점</strong>이 전체 속도를 결정짓습니다.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-[13px] mb-3">
                  <div className="bg-black/20 p-3 rounded-lg border border-white/5">
                    <strong className="text-primary-300 block mb-1">Total Throughput (Tokens/sec)</strong>
                    시스템이 1초에 뱉어내는 <strong>전체 토큰의 합</strong>입니다. 배치(동시 처리 유저 수)가 커질수록 한 번에 메모리에서 가중치를 읽어와 여러 명에게 재사용할 수 있으므로 GPU의 연산 효율이 극대화되어 총 처리량은 증가합니다.
                  </div>
                  <div className="bg-black/20 p-3 rounded-lg border border-white/5">
                    <strong className="text-accent-300 block mb-1">Speed per User (Tokens/s/User)</strong>
                    개별 유저 화면에 글자가 찍히는 <strong>체감 속도</strong>입니다. <code className="text-xs bg-black/40 px-1 rounded">Total Throughput ÷ Batch Size</code>로 계산됩니다. 시스템 전체 효율은 올라도, 파이가 쪼개지므로 동시 접속자가 늘어날수록 유저 1명이 느끼는 속도는 떨어집니다.
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-300">
                <div className="bg-surface-50/50 p-4 rounded-xl border border-white/5">
                  <p className="font-bold text-white mb-2 flex items-center gap-2">
                    <span className="text-accent-400">01.</span> Max Concurrency (OOM Limit)
                  </p>
                  <p className="leading-relaxed">
                    동시에 처리할 수 있는 최대 유저(Batch) 수입니다. 하드웨어의 한계(남은 VRAM 용량 / 1명당 KV 캐시)와 사용자가 설정한 소프트웨어 제한(Batch Size 파라미터) 중 <strong>더 작은 값(병목 구간)</strong>을 실제 동시 처리량으로 채택합니다.
                  </p>
                </div>

                <div className="bg-surface-50/50 p-4 rounded-xl border border-white/5">
                  <p className="font-bold text-white mb-2 flex items-center gap-2">
                    <span className="text-yellow-400">02.</span> Est. TTFT (Time To First Token)
                  </p>
                  <p className="leading-relaxed">
                    새로운 유저의 긴 프롬프트를 한 번에 병렬로 읽어내는 <strong>Prefill 단계의 순수 연산 지연시간</strong>입니다. 큐(Queue) 대기열이 비어있는 상태에서 오직 1건의 요청을 온전히 처리할 때의 최고 속도(Baseline Latency)를 의미합니다.
                  </p>
                </div>

                <div className="bg-surface-50/50 p-4 rounded-xl border border-white/5 md:col-span-2">
                  <p className="font-bold text-white mb-2 flex items-center gap-2">
                    <span className="text-secondary-400">03.</span> Max RPS (Requests Per Second)
                  </p>
                  <p className="leading-relaxed">
                    시스템이 100% 가동률로 쉬지 않고 일할 때 도달할 수 있는 <strong>초당 최대 처리 완료 건수(절대 한계치)</strong>입니다. <code className="bg-black/30 px-1 py-0.5 rounded text-xs text-secondary-300">Total Throughput ÷ Output Length</code> 공식을 사용하여 계산됩니다.
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* Section 3: Reality Factors */}
          <section className="flex flex-col gap-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2 border-b border-white/10 pb-2">
              <Cpu className="w-5 h-5 text-gray-400" />
              3. Reality Factors (vLLM 현실 보정 로직)
            </h3>
            <ul className="space-y-3 text-[13px] text-gray-400 list-disc pl-5 bg-[#141820] border border-white/10 rounded-xl p-5">
              <li><strong className="text-primary-400">Continuous Batching 간섭 페널티:</strong> 기존 유저의 토큰을 한 글자씩 생성(Decode)하는 도중, 새로운 유저의 긴 프롬프트 연산(Prefill)이 끼어들면 GPU 코어가 선점되어 디코딩 효율이 떨어집니다. 입력 길이와 동시 접속자 수에 비례하여 동적 페널티를 부여합니다.</li>
              <li><strong className="text-primary-400">Hybrid Architecture 페널티:</strong> 일반적인 PagedAttention과 달리, Mamba나 Linear Attention 같은 State-Space 커널은 vLLM 환경에서 메모리 I/O 및 컨텍스트 스위칭 오버헤드가 큽니다. 시뮬레이터는 이를 반영하여 Linear Layer가 포함된 모델에 25%의 효율 감소를 적용합니다.</li>
              <li><strong className="text-primary-400">CUDA & System Overhead:</strong> GPU 드라이버, CUDA 컨텍스트, 액티베이션 버퍼 등 실제 구동 시 소비되는 '숨겨진 메모리'를 5GB로 강제 할당하여 안전한 예측을 돕습니다.</li>
            </ul>
          </section>

          {/* Section 4: CRITICAL DISCLAIMERS (주의점) */}
          <section className="flex flex-col gap-4 border-t border-white/10 pt-6 mt-4">
            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 shadow-[0_0_15px_rgba(239,68,68,0.1)]">
              <h3 className="text-lg font-bold text-red-400 flex items-center gap-2 mb-4">
                <AlertTriangle className="w-6 h-6" />
                [필독] 시뮬레이터 결과 해석 시 주의점
              </h3>
              
              <div className="space-y-6 text-[13px]">
                <div>
                  <h4 className="font-bold text-red-300 mb-1 text-sm">🚨 1. TTFT는 '대기열 지연(Queueing Delay)'을 포함하지 않습니다.</h4>
                  <p className="text-red-200/80 leading-relaxed">
                    본 시뮬레이터의 TTFT(예: 299ms)는 대기 줄이 없는 쾌적한 상태에서의 <strong>순수 연산 속도(Baseline)</strong>입니다. 
                    실제 서비스 시 트래픽이 몰려 큐(Queue)에 요청이 쌓이기 시작하면, 사용자가 체감하는 지연시간(P90)은 내 차례를 기다리는 시간 때문에 벤치마크처럼 1초~2초 이상 기하급수적으로 폭증할 수 있습니다.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-bold text-red-300 mb-1 text-sm">🚨 2. 가동률 100%의 함정 (Max RPS대로 부하를 주면 안 됩니다)</h4>
                  <p className="text-red-200/80 leading-relaxed">
                    도출된 <strong>Max RPS</strong>는 GPU가 100% 혹사당할 때 시스템이 뱉어낼 수 있는 절대적인 '물리적 천장'입니다. 현실의 트래픽은 포아송(Poisson) 분포처럼 불규칙하게 몰립니다. 만약 1초 한계치가 2.0건인데 평균 2.0건의 트래픽을 지속적으로 넣는다면, 한 번 트래픽이 몰렸을 때 큐를 비워낼 여유 체력이 없어 시스템 대기열이 끝없이 늘어납니다. 
                    <br/><strong className="text-red-300">→ 안정적인 서비스를 위해서는 목표 트래픽을 Max RPS의 30~50% 수준 이하로 여유롭게 설계해야 합니다.</strong>
                  </p>
                </div>

                <div>
                  <h4 className="font-bold text-red-300 mb-1 text-sm">🚨 3. 소프트웨어 최적화 수준에 따른 편차</h4>
                  <p className="text-red-200/80 leading-relaxed">
                    동일한 파라미터 사이즈(예: 8B)라 하더라도 GQA 비율, Vocab Size 확장비, 활성화 함수 등 내부 아키텍처의 미세한 차이와 추론 엔진(vLLM, TensorRT-LLM)의 버전별 커널 최적화 수준에 따라 실제 벤치마크에서는 ±15% 이상의 편차나 순위 역전이 발생할 수 있습니다. 본 시뮬레이터의 값은 장비 도입을 위한 '가장 확률 높은 기준선'으로 활용하시기 바랍니다.
                  </p>
                </div>

                {/* 추가된 주의 문구 */}
                <div>
                  <h4 className="font-bold text-red-300 mb-1 text-sm">🚨 4. 극한 상황에서의 시뮬레이션 오차 (Edge Cases)</h4>
                  <p className="text-red-200/80 leading-relaxed">
                    본 시뮬레이터는 선형적인 물리 모델을 기반으로 합니다. 따라서 <strong>① 설정한 동시 요청 수(Batch Size)가 VRAM 한계치(OOM)에 극도로 근접하거나, ② 입/출력 토큰 길이가 모델의 최대 처리 길이에 가까워질수록</strong> 예측값과 실측값의 오차가 커질 수 있습니다. 극한의 상황에서는 메모리 파편화 및 CPU 스케줄링 병목이 비선형적으로 증가하여 실제 처리 성능이 예상보다 더 깎이게 됩니다.
                  </p>
                </div>
              </div>
            </div>
          </section>

        </div>
      </div>
    </div>
  );
}