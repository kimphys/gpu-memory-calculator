'use client';

import { useSimulatorStore } from '../store/useSimulatorStore';
import { calculateWeightsMemory, calculateKVCache, calculateTrainingMemory, estimatePerformance, calculateTrainingMetrics, PerformanceMetrics, CUDA_OVERHEAD_GB } from '../utils/calculator';
import { GPUS } from '../data/presets';
import { Cpu, Zap, Users, Activity, CheckCircle, Database, AlertTriangle, Server } from 'lucide-react';

export default function ResultsDashboard() {
  const store = useSimulatorStore();
  const activeModel = store.getActiveModelConfig();
  
  if (!activeModel) {
    return (
      <div className="glass-panel p-6 flex flex-col items-center justify-center h-full text-center">
        <p className="text-gray-400 animate-pulse text-lg">Waiting for model configuration...</p>
        <p className="text-gray-500 text-sm mt-2">Please configure the model on the left panel.</p>
      </div>
    );
  }

  const selectedGpu = GPUS.find(g => g.id === store.gpuId) || GPUS[0];
  const totalVramLimit = selectedGpu.vramGb * store.gpuCount;

  // Math Computations
  const weightsGb = calculateWeightsMemory(activeModel.parametersInB, store.params.quantizationBits);
  const kvGb = calculateKVCache(activeModel, store.params);
  
  // High-Fidelity Training Memory Decomposition
  let optGradGb = 0;
  let activationGb = 0;
  
  if (store.params.isTraining) {
    optGradGb = calculateTrainingMemory(activeModel, store.params.trainingMethod);
    // Activation memory is fetched from the metrics computed later, but we need it for totalUsedGb
    // We'll recompute it briefly here or restructure
    const { calculateTrainingMetrics } = require('../utils/calculator');
    const tMetrics = calculateTrainingMetrics(selectedGpu, activeModel, store.params, store.gpuCount);
    activationGb = tMetrics.activationMemoryGb;
  }
  
  const trainingGb = optGradGb + activationGb;
  const hiddenGb = CUDA_OVERHEAD_GB;
  
  const totalUsedGb = weightsGb + kvGb + trainingGb + hiddenGb;
  
  let metrics: PerformanceMetrics = { totalThroughput: 0, speedPerUser: 0, ttftMs: 0, maxRps: 0 };
  let trainingMetrics: any = null;
  let maxConcurrency = 0;

if (!store.params.isTraining) {
      if (totalUsedGb <= totalVramLimit) {
        metrics = estimatePerformance(
          { 
            bandwidthGbps: selectedGpu.bandwidthGbps, 
            tflops: (selectedGpu as any).tflops || 100,
            vramGb: (selectedGpu as any).vramGb || 24, // 👈 [추가됨] 드디어 VRAM 용량 전달!
            interconnectBandwidthGbps: (selectedGpu as any).interconnectBandwidthGbps || 64 // 👈 [추가됨] 멀티 GPU 통신 속도 전달!
          }, 
          activeModel, store.params, store.gpuCount
        );
      }
      const kvPerUser = kvGb / store.params.batchSize;
      const availableForKV = totalVramLimit - weightsGb - trainingGb - hiddenGb;
      maxConcurrency = kvPerUser > 0 ? Math.floor(availableForKV / kvPerUser) : 0;
  } else {
      trainingMetrics = calculateTrainingMetrics(
        { 
          bandwidthGbps: selectedGpu.bandwidthGbps, 
          tflops: (selectedGpu as any).tflops || 100,
          vramGb: (selectedGpu as any).vramGb || 24, // 👈 [추가됨]
          interconnectBandwidthGbps: (selectedGpu as any).interconnectBandwidthGbps || 64 // 👈 [추가됨]
        },
        activeModel, store.params, store.gpuCount
      );
  }

  // Status computation

  // Status computation
  const isOom = totalUsedGb > totalVramLimit;
  const isWarning = totalUsedGb > totalVramLimit * 0.8 && !isOom;
  
  const statusColor = isOom ? 'text-error' : isWarning ? 'text-warning' : 'text-success';
  const statusBg = isOom ? 'bg-error/20 border-error/50 shadow-[0_0_20px_rgba(239,68,68,0.3)]' : isWarning ? 'bg-warning/20 border-warning/50' : 'bg-success/20 border-success/50';

  // Stacked Bar relative segments (capping to 100% max for visual sanity, unless OOM where overflow gets the rest)
  const maxScale = Math.max(totalUsedGb, totalVramLimit);
  const weightsPct = (weightsGb / maxScale) * 100;
  const kvPct = (kvGb / maxScale) * 100;
  const hiddenPct = (hiddenGb / maxScale) * 100;
  const optGradPct = (optGradGb / maxScale) * 100;
  const activationPct = (activationGb / maxScale) * 100;
  const overflowPct = isOom ? ((totalUsedGb - totalVramLimit) / maxScale) * 100 : 0;

  // Helper for training time display
  const formatTime = (hours: number) => {
    if (hours < 1) return `${Math.round(hours * 60)}분`;
    if (hours < 48) return `${hours.toFixed(1)}시간`;
    return `${(hours / 24).toFixed(1)}일`;
  };

  return (
    <div className="glass-panel p-8 flex flex-col gap-8 w-full h-full">
      <div className="flex justify-between items-start">
        <h2 className="text-2xl font-bold flex items-center gap-3 text-white">
          <Zap className="w-6 h-6 text-accent-500" /> 시뮬레이션 결과
        </h2>
      </div>

      {/* Main Status */}
      <div className={`p-6 rounded-2xl border flex items-center gap-5 transition-all duration-500 ${statusBg}`}>
        {isOom ? <AlertTriangle className="w-12 h-12 text-error flex-shrink-0" /> : <CheckCircle className="w-12 h-12 text-success flex-shrink-0" />}
        <div>
          <h3 className={`text-2xl font-bold tracking-tight ${statusColor}`}>
            {isOom ? '메모리 부족 (OOM)' : isWarning ? '경고: 높은 VRAM 점유율' : '정상: VRAM 용량 충분'}
          </h3>
          <p className="text-gray-300 mt-2 text-sm leading-relaxed">
            {isOom 
              ? `${totalUsedGb.toFixed(1)} GB 가 필요하지만, ${store.gpuCount}x ${selectedGpu.name} 장비의 가용 용량은 ${totalVramLimit} GB 입니다.`
              : store.params.isTraining 
                ? `${totalVramLimit} GB 중 ${totalUsedGb.toFixed(1)} GB 사용 중입니다.`
                : `${totalVramLimit} GB 중 ${totalUsedGb.toFixed(1)} GB 사용 중입니다. (${hiddenGb}GB CUDA/시스템 예약 포함)`}
          </p>
        </div>
      </div>

      {/* VRAM Visualizer */}
      <div className="flex flex-col gap-4 mt-2">
        <div className="flex justify-between items-end">
          <span className="font-medium text-gray-300 flex items-center gap-2">
            VRAM 할당 세부 내역
            <div className="group relative">
              <Activity className="w-3 h-3 text-gray-500 cursor-help" />
              <div className="absolute left-0 bottom-full mb-2 w-64 p-2 bg-surface-200 border border-white/10 rounded-lg text-[10px] text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                PagedAttention 등의 동적 관리 덕분에 실제 사용량은 유동적일 수 있으나, 본 장비는 모든 토큰이 채워졌을 때의 피크 가동 상황을 기준으로 설계되었습니다.
              </div>
            </div>
          </span>
          <span className={`text-sm font-mono font-bold ${isOom ? 'text-error animate-pulse' : 'text-gray-400'}`}>
            {totalUsedGb.toFixed(1)} GB / {totalVramLimit} GB
          </span>
        </div>
        
        {/* Stacked Bar */}
        <div className="h-10 w-full bg-surface-200 rounded-full overflow-hidden flex shadow-inner relative border border-white/5">
          <div 
            className="h-full bg-gray-600 transition-all duration-1000 ease-out"
            style={{ width: `${hiddenPct}%` }}
            title={`CUDA/시스템 예약: ${hiddenGb.toFixed(1)} GB`}
          />
          <div 
            className="h-full bg-primary-500 transition-all duration-1000 ease-out"
            style={{ width: `${weightsPct}%` }}
            title={`모델 가중치: ${weightsGb.toFixed(1)} GB`}
          />
          <div 
            className="h-full bg-accent-500 transition-all duration-1000 ease-out"
            style={{ width: `${kvPct}%` }}
            title={`KV 캐시: ${kvGb.toFixed(1)} GB`}
          />
          {store.params.isTraining && (
            <>
              <div 
                className="h-full bg-warning transition-all duration-1000 ease-out"
                style={{ width: `${optGradPct}%` }}
                title={`옵티마이저/그래디언트: ${optGradGb.toFixed(1)} GB`}
              />
              <div 
                className="h-full bg-rose-400 transition-all duration-1000 ease-out"
                style={{ width: `${activationPct}%` }}
                title={`액티베이션 메모리: ${activationGb.toFixed(1)} GB`}
              />
            </>
          )}
          {isOom && (
             <div 
               className="h-full bg-error transition-all duration-100 pattern-diagonal-lines"
               style={{ width: `${overflowPct}%` }}
               title={`부족분: ${(totalUsedGb - totalVramLimit).toFixed(1)} GB`}
             />
          )}
        </div>

        {/* Legend */}
        <div className="flex gap-4 mt-2 text-sm text-gray-400 flex-wrap">
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-gray-600"></div> 시스템 예약 (CUDA/기타)</div>
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-primary-500"></div> 모델 가중치 ({weightsGb.toFixed(1)} GB)</div>
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-accent-500"></div> KV 캐시 ({kvGb.toFixed(1)} GB)</div>
          {store.params.isTraining && (
            <>
              <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-warning"></div> 학습(Opt/Grad) ({optGradGb.toFixed(1)} GB)</div>
              <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-rose-400"></div> 학습(Activation) ({activationGb.toFixed(1)} GB)</div>
            </>
          )}
          {isOom && (
             <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-error font-bold text-error"></div> 부족분 ({(totalUsedGb - totalVramLimit).toFixed(1)} GB)</div>
          )}
        </div>
      </div>

      {/* 2x2 Performance Matrix */}
      {!store.params.isTraining ? (
        <div className="mt-8 grid grid-cols-2 gap-4">
          <div className="glass-panel !border-white/5 p-4 flex flex-col items-center justify-center text-center">
            <Cpu className={`w-6 h-6 mb-2 ${isOom ? 'text-gray-600' : 'text-primary-500'}`} />
            <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">Throughput (Tokens/sec)</span>
            <span className={`text-2xl md:text-3xl font-black mt-1 ${isOom ? 'text-gray-600' : 'text-white'}`}>
              {metrics.totalThroughput.toLocaleString()} <span className="text-sm text-primary-500 font-normal">t/s</span>
            </span>
            <span className={`text-[10px] mt-1 font-medium ${isOom ? 'text-gray-700' : 'text-gray-400'}`}>
              ≈ 유저당 {metrics.speedPerUser.toLocaleString()} t/s 예상
            </span>
          </div>

          <div className="glass-panel !border-white/5 p-4 flex flex-col items-center justify-center text-center">
            <Users className={`w-6 h-6 mb-2 ${isOom ? 'text-gray-600' : 'text-accent-500'}`} />
            <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">최대 동시 추론 (Concurrency)</span>
            <span className={`text-2xl md:text-3xl font-black mt-1 ${isOom ? 'text-gray-600' : 'text-white'}`}>
              {Math.max(0, maxConcurrency).toLocaleString()} <span className="text-sm text-accent-500 font-normal">명</span>
            </span>
            <span className={`text-[10px] mt-1 font-medium ${isOom ? 'text-gray-700' : 'text-gray-400'}`}>
              OOM 없이 수용 가능한 최대 유저
            </span>
          </div>

          <div className="glass-panel !border-white/5 p-4 flex flex-col items-center justify-center text-center">
            <Zap className={`w-6 h-6 mb-2 ${isOom ? 'text-gray-600' : 'text-yellow-500'}`} />
            <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">예상 TTFT (First Token)</span>
            <span className={`text-2xl md:text-3xl font-black mt-1 ${isOom ? 'text-gray-600' : 'text-white'}`}>
              {metrics.ttftMs.toLocaleString()} <span className="text-sm text-yellow-500 font-normal">ms</span>
            </span>
            <span className={`text-[10px] mt-1 font-medium ${isOom ? 'text-gray-700' : 'text-gray-400'}`}>
              첫 토큰 생성 지연 시간 (Prefill)
            </span>
          </div>

          <div className="glass-panel !border-white/5 p-4 flex flex-col items-center justify-center text-center">
            <Activity className={`w-6 h-6 mb-2 ${isOom ? 'text-gray-600' : 'text-rose-500'}`} />
            <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">최대 RPS 처리 용량</span>
            <span className={`text-2xl md:text-3xl font-black mt-1 ${isOom ? 'text-gray-600' : 'text-white'}`}>
              {metrics.maxRps.toLocaleString()} <span className="text-sm text-rose-500 font-normal">Req/s</span>
            </span>
            <span className={`text-[10px] mt-1 font-medium ${isOom ? 'text-gray-700' : 'text-gray-400'}`}>
              단위 시간당 요청 완결 가능 횟수
            </span>
          </div>
        </div>
      ) : (
        <div className="mt-8 grid grid-cols-2 gap-4">
          <div className="glass-panel !border-white/5 p-4 flex flex-col items-center justify-center text-center">
            <Zap className={`w-6 h-6 mb-2 ${isOom ? 'text-gray-600' : 'text-warning'}`} />
            <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">예상 학습 소요 시간</span>
            <span className={`text-2xl md:text-3xl font-black mt-1 ${isOom ? 'text-gray-600' : 'text-white'}`}>
              {isOom ? '-' : formatTime(trainingMetrics?.estimatedHours || 0)}
            </span>
            <span className={`text-[10px] mt-1 font-medium ${isOom ? 'text-gray-700' : 'text-gray-400'}`}>
              {store.params.numEpochs} Epoch 완료 기준
            </span>
          </div>

          <div className="glass-panel !border-white/5 p-4 flex flex-col items-center justify-center text-center">
            <Database className={`w-6 h-6 mb-2 ${isOom ? 'text-gray-600' : 'text-blue-400'}`} />
            <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">총 학습 토큰 수</span>
            <span className={`text-2xl md:text-3xl font-black mt-1 ${isOom ? 'text-gray-600' : 'text-white'}`}>
               {isOom ? '-' : (trainingMetrics?.totalTokens / 1e6).toFixed(1)} <span className="text-sm text-blue-400 font-normal">M</span>
            </span>
            <span className={`text-[10px] mt-1 font-medium ${isOom ? 'text-gray-700' : 'text-gray-400'}`}>
              Samples * Epochs * Context
            </span>
          </div>

          <div className="glass-panel !border-white/5 p-4 flex flex-col items-center justify-center text-center">
            <Cpu className={`w-6 h-6 mb-2 ${isOom ? 'text-gray-600' : 'text-primary-400'}`} />
            <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">총 연산량 (Total FLOPs)</span>
            <span className={`text-2xl md:text-3xl font-black mt-1 ${isOom ? 'text-gray-600' : 'text-white'}`}>
               {isOom ? '-' : (trainingMetrics?.totalFlops / 1e18).toFixed(1)} <span className="text-sm text-primary-400 font-normal">E</span>
            </span>
            <span className={`text-[10px] mt-1 font-medium ${isOom ? 'text-gray-700' : 'text-gray-400'}`}>
               {isOom ? '-' : 'ExaFLOPs (10^18)'}
            </span>
          </div>

          <div className="glass-panel !border-white/5 p-4 flex flex-col items-center justify-center text-center">
            <Activity className={`w-6 h-6 mb-2 ${isOom ? 'text-gray-600' : 'text-accent-400'}`} />
            <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">학습 Throughput (가동률 반영)</span>
            <span className={`text-2xl md:text-3xl font-black mt-1 ${isOom ? 'text-gray-600' : 'text-white'}`}>
               {isOom ? '-' : Math.floor(trainingMetrics?.totalTokens / (trainingMetrics?.estimatedHours * 3600)).toLocaleString()} <span className="text-sm text-accent-400 font-normal">t/s</span>
            </span>
            <span className={`text-[10px] mt-1 font-medium ${isOom ? 'text-gray-700' : 'text-gray-400'}`}>
              하드웨어 가속기 활용도(MFU): {( 
                (1 - Math.exp(- (0.25 * Math.sqrt(100 / (selectedGpu.tflops || 100))) * (store.params.batchSize || 1))) * 45 
              ).toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      {/* AWS Instance Reference */}

      {/* AWS Instance Reference */}
      {!isOom && (selectedGpu as any).awsInstance && (
        <div className="mt-2 p-4 bg-surface-100/50 rounded-2xl border border-white/5 flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-orange-500/10 rounded-lg">
              <Server className="w-5 h-5 text-orange-500" />
            </div>
            <div>
              <h4 className="text-xs font-bold text-gray-300 uppercase letter tracking-tighter">AWS Infrastructure Reference</h4>
              <p className="text-[11px] text-gray-500">{(selectedGpu as any).awsInstance} 인스턴스 활용 권장</p>
            </div>
          </div>
          
          <a 
            href={(selectedGpu as any).awsUrl} 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-[10px] bg-white/5 hover:bg-white/10 text-gray-300 px-3 py-1.5 rounded-lg border border-white/10 transition-all flex items-center gap-2"
          >
            AWS 공식 문서 ({(selectedGpu as any).awsInstance.split(' ')[0]}) 보기
          </a>
        </div>
      )}

      {/* Recommendations */}
      <div className="mt-auto pt-8 border-t border-white/5">
        <h4 className="font-bold text-gray-200 mb-3 flex items-center gap-2">
           💡 AI 권장 사항 (Recommendations)
        </h4>
        <ul className="list-inside text-sm text-gray-400 space-y-3">
          {isOom && store.params.quantizationBits === 16 && (
            <li className="flex gap-2 items-start"><span className="text-primary-500 mt-0.5">•</span> <span>정밀도를 <strong className="text-accent-500">INT8 또는 INT4 양자화</strong>로 낮춰보세요. 모델 가중치 메모리를 비약적으로 줄일 수 있습니다.</span></li>
          )}
          {isOom && (store.params.inputLength + store.params.outputLength) > 4096 && (
            <li className="flex gap-2 items-start"><span className="text-primary-500 mt-0.5">•</span> <span>총 컨텍스트 길이가 너무 깁니다. <strong>Input Prompt</strong> 또는 <strong>Max Output</strong> 한도를 줄이는 것을 권장합니다.</span></li>
          )}
          {isOom && store.params.isTraining && store.params.trainingMethod === 'full' && (
            <li className="flex gap-2 items-start"><span className="text-primary-500 mt-0.5">•</span> <span>Full Fine-tuning은 막대한 VRAM을 소모합니다. <strong className="text-primary-500">PEFT / LoRA</strong> 방식으로 전환하여 학습 효율을 높여보세요.</span></li>
          )}
          {isOom && (
             <li className="flex gap-2 items-start"><span className="text-primary-500 mt-0.5">•</span> <span><strong>GPU 개수</strong>를 늘리거나, {selectedGpu.name}보다 VRAM이 더 큰 인스턴스(예: H200 141GB)로 업그레이드하세요.</span></li>
          )}
          {!isOom && !isWarning && (
            <li className="flex gap-2 items-start"><span className="text-success mt-0.5">✓</span> <span>VRAM 여유 공간이 충분합니다. <strong>배치 사이즈(Batch Size)</strong>를 늘려 전체 처리량을 극대화할 수 있습니다.</span></li>
          )}
        </ul>
      </div>

    </div>
  );
}
