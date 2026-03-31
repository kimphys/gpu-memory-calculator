'use client';

import { useState, useRef } from 'react';
import { toPng } from 'html-to-image';

import { useSimulatorStore } from '../store/useSimulatorStore';
import {
  calculateWeightsMemory,
  calculateKVCache,
  calculateTrainingMemory,
  estimatePerformance,
  calculateTrainingMetrics,
  PerformanceMetrics,
  CUDA_OVERHEAD_GB
} from '../utils/calculator';
import { GPUS, MODEL_PRESETS } from '../data/presets';
import {
  Cpu,
  Zap,
  Users,
  Activity,
  CheckCircle,
  Database,
  AlertTriangle,
  Server,
  Layers,
  MessageSquare,
  Download,
  Loader2
} from 'lucide-react';

interface CalculationResult {
  activeModel: any;
  selectedGpu: any;
  totalVramLimit: number;
  weightsGb: number;
  kvGb: number;
  optGradGb: number;
  activationGb: number;
  trainingGb: number;
  hiddenGb: number;
  totalUsedGb: number;
  metrics: PerformanceMetrics;
  trainingMetrics: any;
  maxConcurrency: number;
  isOom: boolean;
  isWarning: boolean;
  isTraining: boolean; // Coupled to ensure sync
  statusColor: string;
  statusBg: string;
  weightsPct: number;
  kvPct: number;
  hiddenPct: number;
  optGradPct: number;
  activationPct: number;
  overflowPct: number;
}

export default function ResultsDashboard() {
  const store = useSimulatorStore();
  const dashboardRef = useRef<HTMLDivElement>(null);
  const [isDownloading, setIsDownloading] = useState(false);

  const calculateScenario = (scenarioId: 'A' | 'B'): CalculationResult | null => {
    const data = store.scenarios[scenarioId];
    const activeModel = store.getActiveModelConfig(scenarioId);

    if (!activeModel) return null;

    const selectedGpu = GPUS.find(g => g.id === data.gpuId) || GPUS[0];
    const totalVramLimit = selectedGpu.vramGb * data.gpuCount;

    const weightsGb = calculateWeightsMemory(activeModel.parametersInB, data.params.quantizationBits);
    const kvGb = calculateKVCache(activeModel, data.params);

    let optGradGb = 0;
    let activationGb = 0;
    const isTraining = data.params.isTraining;

    if (isTraining) {
      optGradGb = calculateTrainingMemory(activeModel, data.params.trainingMethod);
      const tMetrics = calculateTrainingMetrics(selectedGpu, activeModel, data.params, data.gpuCount);
      activationGb = tMetrics.activationMemoryGb;
    }

    const trainingGb = optGradGb + activationGb;
    const hiddenGb = CUDA_OVERHEAD_GB;
    const totalUsedGb = weightsGb + kvGb + trainingGb + hiddenGb;

    let metrics: PerformanceMetrics = { totalThroughput: 0, speedPerUser: 0, ttftMs: 0, maxRps: 0 };
    let trainingMetrics: any = null;
    let maxConcurrency = 0;

    if (!isTraining) {
      if (totalUsedGb <= totalVramLimit) {
        metrics = estimatePerformance(
          {
            bandwidthGbps: selectedGpu.bandwidthGbps,
            tflops: (selectedGpu as any).tflops || 100,
            vramGb: (selectedGpu as any).vramGb || 24,
            interconnectBandwidthGbps: (selectedGpu as any).interconnectBandwidthGbps || 64
          },
          activeModel, data.params, data.gpuCount
        );
      }
      const kvPerUser = kvGb / data.params.batchSize;
      const availableForKV = totalVramLimit - weightsGb - trainingGb - hiddenGb;
      maxConcurrency = kvPerUser > 0 ? Math.floor(availableForKV / kvPerUser) : 0;
    } else {
      trainingMetrics = calculateTrainingMetrics(
        {
          bandwidthGbps: selectedGpu.bandwidthGbps,
          tflops: (selectedGpu as any).tflops || 100,
          vramGb: (selectedGpu as any).vramGb || 24,
          interconnectBandwidthGbps: (selectedGpu as any).interconnectBandwidthGbps || 64
        },
        activeModel, data.params, data.gpuCount
      );
    }

    const isOom = totalUsedGb > totalVramLimit;
    const isWarning = totalUsedGb > totalVramLimit * 0.8 && !isOom;
    const statusColor = isOom ? 'text-error' : isWarning ? 'text-warning' : 'text-success';
    const statusBg = isOom ? 'bg-error/20 border-error/50' : isWarning ? 'bg-warning/20 border-warning/50' : 'bg-success/20 border-success/50';

    const maxScale = Math.max(totalUsedGb, totalVramLimit);
    return {
      activeModel, selectedGpu, totalVramLimit, weightsGb, kvGb, optGradGb, activationGb, trainingGb, hiddenGb, totalUsedGb,
      metrics, trainingMetrics, maxConcurrency, isOom, isWarning, isTraining, statusColor, statusBg,
      weightsPct: (weightsGb / maxScale) * 100,
      kvPct: (kvGb / maxScale) * 100,
      hiddenPct: (hiddenGb / maxScale) * 100,
      optGradPct: (optGradGb / maxScale) * 100,
      activationPct: (activationGb / maxScale) * 100,
      overflowPct: isOom ? ((totalUsedGb - totalVramLimit) / maxScale) * 100 : 0
    };
  };

  const resA = calculateScenario('A');
  const resB = store.isCompareMode ? calculateScenario('B') : null;

  if (!resA) {
    return (
      <div className="glass-panel p-6 flex flex-col items-center justify-center h-full text-center">
        <p className="text-gray-400 animate-pulse text-lg">Waiting for model configuration...</p>
      </div>
    );
  }

  const formatTime = (hours: number) => {
    if (hours < 1) return `${Math.round(hours * 60)}분`;
    if (hours < 48) return `${hours.toFixed(1)}시간`;
    return `${(hours / 24).toFixed(1)}일`;
  };

  const renderScenarioResult = (res: CalculationResult, scenarioId: string, isSmall: boolean = false) => {
    const data = store.scenarios[scenarioId as 'A' | 'B'];
    const isTraining = res.isTraining; // Use the value directly from calculation result

    const modelName = data.inputMode === 'PRESET'
      ? (MODEL_PRESETS[data.presetModelId]?.name || 'Unknown Model')
      : data.inputMode === 'HF_URL' ? (data.hfUrl?.split('/').pop() || 'HF Model') : 'Custom Model';

    return (
      <div className="flex flex-col gap-6 w-full h-full">

        {/* Expanded Scenario Header with Config Detail */}
        <div className="flex flex-col gap-3 pb-4 border-b border-white/5 min-h-[100px] shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="px-1.5 py-0.5 bg-primary-500/10 rounded text-[9px] font-black text-primary-500 uppercase tracking-tighter border border-primary-500/20">
                Scenario {scenarioId}
              </span>
              <span className="text-sm font-bold text-white truncate max-w-[150px] xl:max-w-[200px]" title={modelName}>
                {modelName}
              </span>
            </div>
            <span className="text-[10px] font-medium px-2 py-0.5 bg-surface-200 rounded-full text-gray-400 border border-white/5">
              {isTraining ? 'Fine-tuning Mode' : 'Inference Mode'}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-y-2 gap-x-4">
            <div className="flex items-center gap-2 text-[11px]">
              <Server className="w-3.5 h-3.5 text-accent-500 shrink-0" />
              <div className="flex flex-col">
                <span className="text-gray-500 leading-none mb-0.5">Hardware Target</span>
                <span className="text-gray-300 font-medium truncate">{res.selectedGpu.name} × {data.gpuCount}</span>
              </div>
            </div>
            <div className="flex items-center gap-2 text-[11px]">
              <Layers className="w-3.5 h-3.5 text-primary-400 shrink-0" />
              <div className="flex flex-col">
                <span className="text-gray-500 leading-none mb-0.5">Quantization</span>
                <span className="text-gray-300 font-medium">{data.params.quantizationBits}-bit (KV: {data.params.kvQuantizationBits}-bit)</span>
              </div>
            </div>
            <div className="flex items-center gap-2 text-[11px]">
              <MessageSquare className="w-3.5 h-3.5 text-yellow-500 shrink-0" />
              <div className="flex flex-col">
                <span className="text-gray-500 leading-none mb-0.5">Context Window</span>
                <span className="text-gray-300 font-medium">In: {data.params.inputLength.toLocaleString()} / Out: {data.params.outputLength.toLocaleString()}</span>
              </div>
            </div>
            <div className="flex items-center gap-2 text-[11px]">
              <Users className="w-3.5 h-3.5 text-rose-400 shrink-0" />
              <div className="flex flex-col">
                <span className="text-gray-500 leading-none mb-0.5">Throughput Config</span>
                <span className="text-gray-300 font-medium">Batch Size {data.params.batchSize}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Compact Status Banner */}
        <div className={`px-5 py-3 rounded-xl border flex items-center justify-between transition-all duration-500 h-[70px] shrink-0 shadow-lg ${res.statusBg}`}>
          <div className="flex items-center gap-4">
            {res.isOom ? <AlertTriangle className="w-7 h-7 text-error flex-shrink-0 animate-pulse" /> : <CheckCircle className="w-7 h-7 text-success flex-shrink-0" />}
            <div className="flex flex-col">
              <h3 className={`text-sm md:text-base font-black tracking-tight ${res.statusColor}`}>
                {res.isOom ? 'MEM OOM' : res.isWarning ? 'HIGH USAGE' : 'STABLE'}
              </h3>
              <p className="text-[11px] text-gray-300/80 font-medium">
                {res.isOom ? '메모리 한계 초과' : '리소스 상태 정상'}
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className={`text-lg font-black tracking-tighter leading-none ${res.isOom ? 'text-error' : 'text-white'}`}>
              {res.totalUsedGb.toFixed(1)} <span className="text-[10px] opacity-70">GB</span>
            </div>
            <div className="text-[10px] text-gray-400 font-bold opacity-60">
              Total Used / {res.totalVramLimit}GB
            </div>
          </div>
        </div>

        {/* VRAM Visualizer */}
        <div className="flex flex-col gap-4 mt-2 h-auto shrink-0">
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
            <span className={`text-sm font-mono font-bold ${res.isOom ? 'text-error animate-pulse' : 'text-gray-400'}`}>
              {res.totalUsedGb.toFixed(1)} GB / {res.totalVramLimit} GB
            </span>
          </div>

          {/* Stacked Bar */}
          <div className="h-10 w-full bg-surface-200 rounded-full overflow-hidden flex shadow-inner relative border border-white/5 flex-shrink-0">
            <div
              className="h-full bg-gray-600 transition-all duration-1000 ease-out flex-shrink-0"
              style={{ width: `${res.hiddenPct}%` }}
              title={`CUDA/시스템 예약: ${res.hiddenGb.toFixed(1)} GB`}
            />
            <div
              className="h-full bg-primary-500 transition-all duration-1000 ease-out flex-shrink-0"
              style={{ width: `${res.weightsPct}%` }}
              title={`모델 가중치: ${res.weightsGb.toFixed(1)} GB`}
            />
            <div
              className="h-full bg-accent-500 transition-all duration-1000 ease-out flex-shrink-0"
              style={{ width: `${res.kvPct}%` }}
              title={`KV 캐시: ${res.kvGb.toFixed(1)} GB`}
            />
            {isTraining && (
              <>
                <div
                  className="h-full transition-all duration-1000 ease-out flex-shrink-0"
                  style={{ width: `${res.optGradPct}%`, backgroundColor: '#f59e0b' }}
                  title={`옵티마이저/그래디언트: ${res.optGradGb.toFixed(1)} GB`}
                />
                <div
                  className="h-full transition-all duration-1000 ease-out flex-shrink-0"
                  style={{ width: `${res.activationPct}%`, backgroundColor: '#fb7185' }}
                  title={`액티베이션 메모리: ${res.activationGb.toFixed(1)} GB`}
                />
              </>
            )}
            {res.isOom && (
              <div
                className="h-full bg-error transition-all duration-100 pattern-diagonal-lines flex-shrink-0"
                style={{ width: `${res.overflowPct}%` }}
                title={`부족분: ${(res.totalUsedGb - res.totalVramLimit).toFixed(1)} GB`}
              />
            )}
          </div>

          {/* Legend */}
          <div className="flex gap-4 mt-2 text-sm text-gray-400 flex-wrap">
            <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-gray-600"></div> 시스템 예약 (CUDA/기타)</div>
            <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-primary-500"></div> 모델 가중치 ({res.weightsGb.toFixed(1)} GB)</div>
            <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-accent-500"></div> KV 캐시 ({res.kvGb.toFixed(1)} GB)</div>
            {isTraining && (
              <>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#f59e0b' }}></div> 
                  학습(Opt/Grad) ({res.optGradGb.toFixed(1)} GB)
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#fb7185' }}></div> 
                  학습(Activation) ({res.activationGb.toFixed(1)} GB)
                </div>
              </>
            )}
            {res.isOom && (
              <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-error font-bold text-error"></div> 부족분 ({(res.totalUsedGb - res.totalVramLimit).toFixed(1)} GB)</div>
            )}
          </div>
        </div>

        {/* 2x2 Performance Matrix */}
        {!isTraining ? (
          <div className="mt-4 grid grid-cols-2 gap-4">
            <div className="glass-panel !border-white/5 p-4 flex flex-col items-center justify-center text-center">
              <Cpu className={`w-6 h-6 mb-2 ${res.isOom ? 'text-gray-600' : 'text-primary-500'}`} />
              <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">Throughput</span>
              <span className={`text-2xl md:text-3xl font-black mt-1 ${res.isOom ? 'text-gray-600' : 'text-white'}`}>
                {res.metrics.totalThroughput.toLocaleString()} <span className="text-sm text-primary-500 font-normal">t/s</span>
              </span>
              <span className={`text-[10px] mt-1 font-medium ${res.isOom ? 'text-gray-700' : 'text-gray-400'}`}>
                ≈ 유저당 {res.metrics.speedPerUser.toLocaleString()} t/s 예상
              </span>
            </div>

            <div className="glass-panel !border-white/5 p-4 flex flex-col items-center justify-center text-center">
              <Users className={`w-6 h-6 mb-2 ${res.isOom ? 'text-gray-600' : 'text-accent-500'}`} />
              <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">Concurrency</span>
              <span className={`text-2xl md:text-3xl font-black mt-1 ${res.isOom ? 'text-gray-600' : 'text-white'}`}>
                {Math.max(0, res.maxConcurrency).toLocaleString()} <span className="text-sm text-accent-500 font-normal">명</span>
              </span>
              <span className={`text-[10px] mt-1 font-medium ${res.isOom ? 'text-gray-700' : 'text-gray-400'}`}>
                최대 동시 추론
              </span>
            </div>

            <div className="glass-panel !border-white/5 p-4 flex flex-col items-center justify-center text-center">
              <Zap className={`w-6 h-6 mb-2 ${res.isOom ? 'text-gray-600' : 'text-yellow-500'}`} />
              <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">Time To First Token</span>
              <span className={`text-2xl md:text-3xl font-black mt-1 ${res.isOom ? 'text-gray-600' : 'text-white'}`}>
                {res.metrics.ttftMs.toLocaleString()} <span className="text-sm text-yellow-500 font-normal">ms</span>
              </span>
              <span className={`text-[10px] mt-1 font-medium ${res.isOom ? 'text-gray-700' : 'text-gray-400'}`}>
                첫 토큰 생성 지연 시간
              </span>
            </div>

            <div className="glass-panel !border-white/5 p-4 flex flex-col items-center justify-center text-center">
              <Activity className={`w-6 h-6 mb-2 ${res.isOom ? 'text-gray-600' : 'text-rose-500'}`} />
              <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">최대 RPS 처리 용량</span>
              <span className={`text-2xl md:text-3xl font-black mt-1 ${res.isOom ? 'text-gray-600' : 'text-white'}`}>
                {res.metrics.maxRps.toLocaleString()} <span className="text-sm text-rose-500 font-normal">Req/s</span>
              </span>
              <span className={`text-[10px] mt-1 font-medium ${res.isOom ? 'text-gray-700' : 'text-gray-400'}`}>
                1초 당 요청 완결 가능 횟수
              </span>
            </div>
          </div>
        ) : (
          <div className="mt-4 grid grid-cols-2 gap-4">
            <div className="glass-panel !border-white/5 p-4 flex flex-col items-center justify-center text-center">
              <Zap className={`w-6 h-6 mb-2 ${res.isOom ? 'text-gray-600' : 'text-warning'}`} />
              <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">예상 학습 소요 시간</span>
              <span className={`text-2xl md:text-3xl font-black mt-1 ${res.isOom ? 'text-gray-600' : 'text-white'}`}>
                {res.isOom ? '-' : formatTime(res.trainingMetrics?.estimatedHours || 0)}
              </span>
              <span className={`text-[10px] mt-1 font-medium ${res.isOom ? 'text-gray-700' : 'text-gray-400'}`}>
                {data.params.numEpochs} Epoch 완료 기준
              </span>
            </div>

            <div className="glass-panel !border-white/5 p-4 flex flex-col items-center justify-center text-center">
              <Database className={`w-6 h-6 mb-2 ${res.isOom ? 'text-gray-600' : 'text-blue-400'}`} />
              <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">총 학습 토큰 수</span>
              <span className={`text-2xl md:text-3xl font-black mt-1 ${res.isOom ? 'text-gray-600' : 'text-white'}`}>
                {res.isOom ? '-' : (res.trainingMetrics?.totalTokens / 1e6).toFixed(1)} <span className="text-sm text-blue-400 font-normal">M</span>
              </span>
              <span className={`text-[10px] mt-1 font-medium ${res.isOom ? 'text-gray-700' : 'text-gray-400'}`}>
                Samples * Epochs * Context
              </span>
            </div>

            <div className="glass-panel !border-white/5 p-4 flex flex-col items-center justify-center text-center">
              <Cpu className={`w-6 h-6 mb-2 ${res.isOom ? 'text-gray-600' : 'text-primary-400'}`} />
              <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">총 연산량 (Total FLOPs)</span>
              <span className={`text-2xl md:text-3xl font-black mt-1 ${res.isOom ? 'text-gray-600' : 'text-white'}`}>
                {res.isOom ? '-' : (res.trainingMetrics?.totalFlops / 1e18).toFixed(1)} <span className="text-sm text-primary-400 font-normal">E</span>
              </span>
              <span className={`text-[10px] mt-1 font-medium ${res.isOom ? 'text-gray-700' : 'text-gray-400'}`}>
                {res.isOom ? '-' : 'ExaFLOPs (10^18)'}
              </span>
            </div>

            <div className="glass-panel !border-white/5 p-4 flex flex-col items-center justify-center text-center">
              <Activity className={`w-6 h-6 mb-2 ${res.isOom ? 'text-gray-600' : 'text-accent-400'}`} />
              <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">학습 Throughput (가동률 반영)</span>
              <span className={`text-2xl md:text-3xl font-black mt-1 ${res.isOom ? 'text-gray-600' : 'text-white'}`}>
                {res.isOom ? '-' : Math.floor(res.trainingMetrics?.totalTokens / (res.trainingMetrics?.estimatedHours * 3600)).toLocaleString()} <span className="text-sm text-accent-400 font-normal">t/s</span>
              </span>
              <span className={`text-[10px] mt-1 font-medium ${res.isOom ? 'text-gray-700' : 'text-gray-400'}`}>
                하드웨어 가속기 활용도(MFU): {(
                  (1 - Math.exp(- (0.25 * Math.sqrt(100 / (res.selectedGpu.tflops || 100))) * (data.params.batchSize || 1))) * 45
                ).toFixed(1)}%
              </span>
            </div>
          </div>
        )}

        {/* AWS Instance Reference */}
        {!res.isOom && (res.selectedGpu as any).awsInstance && (
          <div className="mt-2 p-4 bg-surface-100/50 rounded-2xl border border-white/5 flex flex-wrap justify-between items-center gap-4 min-h-[85px] shrink-0">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-orange-500/10 rounded-lg">
                <Server className="w-5 h-5 text-orange-500" />
              </div>
              <div>
                <h4 className="text-xs font-bold text-gray-300 uppercase leading-relaxed tracking-tighter">AWS Infrastructure Reference</h4>
                <p className="text-[11px] text-gray-500">{(res.selectedGpu as any).awsInstance} 인스턴스 활용 권장</p>
              </div>
            </div>

            <a
              href={(res.selectedGpu as any).awsUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[10px] bg-white/5 hover:bg-white/10 text-gray-300 px-3 py-1.5 rounded-lg border border-white/10 transition-all flex items-center gap-2 flex-shrink-0"
            >
              AWS 공식 문서 ({(res.selectedGpu as any).awsInstance.split(' ')[0]}) 보기
            </a>
          </div>
        )}

        {/* Detailed Recommendations / Insights */}
        <div className="mt-auto pt-4 border-t border-white/5">
          <h4 className="font-bold text-gray-300 mb-3 flex items-center gap-1.5 text-xs uppercase tracking-wider">
            <Activity className="w-3.5 h-3.5 text-primary-500" /> Infra Insights
          </h4>
          <ul className="text-[11px] text-gray-400 space-y-2 leading-relaxed">
            {res.isOom && store.scenarios[scenarioId as 'A' | 'B'].params.quantizationBits === 16 && (
              <li className="flex gap-2 items-start"><span className="text-primary-500 mt-0.5 leading-none">•</span> <span>정밀도를 <strong className="text-accent-500">INT8 또는 INT4 양자화</strong>로 낮춰보세요. 가중치 로드에 필요한 메모리를 즉각적으로 절반 이하로 줄일 수 있습니다.</span></li>
            )}
            {res.isOom && (store.scenarios[scenarioId as 'A' | 'B'].params.inputLength + store.scenarios[scenarioId as 'A' | 'B'].params.outputLength) > 4096 && (
              <li className="flex gap-2 items-start"><span className="text-primary-500 mt-0.5 leading-none">•</span> <span>요청되는 컨텍스트 길이가 지나치게 길어 <strong className="text-accent-400">KV 캐시</strong> 병목이 발생했습니다. <strong>Input Prompt Limit</strong> 또는 <strong>Max Output</strong>을 줄이는 것을 권장합니다.</span></li>
            )}
            {res.isOom && isTraining && store.scenarios[scenarioId as 'A' | 'B'].params.trainingMethod === 'full' && (
              <li className="flex gap-2 items-start"><span className="text-primary-500 mt-0.5 leading-none">•</span> <span>Full Fine-tuning은 옵티마이저 오버헤드로 인해 막대한 VRAM을 소모합니다. <strong className="text-primary-500">PEFT / LoRA</strong> 방식으로 전환하여 학습 메모리를 획기적으로 줄여보세요.</span></li>
            )}
            {res.isOom && (
              <li className="flex gap-2 items-start"><span className="text-error mt-0.5 leading-none">•</span> <span>현재 설정은 단일 서버의 물리적인 VRAM 한계를 완전히 초과했습니다. <strong>GPU 개수</strong>를 늘리거나, {res.selectedGpu.name}보다 VRAM이 더 큰 인스턴스로의 이관이 필수적입니다.</span></li>
            )}
            {!res.isOom && res.isWarning && (
              <li className="flex gap-2 items-start"><span className="text-warning mt-0.5 leading-none">!</span> <span>메모리가 거의 가득 차 있어 사용량이 급증할 경우 OOM 위험이 있습니다. 안정성을 위해 10~20%의 여유 마진을 확보하세요.</span></li>
            )}
            {!res.isOom && !res.isWarning && (
              <li className="flex gap-2 items-start"><span className="text-success mt-0.5 leading-none">✓</span> <span>시스템 자원이 안정적입니다. 활용도를 높이려면 <strong>배치 사이즈(Batch Size)</strong> 한도를 상향 조정하여 최대 Concurrency를 끌어올려 보세요.</span></li>
            )}
          </ul>
        </div>
      </div>
    );
  };

  const handleDownloadImage = async () => {
    if (!dashboardRef.current) return;
    setIsDownloading(true);

    try {
      const dataUrl = await toPng(dashboardRef.current, {
        backgroundColor: '#0b101a', // Solid dark background as requested
        pixelRatio: 2, // High fidelity
        skipFonts: false,
        cacheBust: true,
      });

      const link = document.createElement('a');
      const dateStr = new Date().toISOString().split('T')[0];
      const modelA = store.scenarios.A.inputMode === 'PRESET' ? store.scenarios.A.presetModelId : 'custom';

      link.download = `GPU_Simulator_Report_${modelA}${store.isCompareMode ? '_Comparison' : ''}_${dateStr}.png`;
      link.href = dataUrl;
      link.click();
    } catch (err) {
      console.error('Failed to download image:', err);
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div
      ref={dashboardRef}
      className={`glass-panel p-6 md:p-8 flex flex-col gap-8 w-full ${store.isCompareMode ? 'h-auto' : 'h-full overflow-y-auto custom-scrollbar'}`}
    >
      <div className="flex justify-between items-center shrink-0">
        <h2 className="text-2xl font-bold flex items-center gap-3 text-white">
          <Zap className="w-6 h-6 text-accent-500" /> 시뮬레이션 결과
        </h2>

        <button
          onClick={handleDownloadImage}
          disabled={isDownloading}
          className="flex items-center gap-2 bg-primary-500/10 hover:bg-primary-500/20 text-primary-400 px-4 py-2 rounded-xl text-sm font-bold border border-primary-500/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed group"
        >
          {isDownloading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Download className="w-4 h-4 group-hover:translate-y-0.5 transition-transform" />
          )}
          <span>{isDownloading ? '생성 중...' : '이미지로 저장'}</span>
        </button>
      </div>

      <div className={`grid ${store.isCompareMode ? 'md:grid-cols-2' : 'grid-cols-1'} flex-1`}>
        <div className={`h-full ${store.isCompareMode ? 'md:pr-8 md:border-r border-white/5 pb-8 md:pb-0' : ''}`}>
          {renderScenarioResult(resA, 'A', store.isCompareMode)}
        </div>

        {store.isCompareMode && (
          <div className="pt-8 md:pt-0 md:pl-8 border-t md:border-t-0 border-white/5 h-full">
            {resB ? renderScenarioResult(resB, 'B', true) : (
              <div className="h-full flex items-center justify-center text-gray-600 italic text-sm">
                Scenario B loading...
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
