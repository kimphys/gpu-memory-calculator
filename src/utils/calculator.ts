export interface ModelConfig {
  parametersInB: number;
  activeParametersInB?: number; // active parameters for MoE (optional)
  hiddenSize: number;
  numLayers: number;
  numAttentionLayers?: number; // for hybrid models (e.g. Qwen 3.5)
  numLinearLayers?: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  maxContextLength?: number;
}

export interface SimulationParams {
  quantizationBits: number;
  kvQuantizationBits?: number; // Added for FP8/INT8 KV Cache support
  batchSize: number;
  inputLength: number;
  outputLength: number;
  isTraining: boolean;
  trainingMethod?: 'full' | 'lora';
  numSamples?: number;
  numEpochs?: number;
}

export interface TrainingMetrics {
  totalTokens: number;
  totalFlops: number;
  estimatedHours: number;
  activationMemoryGb: number; // Added for precise training VRAM modeling
}

/**
 * Calculates VRAM required for model weights in GiB.
 */
export function calculateWeightsMemory(parametersInB: number, quantizationBits: number): number {
  const params = Number(parametersInB) || 0;
  const bits = Number(quantizationBits) || 16; // 누락 시 FP16(16비트) 강제 적용
  const bytesPerParam = bits / 8;
  const totalBytes = params * 1e9 * bytesPerParam;
  return totalBytes / 1e9;
}

/**
 * Calculates VRAM required for KV cache per token batch in GiB.
 */
export function calculateKVCache(config: ModelConfig, params: SimulationParams): number {
  const hiddenSize = Number(config.hiddenSize) || 4096;
  const numAttnHeads = Number(config.numAttentionHeads) || 1;
  const headDim = hiddenSize / numAttnHeads;

  const bits = Number(params.kvQuantizationBits) || 16;
  const bytesPerParam = bits / 8;

  const numKvHeads = Number(config.numKeyValueHeads) || numAttnHeads;
  const numAttnLayers = Number(config.numAttentionLayers ?? config.numLayers) || 1;
  const numLinearLayers = Number(config.numLinearLayers ?? 0);

  const batch = Number(params.batchSize) || 1;
  const inLen = Number(params.inputLength) || 1;
  const outLen = Number(params.outputLength) || 1;

  const kvElements = 2 * batch * (inLen + outLen) * numKvHeads * headDim * numAttnLayers;
  const linearStateElements = batch * (hiddenSize * 2) * numLinearLayers;

  const totalBytes = (kvElements + linearStateElements) * bytesPerParam;
  return (totalBytes / 1e9) || 0;
}

/**
 * Estimates additional memory required for training (Optimizer + Gradients).
 * Note: Activation memory is handled separately in calculateTrainingMetrics.
 */
export function calculateTrainingMemory(config: ModelConfig, method?: 'full' | 'lora'): number {
  const params = config.parametersInB * 1e9;
  if (method === 'full') {
    // AdamW (12 bytes/param: m=4, v=4, backup_weights=4) + Gradients (FP32: 4 bytes/param) = 16 bytes/param
    const bytesPerParam = 16;
    return (params * bytesPerParam) / 1e9;
  } else if (method === 'lora') {
    // LoRA overhead is tiny for weights, but we still have some gradients/states for adapters
    // Roughly 2 bytes/param of the baseline model is a safe enterprise upper bound for LoRA overhead
    const bytesPerParam = 2;
    return (params * bytesPerParam) / 1e9;
  }
  return 0;
}

export interface PerformanceMetrics {
  totalThroughput: number; // System-wide Tokens/sec
  speedPerUser: number;    // Tokens/sec per user
  ttftMs: number;          // Time to First Token (ms)
  maxRps: number;          // Maximum Request Capacity (Req/s)
}

export interface GpuSpec {
  bandwidthGbps: number;
  tflops: number;
  vramGb: number; // 추가됨: VRAM 기반 최대 동시 요청 수(Max Batch) 계산용
  interconnectBandwidthGbps: number; // 추가됨: 멀티 GPU 통신 병목 계산용
}

export const CUDA_OVERHEAD_GB = 5.0; // Base VRAM eaten by CUDA context + driver + initial activations
export const REAL_WORLD_FACTOR = 1.0; // 20% throughput penalty for enterprise logging/monitoring/scheduling

export function estimatePerformance(
  gpu: GpuSpec,
  config: ModelConfig,
  params: SimulationParams,
  numGPUs: number = 1
): PerformanceMetrics {
  // [강제 내장] 외부 상수 참조 에러를 막기 위해 함수 안으로 이동
  const REAL_WORLD_FACTOR = 0.95;
  const CUDA_OVERHEAD_GB = 5.0;

  // 1. 모든 입력을 숫자로 강제 변환
  const batchSize = Number(params.batchSize) || 1;
  const inputLength = Number(params.inputLength) || 1;
  const outputLength = Number(params.outputLength) || 1;
  const gpuUtilization = Number((params as any).gpuUtilization) || 0.9;
  const gpus = Number(numGPUs) || 1;

  const activeParams = Number(config.activeParametersInB || config.parametersInB) || 1;
  const numAttnLayers = Number(config.numAttentionLayers ?? config.numLayers) || 1;
  const numKvHeads = Number(config.numKeyValueHeads || config.numAttentionHeads) || 1;
  const numLinearLayers = Number(config.numLinearLayers ?? 0);
  const hiddenSize = Number(config.hiddenSize) || 4096;

  // 2. 메모리 계산
  const qBits = Number(params.quantizationBits) || 16;
  const weightsMemGb = calculateWeightsMemory(activeParams, qBits) || 0;
  const kvCacheGb = calculateKVCache(config, params) || 0;

  if (weightsMemGb <= 0) {
    return { totalThroughput: 0, speedPerUser: 0, ttftMs: 0, maxRps: 0 };
  }

  // 3. 하드웨어 스펙 세팅
  let interconnectPenalty = 1.0;
  if (gpus > 1) {
    const interconnectBw = Number(gpu.interconnectBandwidthGbps) || 64;
    const interconnectRatio = Math.min(interconnectBw / 900, 1.0);
    interconnectPenalty = 0.6 + (0.4 * interconnectRatio);
  }

  const effectiveBandwidthGBps = (Number(gpu.bandwidthGbps) || 1000) * gpus * 0.85 * interconnectPenalty;
  const totalTflopsStr = (Number(gpu.tflops) || 100) * gpus * 1e12;
  const computeEfficiency = 0.6 * interconnectPenalty;

  // ============================================================================
  // [순서 변경] 4. Max RPS 및 Continuous Batching (OOM 체크)를 먼저 수행합니다.
  // 부하(Load)를 알아야 Loaded TTFT를 계산할 수 있기 때문입니다.
  // ============================================================================
  const vramGb = Number(gpu.vramGb) || 24;
  const availableVramForKv = (vramGb * gpus * gpuUtilization) - weightsMemGb - CUDA_OVERHEAD_GB;

  if (availableVramForKv <= 0) {
    return { totalThroughput: 0, speedPerUser: 0, ttftMs: 0, maxRps: 0 };
  }

  const headDim = hiddenSize / (Number(config.numAttentionHeads) || 1);
  const kvBytesPerParam = (Number(params.kvQuantizationBits) || 16) / 8;
  const bytesPerToken = 2 * numKvHeads * headDim * numAttnLayers * kvBytesPerParam;
  const gbPerToken = bytesPerToken / 1e9;

  if (gbPerToken <= 0) return { totalThroughput: 0, speedPerUser: 0, ttftMs: 0, maxRps: 0 };

  const maxTokensInKv = availableVramForKv / gbPerToken;
  const avgTokensPerRequest = inputLength + outputLength;

  // 하드웨어적 최대한도 (VRAM 기준)
  const maxConcurrentRequests = Math.floor(maxTokensInKv / avgTokensPerRequest);
  // 하드웨어 한계와 소프트웨어 설정 중 작은 값을 실제 동시 처리량으로 결정
  const effectiveBatchSize = Math.max(1, Math.min(batchSize, maxConcurrentRequests));

  // ============================================================================
  // [로직 보정] 5. 부하 상태의 TTFT (Loaded TTFT) 계산
  // ============================================================================
  const prefillMfu = 0.6 * interconnectPenalty;

  // vLLM의 Chunked Prefill 및 큐 스케줄링 특성 반영 (배치가 커도 한 번에 다 읽지 않음)
  // 수학적 모델링: 동시 Prefill 부하는 전체 배치의 제곱근(Square Root)에 비례하여 증가
  let concurrentPrefills = Math.max(1, Math.floor(Math.sqrt(effectiveBatchSize)));

  const MAX_BATCHED_TOKENS = 2048; // vLLM 기본/권장 한계치
  if (concurrentPrefills * inputLength > MAX_BATCHED_TOKENS) {
    concurrentPrefills = Math.max(1, Math.floor(MAX_BATCHED_TOKENS / inputLength));
  }

  const linearFlops = 2 * (activeParams * 1e9) * inputLength * concurrentPrefills;
  const attentionFlops = 4 * concurrentPrefills * numAttnLayers * hiddenSize * Math.pow(inputLength, 2);
  const prefillFlops = linearFlops + attentionFlops;
  const prefillComputeTime = prefillFlops / (totalTflopsStr * prefillMfu);

  const prefillKvCacheGb = calculateKVCache(config, { ...params, batchSize: concurrentPrefills }) || 0;
  const kvCacheInputOnlyGb = prefillKvCacheGb * (inputLength / (inputLength + outputLength));
  const prefillMemoryTime = (weightsMemGb + kvCacheInputOnlyGb) / effectiveBandwidthGBps;

  // 현실적인 네트워크 및 토크나이저 지연 시간 (기본 10ms + 배치가 커질수록 파이썬 GIL 병목 증가)
  const queueingAndTokenizerOverheadSec = 0.01 + (effectiveBatchSize * 0.0002);

  const ttftSec = Math.max(prefillComputeTime, prefillMemoryTime) + queueingAndTokenizerOverheadSec;
  const ttftMs = Math.round(ttftSec * 1000);

  // ============================================================================
  // 6. 최종 성능 도출 및 디코딩 페널티
  // ============================================================================
  const effectiveKVCacheGb = gbPerToken * effectiveBatchSize * avgTokensPerRequest;
  const effectiveMemoryBoundSec = (weightsMemGb + effectiveKVCacheGb) / effectiveBandwidthGBps;
  const effectiveFlopsPerStep = 2 * (activeParams * 1e9) * effectiveBatchSize;
  const effectiveComputeTime = effectiveFlopsPerStep / (totalTflopsStr * computeEfficiency);

  const decodeSoftwareOverheadSec = 0.002;
  const effectiveStepTime = Math.max(effectiveMemoryBoundSec, effectiveComputeTime) + decodeSoftwareOverheadSec;

  // Continuous Batching 간섭 페널티: 입력이 길고 동시 처리량이 많을수록 Decode 효율 감소
  const interferenceFactor = (inputLength * effectiveBatchSize) / 500000;
  let dynamicPenalty = REAL_WORLD_FACTOR * Math.max(0.65, 1.0 - interferenceFactor);

  // [하이브리드 아키텍처 페널티]
  if (numLinearLayers > 0) {
    dynamicPenalty *= 0.75; // 하이브리드 커널 오버헤드로 인한 약 25% 효율 강제 감소 적용
  }

  const effectiveThroughput = (effectiveBatchSize / effectiveStepTime) * dynamicPenalty;
  const speedPerUser = (1 / effectiveStepTime) * dynamicPenalty;

  const maxRps = outputLength > 0 ? (effectiveThroughput / outputLength) : 0;

  return {
    totalThroughput: Math.floor(effectiveThroughput) || 0,
    speedPerUser: Math.max(1, Math.floor(speedPerUser)) || 0,
    ttftMs: Math.max(1, ttftMs) || 0,
    maxRps: Number(maxRps.toFixed(2)) || 0
  };
}

/**
 * Estimates Training Duration and FLOPs.
 * Standard rule of thumb: 6 * Parameters * Tokens for dense forward/backward.
 */
export function calculateTrainingMetrics(
  gpu: GpuSpec,
  config: ModelConfig,
  params: SimulationParams,
  numGPUs: number = 1
): TrainingMetrics {
  const totalParams = config.parametersInB * 1e9;
  const samples = params.numSamples || 1000;
  const epochs = params.numEpochs || 1;
  const tokensPerSample = params.inputLength + params.outputLength;

  const totalTokens = samples * epochs * tokensPerSample;

  // 6 * P * Tokens is the standard approximation for training FLOPs (3 for forward, 3 for backward)
  // For LoRA, P is still largely the full model size as we do forward/backward through all layers,
  // even if only LoRA weights are updated.
  const totalFlops = 6 * totalParams * totalTokens;

  const systemTflops = gpu.tflops * numGPUs * 1e12;

  // Model Flops Utilization (MFU) Scaling with Batch Size & GPU Power
  // Higher TFLOPS GPUs (e.g., H100 with 989 TFLOPS) have more compute units 
  // that require LARGER batch sizes to fully saturate compared to smaller GPUs (e.g., RTX 4090).
  const maxMfu = 0.45;
  // Saturation factor 'k': larger k = faster saturation. 
  // We scale k inversely with the GPU's compute power.
  // Base k = 0.25 (for 100 TFLOPS), scales down for more powerful GPUs.
  const k = 0.25 * Math.sqrt(100 / (gpu.tflops || 100));
  const batchEfficiency = 1 - Math.exp(-k * (params.batchSize || 1));
  const trainingMfu = maxMfu * batchEfficiency;

  const effectiveTflops = systemTflops * trainingMfu;

  const totalSeconds = totalFlops / effectiveTflops;
  const estimatedHours = totalSeconds / 3600;

  // --- Activation Memory Model (for Training VRAM) ---
  // Activation memory is roughly: 2 * Batch * Seq * HiddenSize * Layers * Bits (Training usually FP16/BF16 = 2 bytes)
  // We use a factor of 2 to account for intermediate activations (MLP expansion, etc.)
  const seqLen = params.inputLength + params.outputLength;
  const activationElements = 2 * (params.batchSize || 1) * seqLen * config.hiddenSize * config.numLayers;
  const activationMemoryGb = (activationElements * 2) / 1e9;

  return {
    totalTokens,
    totalFlops,
    estimatedHours: Number(estimatedHours.toFixed(2)),
    activationMemoryGb: Number(activationMemoryGb.toFixed(2))
  };
}
