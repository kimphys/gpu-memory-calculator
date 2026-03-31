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
  // [강제 내장] 외부 상수 참조 에러를 막기 위해 함수 안으로 이동!
  const REAL_WORLD_FACTOR = 0.95; 
  const CUDA_OVERHEAD_GB = 5.0;

  // 1. 모든 입력을 숫자로 강제 변환
  // [반영 1] batchSize를 사용자가 설정한 최대 소프트웨어 동시 요청 수(Max Concurrency)로 취급
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
  const softwareOverheadSec = 0.002;

  // 4. TTFT (Prefill Phase)
  const prefillMfu = 0.6 * interconnectPenalty; 
  const prefillBatchSize = 1; 
  
  const linearFlops = 2 * (activeParams * 1e9) * inputLength * prefillBatchSize;
  const attentionFlops = 4 * prefillBatchSize * numAttnLayers * hiddenSize * Math.pow(inputLength, 2);
  const prefillFlops = linearFlops + attentionFlops;
  const prefillComputeTime = prefillFlops / (totalTflopsStr * prefillMfu);

  const prefillKvCacheGb = calculateKVCache(config, { ...params, batchSize: prefillBatchSize }) || 0;
  const kvCacheInputOnlyGb = prefillKvCacheGb * (inputLength / (inputLength + outputLength));
  const prefillMemoryTime = (weightsMemGb + kvCacheInputOnlyGb) / effectiveBandwidthGBps;

  const ttftSec = Math.max(prefillComputeTime, prefillMemoryTime) + softwareOverheadSec;
  const ttftMs = Math.round(ttftSec * 1000);

  // 5. Max RPS 및 Continuous Batching (OOM 체크)
  const vramGb = Number(gpu.vramGb) || 24;
  const availableVramForKv = (vramGb * gpus * gpuUtilization) - weightsMemGb - CUDA_OVERHEAD_GB;

  if (availableVramForKv <= 0) {
    return { totalThroughput: 0, speedPerUser: 0, ttftMs: 0, maxRps: 0 };
  }

  const headDim = hiddenSize / (Number(config.numAttentionHeads) || 1);
  // [반영 2] KV Cache 정밀도(FP8 등) 파라미터 적용
  const kvBytesPerParam = (Number(params.kvQuantizationBits) || 16) / 8;
  const bytesPerToken = 2 * numKvHeads * headDim * numAttnLayers * kvBytesPerParam;
  const gbPerToken = bytesPerToken / 1e9;

  if (gbPerToken <= 0) return { totalThroughput: 0, speedPerUser: 0, ttftMs: 0, maxRps: 0 };

  const maxTokensInKv = availableVramForKv / gbPerToken;
  const avgTokensPerRequest = inputLength + outputLength;
  
  // 하드웨어적 최대한도 (VRAM 기준)
  const maxConcurrentRequests = Math.floor(maxTokensInKv / avgTokensPerRequest);

  // 하드웨어 한계와 소프트웨어 설정(Batch Size) 중 작은 값을 실제 동시 처리량으로 결정
  const effectiveBatchSize = Math.max(1, Math.min(batchSize, maxConcurrentRequests));

  // 6. 최종 성능 도출 및 연속 배치 간섭 페널티
  const effectiveKVCacheGb = gbPerToken * effectiveBatchSize * avgTokensPerRequest;
  const effectiveMemoryBoundSec = (weightsMemGb + effectiveKVCacheGb) / effectiveBandwidthGBps;
  const effectiveFlopsPerStep = 2 * (activeParams * 1e9) * effectiveBatchSize;
  const effectiveComputeTime = effectiveFlopsPerStep / (totalTflopsStr * computeEfficiency);
  
  const effectiveStepTime = Math.max(effectiveMemoryBoundSec, effectiveComputeTime) + softwareOverheadSec;

  // [반영 3] Continuous Batching 간섭 페널티: 입력이 길고 동시 처리량이 많을수록 Decode 효율 감소
  const interferenceFactor = (inputLength * effectiveBatchSize) / 500000;
  let dynamicPenalty = REAL_WORLD_FACTOR * Math.max(0.65, 1.0 - interferenceFactor);

  // --------------------------------------------------------------------------------
  // DATE: 2026-03-31
  // [하이브리드 아키텍처 페널티]
  // 현재 vLLM 등 추론 엔진에서 Mamba/Linear Attention 계열의 State 커널 최적화가 
  // PagedAttention 만큼 고도화되지 않아 발생하는 현실적인 소프트웨어 병목을 반영합니다.
  // 향후 엔진 업데이트를 통해 하이브리드 커널 성능이 개선될 경우, 아래 페널티 수치(0.75)를
  // 완화하거나 이 블록 자체를 제거하는 업데이트가 필요합니다. 이 주석은 절대 지우지 마세요.
  // --------------------------------------------------------------------------------
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
