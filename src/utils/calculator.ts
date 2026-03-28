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
  const bytesPerParam = quantizationBits / 8;
  const totalBytes = parametersInB * 1e9 * bytesPerParam;
  return totalBytes / 1e9; // Convert to GB (10^9)
}

/**
 * Calculates VRAM required for KV cache per token batch in GiB.
 * Formula: 2 (K,V) * batchSize * contextLength * numKeyValueHeads * headDim * numLayers * bytesPerParam
 */
export function calculateKVCache(config: ModelConfig, params: SimulationParams): number {
  const headDim = config.hiddenSize / config.numAttentionHeads;
  const bits = params.kvQuantizationBits || 16;
  const bytesPerParam = bits / 8; 

  // Handle Hybrid Architectures (e.g. Qwen 3.5)
  // If not specified, assume all layers are standard attention
  const numAttnLayers = config.numAttentionLayers ?? config.numLayers;
  const numLinearLayers = config.numLinearLayers ?? 0;

  // 1. Standard Self-Attention KV Cache (O(SeqLen))
  const kvElements = 2 * params.batchSize * (params.inputLength + params.outputLength) * config.numKeyValueHeads * headDim * numAttnLayers;
  
  // 2. Linear Attention / Mamba State (O(1) per token, O(Hidden) per request)
  // Heuristic: Linear layers usually have a recurrent state approx 2x HiddenSize
  const linearStateElements = params.batchSize * (config.hiddenSize * 2) * numLinearLayers;

  const totalBytes = (kvElements + linearStateElements) * bytesPerParam;
  return totalBytes / 1e9; // Convert to GB (10^9)
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
}

export const CUDA_OVERHEAD_GB = 5.0; // Base VRAM eaten by CUDA context + driver + initial activations
export const REAL_WORLD_FACTOR = 0.8; // 20% throughput penalty for enterprise logging/monitoring/scheduling

/**
 * Estimates Performance (Throughput & Latency) using an HPC Roofline Model.
 * Considers Memory Bandwidth bounds, Compute (TFLOPS) bounds, and Kernel Overheads.
 */
export function estimatePerformance(
  gpu: GpuSpec,
  config: ModelConfig,
  params: SimulationParams,
  numGPUs: number = 1
): PerformanceMetrics {
  const activeParams = config.activeParametersInB || config.parametersInB;
  const weightsMemGb = calculateWeightsMemory(activeParams, params.quantizationBits);
  const kvCacheGb = calculateKVCache(config, params);
  
  if (weightsMemGb === 0) return { totalThroughput: 0, speedPerUser: 0, ttftMs: 0, maxRps: 0 };

  // --- 1. Memory Bound Limits (Y-axis of Roofline) ---
  // ALL UNITS Standardized to GB (10^9)
  const effectiveBandwidthGBps = gpu.bandwidthGbps * numGPUs * 0.85 * (numGPUs > 1 ? 0.8 : 1.0);
  
  // Memory Bound Time: Data to read per step = Weights (read once) + KV Cache (for all users)
  const timeMemoryBoundSec = (weightsMemGb + kvCacheGb) / effectiveBandwidthGBps;
  
  // --- 2. Compute Bound Limits (X-axis Slope of Roofline) ---
  // To generate a token, the model does MACs for every active parameter.
  // MAC = 1 Multiply + 1 Add = 2 FLOPs.
  // Total FLOPs per step = 2 * Active Params * Batch Size
  const flopsPerStep = 2 * (activeParams * 1e9) * params.batchSize;
  const totalTflopsStr = gpu.tflops * numGPUs * 1e12; 
  // Real world MFU (Model Flops Utilization) during decoding is terribly low (10-30%) but we set a generous structural ceiling
  const computeEfficiency = 0.4; 
  const timeComputeBoundSec = flopsPerStep / (totalTflopsStr * computeEfficiency);
  
  // --- 3. Software & Kernel Launch Overhead ---
  // Python, vLLM scheduler, and CUDA kernel launch latency (constant per step)
  const softwareOverheadSec = 0.002; // ~2ms overhead per generation step

  // ROOFLINE EXECUTION TIME
  // Step time is dictated by whichever hardware bottleneck is slower, plus the inescapable software overhead.
  const timePerStepSec = Math.max(timeMemoryBoundSec, timeComputeBoundSec) + softwareOverheadSec;

  const totalThroughput = (params.batchSize / timePerStepSec) * REAL_WORLD_FACTOR;
  const speedPerUser = (1 / timePerStepSec) * REAL_WORLD_FACTOR;
  
  // --- 4. TTFT (Time To First Token / Prefill Phase) ---
  const prefillMfu = 0.6; // Higher MFU for massive dense gemm
  const prefillFlops = 2 * (activeParams * 1e9) * params.inputLength * params.batchSize;
  const ttftSec = (prefillFlops / (totalTflopsStr * prefillMfu) + softwareOverheadSec) / REAL_WORLD_FACTOR; // Also penalize TTFT slightly
  const ttftMs = Math.round(ttftSec * 1000);

  // --- 5. Max RPS (Requests Per Second) ---
  // A more realistic RPS considers that Input tokens are processed in one fast prefill step (TTFT),
  // and only Output tokens are generated at the slower decoding speed.
  // Total Request Time = TTFT + (Output Length / Speed Per User)
  const decodeTimeSec = params.outputLength / speedPerUser;
  const totalRequestTimeSec = (ttftMs / 1000) + decodeTimeSec;
  const maxRps = params.batchSize / totalRequestTimeSec;

  return {
    totalThroughput: Math.floor(totalThroughput),
    speedPerUser: Math.max(1, Math.floor(speedPerUser)),
    ttftMs: Math.max(1, ttftMs),
    maxRps: Number(maxRps.toFixed(2))
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
