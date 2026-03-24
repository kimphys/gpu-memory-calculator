export interface ModelConfig {
  parametersInB: number;
  activeParametersInB?: number; // active parameters for MoE (optional)
  hiddenSize: number;
  numLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
}

export interface SimulationParams {
  quantizationBits: number;
  batchSize: number;
  contextLength: number;
  isTraining: boolean;
  trainingMethod?: 'full' | 'lora';
}

/**
 * Calculates VRAM required for model weights in GiB.
 */
export function calculateWeightsMemory(parametersInB: number, quantizationBits: number): number {
  const bytesPerParam = quantizationBits / 8;
  const totalBytes = parametersInB * 1e9 * bytesPerParam;
  return totalBytes / (1024 ** 3); // Convert to GiB
}

/**
 * Calculates VRAM required for KV cache per token batch in GiB.
 * Formula: 2 (K,V) * batchSize * contextLength * numKeyValueHeads * headDim * numLayers * bytesPerParam
 */
export function calculateKVCache(config: ModelConfig, params: SimulationParams): number {
  const headDim = config.hiddenSize / config.numAttentionHeads;
  // KV Cache typically uses FP16 (2 bytes) in default models.
  const bytesPerParam = 2; 

  const kvElements = 2 * params.batchSize * params.contextLength * config.numKeyValueHeads * headDim * config.numLayers;
  const totalBytes = kvElements * bytesPerParam;
  return totalBytes / (1024 ** 3); // Convert to GiB
}

/**
 * Estimates additional memory required for training.
 */
export function calculateTrainingMemory(weightsMemGb: number, method?: 'full' | 'lora'): number {
  if (method === 'full') {
    // AdamW optimizer states + gradients + activations ~ 4x weights overhead
    return weightsMemGb * 4;
  } else if (method === 'lora') {
    // LoRA overhead is much smaller, roughly 50% extra overhead
    return weightsMemGb * 0.5;
  }
  return 0;
}

/**
 * Estimates Tokens Per Second (TPS) purely based on Memory Bandwidth Bound heuristics.
 */
export function estimateTPS(
  bandwidthGbps: number, 
  parametersInB: number, 
  quantizationBits: number, 
  numGPUs: number = 1
): number {
  const weightsMemGb = calculateWeightsMemory(parametersInB, quantizationBits);
  const totalBandwidth = bandwidthGbps * numGPUs;
  
  // Rule of thumb: Tensor parallelism causes ~20% overhead (80% efficiency)
  const efficiency = numGPUs > 1 ? 0.8 : 1.0;
  
  if (weightsMemGb === 0) return 0;
  const maxTps = (totalBandwidth * efficiency) / weightsMemGb;
  return Math.floor(maxTps);
}
