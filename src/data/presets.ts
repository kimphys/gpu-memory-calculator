import { ModelConfig } from '../utils/calculator';

export const GPUS = [
  { id: 'h200-141', name: 'NVIDIA H200 (141GB)', vramGb: 141, bandwidthGbps: 4800 },
  { id: 'h100-80', name: 'NVIDIA H100 (80GB)', vramGb: 80, bandwidthGbps: 3350 },
  { id: 'a100-80', name: 'NVIDIA A100 (80GB)', vramGb: 80, bandwidthGbps: 2039 },
  { id: 'a100-40', name: 'NVIDIA A100 (40GB)', vramGb: 40, bandwidthGbps: 1555 },
  { id: 'l40s-48', name: 'NVIDIA L40S (48GB)', vramGb: 48, bandwidthGbps: 864 },
  { id: 'a30-24', name: 'NVIDIA A30 (24GB)', vramGb: 24, bandwidthGbps: 933 },
  { id: 'rtx4090-24', name: 'RTX 4090 (24GB)', vramGb: 24, bandwidthGbps: 1008 },
  { id: 'rtx3090-24', name: 'RTX 3090 (24GB)', vramGb: 24, bandwidthGbps: 936 },
];

export const QUANTIZATION_OPTIONS = [
  { id: 'fp16', name: 'FP16 / BF16 (16-bit)', bits: 16 },
  { id: 'int8', name: 'INT8 (8-bit)', bits: 8 },
  { id: 'int4', name: 'INT4 (AWQ / GPTQ)', bits: 4 },
];

export const MODEL_PRESETS: Record<string, ModelConfig & { name: string }> = {
  'llama-3-8b': {
    name: 'Llama 3 (8B)',
    parametersInB: 8.03,
    hiddenSize: 4096,
    numLayers: 32,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
  },
  'llama-3-70b': {
    name: 'Llama 3 (70B)',
    parametersInB: 70.6,
    hiddenSize: 8192,
    numLayers: 80,
    numAttentionHeads: 64,
    numKeyValueHeads: 8,
  },
  'mistral-7b': {
    name: 'Mistral v0.2/v0.3 (7B)',
    parametersInB: 7.24,
    hiddenSize: 4096,
    numLayers: 32,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
  },
  'qwen-2-72b': {
    name: 'Qwen 1.5/2 (72B)',
    parametersInB: 72.7,
    hiddenSize: 8192,
    numLayers: 80,
    numAttentionHeads: 64,
    numKeyValueHeads: 8,
  },
  'qwen-2-7b': {
    name: 'Qwen 2 (7B)',
    parametersInB: 7.6,
    hiddenSize: 3584,
    numLayers: 28,
    numAttentionHeads: 28,
    numKeyValueHeads: 4,
  },
  'gemma-2-27b': {
    name: 'Gemma 2 (27B)',
    parametersInB: 27.2,
    hiddenSize: 4608,
    numLayers: 46,
    numAttentionHeads: 32,
    numKeyValueHeads: 16,
  },
  'gemma-2-9b': {
    name: 'Gemma 2 (9B)',
    parametersInB: 9.24,
    hiddenSize: 3584,
    numLayers: 42,
    numAttentionHeads: 16,
    numKeyValueHeads: 8,
  },
  'mixtral-8x7b': {
    name: 'Mixtral 8x7B (MoE)',
    parametersInB: 46.7,
    activeParametersInB: 12.9,
    hiddenSize: 4096,
    numLayers: 32,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
  },
  'gpt-oss-120b': {
    name: 'GPT-OSS 120B (MoE)',
    parametersInB: 117,
    activeParametersInB: 5.1,
    hiddenSize: 8192,
    numLayers: 64,
    numAttentionHeads: 64,
    numKeyValueHeads: 8,
  }
};
