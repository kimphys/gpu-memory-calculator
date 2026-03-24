import { ModelConfig } from './calculator';

export interface HFModelConfigRaw {
  architectures?: string[];
  hidden_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  vocab_size?: number;
  num_local_experts?: number;
  num_experts_per_tok?: number;
}

export async function fetchHFModelConfig(modelUrlOrId: string): Promise<ModelConfig | null> {
  const parsedId = modelUrlOrId.replace('https://huggingface.co/', '').split('?')[0].trim();

  if (!parsedId) return null;

  try {
    const response = await fetch(`https://huggingface.co/${parsedId}/resolve/main/config.json`);
    
    if (!response.ok) {
        throw new Error(`Failed to fetch config for ${parsedId}`);
    }

    const data: HFModelConfigRaw = await response.json();
    
    const hiddenSize = data.hidden_size || 4096;
    const numLayers = data.num_hidden_layers || 32;
    const numAttentionHeads = data.num_attention_heads || 32;
    const numKeyValueHeads = data.num_key_value_heads || numAttentionHeads; // fallback for non-GQA

    let parametersInB = extractParamsFromModelId(parsedId);
    let activeParametersInB: number | undefined = undefined;
    
    if (!parametersInB) {
      const roughParams = (12 * numLayers * (hiddenSize ** 2)) / 1e9;
      parametersInB = Math.round(roughParams);
    }

    // MoE Architecture Detection (e.g. Mixtral, Qwen MoE)
    if (data.num_local_experts && data.num_experts_per_tok) {
      // Rough heuristic for Active Parameters if Safetensors sizes are missing:
      // Ratio modeling shared attention vs split MLPs
      const ratio = (data.num_experts_per_tok + 1) / (data.num_local_experts + 1);
      activeParametersInB = Math.round(parametersInB * ratio * 10) / 10;
    }

    return {
      parametersInB: parametersInB || 8, // fallback to 8B if estimation fails
      activeParametersInB,
      hiddenSize,
      numLayers,
      numAttentionHeads,
      numKeyValueHeads,
    };
  } catch (error) {
    console.error("Error parsing HF config:", error);
    return null;
  }
}

/**
 * Extracts param count from strings like "Meta-Llama-3-8B" or "Mixtral-8x7B"
 */
function extractParamsFromModelId(modelId: string): number | null {
  const match = modelId.match(/(\d+)(b|x\d+b)/i);
  if (match) {
    if (match[2].toLowerCase() === 'b') {
      return parseInt(match[1], 10);
    } else if (match[2].toLowerCase().startsWith('x')) {
      const parts = match[2].toLowerCase().split('b')[0].replace('x', '');
      return parseInt(match[1], 10) * parseInt(parts, 10);
    }
  }
  return null;
}
