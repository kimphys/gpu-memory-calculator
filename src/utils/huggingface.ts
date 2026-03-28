import { ModelConfig } from './calculator';

export interface HFModelConfigRaw {
  architectures?: string[];
  hidden_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  vocab_size?: number;
  intermediate_size?: number;
  num_local_experts?: number;
  num_experts_per_tok?: number;
  max_position_embeddings?: number;
  // Nested config for multimodal/hybrid models (Qwen 2.5/3.5)
  text_config?: HFModelConfigRaw;
  layer_types?: string[];
}

export function parseRawConfig(data: HFModelConfigRaw, modelId?: string): ModelConfig {
    // Drill down into text_config if it's a multimodal wrapper (like Qwen 2-VL or 3.5)
    const config = data.text_config ? { ...data, ...data.text_config } : data;

    const hiddenSize = config.hidden_size || 4096;
    const numLayers = config.num_hidden_layers || 32;
    const numAttentionHeads = config.num_attention_heads || 32;
    const numKeyValueHeads = config.num_key_value_heads || numAttentionHeads;
    const intermediateSize = config.intermediate_size || (hiddenSize * 4); 
    const vocabSize = config.vocab_size || 32000;
    const numExperts = config.num_local_experts || 0;
    const numActiveExperts = config.num_experts_per_tok || 0;
    const maxContextLength = config.max_position_embeddings || 2048;

    // Detect Hybrid Layers (e.g. Qwen 3.5)
    const layerTypes = config.layer_types || [];
    let numAttentionLayers = numLayers;
    let numLinearLayers = 0;

    if (layerTypes.length > 0) {
      numAttentionLayers = layerTypes.filter(t => t.includes('full_attention')).length;
      numLinearLayers = layerTypes.filter(t => t.includes('linear_attention') || t.includes('mamba')).length;
      // Fallback: if they don't add up to numLayers, use the types array directly
      if (numAttentionLayers + numLinearLayers === 0) {
        numAttentionLayers = numLayers;
      }
    }

    let parametersInB = modelId ? extractParamsFromModelId(modelId) : null;
    let activeParametersInB: number | undefined = undefined;
    
    // Advanced Parameter Estimation Heuristic
    if (!parametersInB) {
      const embeddingParams = vocabSize * hiddenSize;
      const attentionParams = numLayers * 4 * (hiddenSize ** 2);
      
      let mlpParams = 0;
      if (numExperts > 0) {
        mlpParams = numLayers * numExperts * 3 * hiddenSize * intermediateSize;
      } else {
        mlpParams = numLayers * 3 * hiddenSize * intermediateSize;
      }

      parametersInB = Math.round((embeddingParams + attentionParams + mlpParams) / 1e8) / 10;
    }

    // Active Parameters Estimation
    if (numExperts > 0 && numActiveExperts > 0) {
        const embeddingParams = vocabSize * hiddenSize;
        const attentionParams = numLayers * 4 * (hiddenSize ** 2);
        const activeMlpParams = numLayers * numActiveExperts * 3 * hiddenSize * intermediateSize;
        activeParametersInB = Math.round((embeddingParams + attentionParams + activeMlpParams) / 1e8) / 10;
    }

    return {
      parametersInB: parametersInB || 8,
      activeParametersInB,
      hiddenSize,
      numLayers,
      numAttentionLayers,
      numLinearLayers,
      numAttentionHeads,
      numKeyValueHeads,
      maxContextLength,
    };
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
    return parseRawConfig(data, parsedId);
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
