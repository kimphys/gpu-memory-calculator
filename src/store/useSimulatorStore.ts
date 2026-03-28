import { create } from 'zustand';
import { ModelConfig, SimulationParams } from '../utils/calculator';
import { MODEL_PRESETS, GPUS } from '../data/presets';

type InputMode = 'PRESET' | 'HF_URL' | 'CUSTOM';

interface SimulatorState {
  isInfoOpen: boolean;
  setIsInfoOpen: (open: boolean) => void;
  inputMode: InputMode;
  setInputMode: (mode: InputMode) => void;

  // Preset Selection
  presetModelId: string;
  setPresetModelId: (id: string) => void;
  
  // HF URL Logic
  hfUrl: string;
  setHfUrl: (url: string) => void;
  hfConfig: ModelConfig | null;
  setHfConfig: (config: ModelConfig | null) => void;
  isFetchingHf: boolean;
  setIsFetchingHf: (fetching: boolean) => void;
  hfError: string | null;
  setHfError: (error: string | null) => void;
  
  // Custom Config
  customConfig: ModelConfig;
  setCustomConfig: (config: Partial<ModelConfig>) => void;

  // Active Model Getter
  getActiveModelConfig: () => ModelConfig | null;

  // Hardware
  gpuId: string;
  setGpuId: (id: string) => void;
  gpuCount: number;
  setGpuCount: (count: number) => void;
  // Task & Params
  params: SimulationParams;
  setParams: (params: Partial<SimulationParams>) => void;
  
  // UI Interaction
  editingField: string | null;
  setEditingField: (field: string | null) => void;
}

export const useSimulatorStore = create<SimulatorState>((set, get) => ({
  isInfoOpen: false,
  setIsInfoOpen: (isInfoOpen) => set({ isInfoOpen }),
  inputMode: 'PRESET',
  setInputMode: (mode) => set({ inputMode: mode }),

  presetModelId: 'ministral-8b',
  setPresetModelId: (presetModelId) => set({ presetModelId }),

  hfUrl: '',
  setHfUrl: (hfUrl) => set({ hfUrl }),
  hfConfig: null,
  setHfConfig: (hfConfig) => set({ hfConfig }),
  isFetchingHf: false,
  setIsFetchingHf: (isFetchingHf) => set({ isFetchingHf }),
  hfError: null,
  setHfError: (hfError) => set({ hfError }),

  customConfig: {
    parametersInB: 8,
    hiddenSize: 4096,
    numLayers: 32,
    numAttentionHeads: 32,
    numKeyValueHeads: 32,
  },
  setCustomConfig: (config) => set((state) => ({ customConfig: { ...state.customConfig, ...config } })),

  getActiveModelConfig: () => {
    const s = get();
    if (s.inputMode === 'PRESET') return MODEL_PRESETS[s.presetModelId];
    if (s.inputMode === 'HF_URL') return s.hfConfig;
    return s.customConfig;
  },

  gpuId: 'a100-80',
  setGpuId: (gpuId) => set({ gpuId }),
  gpuCount: 1,
  setGpuCount: (gpuCount) => set({ gpuCount }),

  params: {
    quantizationBits: 16,
    kvQuantizationBits: 16,
    batchSize: 1,
    inputLength: 1024,
    outputLength: 1024,
    isTraining: false,
    trainingMethod: 'lora',
    numSamples: 1000,
    numEpochs: 1,
  },
  setParams: (params) => set((state) => ({ params: { ...state.params, ...params } })),

  editingField: null,
  setEditingField: (editingField) => set({ editingField }),
}));
