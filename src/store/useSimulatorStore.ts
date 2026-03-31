import { create } from 'zustand';
import { ModelConfig, SimulationParams } from '../utils/calculator';
import { MODEL_PRESETS, GPUS } from '../data/presets';

type InputMode = 'PRESET' | 'HF_URL' | 'CUSTOM';

export interface ScenarioData {
  inputMode: InputMode;
  presetModelId: string;
  hfUrl: string;
  hfConfig: ModelConfig | null;
  isFetchingHf: boolean;
  hfError: string | null;
  customConfig: ModelConfig;
  gpuId: string;
  gpuCount: number;
  params: SimulationParams;
}

export interface SimulatorState {
  isInfoOpen: boolean;
  setIsInfoOpen: (open: boolean) => void;
  
  // Compare Mode State
  isCompareMode: boolean;
  setIsCompareMode: (enabled: boolean) => void;
  activeScenario: 'A' | 'B';
  setActiveScenario: (id: 'A' | 'B') => void;
  
  scenarios: {
    A: ScenarioData;
    B: ScenarioData;
  };

  // Setters (Target active scenario)
  setInputMode: (mode: InputMode) => void;
  setPresetModelId: (id: string) => void;
  setHfUrl: (url: string) => void;
  setHfConfig: (config: ModelConfig | null) => void;
  setIsFetchingHf: (fetching: boolean) => void;
  setHfError: (error: string | null) => void;
  setCustomConfig: (config: Partial<ModelConfig>) => void;
  setGpuId: (id: string) => void;
  setGpuCount: (count: number) => void;
  setParams: (params: Partial<SimulationParams>) => void;
  
  copyScenario: (from: 'A' | 'B', to: 'A' | 'B') => void;

  // Active Model Getter
  getActiveModelConfig: (scenarioId?: 'A' | 'B') => ModelConfig | null;

  // UI Interaction
  editingField: string | null;
  setEditingField: (field: string | null) => void;
}

const initialScenarioData: ScenarioData = {
  inputMode: 'PRESET',
  presetModelId: 'ministral-8b',
  hfUrl: '',
  hfConfig: null,
  isFetchingHf: false,
  hfError: null,
  customConfig: {
    parametersInB: 8,
    hiddenSize: 4096,
    numLayers: 32,
    numAttentionHeads: 32,
    numKeyValueHeads: 32,
    maxContextLength: 8192,
  },
  gpuId: 'a100-80',
  gpuCount: 1,
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
};

export const useSimulatorStore = create<SimulatorState>((set, get) => ({
  isInfoOpen: false,
  setIsInfoOpen: (isInfoOpen) => set({ isInfoOpen }),

  isCompareMode: false,
  setIsCompareMode: (isCompareMode) => set({ isCompareMode }),
  activeScenario: 'A',
  setActiveScenario: (activeScenario) => set({ activeScenario }),

  scenarios: {
    A: { ...initialScenarioData },
    B: { ...initialScenarioData },
  },

  setInputMode: (mode) => set((s) => ({ 
    scenarios: { ...s.scenarios, [s.activeScenario]: { ...s.scenarios[s.activeScenario], inputMode: mode } } 
  })),

  setPresetModelId: (id) => set((s) => ({ 
    scenarios: { ...s.scenarios, [s.activeScenario]: { ...s.scenarios[s.activeScenario], presetModelId: id } } 
  })),

  setHfUrl: (url) => set((s) => ({ 
    scenarios: { ...s.scenarios, [s.activeScenario]: { ...s.scenarios[s.activeScenario], hfUrl: url } } 
  })),

  setHfConfig: (config) => set((s) => ({ 
    scenarios: { ...s.scenarios, [s.activeScenario]: { ...s.scenarios[s.activeScenario], hfConfig: config } } 
  })),

  setIsFetchingHf: (fetching) => set((s) => ({ 
    scenarios: { ...s.scenarios, [s.activeScenario]: { ...s.scenarios[s.activeScenario], isFetchingHf: fetching } } 
  })),

  setHfError: (error) => set((s) => ({ 
    scenarios: { ...s.scenarios, [s.activeScenario]: { ...s.scenarios[s.activeScenario], hfError: error } } 
  })),

  setCustomConfig: (config) => set((s) => ({
    scenarios: { 
      ...s.scenarios, 
      [s.activeScenario]: { 
        ...s.scenarios[s.activeScenario], 
        customConfig: { ...s.scenarios[s.activeScenario].customConfig, ...config } 
      } 
    }
  })),

  setGpuId: (id) => set((s) => ({ 
    scenarios: { ...s.scenarios, [s.activeScenario]: { ...s.scenarios[s.activeScenario], gpuId: id } } 
  })),

  setGpuCount: (count) => set((s) => ({ 
    scenarios: { ...s.scenarios, [s.activeScenario]: { ...s.scenarios[s.activeScenario], gpuCount: count } } 
  })),

  setParams: (params) => set((s) => ({
    scenarios: { 
      ...s.scenarios, 
      [s.activeScenario]: { 
        ...s.scenarios[s.activeScenario], 
        params: { ...s.scenarios[s.activeScenario].params, ...params } 
      } 
    }
  })),

  copyScenario: (from, to) => set((s) => ({
    scenarios: {
      ...s.scenarios,
      [to]: JSON.parse(JSON.stringify(s.scenarios[from]))
    }
  })),

  getActiveModelConfig: (scenarioId) => {
    const s = get();
    const targetId = scenarioId || s.activeScenario;
    const data = s.scenarios[targetId];
    if (data.inputMode === 'PRESET') return MODEL_PRESETS[data.presetModelId];
    if (data.inputMode === 'HF_URL') return data.hfConfig;
    return data.customConfig;
  },

  editingField: null,
  setEditingField: (editingField) => set({ editingField }),
}));
