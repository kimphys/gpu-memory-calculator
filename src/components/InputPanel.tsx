'use client';

import { useSimulatorStore } from '../store/useSimulatorStore';
import { MODEL_PRESETS, GPUS, QUANTIZATION_OPTIONS } from '../data/presets';
import { fetchHFModelConfig } from '../utils/huggingface';
import { Server, Database, Activity, UploadCloud } from 'lucide-react';

export default function InputPanel() {
  const store = useSimulatorStore();
  
  const handleHfFetch = async () => {
    if (!store.hfUrl) return;
    store.setIsFetchingHf(true);
    store.setHfError(null);
    const config = await fetchHFModelConfig(store.hfUrl);
    if (config) {
      store.setHfConfig(config);
    } else {
      store.setHfError('Failed to fetch from HuggingFace. Check the URL/ID.');
    }
    store.setIsFetchingHf(false);
  };

  return (
    <div className="glass-panel p-6 flex flex-col gap-8 h-full overflow-y-auto w-full">
      
      {/* 1. Model Selection */}
      <section className="flex flex-col gap-4">
        <h2 className="text-xl font-semibold flex items-center gap-2 text-white">
          <Database className="w-5 h-5 text-primary-500" /> Model Definition
        </h2>
        
        <div className="flex bg-surface-100 p-1 rounded-xl">
          {['PRESET', 'HF_URL', 'CUSTOM'].map((m) => (
            <button
              key={m}
              onClick={() => store.setInputMode(m as any)}
              className={`flex-1 py-1.5 text-sm font-medium rounded-lg transition-all ${
                store.inputMode === m 
                  ? 'bg-primary-600 text-white shadow-md' 
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {m === 'PRESET' ? 'Preset' : m === 'HF_URL' ? 'HF URL' : 'Custom'}
            </button>
          ))}
        </div>

        <div className="p-4 bg-surface-50 rounded-xl border border-white/5 space-y-4">
          {store.inputMode === 'PRESET' && (
            <div className="flex flex-col gap-2">
              <label className="text-sm text-gray-400">Select Model</label>
              <select 
                title="Select Preset Model"
                className="glass-input" 
                value={store.presetModelId} 
                onChange={(e) => store.setPresetModelId(e.target.value)}
              >
                {Object.entries(MODEL_PRESETS).map(([id, model]) => (
                  <option key={id} value={id} className="bg-[#1a1d24]">{model.name}</option>
                ))}
              </select>
            </div>
          )}

          {store.inputMode === 'HF_URL' && (
            <div className="flex flex-col gap-2">
              <label className="text-sm text-gray-400">HuggingFace Model ID or URL</label>
              <div className="flex flex-col gap-3">
                <input
                  type="text"
                  placeholder="e.g. meta-llama/Meta-Llama-3-8B"
                  className="glass-input"
                  value={store.hfUrl}
                  onChange={(e) => store.setHfUrl(e.target.value)}
                />
                <button 
                  onClick={handleHfFetch}
                  disabled={store.isFetchingHf || !store.hfUrl}
                  className="glass-button w-full"
                >
                  {store.isFetchingHf ? 'Parsing config...' : <><UploadCloud className="w-4 h-4" /> Fetch Config</>}
                </button>
              </div>
              {store.hfError && <p className="text-red-400 text-sm mt-1">{store.hfError}</p>}
              {store.hfConfig && (
                <div className="mt-2 text-sm text-green-400 bg-green-500/10 p-2 rounded border border-green-500/20">
                  Successfully loaded! (~{store.hfConfig.parametersInB}B params)
                </div>
              )}
            </div>
          )}

          {store.inputMode === 'CUSTOM' && (
            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">Total Params (B)</label>
                <input 
                  type="number" title="Parameters (B)" className="glass-input !py-1"
                  value={store.customConfig.parametersInB}
                  onChange={(e) => store.setCustomConfig({ parametersInB: Number(e.target.value) })}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-accent-500 font-medium">Active Params (MoE)</label>
                <input 
                  type="number" title="Active Parameters (Optional for MoE)" 
                  className="glass-input !py-1"
                  placeholder="Dense면 비워두세요"
                  value={store.customConfig.activeParametersInB || ''}
                  onChange={(e) => store.setCustomConfig({ activeParametersInB: e.target.value ? Number(e.target.value) : undefined })}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">Hidden Size</label>
                <input 
                  type="number" title="Hidden Size" className="glass-input !py-1"
                  value={store.customConfig.hiddenSize}
                  onChange={(e) => store.setCustomConfig({ hiddenSize: Number(e.target.value) })}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">Layers</label>
                <input 
                  type="number" title="Layers" className="glass-input !py-1"
                  value={store.customConfig.numLayers}
                  onChange={(e) => store.setCustomConfig({ numLayers: Number(e.target.value) })}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">KV Heads</label>
                <input 
                  type="number" title="KV Heads" className="glass-input !py-1"
                  value={store.customConfig.numKeyValueHeads}
                  onChange={(e) => store.setCustomConfig({ numKeyValueHeads: Number(e.target.value) })}
                />
              </div>
            </div>
          )}
        </div>
        
        <div className="flex flex-col gap-2">
          <label className="text-sm text-gray-400">Weight Precision / Quantization</label>
          <select 
            title="Precision" className="glass-input"
            value={store.params.quantizationBits}
            onChange={(e) => store.setParams({ quantizationBits: Number(e.target.value) })}
          >
            {QUANTIZATION_OPTIONS.map((q) => (
              <option key={q.id} value={q.bits} className="bg-[#1a1d24]">{q.name}</option>
            ))}
          </select>
        </div>
      </section>

      {/* 2. Hardware Selection */}
      <section className="flex flex-col gap-4">
        <h2 className="text-xl font-semibold flex items-center gap-2 text-white">
          <Server className="w-5 h-5 text-accent-500" /> Hardware Config
        </h2>
        
        <div className="grid grid-cols-3 gap-4">
          <div className="col-span-2 flex flex-col gap-2">
            <label className="text-sm text-gray-400">GPU Type</label>
            <select 
              title="GPU" className="glass-input"
              value={store.gpuId}
              onChange={(e) => store.setGpuId(e.target.value)}
            >
              {GPUS.map((g) => (
                <option key={g.id} value={g.id} className="bg-[#1a1d24]">{g.name}</option>
              ))}
            </select>
          </div>
          <div className="col-span-1 flex flex-col gap-2">
            <label className="text-sm text-gray-400">Count</label>
            <input 
              type="number" title="GPU Count" min="1" max="128"
              className="glass-input"
              value={store.gpuCount}
              onChange={(e) => store.setGpuCount(Number(e.target.value))}
            />
          </div>
        </div>
      </section>

      {/* 3. Task Parameters */}
      <section className="flex flex-col gap-4">
        <h2 className="text-xl font-semibold flex items-center gap-2 text-white">
          <Activity className="w-5 h-5 text-warning" /> Execution Task
        </h2>
        
        <div className="flex bg-surface-100 p-1 rounded-xl">
          <button
            onClick={() => store.setParams({ isTraining: false })}
            className={`flex-1 py-1.5 text-sm font-medium rounded-lg transition-all ${
              !store.params.isTraining ? 'bg-primary-600 text-white shadow-md' : 'text-gray-400 hover:text-white'
            }`}
          >
            Inference
          </button>
          <button
            onClick={() => store.setParams({ isTraining: true })}
            className={`flex-1 py-1.5 text-sm font-medium rounded-lg transition-all ${
              store.params.isTraining ? 'bg-accent-600 text-white shadow-md' : 'text-gray-400 hover:text-white'
            }`}
          >
            Training
          </button>
        </div>

        <div className="flex flex-col gap-4 p-4 bg-surface-50 rounded-xl border border-white/5">
          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center text-sm">
              <span className="text-gray-400">Context Length (Tokens)</span>
              <span className="text-white font-mono">{store.params.contextLength.toLocaleString()}</span>
            </div>
            <input 
              title="Context Length" type="range" 
              min="1024" max="131072" step="1024"
              value={store.params.contextLength}
              onChange={(e) => store.setParams({ contextLength: Number(e.target.value) })}
              className="w-full accent-primary-500"
            />
          </div>
          
          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center text-sm">
              <span className="text-gray-400">Batch Size</span>
              <span className="text-white font-mono">{store.params.batchSize}</span>
            </div>
            <input 
              title="Batch Size" type="range" 
              min="1" max="256" step="1"
              value={store.params.batchSize}
              onChange={(e) => store.setParams({ batchSize: Number(e.target.value) })}
              className="w-full accent-primary-500"
            />
          </div>
          
          {store.params.isTraining && (
            <div className="flex flex-col gap-2 pt-2 border-t border-white/10">
              <label className="text-sm text-gray-400">Training Method</label>
              <select 
                title="Training Method" className="glass-input"
                value={store.params.trainingMethod}
                onChange={(e) => store.setParams({ trainingMethod: e.target.value as any })}
              >
                <option value="lora" className="bg-[#1a1d24]">PEFT / LoRA</option>
                <option value="full" className="bg-[#1a1d24]">Full Fine-Tuning</option>
              </select>
            </div>
          )}
        </div>
      </section>

    </div>
  );
}
