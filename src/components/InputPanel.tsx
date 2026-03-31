'use client';

import { useSimulatorStore } from '../store/useSimulatorStore';
import { MODEL_PRESETS, GPUS, QUANTIZATION_OPTIONS } from '../data/presets';
import { fetchHFModelConfig } from '../utils/huggingface';
import { Server, Database, Activity, UploadCloud, CheckCircle, Split, Copy, AlertTriangle } from 'lucide-react';

export default function InputPanel() {
  const store = useSimulatorStore();
  const data = store.scenarios[store.activeScenario];
  
  const config = store.getActiveModelConfig();
  const maxContext = config?.maxContextLength || 0;
  const totalTokens = data.params.inputLength + data.params.outputLength;
  const isLimitExceeded = maxContext > 0 && totalTokens > maxContext;

  const handleHfFetch = async () => {
    if (!data.hfUrl) return;
    store.setIsFetchingHf(true);
    store.setHfError(null);
    const fetchedConfig = await fetchHFModelConfig(data.hfUrl);
    if (fetchedConfig) {
      store.setHfConfig(fetchedConfig);
    } else {
      store.setHfError('Failed to fetch from HuggingFace. Check the URL/ID.');
    }
    store.setIsFetchingHf(false);
  };

  return (
    <div className="glass-panel p-6 md:p-8 flex flex-col gap-6 w-full h-full overflow-y-auto custom-scrollbar">
      
      {/* Compare Mode & Scenario Tabs */}
      <div className="flex flex-col gap-4 pb-4 border-b border-white/5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Split className="w-4 h-4 text-primary-400" />
            <span className="text-sm font-bold text-white uppercase tracking-wider">Compare Mode</span>
          </div>
          <button 
            onClick={() => store.setIsCompareMode(!store.isCompareMode)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${store.isCompareMode ? 'bg-primary-600' : 'bg-surface-200'}`}
          >
            <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${store.isCompareMode ? 'translate-x-6' : 'translate-x-1'}`} />
          </button>
        </div>

        {store.isCompareMode && (
          <div className="flex items-center gap-2">
            <div className="flex-1 flex bg-surface-100 p-1 rounded-xl items-center">
              {(['A', 'B'] as const).map((id) => (
                <button
                  key={id}
                  onClick={() => store.setActiveScenario(id)}
                  className={`flex-1 py-1.5 text-xs font-bold rounded-lg transition-all ${
                    store.activeScenario === id 
                      ? 'bg-primary-600 text-white shadow-lg' 
                      : 'text-gray-500 hover:text-gray-300'
                  }`}
                >
                  Scenario {id}
                </button>
              ))}
            </div>
            <button 
              title={`Copy Scenario ${store.activeScenario} to ${store.activeScenario === 'A' ? 'B' : 'A'}`}
              onClick={() => store.copyScenario(store.activeScenario, store.activeScenario === 'A' ? 'B' : 'A')}
              className="p-2.5 bg-surface-100 hover:bg-surface-200 text-gray-400 hover:text-white rounded-xl transition-all border border-white/5"
            >
              <Copy className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

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
                data.inputMode === m 
                  ? 'bg-primary-600 text-white shadow-md' 
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {m === 'PRESET' ? 'Preset' : m === 'HF_URL' ? 'HF URL' : 'Custom'}
            </button>
          ))}
        </div>

        <div className="p-4 bg-surface-50 rounded-xl border border-white/5 space-y-4">
          {data.inputMode === 'PRESET' && (
            <>
              <div className="flex flex-col gap-2">
                <label className="text-sm text-gray-400">Select Model</label>
                <select 
                  title="Select Preset Model"
                  className="glass-input" 
                  value={data.presetModelId} 
                  onChange={(e) => store.setPresetModelId(e.target.value)}
                >
                  {Object.entries(MODEL_PRESETS).map(([id, model]) => (
                    <option key={id} value={id} className="bg-[#1a1d24]">{model.name}</option>
                  ))}
                </select>
              </div>
              <div className="mt-2 px-2 py-1.5 bg-primary-500/5 rounded-lg border border-primary-500/10">
                <p className="text-[10px] text-primary-400 flex items-center gap-1.5">
                  <Activity className="w-3 h-3" />
                  Max Context: <span className="font-bold">{MODEL_PRESETS[data.presetModelId]?.maxContextLength?.toLocaleString() || 'N/A'} tokens</span>
                </p>
              </div>
            </>
          )}

          {data.inputMode === 'HF_URL' && (
            <div className="flex flex-col gap-2">
              <label className="text-sm text-gray-400">HuggingFace Model ID or URL</label>
              <div className="flex flex-col gap-3">
                <input
                  type="text"
                  placeholder="e.g. meta-llama/Meta-Llama-3-8B"
                  className="glass-input"
                  value={data.hfUrl}
                  onChange={(e) => store.setHfUrl(e.target.value)}
                />
                <button 
                  onClick={handleHfFetch}
                  disabled={data.isFetchingHf || !data.hfUrl}
                  className="glass-button w-full"
                >
                  {data.isFetchingHf ? 'Parsing config...' : <><UploadCloud className="w-4 h-4" /> Fetch Config</>}
                </button>
              </div>
              
              <div className="flex flex-col gap-2 mt-4 px-1">
                <div className="flex items-center gap-2 text-[10px] text-gray-500 font-bold uppercase tracking-wider">
                  <div className="h-px flex-1 bg-white/5"></div>
                  <span>OR</span>
                  <div className="h-px flex-1 bg-white/5"></div>
                </div>
                
                <label className="flex flex-col items-center justify-center w-full py-3 px-4 border-2 border-dashed border-white/5 rounded-xl bg-white/2 cursor-pointer hover:bg-white/5 hover:border-primary-500/30 transition-all group">
                  <div className="flex items-center gap-3">
                    <Database className="w-5 h-5 text-gray-500 group-hover:text-primary-400 transition-colors" />
                    <div className="flex flex-col">
                      <span className="text-xs font-bold text-gray-400 group-hover:text-white transition-colors">Upload config.json</span>
                      <span className="text-[10px] text-gray-600">HuggingFace format JSON</span>
                    </div>
                  </div>
                  <input 
                    type="file" 
                    accept=".json"
                    className="hidden" 
                    onChange={async (e) => {
                      const file = e.target.files?.[0];
                      if (!file) return;
                      const reader = new FileReader();
                      reader.onload = (event) => {
                        try {
                          const json = JSON.parse(event.target?.result as string);
                          const { parseRawConfig } = require('../utils/huggingface');
                          const fetchedConfig = parseRawConfig(json);
                          store.setHfConfig(fetchedConfig);
                          store.setHfError(null);
                        } catch (err) {
                          store.setHfError('Invalid config.json file.');
                        }
                      };
                      reader.readAsText(file);
                    }}
                  />
                </label>
              </div>

              {data.hfError && <p className="text-red-400 text-sm mt-3 bg-red-500/10 p-2 rounded border border-red-500/20">{data.hfError}</p>}
              {data.hfConfig && (
                <div className="mt-3 text-xs text-green-400 bg-green-500/10 p-3 rounded border border-green-500/20 flex flex-col gap-1">
                  <div className="font-bold flex items-center gap-1.5"><CheckCircle className="w-3.5 h-3.5" /> Model Loaded Successfully</div>
                  <div className="text-gray-400 text-[10px] mt-0.5">
                    {data.hfConfig.parametersInB}B Params • {data.hfConfig.hiddenSize} Hidden • {data.hfConfig.numLayers} Layers
                    {data.hfConfig.numLinearLayers !== undefined && data.hfConfig.numLinearLayers > 0 && (
                      <span className="text-accent-400"> ({data.hfConfig.numAttentionLayers} Self / {data.hfConfig.numLinearLayers} Linear)</span>
                    )}
                    {data.hfConfig.maxContextLength && (
                      <div className="mt-1 text-primary-400 font-bold">
                        Max Context: {data.hfConfig.maxContextLength.toLocaleString()} tokens
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {data.inputMode === 'CUSTOM' && (
            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">Total Params (B)</label>
                <div onDoubleClick={() => store.setEditingField('custom_params')}>
                  {store.editingField === 'custom_params' ? (
                    <input 
                      autoFocus type="text" inputMode="numeric"
                      className="glass-input !py-1 w-full text-right"
                      value={data.customConfig.parametersInB}
                      onChange={(e) => store.setCustomConfig({ parametersInB: Number(e.target.value.replace(/\D/g, '')) })}
                      onBlur={() => store.setEditingField(null)}
                      onKeyDown={(e) => e.key === 'Enter' && store.setEditingField(null)}
                    />
                  ) : (
                    <div className="glass-input !py-1 text-right flex justify-between px-3 cursor-text">
                      <span className="text-gray-500 text-[10px] uppercase">Billion</span>
                      <span className="text-white font-mono">{data.customConfig.parametersInB}</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs text-accent-400 font-medium">Active Params (MoE)</label>
                <div onDoubleClick={() => store.setEditingField('custom_active')}>
                  {store.editingField === 'custom_active' ? (
                    <input 
                      autoFocus type="text" inputMode="numeric"
                      className="glass-input !py-1 w-full text-right"
                      placeholder="Dense면 비워두세요"
                      value={data.customConfig.activeParametersInB || ''}
                      onChange={(e) => store.setCustomConfig({ activeParametersInB: e.target.value ? Number(e.target.value.replace(/\D/g, '')) : undefined })}
                      onBlur={() => store.setEditingField(null)}
                      onKeyDown={(e) => e.key === 'Enter' && store.setEditingField(null)}
                    />
                  ) : (
                    <div className="glass-input !py-1 text-right flex justify-between px-3 cursor-text">
                      <span className="text-accent-500/50 text-[10px] uppercase">Active</span>
                      <span className="text-accent-400 font-mono italic">{data.customConfig.activeParametersInB || 'Dense'}</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">Hidden Size</label>
                <div onDoubleClick={() => store.setEditingField('custom_hidden')}>
                  {store.editingField === 'custom_hidden' ? (
                    <input 
                      autoFocus type="text" inputMode="numeric"
                      className="glass-input !py-1 w-full text-right"
                      value={data.customConfig.hiddenSize}
                      onChange={(e) => store.setCustomConfig({ hiddenSize: Number(e.target.value.replace(/\D/g, '')) })}
                      onBlur={() => store.setEditingField(null)}
                      onKeyDown={(e) => e.key === 'Enter' && store.setEditingField(null)}
                    />
                  ) : (
                    <div className="glass-input !py-1 text-right flex justify-between px-3 cursor-text">
                       <span className="text-gray-500 text-[10px] uppercase">Dim</span>
                       <span className="text-white font-mono">{data.customConfig.hiddenSize}</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">Layers</label>
                <div onDoubleClick={() => store.setEditingField('custom_layers')}>
                  {store.editingField === 'custom_layers' ? (
                    <input 
                      autoFocus type="text" inputMode="numeric"
                      className="glass-input !py-1 w-full text-right"
                      value={data.customConfig.numLayers}
                      onChange={(e) => store.setCustomConfig({ numLayers: Number(e.target.value.replace(/\D/g, '')) })}
                      onBlur={() => store.setEditingField(null)}
                      onKeyDown={(e) => e.key === 'Enter' && store.setEditingField(null)}
                    />
                  ) : (
                    <div className="glass-input !py-1 text-right flex justify-between px-3 cursor-text">
                      <span className="text-gray-500 text-[10px] uppercase">Blocks</span>
                      <span className="text-white font-mono">{data.customConfig.numLayers}</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">Attention Heads</label>
                <div onDoubleClick={() => store.setEditingField('custom_att_heads')}>
                  {store.editingField === 'custom_att_heads' ? (
                    <input 
                      autoFocus type="text" inputMode="numeric"
                      className="glass-input !py-1 w-full text-right"
                      value={data.customConfig.numAttentionHeads}
                      onChange={(e) => store.setCustomConfig({ numAttentionHeads: Number(e.target.value.replace(/\D/g, '')) })}
                      onBlur={() => store.setEditingField(null)}
                      onKeyDown={(e) => e.key === 'Enter' && store.setEditingField(null)}
                    />
                  ) : (
                    <div className="glass-input !py-1 text-right flex justify-between px-3 cursor-text">
                      <span className="text-gray-500 text-[10px] uppercase">Query</span>
                      <span className="text-white font-mono">{data.customConfig.numAttentionHeads}</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">KV Heads</label>
                <div onDoubleClick={() => store.setEditingField('custom_kv_heads')}>
                  {store.editingField === 'custom_kv_heads' ? (
                    <input 
                      autoFocus type="text" inputMode="numeric"
                      className="glass-input !py-1 w-full text-right"
                      value={data.customConfig.numKeyValueHeads}
                      onChange={(e) => store.setCustomConfig({ numKeyValueHeads: Number(e.target.value.replace(/\D/g, '')) })}
                      onBlur={() => store.setEditingField(null)}
                      onKeyDown={(e) => e.key === 'Enter' && store.setEditingField(null)}
                    />
                  ) : (
                    <div className="glass-input !py-1 text-right flex justify-between px-3 cursor-text">
                      <span className="text-gray-500 text-[10px] uppercase">GQA/MQA</span>
                      <span className="text-white font-mono">{data.customConfig.numKeyValueHeads}</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs text-primary-400 font-medium">Max Context</label>
                <div onDoubleClick={() => store.setEditingField('custom_context')}>
                  {store.editingField === 'custom_context' ? (
                    <input 
                      autoFocus type="text" inputMode="numeric"
                      className="glass-input !py-1 w-full text-right"
                      value={data.customConfig.maxContextLength || ''}
                      onChange={(e) => store.setCustomConfig({ maxContextLength: Number(e.target.value.replace(/\D/g, '')) })}
                      onBlur={() => store.setEditingField(null)}
                      onKeyDown={(e) => e.key === 'Enter' && store.setEditingField(null)}
                    />
                  ) : (
                    <div className="glass-input !py-1 text-right flex justify-between px-3 cursor-text">
                      <span className="text-primary-500/50 text-[10px] uppercase">Tokens</span>
                      <span className="text-primary-400 font-mono">{data.customConfig.maxContextLength?.toLocaleString() || 'N/A'}</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="col-span-2 mt-2 px-3 py-2 bg-white/5 rounded-lg border border-white/10">
                <p className="text-[10px] text-gray-500 leading-relaxed italic">
                  * Custom 모드에서는 최대 컨텍스트 길이를 자동으로 알 수 없습니다. 해당 모델의 config.json이나 공식 기술 문서를 참고하여 슬라이더 범위를 조절해 주세요.
                </p>
              </div>
            </div>
          )}
        </div>
        
        <div className="flex flex-col gap-2">
          <label className="text-sm text-gray-400">Weight Precision / Quantization</label>
          <select 
            title="Weights Quantization" className="glass-input"
            value={data.params.quantizationBits}
            onChange={(e) => store.setParams({ quantizationBits: Number(e.target.value) })}
          >
            {QUANTIZATION_OPTIONS.map((q) => (
              <option key={q.id} value={q.bits} className="bg-[#1a1d24]">{q.name}</option>
            ))}
          </select>
        </div>
        <div className="flex flex-col gap-2">
          <label className="text-sm text-gray-400">KV Quantization</label>
          <select 
            title="KV Cache Quantization" className="glass-input"
            value={data.params.kvQuantizationBits}
            onChange={(e) => store.setParams({ kvQuantizationBits: Number(e.target.value) })}
          >
            <option value={16} className="bg-[#1a1d24]">16-bit (Default)</option>
            <option value={8} className="bg-[#1a1d24]">8-bit (FP8/INT8)</option>
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
              value={data.gpuId}
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
              value={data.gpuCount}
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
              !data.params.isTraining ? 'bg-primary-600 text-white shadow-md' : 'text-gray-400 hover:text-white'
            }`}
          >
            Inference
          </button>
          <button
            onClick={() => store.setParams({ isTraining: true })}
            className={`flex-1 py-1.5 text-sm font-medium rounded-lg transition-all ${
              data.params.isTraining ? 'bg-accent-600 text-white shadow-md' : 'text-gray-400 hover:text-white'
            }`}
          >
            Training
          </button>
        </div>

        <div className="flex flex-col gap-4 p-4 bg-surface-50 rounded-xl border border-white/5">
          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center text-sm">
              <span className="text-gray-400">Input Prompt Limit (Tokens)</span>
              <div onDoubleClick={() => store.setEditingField('inputLength')}>
                {store.editingField === 'inputLength' ? (
                  <input 
                    autoFocus
                    type="text"
                    inputMode="numeric"
                    className="w-24 bg-primary-600/20 border border-primary-500/50 rounded px-2 py-0.5 text-right text-primary-400 font-mono font-bold text-xs outline-none"
                    value={data.params.inputLength}
                    onChange={(e) => {
                      const val = e.target.value.replace(/\D/g, '');
                      store.setParams({ inputLength: val ? Number(val) : 0 });
                    }}
                    onBlur={() => store.setEditingField(null)}
                    onKeyDown={(e) => e.key === 'Enter' && store.setEditingField(null)}
                  />
                ) : (
                  <span className={`${isLimitExceeded ? 'text-red-500' : 'text-primary-400'} font-mono font-bold cursor-help border-b border-dotted ${isLimitExceeded ? 'border-red-500/50' : 'border-primary-500/50'}`} title="Double-click to type">
                    {data.params.inputLength.toLocaleString()}
                  </span>
                )}
              </div>
            </div>
            <input 
              title="Input Prompt Length" type="range" 
              min="128" max={maxContext || 131072} step="128"
              value={data.params.inputLength}
              onChange={(e) => store.setParams({ inputLength: Number(e.target.value) })}
              className="w-full accent-primary-500"
            />
          </div>

          {isLimitExceeded && (
            <div className="flex items-start gap-2.5 p-3 bg-red-500/10 border border-red-500/20 rounded-xl animate-in fade-in slide-in-from-top-1 duration-300">
              <AlertTriangle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
              <div className="flex flex-col gap-0.5">
                <p className="text-[11px] font-bold text-red-400">Token Limit Exceeded</p>
                <p className="text-[10px] text-red-400/80 leading-tight">
                  Sum of input ({data.params.inputLength.toLocaleString()}) and output ({data.params.outputLength.toLocaleString()}) tokens 
                  is <span className="font-bold underline">{totalTokens.toLocaleString()}</span>, which exceeds the model's 
                  Max Context ({maxContext.toLocaleString()}).
                </p>
              </div>
            </div>
          )}

          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center text-sm">
              <span className="text-gray-400">Max Generated Output (Tokens)</span>
              <div onDoubleClick={() => store.setEditingField('outputLength')}>
                {store.editingField === 'outputLength' ? (
                  <input 
                    autoFocus
                    type="text"
                    inputMode="numeric"
                    className="w-24 bg-accent-600/20 border border-accent-500/50 rounded px-2 py-0.5 text-right text-accent-400 font-mono font-bold text-xs outline-none"
                    value={data.params.outputLength}
                    onChange={(e) => {
                      const val = e.target.value.replace(/\D/g, '');
                      store.setParams({ outputLength: val ? Number(val) : 0 });
                    }}
                    onBlur={() => store.setEditingField(null)}
                    onKeyDown={(e) => e.key === 'Enter' && store.setEditingField(null)}
                  />
                ) : (
                  <span className={`${isLimitExceeded ? 'text-red-500' : 'text-accent-400'} font-mono font-bold cursor-help border-b border-dotted ${isLimitExceeded ? 'border-red-500/50' : 'border-accent-500/50'}`} title="Double-click to type">
                    {data.params.outputLength.toLocaleString()}
                  </span>
                )}
              </div>
            </div>
            <input 
              title="Max Output Length" type="range" 
              min="128" max={maxContext || 131072} step="128"
              value={data.params.outputLength}
              onChange={(e) => store.setParams({ outputLength: Number(e.target.value) })}
              className="w-full accent-accent-500"
            />
          </div>
          
          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center text-sm">
              <span className="text-gray-400 font-medium">Batch Size</span>
              <div onDoubleClick={() => store.setEditingField('batchSize')}>
                {store.editingField === 'batchSize' ? (
                  <input 
                    autoFocus
                    type="text"
                    inputMode="numeric"
                    className="w-16 bg-white/10 border border-white/20 rounded px-2 py-0.5 text-right text-white font-mono font-bold text-xs outline-none"
                    value={data.params.batchSize}
                    onChange={(e) => {
                      const val = e.target.value.replace(/\D/g, '');
                      store.setParams({ batchSize: val ? Number(val) : 0 });
                    }}
                    onBlur={() => store.setEditingField(null)}
                    onKeyDown={(e) => e.key === 'Enter' && store.setEditingField(null)}
                  />
                ) : (
                  <span className="text-white font-mono font-bold cursor-help border-b border-dotted border-white/30" title="Double-click to type">
                    {data.params.batchSize}
                  </span>
                )}
              </div>
            </div>
            <input 
              title="Batch Size" type="range" 
              min="1" max="256" step="1"
              value={data.params.batchSize}
              onChange={(e) => store.setParams({ batchSize: Number(e.target.value) })}
              className="w-full accent-primary-500"
            />
          </div>
          
          {data.params.isTraining && (
            <>
              <div className="flex flex-col gap-2 pt-4 border-t border-white/10">
                <div className="flex justify-between items-center text-sm">
                  <span className="text-gray-400 font-medium">Dataset Size (Samples)</span>
                  <div onDoubleClick={() => store.setEditingField('numSamples')}>
                    {store.editingField === 'numSamples' ? (
                      <input 
                        autoFocus type="text" inputMode="numeric"
                        className="w-24 bg-accent-600/20 border border-accent-500/50 rounded px-2 py-0.5 text-right text-accent-400 font-mono font-bold text-xs outline-none"
                        value={data.params.numSamples}
                        onChange={(e) => store.setParams({ numSamples: Number(e.target.value.replace(/\D/g, '')) })}
                        onBlur={() => store.setEditingField(null)}
                        onKeyDown={(e) => e.key === 'Enter' && store.setEditingField(null)}
                      />
                    ) : (
                      <span className="text-accent-400 font-mono font-bold cursor-help border-b border-dotted border-accent-500/50">
                        {data.params.numSamples?.toLocaleString()}
                      </span>
                    )}
                  </div>
                </div>
                <input 
                  type="range" min="100" max="1000000" step="100"
                  value={data.params.numSamples}
                  onChange={(e) => store.setParams({ numSamples: Number(e.target.value) })}
                  className="w-full accent-accent-500"
                />
              </div>

              <div className="flex flex-col gap-2">
                <div className="flex justify-between items-center text-sm">
                  <span className="text-gray-400 font-medium">Training Epochs</span>
                  <div onDoubleClick={() => store.setEditingField('numEpochs')}>
                    {store.editingField === 'numEpochs' ? (
                      <input 
                        autoFocus type="text" inputMode="numeric"
                        className="w-16 bg-accent-600/20 border border-accent-500/50 rounded px-2 py-0.5 text-right text-accent-400 font-mono font-bold text-xs outline-none"
                        value={data.params.numEpochs}
                        onChange={(e) => store.setParams({ numEpochs: Number(e.target.value.replace(/\D/g, '')) })}
                        onBlur={() => store.setEditingField(null)}
                        onKeyDown={(e) => e.key === 'Enter' && store.setEditingField(null)}
                      />
                    ) : (
                      <span className="text-accent-400 font-mono font-bold cursor-help border-b border-dotted border-accent-500/50">
                        {data.params.numEpochs}
                      </span>
                    )}
                  </div>
                </div>
                <input 
                  type="range" min="1" max="100" step="1"
                  value={data.params.numEpochs}
                  onChange={(e) => store.setParams({ numEpochs: Number(e.target.value) })}
                  className="w-full accent-accent-500"
                />
              </div>

              <div className="flex flex-col gap-2 pt-2">
                <label className="text-sm text-gray-400">Training Method</label>
                <select 
                  title="Training Method" className="glass-input"
                  value={data.params.trainingMethod}
                  onChange={(e) => store.setParams({ trainingMethod: e.target.value as any })}
                >
                  <option value="lora" className="bg-[#1a1d24]">PEFT / LoRA (Efficient)</option>
                  <option value="full" className="bg-[#1a1d24]">Full Fine-Tuning (Heavy)</option>
                </select>
              </div>
            </>
          )}
        </div>
      </section>

    </div>
  );
}
