'use client';

import { useSimulatorStore } from '../store/useSimulatorStore';
import { calculateWeightsMemory, calculateKVCache, calculateTrainingMemory, estimateTPS } from '../utils/calculator';
import { GPUS } from '../data/presets';
import { Cpu, Zap, AlertTriangle, CheckCircle, Database } from 'lucide-react';

export default function ResultsDashboard() {
  const store = useSimulatorStore();
  const activeModel = store.getActiveModelConfig();
  
  if (!activeModel) {
    return (
      <div className="glass-panel p-6 flex flex-col items-center justify-center h-full text-center">
        <p className="text-gray-400 animate-pulse text-lg">Waiting for model configuration...</p>
        <p className="text-gray-500 text-sm mt-2">Please configure the model on the left panel.</p>
      </div>
    );
  }

  const selectedGpu = GPUS.find(g => g.id === store.gpuId) || GPUS[0];
  const totalVramLimit = selectedGpu.vramGb * store.gpuCount;

  // Math Computations
  const weightsGb = calculateWeightsMemory(activeModel.parametersInB, store.params.quantizationBits);
  const kvGb = calculateKVCache(activeModel, store.params);
  const trainingGb = store.params.isTraining ? calculateTrainingMemory(weightsGb, store.params.trainingMethod) : 0;
  
  const totalUsedGb = weightsGb + kvGb + trainingGb;
  
  let tps = 0;
  if (!store.params.isTraining && totalUsedGb <= totalVramLimit) {
      const activeParams = activeModel.activeParametersInB || activeModel.parametersInB;
      tps = estimateTPS(selectedGpu.bandwidthGbps, activeParams, store.params.quantizationBits, store.gpuCount);
  }

  // Status computation
  const isOom = totalUsedGb > totalVramLimit;
  const isWarning = totalUsedGb > totalVramLimit * 0.8 && !isOom;
  
  const statusColor = isOom ? 'text-error' : isWarning ? 'text-warning' : 'text-success';
  const statusBg = isOom ? 'bg-error/20 border-error/50 shadow-[0_0_20px_rgba(239,68,68,0.3)]' : isWarning ? 'bg-warning/20 border-warning/50' : 'bg-success/20 border-success/50';

  // Stacked Bar relative segments (capping to 100% max for visual sanity, unless OOM where overflow gets the rest)
  const maxScale = Math.max(totalUsedGb, totalVramLimit);
  const weightsPct = (weightsGb / maxScale) * 100;
  const kvPct = (kvGb / maxScale) * 100;
  const trainingPct = (trainingGb / maxScale) * 100;
  const overflowPct = isOom ? ((totalUsedGb - totalVramLimit) / maxScale) * 100 : 0;

  return (
    <div className="glass-panel p-8 flex flex-col gap-8 h-full w-full overflow-y-auto">
      <div className="flex justify-between items-start">
        <h2 className="text-2xl font-bold flex items-center gap-3 text-white">
          <Zap className="w-6 h-6 text-accent-500" /> Simulation Results
        </h2>
      </div>

      {/* Main Status */}
      <div className={`p-6 rounded-2xl border flex items-center gap-5 transition-all duration-500 ${statusBg}`}>
        {isOom ? <AlertTriangle className="w-12 h-12 text-error flex-shrink-0" /> : <CheckCircle className="w-12 h-12 text-success flex-shrink-0" />}
        <div>
          <h3 className={`text-2xl font-bold tracking-tight ${statusColor}`}>
            {isOom ? 'OUT OF MEMORY (OOM)' : isWarning ? 'WARNING: HIGH VRAM USAGE' : 'SUCCESS: VRAM SUFFICIENT'}
          </h3>
          <p className="text-gray-300 mt-2 text-sm leading-relaxed">
            {isOom 
              ? `Requires ${totalUsedGb.toFixed(1)} GB, but only ${totalVramLimit} GB is available on ${store.gpuCount}x ${selectedGpu.name}.`
              : `Using ${totalUsedGb.toFixed(1)} GB out of ${totalVramLimit} GB available.`}
          </p>
        </div>
      </div>

      {/* VRAM Visualizer */}
      <div className="flex flex-col gap-4 mt-2">
        <div className="flex justify-between items-end">
          <span className="font-medium text-gray-300">VRAM Allocation Breakdown</span>
          <span className={`text-sm font-mono font-bold ${isOom ? 'text-error animate-pulse' : 'text-gray-400'}`}>
            {totalUsedGb.toFixed(1)} GB / {totalVramLimit} GB
          </span>
        </div>
        
        {/* Stacked Bar */}
        <div className="h-10 w-full bg-surface-200 rounded-full overflow-hidden flex shadow-inner relative border border-white/5">
          <div 
            className="h-full bg-primary-500 transition-all duration-1000 ease-out"
            style={{ width: `${weightsPct}%` }}
            title={`Weights: ${weightsGb.toFixed(1)} GB`}
          />
          <div 
            className="h-full bg-accent-500 transition-all duration-1000 ease-out"
            style={{ width: `${kvPct}%` }}
            title={`KV Cache: ${kvGb.toFixed(1)} GB`}
          />
          {store.params.isTraining && (
            <div 
              className="h-full bg-warning transition-all duration-1000 ease-out"
              style={{ width: `${trainingPct}%` }}
              title={`Training Overhead: ${trainingGb.toFixed(1)} GB`}
            />
          )}
          {isOom && (
             <div 
               className="h-full bg-error transition-all duration-1000 pattern-diagonal-lines"
               style={{ width: `${overflowPct}%` }}
               title={`Shortfall: ${(totalUsedGb - totalVramLimit).toFixed(1)} GB`}
             />
          )}
        </div>

        {/* Legend */}
        <div className="flex gap-4 mt-2 text-sm text-gray-400 flex-wrap">
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-primary-500"></div> Weights ({weightsGb.toFixed(1)} GB)</div>
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-accent-500"></div> KV Cache ({kvGb.toFixed(1)} GB)</div>
          {store.params.isTraining && (
            <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-warning"></div> Training/Activations ({trainingGb.toFixed(1)} GB)</div>
          )}
          {isOom && (
             <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-error font-bold text-error"></div> Shortfall ({(totalUsedGb - totalVramLimit).toFixed(1)} GB)</div>
          )}
        </div>
      </div>

      {/* Performance Matrix */}
      {!store.params.isTraining && (
        <div className="mt-8 grid grid-cols-2 gap-4">
          <div className="glass-panel !border-white/5 p-6 flex flex-col items-center justify-center text-center">
            <Cpu className={`w-8 h-8 mb-2 ${isOom ? 'text-gray-600' : 'text-primary-500'}`} />
            <span className="text-gray-400 text-sm uppercase tracking-wider font-semibold">Estimated Speed</span>
            <span className={`text-4xl font-black mt-2 ${isOom ? 'text-gray-600' : 'text-white'}`}>
              {tps.toLocaleString()} <span className="text-lg text-primary-500 font-normal tracking-tight">Tokens / sec</span>
            </span>
          </div>
          
          <div className="glass-panel !border-white/5 p-6 flex flex-col items-center justify-center text-center">
            <Database className="w-8 h-8 text-accent-500 mb-2 opacity-80" />
            <span className="text-gray-400 text-sm uppercase tracking-wider font-semibold">Memory Bandwidth</span>
            <span className="text-4xl font-black text-white mt-2">
              {(selectedGpu.bandwidthGbps * store.gpuCount).toLocaleString()} <span className="text-xl text-accent-500 font-normal">GB/s</span>
            </span>
          </div>
        </div>
      )}

      {/* Recommendations */}
      <div className="mt-auto pt-8 border-t border-white/5">
        <h4 className="font-bold text-gray-200 mb-3 flex items-center gap-2">
           💡 AI Recommendations
        </h4>
        <ul className="list-inside text-sm text-gray-400 space-y-3">
          {isOom && store.params.quantizationBits === 16 && (
            <li className="flex gap-2 items-start"><span className="text-primary-500 mt-0.5">•</span> <span>Try reducing precision to <strong className="text-accent-500">INT8 or INT4 quantization</strong>. This will significantly drop the weights memory.</span></li>
          )}
          {isOom && store.params.contextLength > 4096 && (
            <li className="flex gap-2 items-start"><span className="text-primary-500 mt-0.5">•</span> <span>Your KV Cache is large. Consider reducing the <strong>Context Length</strong>.</span></li>
          )}
          {isOom && store.params.isTraining && store.params.trainingMethod === 'full' && (
            <li className="flex gap-2 items-start"><span className="text-primary-500 mt-0.5">•</span> <span>Full Fine-tuning takes massive VRAM. Switch to <strong className="text-primary-500">PEFT / LoRA</strong> to train efficiently on fewer GPUs.</span></li>
          )}
          {isOom && (
             <li className="flex gap-2 items-start"><span className="text-primary-500 mt-0.5">•</span> <span>Increase the <strong>GPU Count</strong> or upgrade from {selectedGpu.name} to a larger VRAM instance (e.g. H200 141GB).</span></li>
          )}
          {!isOom && !isWarning && (
            <li className="flex gap-2 items-start"><span className="text-success mt-0.5">✓</span> <span>Plenty of VRAM headroom available. You could increase the <strong>Batch Size</strong> to maximize throughput.</span></li>
          )}
        </ul>
      </div>

    </div>
  );
}
