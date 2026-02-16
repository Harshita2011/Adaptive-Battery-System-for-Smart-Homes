import React, { useState, useEffect, useCallback } from 'react';
import { ShieldAlert, Cpu, List, Settings, ShieldCheck } from 'lucide-react';

function App() {
  const [data, setData] = useState(null);
  const [simLoad, setSimLoad] = useState(22);
  const [mode, setMode] = useState('SIM');
  const [devices, setDevices] = useState([]);
  const [outbox, setOutbox] = useState([]);
  const [adapter, setAdapter] = useState(null);

  const fetchData = useCallback(async () => {
    try {
      const [statusRes, devicesRes, outboxRes, adapterRes] = await Promise.all([
        fetch(`http://localhost:8000/api/system-status?manual_load=${simLoad}`),
        fetch('http://localhost:8000/api/devices'),
        fetch('http://localhost:8000/api/control/outbox'),
        fetch('http://localhost:8000/api/control/adapter'),
      ]);

      const [statusJson, devicesJson, outboxJson, adapterJson] = await Promise.all([
        statusRes.json(),
        devicesRes.json(),
        outboxRes.json(),
        adapterRes.json(),
      ]);

      setData(statusJson);
      setDevices(Array.isArray(devicesJson.devices) ? devicesJson.devices : []);
      setOutbox(Array.isArray(outboxJson.commands) ? outboxJson.commands : []);
      setAdapter(adapterJson || null);

      if (statusJson.mode) {
        setMode(statusJson.mode);
      }
    } catch (err) {
      console.error('Connection to Digital Twin lost...');
    }
  }, [simLoad]);

  const changeMode = async (nextMode) => {
    try {
      await fetch(`http://localhost:8000/api/mode?name=${nextMode}`, { method: 'POST' });
      setMode(nextMode);
      fetchData();
    } catch (err) {
      console.error('Failed to change mode');
    }
  };

  const clearOutbox = async () => {
    try {
      await fetch('http://localhost:8000/api/control/outbox/clear', { method: 'POST' });
      fetchData();
    } catch (err) {
      console.error('Failed to clear command outbox');
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, [fetchData]);

  if (!data) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center text-blue-500 font-mono animate-pulse">
        CONNECTING TO REAL-TIME DIGITAL TWIN...
      </div>
    );
  }

  const isProtected = data.metrics.status !== 'OPTIMAL';
  const thermalStress = Math.min(((parseFloat(data.metrics.temp) || 30) / 65) * 100, 100);
  const usingLiveFeed = data.source === 'LIVE';
  const publishedCount = outbox.filter((c) => c.status === 'published').length;
  const failedCount = outbox.filter((c) => c.status === 'failed').length;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8 font-sans flex flex-col md:flex-row gap-8">
      <aside className="w-full md:w-80 bg-slate-900 p-6 rounded-2xl border border-slate-800 h-fit shadow-2xl">
        <h2 className="text-lg font-bold mb-6 flex items-center gap-2"><Settings size={18}/> System Inputs</h2>
        <div className="space-y-8">
          <div>
            <label className="text-xs font-bold text-slate-500 uppercase">Runtime Mode</label>
            <div className="grid grid-cols-2 gap-2 mt-3">
              <button
                onClick={() => changeMode('SIM')}
                className={`px-3 py-2 rounded text-xs font-bold border ${mode === 'SIM' ? 'bg-blue-600 border-blue-400 text-white' : 'bg-slate-800 border-slate-700 text-slate-300'}`}
              >
                SIM MODE
              </button>
              <button
                onClick={() => changeMode('LIVE')}
                className={`px-3 py-2 rounded text-xs font-bold border ${mode === 'LIVE' ? 'bg-emerald-600 border-emerald-400 text-white' : 'bg-slate-800 border-slate-700 text-slate-300'}`}
              >
                LIVE MODE
              </button>
            </div>
            <p className="text-[10px] text-slate-400 mt-2 italic">
              LIVE expects telemetry via POST /api/telemetry.
            </p>
          </div>

          <div className={mode === 'LIVE' ? 'opacity-50' : ''}>
            <label className="text-xs font-bold text-slate-500 uppercase">Input Base Load (Amps)</label>
            <input
              type="range"
              min="15"
              max="35"
              step="0.5"
              value={simLoad}
              disabled={mode === 'LIVE'}
              onChange={(e) => setSimLoad(e.target.value)}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500 mt-4 disabled:cursor-not-allowed"
            />
            <div className="flex justify-between text-mono text-sm mt-2 font-bold text-blue-400">
              <span>15A</span>
              <span className="bg-blue-500/20 px-2 rounded">{simLoad}A</span>
              <span>35A</span>
            </div>
          </div>

          <div className="pt-4 border-t border-slate-800">
            <label className="text-xs font-bold text-slate-500 uppercase mb-3 block">Hardware Thermal Stress</label>
            <div className="h-4 w-full bg-slate-800 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${thermalStress > 70 ? 'bg-red-500' : 'bg-blue-500'}`}
                style={{ width: `${thermalStress}%` }}
              />
            </div>
            <p className="text-[10px] text-slate-400 mt-2 italic">Monitoring SEI layer stability...</p>
          </div>
        </div>
      </aside>

      <main className="flex-1">
        <header className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-3">
              <Cpu className={isProtected ? 'text-red-500 animate-spin-slow' : 'text-blue-500'} />
              AI Energy Controller v2.0
            </h1>
            <p className="text-slate-500 text-sm mt-1">Autonomous Multi-Agent Governance Active</p>
          </div>
          <div className="flex gap-4">
            <div className="bg-blue-900/20 border border-blue-500/50 px-4 py-2 rounded-lg text-blue-400 font-mono text-sm">
              SOH: {data.metrics.health}
            </div>
            <div className={`px-4 py-2 rounded-lg font-mono text-sm border ${usingLiveFeed ? 'bg-emerald-900/20 border-emerald-500 text-emerald-400' : 'bg-amber-900/20 border-amber-500 text-amber-300'}`}>
              Source: {data.source}
            </div>
          </div>
        </header>

        <div className={`mb-8 p-6 rounded-2xl border-2 flex items-center gap-6 transition-all duration-500 ${isProtected ? 'bg-red-950/30 border-red-500/50' : 'bg-slate-900 border-slate-800'}`}>
          <div className={`p-4 rounded-full ${isProtected ? 'bg-red-500 text-white animate-pulse' : 'bg-slate-800 text-blue-400'}`}>
            {isProtected ? <ShieldAlert size={32} /> : <ShieldCheck size={32} />}
          </div>
          <div>
            <h2 className="text-xl font-bold">{isProtected ? 'Hardware Protection Active' : 'Nominal Operation'}</h2>
            <p className="text-slate-400 text-sm">
              {isProtected
                ? `Governor engaged load-shedding to mitigate ${data.metrics.temp}C thermal surge.`
                : 'Controller reports nominal battery operation.'}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <StatCard label="Live Current Draw" value={data.metrics.load} sub="Stochastic Fluctuations" color="text-yellow-400" />
          <StatCard label="Internal Temp" value={`${data.metrics.temp}C`} sub="Physics-Informed Input" color="text-orange-400" />
          <StatCard label="Fuzzy Power Limit" value={data.metrics.limit || '100%'} sub="Governance Constraint" color="text-red-400" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          {data.appliances.map((app, i) => (
            <div key={i} className={`p-5 rounded-xl border transition-all duration-300 ${app.status.includes('OFF') ? 'border-red-500/50 bg-red-950/10' : 'border-slate-800 bg-slate-900'}`}>
              <div className="flex justify-between items-center mb-4">
                <span className="text-xs font-bold uppercase text-slate-500">{app.name}</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${app.status.includes('OFF') ? 'bg-red-500 text-white' : 'bg-blue-500 text-white'}`}>
                  {app.status}
                </span>
              </div>
              <p className="text-3xl font-mono tracking-tighter">{app.status.includes('OFF') ? '0.00' : app.usage} <span className="text-sm text-slate-500">kW</span></p>
            </div>
          ))}
        </div>

        <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-inner">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-sm font-bold text-blue-500 uppercase flex items-center gap-2"><List size={16}/> AI Reasoning and Logic Narrative</h3>
            <div className="flex items-center gap-2 text-[10px] text-slate-500">
              <span className="w-2 h-2 bg-blue-500 rounded-full animate-ping"></span>
              POLLING BACKEND (1Hz)
            </div>
          </div>
          <div className="space-y-3 font-mono text-[11px]">
            {data.recent_actions.map((act, i) => (
              <div key={i} className="group p-3 bg-slate-950/50 rounded-lg border-l-4 border-blue-500 flex flex-col md:flex-row md:items-center justify-between hover:bg-slate-950 transition-colors">
                <span className="text-slate-600 mb-1 md:mb-0">[{act.time}]</span>
                <span className="text-slate-300 flex-1 md:ml-6 italic">{act.event}</span>
                <span className={`font-bold ${act.action.includes('limit') ? 'text-red-400' : 'text-green-400'}`}>
                  {'>'} {act.action}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-inner mt-8">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-sm font-bold text-emerald-400 uppercase">Control Monitor</h3>
            <div className="text-[10px] text-slate-500 font-mono">DEVICES / OUTBOX / ADAPTER</div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
            <div className="bg-slate-950/60 border border-slate-800 rounded-lg p-4">
              <p className="text-[10px] text-slate-500 uppercase mb-2">Adapter</p>
              <p className="text-sm font-mono text-slate-200">Type: {adapter?.config?.type || 'n/a'}</p>
              <p className={`text-sm font-mono ${adapter?.mqtt_connected ? 'text-emerald-400' : 'text-amber-400'}`}>
                MQTT: {adapter?.mqtt_connected ? 'connected' : 'not connected'}
              </p>
              {adapter?.mqtt_error ? (
                <p className="text-[11px] text-red-400 mt-2 break-all">Error: {adapter.mqtt_error}</p>
              ) : null}
            </div>

            <div className="bg-slate-950/60 border border-slate-800 rounded-lg p-4">
              <p className="text-[10px] text-slate-500 uppercase mb-2">Normalized Devices ({devices.length})</p>
              <div className="space-y-2 text-[11px] font-mono">
                {devices.slice(0, 5).map((d) => (
                  <div key={d.device_id} className="flex items-center justify-between border-b border-slate-800 pb-1">
                    <span className="text-slate-300">{d.device_id}</span>
                    <span className="text-slate-500">{(Number(d.power_w || 0) / 1000).toFixed(2)} kW</span>
                  </div>
                ))}
                {!devices.length ? <p className="text-slate-500">No devices ingested.</p> : null}
              </div>
            </div>

            <div className="bg-slate-950/60 border border-slate-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <p className="text-[10px] text-slate-500 uppercase">Command Outbox ({outbox.length})</p>
                <button
                  onClick={clearOutbox}
                  className="px-2 py-1 text-[10px] font-bold rounded border border-slate-600 text-slate-300 hover:border-slate-400"
                >
                  Clear
                </button>
              </div>
              <div className="flex items-center gap-2 mb-3">
                <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-900/40 text-emerald-300 border border-emerald-700">
                  published: {publishedCount}
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-red-900/40 text-red-300 border border-red-700">
                  failed: {failedCount}
                </span>
              </div>
              <div className="space-y-2 text-[11px] font-mono">
                {outbox.slice(0, 5).map((c) => (
                  <div key={c.command_id} className="border-b border-slate-800 pb-1">
                    <p className="text-slate-300">{c.device_id}: {c.command}</p>
                    <p className={`${c.status === 'published' ? 'text-emerald-400' : c.status === 'failed' ? 'text-red-400' : 'text-amber-400'}`}>
                      {c.status} ({c.adapter})
                    </p>
                    <p className="text-slate-500 text-[10px]">{c.timestamp || 'no timestamp'}</p>
                  </div>
                ))}
                {!outbox.length ? <p className="text-slate-500">No commands dispatched.</p> : null}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

const StatCard = ({ label, value, sub, color }) => (
  <div className="bg-slate-900 p-5 rounded-xl border border-slate-800 hover:border-slate-700 transition-all">
    <p className="text-[10px] uppercase font-bold text-slate-500 mb-1">{label}</p>
    <p className={`text-3xl font-mono font-bold tracking-tighter ${color}`}>{value}</p>
    <p className="text-[9px] text-slate-600 mt-2 font-mono uppercase">{sub}</p>
  </div>
);

export default App;
