// Owns the harness worker; state on window.DG for short polls. Autostarts from the query string:
//   index.html?start=0&end=800[&redo=1][&local=0]   (local=0 loads ORT from jsDelivr instead of /ortdist; same sha256)
const out = document.getElementById('out');
const w = new Worker('worker.mjs', { type: 'module' });
const DG = (window.DG = { status: {}, setup: null, done: null, errors: [], info: { crossOriginIsolated, ua: navigator.userAgent },
  run(opts = {}) { DG.done = null; w.postMessage({ type: 'run', ...opts }); return 'sent'; } });
const show = () => { out.textContent = JSON.stringify({ setup: DG.setup && { created: DG.setup.created, threads: DG.setup.threads },
  status: DG.status, done: DG.done, errors: DG.errors.map((e) => e.error.split('\n')[0]) }, null, 1); };
w.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'status') { if (m.setup) DG.setup = m.setup; if (m.run) DG.status = m.run; }
  else if (m.type === 'done') DG.done = m;
  else if (m.type === 'error') DG.errors.push(m);
  show();
};
w.onerror = (e) => { DG.errors.push({ error: `worker: ${e.message}` }); show(); };
const q = new URLSearchParams(location.search);
if (q.has('start') || q.has('end')) {
  DG.run({ start: Number(q.get('start') ?? 0), end: Number(q.get('end') ?? 800), redo: q.get('redo') === '1', localRuntime: q.get('local') !== '0' });
}
