// Scratch: owns a module worker that replays uploaded studio captures through the shipped engine.js browserRuntime.
const w = new Worker('worker.mjs', { type: 'module' });
window.RP = { msgs: [], done: false, errors: [], send(m) { w.postMessage(m); return 'sent'; } };
w.onmessage = (e) => { const m = e.data; RP.msgs.push(m); if (m.type === 'done') RP.done = true; if (m.type === 'error') RP.errors.push(m); };
w.onerror = (e) => RP.errors.push({ error: e.message });
