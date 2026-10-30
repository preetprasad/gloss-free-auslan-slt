// Gate B: demo/static/engine/tracklets.js vs tracker_v2.choose on every frame of the 60 ground-truth clips.
// Usage: node eval_port.mjs FLAT_BOX_DIR V2_REFERENCE.json [--module path/to/tracklets.js]
// Exits non-zero on any mismatch (both weight sets are checked; TUNED is the Gate B criterion).
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const [dir, refPath] = process.argv.slice(2);
const mi = process.argv.indexOf('--module');
const mod = mi > 0 ? resolve(process.argv[mi + 1]) : new URL('../../../demo/static/engine/tracklets.js', import.meta.url).pathname;
const { choose, TUNED, DEFAULT } = await import(pathToFileURL(mod));
const ref = JSON.parse(readFileSync(refPath));
const f32 = (p) => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };

export function loadClip(dir, c) {
  const m = JSON.parse(readFileSync(`${dir}/${c}.json`));
  const kp = f32(`${dir}/${c}.kp.f32`), sc = f32(`${dir}/${c}.sc.f32`), box = f32(`${dir}/${c}.box.f32`);
  const { T, N } = m, K = 133;
  const frames = [];
  for (let t = 0; t < T; t++) {
    const n = m.NDET[t];
    frames.push(n ? { n, kp: kp.subarray(t * N * K * 2, (t * N + n) * K * 2), sc: sc.subarray(t * N * K, (t * N + n) * K),
      box: box.subarray(t * N * 4, (t * N + n) * 4) } : undefined);
  }
  return { frames, W: m.W, H: m.H };
}

let bad = 0;
for (const [name, w] of [['TUNED', TUNED], ['DEFAULT', DEFAULT]]) {
  let frames = 0, mism = 0; const clipsBad = [];
  for (const [c, r] of Object.entries(ref.clips)) {
    const { frames: fr, W, H } = loadClip(dir, c);
    const got = choose(fr, W, H, w), exp = r[name];
    if (got.length !== exp.length) throw new Error(`${c}: length ${got.length} vs ${exp.length}`);
    let m = 0;
    for (let t = 0; t < exp.length; t++) if (got[t] !== exp[t]) m++;
    frames += exp.length; mism += m; if (m) clipsBad.push(`${c}(${m})`);
  }
  console.log(`${name}: clips ${Object.keys(ref.clips).length}, frames compared ${frames}, mismatches ${mism}` +
    (clipsBad.length ? ` in ${clipsBad.join(' ')}` : ''));
  bad += mism;
}
process.exit(bad ? 1 : 0);
