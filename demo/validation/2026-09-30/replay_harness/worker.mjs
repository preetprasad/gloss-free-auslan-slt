// Scratch replay worker. Imports the SHIPPED engine (mounted next to it at /static/engine/rp/), replays the exact JPEGs +
// captured_at a studio run sent, and records the raw hypothesis (engine.js blanks the text of declined results).
import { browserRuntime } from '../engine.js';   // its own onmessage is replaced below
const post = (m) => postMessage(m);
const upload = async (name, obj) => {
  const r = await fetch(`/__upload/${name}`, { method: 'POST', headers: { 'X-Dev-Upload': '1' }, body: JSON.stringify(obj) });
  if (!r.ok) throw new Error(`upload ${name}: HTTP ${r.status}`);
};
const PREFIX = 'data:image/jpeg;base64,';
const toBytes = (d) => Uint8Array.from(atob(d.slice(PREFIX.length)), (c) => c.charCodeAt(0));
let rt = null;
async function replay({ model, files }) {
  if (!rt) { rt = browserRuntime(); await rt.init(); }
  const loaded = await rt.load(model, () => {}, (fn) => fn());
  for (const name of files) {
    const cap = await (await fetch(`/rpup/${name}`)).json();
    if (cap.model !== model) throw new Error(`${name} is a ${cap.model} capture`);
    const track = {}, kps = [], times = [], jpegs = [], counts = [];
    for (const fr of cap.frames) {
      if (!fr.data.startsWith(PREFIX)) throw new Error('not a JPEG data URL');
      const jpeg = toBytes(fr.data);
      const { keypoints, count } = await rt.extract(jpeg, track);
      kps.push(keypoints); times.push(fr.captured_at); jpegs.push(jpeg); counts.push(count);
    }
    let out;
    try { out = await rt.translate(model, kps, times, jpegs); } catch (e) { out = { error: e.message }; }
    await upload(name.replace('cap_', 'br_'), { clip: cap.clip, model, loaded, ...out, counts, keypoints: kps.map((k) => Array.from(k)) });
    post({ type: 'status', name, text: out.text, confidence: out.confidence, error: out.error });
  }
  post({ type: 'done', model });
}
onmessage = async (e) => { try { await replay(e.data); } catch (err) { post({ type: 'error', error: String(err?.stack ?? err) }); } };
