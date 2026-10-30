// Model and runtime downloads for the in-browser engine: streamed with progress, SHA-256 verified (crypto.subtle)
// before anything is used, and kept in Cache Storage under the sha256 so a second visit needs no network.
// fetch and caches are injectable (demo/tests/engine_loader.test.mjs).
import { CACHE_NAME, HF_BASE, MODEL_BASE, RUNTIME } from './config.js';

export async function sha256hex(bytes) {
  const d = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(d, (b) => b.toString(16).padStart(2, '0')).join('');
}

const cacheKey = (sha) => `https://model-cache.invalid/sha256/${sha}`;

// Body with progress; written into one buffer of the expected size when it is known (the models are up to 229 MB).
async function readAll(res, total, onProgress) {
  if (!res.body) {
    const b = new Uint8Array(await res.arrayBuffer());
    onProgress(b.length, total || b.length);
    return b;
  }
  const reader = res.body.getReader();
  let out = total ? new Uint8Array(total) : null, chunks = [], n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (out && n + value.length <= out.length) out.set(value, n);
    else {                                   // size unknown or wrong: fall back to chunks (the hash still decides)
      if (out) { chunks = [out.subarray(0, n)]; out = null; }
      chunks.push(value);
    }
    n += value.length;
    onProgress(n, Math.max(total, n));
  }
  if (out) return n === out.length ? out : out.subarray(0, n);
  const all = new Uint8Array(n);
  let o = 0;
  for (const c of chunks) { all.set(c, o); o += c.length; }
  return all;
}

// Models may only come from this site (local dev) or a commit-pinned Hugging Face revision, never a moving branch.
const PINNED_HF = /^https:\/\/huggingface\.co\/[^/]+\/[^/]+\/resolve\/[0-9a-f]{40}\/$/;
export function checkModelBase(base, origin = globalThis.location?.origin) {
  const sameOrigin = origin ? new URL(base, origin).origin === origin : /^\/(?!\/)/.test(base);
  if (!sameOrigin && !PINNED_HF.test(base)) {
    throw new Error(`MODEL_BASE ${base} is neither this site nor a commit-pinned Hugging Face revision; no model is loaded from it.`);
  }
}

export function createLoader({ fetch: get = (...a) => globalThis.fetch(...a), caches: store = globalThis.caches, cacheName = CACHE_NAME } = {}) {
  const open = async () => { try { return store ? await store.open(cacheName) : null; } catch { return null; } };
  const cached = async (sha) => (await open())?.match(cacheKey(sha));
  let closed = false;                        // after clear(): nothing is written to Cache Storage again
  const writes = new Set();

  return {
    /** Delete cached files whose sha256 is not in `keep`, so a model update does not leave the old bytes behind. */
    async prune(keep) {
      const c = await open();
      for (const req of (await c?.keys()) ?? []) if (!keep.has(req.url.split('/').pop())) await c.delete(req);
    },

    /** Remove every cached file and cache nothing more (an in-flight download is still verified and used, not kept). */
    async clear() {
      closed = true;
      await Promise.all(writes);             // a write that started before clear() is deleted, not left behind
      await this.prune(new Set());
    },

    /** True when the file is already in Cache Storage (it is re-verified when read). */
    async has(sha256) { return !!(await cached(sha256)); },

    /** Size for display: the manifest's `bytes`, else the server's Content-Length (HEAD, redirects followed). */
    async size(file) {
      if (file.bytes) return file.bytes;
      try { const r = await get(file.url, { method: 'HEAD' }); return Number(r.headers.get('content-length')) || 0; } catch { return 0; }
    },

    /**
     * Verified bytes of one file. Never returns bytes whose SHA-256 differs from `sha256`. onProgress(loaded, total)
     * reports network bytes only: a verified cache hit reports nothing.
     */
    async bytes(file, onProgress = () => {}) {
      const { url, sha256, bytes: size = 0 } = file;
      const hit = await cached(sha256);
      if (hit) {
        const b = new Uint8Array(await hit.arrayBuffer());
        if ((await sha256hex(b)) === sha256) return b;
        await (await open())?.delete(cacheKey(sha256));        // corrupted entry: drop it and download again
      }
      const res = await get(url, { cache: 'no-store' });
      if (!res.ok) throw new Error(`Could not download ${url} (HTTP ${res.status}).`);
      const b = await readAll(res, size || Number(res.headers.get('content-length')) || 0, onProgress);
      if ((await sha256hex(b)) !== sha256) throw new Error(`Checksum mismatch for ${url}; the file was not used.`);
      if (!closed) {
        const w = (async () => { try { await (await open())?.put(cacheKey(sha256), new Response(b)); } catch { /* quota: keep working uncached */ } })();
        writes.add(w); await w; writes.delete(w);
      }
      return b;
    },
  };
}

/**
 * Every file an operating point needs, with its expected SHA-256 and size (manifest `bytes` where present).
 * The pose models and runtimes are shared; callers skip what is already loaded.
 */
export function modelFiles(manifest, model, { base = MODEL_BASE, hfBase = HF_BASE } = {}) {
  const spec = manifest.models[model];
  if (!spec) throw new Error(`unknown model ${model}`);
  const files = [];
  const add = (key, url, sha256, bytes) => files.push({ key, url, sha256, bytes: bytes || 0 });
  add('ort', RUNTIME.ort.url, RUNTIME.ort.sha256, RUNTIME.ort.bytes);
  add('ortWasm', RUNTIME.ortWasm.url, RUNTIME.ortWasm.sha256, RUNTIME.ortWasm.bytes);
  for (const k of ['detector', 'pose']) {
    const p = manifest.pose[k];
    add(`pose.${k}`, base + p.file, p.sha256, p.bytes);
  }
  for (const m of spec.members) {
    for (const k of ['encoder', 'decoder', 'vocab']) add(m[k], base + m[k], m.sha256[k], m.bytes?.[k]);
  }
  if (spec.rgb) {
    const d = manifest.dinov2;
    add('dinov2', d.file ? base + d.file : `${hfBase}${d.hf_model}/resolve/${d.revision}/${d.path}`, d.sha256, d.bytes);
  }
  return files;
}
