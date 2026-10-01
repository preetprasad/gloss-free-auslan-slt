// Plain logic shared by worker.mjs and the Node self-test (no browser APIs): payload parsing (layout in serve.py),
// RGB -> RGBA frames as rgbFeatures expects them, and the per-image comparison of two feature arrays.

export function parseClip(buf) {
  const dv = new DataView(buf);
  if (dv.getUint32(0, false) !== 0x44474631) throw new Error('bad magic (want DGF1)');
  const [n, W, H, T, nVideo, nKps] = Array.from({ length: 6 }, (_, i) => dv.getUint32(4 + 4 * i, true));
  let o = 28;
  const idx = Array.from(new Int32Array(buf, o, n)); o += 4 * n;                 // little-endian host (all targets)
  const kps = new Float32Array(buf, o, n * 399); o += 4 * n * 399;
  const rgb = new Uint8Array(buf, o, n * H * W * 3); o += n * H * W * 3;
  if (o !== buf.byteLength) throw new Error(`payload is ${buf.byteLength} bytes, layout says ${o}`);
  return { n, W, H, T, nVideo, nKps, idx, kps, rgb };
}

// RGBA ImageData-likes (alpha 255), as the G5 test builds them; rgb.js rgbFromRGBA drops alpha again.
export function toFrames({ n, W, H, rgb }) {
  return Array.from({ length: n }, (_, f) => {
    const data = new Uint8ClampedArray(H * W * 4), src = rgb.subarray(f * H * W * 3, (f + 1) * H * W * 3);
    for (let p = 0; p < H * W; p++) { data[4 * p] = src[3 * p]; data[4 * p + 1] = src[3 * p + 1]; data[4 * p + 2] = src[3 * p + 2]; data[4 * p + 3] = 255; }
    return { data, width: W, height: H };
  });
}

// Per 384-d image vector: cosine (null when either side is all zero), max |a - b|, zero flags. Float64 accumulation.
export function compare(a, b, D = 384) {
  if (a.length !== b.length || a.length % D) throw new Error(`compare: lengths ${a.length} / ${b.length}`);
  const rows = [];
  for (let i = 0; i < a.length / D; i++) {
    let dot = 0, aa = 0, bb = 0, mx = 0;
    for (let d = i * D; d < (i + 1) * D; d++) { dot += a[d] * b[d]; aa += a[d] * a[d]; bb += b[d] * b[d]; mx = Math.max(mx, Math.abs(a[d] - b[d])); }
    rows.push({ cos: aa && bb ? dot / Math.sqrt(aa * bb) : null, max_abs: mx, zero_a: aa === 0, zero_b: bb === 0 });
  }
  const valid = rows.filter((r) => r.cos !== null);
  return { rows, min_cos: valid.length ? Math.min(...valid.map((r) => r.cos)) : null,
    max_abs: Math.max(0, ...rows.map((r) => r.max_abs)), zero_mismatch: rows.filter((r) => r.zero_a !== r.zero_b).length };
}
