// The native cv2 image kernels the research pipeline calls, ported to plain JavaScript and bit-identical to the cv2
// 4.13 the Python service runs (pyenv, arm64): resize (rtmlib's YOLOX letterbox, the 224 x 224 crops of
// demo/rgb_features.py), warpAffine and getAffineTransform (rtmlib's top-down crop), and the RGBA -> BGR / RGB
// channel orders. No OpenCV.js: its WASM build rounds differently from native cv2 in warpAffine and
// getAffineTransform (no FMA), and its embind glue needs 'unsafe-eval'.
// Images are plain { data: Uint8Array (HWC, row-major), width, height, channels }.

// cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR) on uint8 3-channel images, as native cv2 4.13 runs it
// on arm64. Both custom HALs in that build decline 3-channel images (carotene's resizeLinearOpenCV takes 1 or 4
// channels, KleidiCV 0.7.0 only 1), so this is OpenCV's own code, modules/imgproc/src/resize.cpp: cv::resize (same size
// -> copy) -> hal::resize -> resizeGeneric_ with 11-bit fixed-point weights: horizontal pass
// S[sx] * a0 + S[sx + cn] * a1 (S[sx] * 2048 past the right edge), vertical pass
// ((b0 * (R0 >> 4)) >> 16) + ((b1 * (R1 >> 4)) >> 16) + 2) >> 2, the formula of both the NEON path
// (VResizeLinearVec_32s8u) and its scalar tail. The weights: fx = (float)((dx + 0.5) * scale - 0.5), which the
// arm64 cv2.abi3.so computes as one FMA (fadd; fmadd with -0.5; fcvt; read in the cv::hal::resize disassembly),
// floor, the fraction, then saturate_cast<short>(w * 2048) (frintx: ties to even). x is clamped at the edges, y is
// not (its rows are clipped instead). An exact 2x downscale in both axes becomes INTER_AREA inside cv2 (resizeAreaFast_:
// (a + b + c + d + 2) >> 2); there every weight is 1024, so the formula above gives exactly that and needs no branch.
// For every size the engine meets (source <= 1280 x 960, target <= 640) the FMA and
// an unfused multiply-add give the same effective taps (exhaustive scan); it is kept because native cv2 fuses.
// Parity: engine_pose.test.mjs "resize fuzz" (2,549 size pairs against native
// cv2, demo/tests/fixtures/make_resize_fixtures.py) plus the kernel set, the G4 letterboxes and the G5 crops.
const COEF = 2048;                               // INTER_RESIZE_COEF_SCALE

export function resize(img, width, height) {
  const { data: S, width: W, height: H } = img, cn = img.channels ?? 3;
  if (cn !== 3) throw new Error(`resize: 3-channel images only (native cv2 hands ${cn}-channel images to a HAL)`);
  if (!(W >= 1 && H >= 1 && width >= 1 && height >= 1)) throw new Error(`resize: empty size ${W}x${H} -> ${width}x${height} (cv2 asserts too)`);
  const out = new Uint8Array(width * height * cn), row = W * cn, dw = width * cn;
  if (width === W && height === H) { out.set(S.subarray(0, out.length)); return { data: out, width, height, channels: cn }; }
  const scaleX = 1 / (width / W), scaleY = 1 / (height / H);
  const weights = (d, scale) => {               // [floor, fraction] for one output column or row
    const f = Math.fround(fma(d + 0.5, scale, -0.5)), s = Math.floor(f), t = Math.fround(f - s);
    return [s, t];
  };
  const q = (w) => sat16(roundEven(w * COEF));
  const xofs = new Int32Array(width), a0 = new Int32Array(width), a1 = new Int32Array(width);
  let xmax = width;
  for (let dx = 0; dx < width; dx++) {
    let [s, f] = weights(dx, scaleX);
    if (s < 0) { s = 0; f = 0; }
    if (s + 1 >= W) { xmax = Math.min(xmax, dx); if (s >= W - 1) { s = W - 1; f = 0; } }
    xofs[dx] = s * cn; a0[dx] = q(Math.fround(1 - f)); a1[dx] = q(f);
  }
  const hresize = (y, D) => {
    const o = y * row;
    for (let dx = 0, x = 0; dx < width; dx++) {
      const p = o + xofs[dx];
      if (dx < xmax) for (let c = 0; c < cn; c++, x++) D[x] = S[p + c] * a0[dx] + S[p + c + cn] * a1[dx];
      else for (let c = 0; c < cn; c++, x++) D[x] = S[p + c] * COEF;
    }
  };
  const R0 = new Int32Array(dw), R1 = new Int32Array(dw), clip = (y) => (y < 0 ? 0 : y >= H ? H - 1 : y);
  for (let dy = 0, o = 0; dy < height; dy++) {
    const [s, f] = weights(dy, scaleY), b0 = q(Math.fround(1 - f)), b1 = q(f);
    hresize(clip(s), R0); hresize(clip(s + 1), R1);   // native reuses rows between output rows: same values
    for (let x = 0; x < dw; x++, o++) out[o] = (((b0 * (R0[x] >> 4)) >> 16) + ((b1 * (R1[x] >> 4)) >> 16) + 2) >> 2;
  }
  return { data: out, width, height, channels: cn };
}

// cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR) with the default BORDER_CONSTANT / Scalar() — the one
// call rtmlib makes (top_down_affine) — ported from OpenCV 4.13 modules/imgproc/src/imgwarp.cpp (cv::warpAffine ->
// hal::warpAffine -> WarpAffineInvoker -> remapBilinear<FixedPtCast<int, uchar, 15>>). OpenCV.js computes the inverse
// map without the FMAs native cv2 uses (as in getAffineTransform below), so its crops differ; this port does not.
// Every output pixel: inverse map in 1/1024 px, rounded to 1/32 px, Q15 bilinear weights from a 32x32 table,
// neighbours outside the image read as 0, (sum + 2^14) >> 15. The float steps (the FMAs in the inverse and in X0/Y0,
// cvRound) were read in the arm64 cv2.abi3.so 4.13 disassembly (cv::warpAffine, cv::hal::warpAffine,
// WarpAffineInvoker); this build has no warpAffine HAL override. Parity: engine_pose.test.mjs "warpAffine fuzz"
// (3,024 real-frame boxes + 409 degenerate boxes through rtmlib, plus 10 extreme matrices: NaN, Inf, overflow).
// M: 6 doubles, row-major 2x3 (cv2's float64 matrix). img: { data: Uint8Array HWC, width, height, channels }.
const INTER_BITS = 5, TAB = 1 << INTER_BITS, AB_BITS = 10, AB_SCALE = 1 << AB_BITS, COEF_BITS = 15;

// BilinearTab_i (initInterTab2D, INTER_LINEAR, fixpt), a literal port: entry 0's weight 32768 saturates to 32767 and
// the sum fix-up then adds 1 to its [1][1] weight, reading the (still zero) next entry as it searches — hence the pad.
const BILINEAR = (() => {
  const tab = new Float32Array(2 * TAB), itab = new Int16Array(TAB * TAB * 4 + 4), scale = Math.fround(1 / TAB);
  for (let i = 0; i < TAB; i++) { const x = Math.fround(i * scale); tab[2 * i] = Math.fround(1 - x); tab[2 * i + 1] = x; }
  for (let i = 0, e = 0; i < TAB; i++) {
    for (let j = 0; j < TAB; j++, e += 4) {
      let isum = 0;
      for (let k1 = 0; k1 < 2; k1++) {
        for (let k2 = 0; k2 < 2; k2++) {
          const v = Math.fround(Math.fround(tab[2 * i + k1] * tab[2 * j + k2]) * 32768);
          isum += itab[e + k1 * 2 + k2] = Math.max(-32768, Math.min(32767, roundEven(v)));
        }
      }
      if (isum !== 32768) {
        const diff = isum - 32768;
        let Mk = 3, mk = 3;
        for (let k1 = 1; k1 < 3; k1++) {
          for (let k2 = 1; k2 < 3; k2++) {
            const k = k1 * 2 + k2;
            if (itab[e + k] < itab[e + mk]) mk = k; else if (itab[e + k] > itab[e + Mk]) Mk = k;
          }
        }
        if (diff < 0) itab[e + Mk] -= diff; else itab[e + mk] -= diff;
      }
    }
  }
  return itab.subarray(0, TAB * TAB * 4);
})();

function roundEven(v) {                   // nearest integer, ties to even (lrint in the default rounding mode)
  let r = Math.round(v);
  if (r - v === 0.5 && r % 2) r -= 1;
  return r;
}

// cvRound(double) = (int)lrint(v), which the arm64 cv2 4.13 build compiles to frintx (ties to even), fcvtzs x
// (saturating to int64, NaN -> 0) and a 32-bit store (the low 32 bits). Read in the cv2.abi3.so disassembly; the
// saturating and NaN cases only arise for matrices no detector box produces, and still match native (see the note
// at warpAffine).
function cvRound(v) {
  if (v !== v) return 0;
  const r = roundEven(v);
  if (r >= 2 ** 63) return -1;            // INT64_MAX: low 32 bits all ones
  if (r <= -(2 ** 63)) return 0;          // INT64_MIN: low 32 bits zero
  return Math.abs(r) < 2 ** 31 ? r : Number(BigInt.asIntN(32, BigInt(r)));
}

const sat16 = (v) => (v < -32768 ? -32768 : v > 32767 ? 32767 : v);

export function warpAffine(img, M, width, height) {
  const { data: S, width: W, height: H } = img, cn = img.channels ?? 3, row = W * cn;
  // cv::warpAffine inverts M itself (no WARP_INVERSE_MAP); Apple clang contracts a*b - c*d into fma(a, b, -(c*d)).
  let [m0, m1, m2, m3, m4, m5] = M;
  let D = fma(m0, m4, -(m1 * m3));
  D = D !== 0 ? 1 / D : 0;
  const a11 = m4 * D, a22 = m0 * D;
  m0 = a11; m1 *= -D; m3 *= -D; m4 = a22;
  const b1 = fma(-m0, m2, -(m1 * m5)), b2 = fma(-m3, m2, -(m4 * m5));
  m2 = b1; m5 = b2;
  const adelta = new Int32Array(width), bdelta = new Int32Array(width);
  for (let x = 0; x < width; x++) { adelta[x] = cvRound(m0 * x * AB_SCALE); bdelta[x] = cvRound(m3 * x * AB_SCALE); }
  const out = new Uint8Array(width * height * cn), half = AB_SCALE / TAB / 2;
  for (let y = 0, o = 0; y < height; y++) {
    const X0 = (cvRound(fma(m1, y, m2) * AB_SCALE) + half) | 0;   // (M[1]*y + M[2])*AB_SCALE: fused multiply-add
    const Y0 = (cvRound(fma(m4, y, m5) * AB_SCALE) + half) | 0;
    for (let x = 0; x < width; x++, o += cn) {
      const X = ((X0 + adelta[x]) | 0) >> (AB_BITS - INTER_BITS), Y = ((Y0 + bdelta[x]) | 0) >> (AB_BITS - INTER_BITS);
      const sx = sat16(X >> INTER_BITS), sy = sat16(Y >> INTER_BITS), wi = ((Y & (TAB - 1)) * TAB + (X & (TAB - 1))) * 4;
      if (sx >= W || sx + 1 < 0 || sy >= H || sy + 1 < 0) continue;   // all four neighbours outside: border value 0
      const x0 = sx >= 0, x1 = sx + 1 < W, y0 = sy >= 0, y1 = sy + 1 < H, p = sy * row + sx * cn;
      const w0 = y0 && x0 ? BILINEAR[wi] : 0, w1 = y0 && x1 ? BILINEAR[wi + 1] : 0;
      const w2 = y1 && x0 ? BILINEAR[wi + 2] : 0, w3 = y1 && x1 ? BILINEAR[wi + 3] : 0;
      for (let c = 0; c < cn; c++) {
        const q = p + c;
        const t = (w0 && S[q] * w0) + (w1 && S[q + cn] * w1) + (w2 && S[q + row] * w2) + (w3 && S[q + row + cn] * w3);
        out[o + c] = (t + (1 << (COEF_BITS - 1))) >> COEF_BITS;
      }
    }
  }
  return { data: out, width, height, channels: cn };
}

// Exact fused multiply-add a*b + c (one rounding), for finite doubles with normal results. JS has no Math.fma.
const F64 = new Float64Array(1), U64 = new BigUint64Array(F64.buffer);
function parts(x) {                        // x = m * 2^e with m an integer BigInt
  F64[0] = x;
  const bits = U64[0], ex = Number((bits >> 52n) & 0x7ffn), frac = bits & 0xfffffffffffffn;
  const m = ex ? frac | 0x10000000000000n : frac;
  return [bits >> 63n ? -m : m, (ex || 1) - 1075];
}
export function fma(a, b, c) {
  if (!(Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(c))) return a * b + c;   // inf/NaN: no rounding
  const [ma, ea] = parts(a), [mb, eb] = parts(b), [mc, ec] = parts(c);
  const mp = ma * mb, ep = ea + eb;
  if (mp === 0n || mc === 0n) return a * b + c;   // one term is exactly zero: a single rounding already
  const e = Math.min(ep, ec);
  let n = (mp << BigInt(ep - e)) + (mc << BigInt(ec - e));
  if (n === 0n) return 0;
  const neg = n < 0n;
  if (neg) n = -n;
  const shift = Math.max(0, n.toString(2).length - 55);   // keep 53 bits + guard + round, fold the rest into sticky
  if (shift) n = (n >> BigInt(shift)) | ((n & ((1n << BigInt(shift)) - 1n)) ? 1n : 0n);
  const r = Number(n) * 2 ** (e + shift);                  // Number(BigInt) rounds to nearest, ties to even
  return neg ? -r : r;
}

// cv::getAffineTransform = solve(6x6, DECOMP_LU) -> LUImpl (modules/core/src/matrix_decomp.cpp, OpenCV 4.13).
// Native cv2 (Apple clang, default -ffp-contract=on) fuses `A += alpha*B` and `s -= A*b` into FMAs; OpenCV.js
// (WASM, no FMA) does not, so its matrix differs in the last bits and 41 of 120 warped crops then differ by 1.
// This port uses the fused form: bit-identical to native cv2 4.13 on 300 random point sets and 120 real crops
// (engine_pose.test.mjs); OpenCV.js's own solver matched 29/300.
// src, dst: 3 points as 6 float32 values (x0, y0, x1, y1, x2, y2). Returns the 2x3 float64 matrix, row-major.
export function getAffineTransform(src, dst) {
  const a = new Float64Array(36), b = new Float64Array(6), m = 6;
  for (let i = 0; i < 3; i++) {
    const j = i * 12, k = i * 12 + 6;
    a[j] = a[k + 3] = src[2 * i]; a[j + 1] = a[k + 4] = src[2 * i + 1]; a[j + 2] = a[k + 5] = 1;
    b[2 * i] = dst[2 * i]; b[2 * i + 1] = dst[2 * i + 1];
  }
  for (let i = 0; i < m; i++) {
    let k = i;
    for (let j = i + 1; j < m; j++) if (Math.abs(a[j * m + i]) > Math.abs(a[k * m + i])) k = j;
    if (Math.abs(a[k * m + i]) < 2.220446049250313e-16 * 100) return new Float64Array(6);   // solve(): singular -> dst = 0
    if (k !== i) {
      for (let j = i; j < m; j++) [a[i * m + j], a[k * m + j]] = [a[k * m + j], a[i * m + j]];
      [b[i], b[k]] = [b[k], b[i]];
    }
    const d = -1 / a[i * m + i];
    for (let j = i + 1; j < m; j++) {
      const alpha = a[j * m + i] * d;
      for (let kk = i + 1; kk < m; kk++) a[j * m + kk] = fma(alpha, a[i * m + kk], a[j * m + kk]);
      b[j] = fma(alpha, b[i], b[j]);
    }
  }
  for (let i = m - 1; i >= 0; i--) {
    let s = b[i];
    for (let k = i + 1; k < m; k++) s = fma(-a[i * m + k], b[k], s);
    b[i] = s / a[i * m + i];
  }
  return b;
}

// Browser frames arrive as RGBA ImageData; the service decodes JPEG -> RGB -> BGR (demo/service.py decode_frame) and
// rgb_features.py works on RGB. cv2.cvtColor RGBA2BGR / RGBA2RGB on uint8 only reorders and drops channels.
function fromRGBA({ data, width, height }, order) {
  const n = width * height, out = new Uint8Array(3 * n);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) out[3 * i + c] = data[4 * i + order[c]];
  return { data: out, width, height, channels: 3 };
}
export const bgrFromRGBA = (imageData) => fromRGBA(imageData, [2, 1, 0]);
export const rgbFromRGBA = (imageData) => fromRGBA(imageData, [0, 1, 2]);
