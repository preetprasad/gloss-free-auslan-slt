// DWPose as the INSTALLED rtmlib 0.0.16 runs it for Wholebody(mode='balanced') — YOLOX-m 640x640 detector
// (NMS inside the ONNX graph) + RTMW-DW-x-l 256x192 top-down SimCC — plus demo/service.py ModelService.extract:
// no detection -> zero frame without running pose; otherwise follow one signer with the tracker.
// Numerics follow numpy: float32 where rtmlib's arrays are float32 (Math.fround), float64 elsewhere.
// ONNX Runtime is injected (onnxruntime-node in tests, onnxruntime-web in the browser).

import { resize, warpAffine, getAffineTransform } from './cv.js';
import { torso, trackIndex } from './track.js';

const f = Math.fround;
export const K = 133;
export const DET_SIZE = [640, 640];          // YOLOX model_input_size (rows, cols)
export const POSE_SIZE = [192, 256];         // RTMPose model_input_size (w, h)
export const MEAN = [123.675, 116.28, 103.53];  // applied to the BGR image's channels 0, 1, 2 as rtmlib does
export const STD = [58.395, 57.12, 57.375];
export const DET_SCORE = 0.3;                // yolox.py, NMS-in-graph branch: final_scores > 0.3
export const PADDING = 1.25;                 // rtmpose.py: bbox_xyxy2cs(bbox, padding=1.25)
export const SIMCC_SPLIT = 2.0;
const PAD_VALUE = 114;

export async function createPose({ ort, detector, pose, sessionOptions = {} }) {
  const [det, est] = await Promise.all([
    ort.InferenceSession.create(detector, sessionOptions),
    ort.InferenceSession.create(pose, sessionOptions),
  ]);
  return new DWPose(ort, det, est);
}

export class DWPose {
  constructor(ort, det, est) { Object.assign(this, { ort, det, est }); }

  /** YOLOX.preprocess: letterbox into a 640x640 canvas of 114 (top-left aligned). img: BGR. */
  letterbox(img) {
    const [H, W] = DET_SIZE;
    if (img.height === H && img.width === W) return { padded: new Uint8Array(img.data), ratio: 1.0 };
    const ratio = Math.min(H / img.height, W / img.width);
    const w = Math.trunc(img.width * ratio), h = Math.trunc(img.height * ratio);
    const small = resize(img, w, h);             // cv2.resize(..., interpolation=cv2.INTER_LINEAR)
    const padded = new Uint8Array(H * W * 3).fill(PAD_VALUE);
    for (let y = 0; y < h; y++) padded.set(small.data.subarray(y * w * 3, (y + 1) * w * 3), y * W * 3);
    return { padded, ratio };
  }

  /** YOLOX.__call__ -> boxes Float32Array(n*4) xyxy in image pixels (float32, as rtmlib returns them). */
  async detect(img) {
    const { padded, ratio } = this.letterbox(img);
    const [H, W] = DET_SIZE;
    const chw = new Float32Array(3 * H * W);
    for (let i = 0, p = 0; i < H * W; i++, p += 3) {
      chw[i] = padded[p]; chw[H * W + i] = padded[p + 1]; chw[2 * H * W + i] = padded[p + 2];
    }
    const out = await this.det.run({ [this.det.inputNames[0]]: new this.ort.Tensor('float32', chw, [1, 3, H, W]) });
    const dets = out[this.det.outputNames[0]];
    return { boxes: this.decodeDets(dets.data, dets.dims, ratio), dets };
  }

  /** YOLOX.postprocess, branch `outputs.shape[-1] == 5` (this ONNX carries NMS): boxes /= ratio; keep score > 0.3. */
  decodeDets(data, dims, ratio) {
    if (dims.length !== 3 || dims[2] !== 5) throw new Error(`YOLOX output ${dims}: only the NMS-in-graph export is supported`);
    const r = f(ratio), thr = f(DET_SCORE), keep = [];
    for (let i = 0; i < dims[1]; i++) {
      if (!(data[i * 5 + 4] > thr)) continue;
      for (let k = 0; k < 4; k++) keep.push(f(data[i * 5 + k] / r));
    }
    return Float32Array.from(keep);
  }

  /** RTMPose.preprocess for one xyxy float32 box: center/scale, aspect fix, warp matrix, warpAffine, mean/std. */
  topDown(img, box) {
    const [x1, y1, x2, y2] = box;
    const cx = f(f(x1 + x2) * 0.5), cy = f(f(y1 + y2) * 0.5);
    const bw = f(f(x2 - x1) * PADDING), bh = f(f(y2 - y1) * PADDING);
    const [w, h] = POSE_SIZE, aspect = w / h;
    const scale = bw > f(bh * aspect) ? [bw, f(bw / aspect)] : [f(bh * aspect), bh];
    // get_warp_matrix(center, scale, rot=0, output_size=(w, h)); src/dst are float32 arrays in rtmlib.
    const s1y = f(cy + -0.5 * scale[0]);
    const d1 = f(cy - s1y);
    const src = Float32Array.of(cx, cy, cx, s1y, f(cx - d1), s1y);
    const dst = Float32Array.of(w * 0.5, h * 0.5, w * 0.5, h * 0.5 - w * 0.5, 0, h * 0.5 - w * 0.5);
    const M = getAffineTransform(src, dst);
    const crop = warpAffine(img, M, w, h);
    const input = new Float32Array(3 * h * w);
    for (let i = 0, p = 0; i < h * w; i++, p += 3) {
      for (let c = 0; c < 3; c++) input[c * h * w + i] = f((crop.data[p + c] - MEAN[c]) / STD[c]);
    }
    return { center: [cx, cy], scale, src, M, crop, input };
  }

  /** get_simcc_maximum + RTMPose.postprocess for one person -> kp (K*2), sc (K). */
  decodeSimcc(sx, sy, Wx, Wy, center, scale, kp, sc, o = 0) {
    const [w, h] = POSE_SIZE;
    for (let k = 0; k < K; k++) {
      let ix = 0, mx = sx[k * Wx], iy = 0, my = sy[k * Wy];
      for (let j = 1; j < Wx; j++) if (sx[k * Wx + j] > mx) { mx = sx[k * Wx + j]; ix = j; }
      for (let j = 1; j < Wy; j++) if (sy[k * Wy + j] > my) { my = sy[k * Wy + j]; iy = j; }
      const v = f(0.5 * f(mx + my));
      const lx = v <= 0 ? -1 : ix, ly = v <= 0 ? -1 : iy;
      const kx = f(lx / SIMCC_SPLIT), ky = f(ly / SIMCC_SPLIT);
      kp[(o + k) * 2] = f(kx / w * scale[0] + center[0] - scale[0] / 2);
      kp[(o + k) * 2 + 1] = f(ky / h * scale[1] + center[1] - scale[1] / 2);
      sc[o + k] = v;
    }
  }

  /** RTMPose.__call__(image, bboxes): one person at a time (batch 1, as rtmlib). */
  async estimate(img, boxes) {
    const n = boxes.length / 4, kp = new Float32Array(n * K * 2), sc = new Float32Array(n * K), persons = [];
    const [w, h] = POSE_SIZE;
    for (let i = 0; i < n; i++) {
      const td = this.topDown(img, boxes.subarray(i * 4, i * 4 + 4));
      const out = await this.est.run({ [this.est.inputNames[0]]: new this.ort.Tensor('float32', td.input, [1, 3, h, w]) });
      const sx = out[this.est.outputNames[0]], sy = out[this.est.outputNames[1]];
      this.decodeSimcc(sx.data, sy.data, sx.dims[2], sy.dims[2], td.center, td.scale, kp, sc, i * K);
      persons.push({ ...td, simx: sx, simy: sy });
    }
    return { kp, sc, persons };
  }

  /**
   * ModelService.extract: -> { keypoints Float32Array(133*3) (x, y, score), count }. Mutates track {mid, wid}.
   * A run that threw returns the same empty frame plus `error` (the exception), so the engine can tell a broken
   * runtime from "no person detected"; the frame itself is what service.py produces either way.
   */
  async extract(img, track = {}) {
    const empty = () => ({ keypoints: new Float32Array(K * 3), count: 0 });
    let kp, sc, count;
    try {
      const { boxes } = await this.detect(img);
      count = boxes.length / 4;
      if (count === 0) return empty();   // masked, never posed: empty input decodes to 'one two three .'
      ({ kp, sc } = await this.estimate(img, boxes));
    } catch (error) {                    // as in the research workers: a failed frame is an empty frame
      return { ...empty(), error };
    }
    const [i] = trackIndex(kp, sc, count, track.mid ?? null, track.wid ?? null, [img.width, img.height]);
    const t = torso(kp, sc, i);
    if (t.ok) { track.mid = t.mid; track.wid = t.wid; }
    const keypoints = new Float32Array(K * 3);
    for (let k = 0; k < K; k++) {
      keypoints[k * 3] = kp[(i * K + k) * 2]; keypoints[k * 3 + 1] = kp[(i * K + k) * 2 + 1]; keypoints[k * 3 + 2] = sc[i * K + k];
    }
    // `all`: every detection's keypoints (persons.js follows them), `picked`: the detection the tracker above chose.
    return { keypoints, count, picked: i, all: { kp, sc, n: count, width: img.width } };
  }
}
