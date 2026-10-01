// Injected into the studio page (javascript_tool). Stand-in camera = captureStream() of a paused H.264 transcode,
// returned by an overridden navigator.mediaDevices.getUserMedia. The pane is hidden, so requestVideoFrameCallback never
// fires: each source frame is seeked, the studio's <video> is awaited until it shows that frame, then the studio's own
// sendFrame(mediaTime) captures it (canvas -> JPEG q0.75 -> queue -> engine worker). The wall-clock 10 s timer is
// cleared because stepping is slower than real time (captured_at still enforces the 10 s limit).
window.G6S ??= { runs: [] };
G6S.thumb = (el, c) => { const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(el, 0, 0, c.width, c.height); return g.getImageData(0, 0, c.width, c.height).data; };
G6S.mad = (a, b) => { let s = 0; for (let i = 0; i < a.length; i += 4) s += Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]); return s / (a.length / 4 * 3); };
G6S.sleep = (ms) => new Promise((r) => setTimeout(r, ms));
G6S.until = async (cond, ms, what) => { const t0 = performance.now(); while (!cond()) { if (performance.now() - t0 > ms) throw new Error(`timeout: ${what}`); await G6S.sleep(20); } };
if (!G6S.hooked) {
  const orig = Worker.prototype.postMessage;
  Worker.prototype.postMessage = function (m, ...rest) {
    if (m && m.type === 'frame' && G6S.cur) G6S.cur.frames_sent.push({ captured_at: m.captured_at, data: m.data });
    return orig.call(this, m, ...rest);
  };
  G6S.hooked = true;
}
G6S.run = async function (clip, model, nFrames, fps = 25) {
  const $ = (id) => document.getElementById(id);
  const R = G6S.cur = { clip, model, frames_sent: [], stale: 0, sync_mad: [], phase: 'setup' };
  let src;
  try {
    if (state.stream) $('cameraOffButton').click();
    $(model === 'single' ? 'modelSingle' : 'modelEnsemble').click();
    src = document.createElement('video');
    Object.assign(src, { muted: true, playsInline: true, preload: 'auto', src: (G6S.srcBase ?? '/examples/') + `${clip}` + (G6S.srcSuffix ?? '/clip.mp4') });
    await new Promise((res, rej) => { src.onloadeddata = res; src.onerror = () => rej(new Error('source video failed to load')); });
    const seek = (t) => new Promise((res) => { src.onseeked = res; src.currentTime = t; });
    await seek(0.5 / fps);
    const stream = src.captureStream();
    navigator.mediaDevices.getUserMedia = async () => stream;
    R.phase = 'camera';
    $('enableCameraButton').click();
    await G6S.until(() => state.stream && state.phase === 'ready', 20000, 'camera ready');
    // The ensemble is downloaded only on request (the studio's Download button); time it from the click.
    if (!state.loaded[model]) {
      const t0 = performance.now();
      if (state.download?.model !== model) {   // the single model loads by itself at boot
        await G6S.until(() => !$('downloadButton').hidden && !$('downloadButton').disabled, 120000, 'download button');
        $('downloadButton').click();
      }
      await G6S.until(() => state.loaded[model] || state.unavailable || state.loadError?.[model], 600000, 'model loaded');
      if (!state.loaded[model]) throw new Error(`load failed: ${state.unavailable || JSON.stringify(state.loadError)}`);
      R.download_ms = performance.now() - t0;
    }
    await G6S.until(() => state.connected, 600000, 'model loaded');
    R.camera = $('cameraDetail').textContent; R.canvas = [canvas.width, canvas.height]; R.providers = { ...state.providers };
    // The stand-in camera delivers frames ONLY through sendFrame below. On the hidden pane the studio's own
    // requestVideoFrameCallback still fires sporadically, with captureTime on the performance.now() clock; one such
    // frame set t0 and every stepped frame after it came out negative and was skipped (runs with 1 captured frame).
    video.requestVideoFrameCallback = () => 0; video.cancelVideoFrameCallback = () => {};
    R.phase = 'start';
    $('recordButton').click();
    await G6S.until(() => state.phase === 'recording', 20000, 'recording');
    clearInterval(state.clockTimer);
    const A = document.createElement('canvas'), B = document.createElement('canvas');
    A.width = B.width = 48; A.height = B.height = 48;
    R.phase = 'capture';
    // Sync: the studio <video> must show source frame i (MAD < 3 on 48x48) AND be closer to frame i than to frame i-1
    // when those two differ (a MAD < 3 test alone accepted the previous frame). Frame identity is re-checked offline
    // against the cv2-decoded source (frame_identity.py): on the hidden pane both elements sometimes stop repainting.
    let prev = null;
    for (let i = 0; i < nFrames; i++) {
      await seek((i + 0.5) / fps);
      const want = G6S.thumb(src, A), step = prev ? G6S.mad(want, prev) : Infinity;
      let d, dp, ok, t0 = performance.now();
      for (;;) {
        const v = G6S.thumb(video, B);
        d = G6S.mad(v, want); dp = prev ? G6S.mad(v, prev) : Infinity;
        ok = d < 3 && (step < 1 || d < dp);
        if (ok || performance.now() - t0 > 1500) break;
        await G6S.sleep(10);
      }
      if (!ok) R.stale++;
      R.sync_mad.push(d); (R.src_step ??= []).push(step);
      prev = want;
      sendFrame(i / fps);
    }
    R.captured = state.captured; R.skipped = state.skipped;
    R.phase = 'drain';
    await G6S.until(() => !state.pendingFrame && !state.queue.length && state.frames === state.captured, 600000, 'pose drain');
    R.phase = 'finish';
    $('recordButton').click();
    await G6S.until(() => state.phase === 'result' || state.phase === 'ready' || state.phase === 'off', 600000, 'result');
    Object.assign(R, { result: state.result, poseMs: [...state.poseMs], queueMs: [...state.queueMs], finishMs: state.finishMs,
      message: $('captureMessage').textContent, shownText: $('resultContent').hidden ? '' : $('resultText').textContent });
    const hex = async (s) => [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s)))].map((b) => b.toString(16).padStart(2, '0')).join('');
    R.frame_sha = []; for (const f of R.frames_sent) R.frame_sha.push((await hex(f.data)).slice(0, 16));
    R.frames_digest = await hex(R.frame_sha.join(''));
    R.captured_at = R.frames_sent.map((f) => f.captured_at);
    if (G6S.upload) {
      const up = (name, obj) => fetch(`/__upload/${name}`, { method: 'POST', headers: { 'X-Dev-Upload': '1' }, body: JSON.stringify(obj) });
      await up(`cap_${clip}_${model}.json`, { clip, model, frames: R.frames_sent });
      const { frames_sent, ...rest } = R;
      await up(`st_${clip}_${model}.json`, { ...rest, frames_sent: frames_sent.map((f) => f.captured_at) });
    }
    R.frames_sent = R.frames_sent.map((f) => ({ captured_at: f.captured_at }));   // drop JPEG data from memory
    R.phase = 'done';
  } catch (e) { R.phase = 'error'; R.error = String(e.stack || e); }
  if (src) { src.pause(); src.removeAttribute('src'); src.load(); }   // release the decoder
  G6S.runs.push({ clip, model, phase: R.phase, error: R.error, stale: R.stale, skipped: R.skipped, n: R.frames_sent.length, result: R.result, shown: R.shownText, message: R.message, digest: R.frames_digest, frame_sha: R.frame_sha, captured_at: R.captured_at, finishMs: R.finishMs, poseMs: R.poseMs, download_ms: R.download_ms, providers: R.providers, canvas: R.canvas, sync_mad_max: Math.max(...(R.sync_mad ?? [0])) });
  return R.phase;
};
// Sequential runs; poll G6S.runs / G6S.cur.phase (javascript_tool calls time out at 45 s).
G6S.batch = async (jobs) => { G6S.batchDone = false; for (const [clip, model, n] of jobs) await G6S.run(clip, model, n); G6S.batchDone = true; };
'installed';
