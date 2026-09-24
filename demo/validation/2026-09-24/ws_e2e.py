# Protocol-level end-to-end against the running live service: real models, real websockets, browser-like frames.
import asyncio, base64, json, sys, time, secrets
import cv2, websockets
URL, ORIGIN = 'ws://localhost:8010', 'http://localhost:8010'
clip = sys.argv[1]
cap = cv2.VideoCapture(f'./results/dwpose_experiment/rgb_clips/Signer/{clip}_signer.mp4'); frames = []
while (r := cap.read())[0]:
    f = r[1]; h, w = f.shape[:2]; s = min(1, 960 / w, 720 / h)
    f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else f
    frames.append('data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 75])[1]).decode())

async def utter(model):
    sid = 'e2e' + secrets.token_hex(10); rid = 'utt' + secrets.token_hex(10)
    async with websockets.connect(f'{URL}/ws/translate?session_id={sid}', origin=ORIGIN) as tr, \
               websockets.connect(f'{URL}/ws/pose?session_id={sid}', origin=ORIGIN) as po:
        await tr.send(json.dumps({'type': 'start_recording', 'request_id': rid, 'model': model}))
        print(' ', json.loads(await tr.recv()))
        t0 = time.perf_counter(); pose_ms = []
        for i, d in enumerate(frames):
            await po.send(json.dumps({'type': 'frame', 'request_id': rid, 'captured_at': i * 40, 'data': d}))
            ack = json.loads(await po.recv()); pose_ms.append(ack['pose_ms'])
        signed = len(frames) * 40
        drain = (time.perf_counter() - t0) * 1000 - signed
        t1 = time.perf_counter()
        await tr.send(json.dumps({'type': 'translate', 'request_id': rid}))
        res = json.loads(await tr.recv())
        fin = (time.perf_counter() - t1) * 1000
        keep = {k: res.get(k) for k in ('text', 'accepted', 'reason', 'confidence', 'threshold', 'capture_fps', 'frames', 'frames_used', 'translate_ms', 'rgb_ms', 'model_label')}
        pose_ms.sort()
        print(f'  {model}: {keep}\n  pose median {pose_ms[len(pose_ms)//2]:.0f} ms/frame; backlog if streamed at 25 fps ≈ {max(drain,0)/1000:.1f} s; translate request → result {fin:.0f} ms')

async def main():
    print(clip, len(frames), 'frames')
    for m in ('single', 'ensemble'): await utter(m)
asyncio.run(main())
