"""Auslan clip in, spoken English out -- the demonstrator, end to end, on one clip.

This is the Phase-3 deliverable in its simplest runnable form: pose -> English -> speech, with the
model's own confidence deciding whether to voice the sentence or ask the signer to sign again.

    python tools/speak_clip.py video_48_27                 # from the test split, speaks aloud
    python tools/speak_clip.py video_48_27 --threshold 0.7 # gate on confidence
    python tools/speak_clip.py --clip_path /path/to.npz    # any DWPose (T,133,3) npz

Contract, per the 2026-09-09 interface brief: the translation is UTTERANCE-FINAL. The clip is
consumed whole and the sentence is produced whole. This is not simultaneous interpreting and must
not be described as such.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
from config import Config
from slt.data import normalize_clip
from slt.reference_model import build_model
from slt.speech import Speaker
sys.path.insert(0, os.path.join(HERE, "tools"))
from speech_pipeline_bench import translate, POSE_FPS_CPU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", help="clip name in the official test split, e.g. video_48_27")
    ap.add_argument("--clip_path", help="path to any DWPose npz instead")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "results/ckpts/comm_dwpose_vlp_ctc.pt"))
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--length_penalty", type=float, default=0.6)
    ap.add_argument("--threshold", type=float, default=None,
                    help="confidence gate in nats; above this the system declines rather than "
                         "voicing a guess. Omit to always speak.")
    ap.add_argument("--no_play", action="store_true", help="synthesise but do not play")
    a = ap.parse_args()

    path = a.clip_path or os.path.join(ROOT, "results/test_clips_dwpose", f"{a.clip}.npz")
    if not os.path.exists(path):
        sys.exit(f"no such clip: {path}")
    ref = None
    man = os.path.join(ROOT, "results/TEST_manifest_official.tsv")
    if os.path.exists(man):
        for line in open(man):
            if "\t" in line and os.path.basename(line.split("\t")[0]) == os.path.basename(path):
                ref = line.split("\t")[1].strip()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config().to_dict()})
    d = np.load(path)
    feats, mask = normalize_clip(d["KPS"], d["BBOX"], cfg)
    model = build_model(feats.shape[1], len(ck["vocab"]), cfg)
    model.load_state_dict(ck["model"]); model.eval()

    t0 = time.perf_counter()
    hyp, ce = translate(model, feats, mask, ck["vocab"], a.beam, a.length_penalty)
    tr_ms = (time.perf_counter() - t0) * 1000.0

    speaker = Speaker(threshold=a.threshold)
    u = speaker.speak(hyp, confidence=ce, play=not a.no_play)

    T = len(d["KPS"])
    print(f"clip            {os.path.basename(path)[:-4]}  ({T} frames, {T/25:.1f} s at 25 fps)")
    if ref:
        print(f"signed          “{ref}”")
    print(f"translation     “{hyp}”")
    print(f"confidence      {ce:.3f} nats" + (f"  (gate {a.threshold})" if a.threshold else ""))
    print(f"decision        {u.reason}")
    print(f"latency         pose {T/POSE_FPS_CPU*1000:.0f} ms (est.) + translate {tr_ms:.0f} ms"
          + (f" + synthesise {u.synth_ms:.0f} ms" if u.spoken else "")
          + f"  =  {T/POSE_FPS_CPU*1000 + tr_ms + (u.synth_ms if u.spoken else 0):.0f} ms")
    if not u.spoken and u.reason.startswith("declined"):
        print('\n  the interface would say: "I did not catch that -- please sign again."')


if __name__ == "__main__":
    main()
