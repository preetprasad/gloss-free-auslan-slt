"""End-to-end Auslan-to-speech: measure the whole chain, and what a user actually hears.

This closes the two deployment gaps together. G1 asks what the system's latency is; G2 asks whether
there is a speech stage at all. Both are answered here on real test clips, with the translator's own
confidence deciding whether the sentence is voiced or the signer is asked to sign again.

What is measured, per clip:
  translate_ms   pose (T,133,3) -> English, on this machine's CPU
  synth_ms       English -> rendered audio (macOS `say`, en_AU)
  audio_s        length of the resulting speech
and, over the split, the coverage/quality trade-off: at each confidence threshold, what fraction of
utterances are voiced and what the BLEU-4 of the voiced subset is. That second table is the point --
a text system reports one number for everything it emits, but a speech system can decline, and what
the user hears is the score of the subset it chose to say.

Pose extraction is NOT timed here; it is a separate stage already measured at 25.4 fps on this CPU
(CLAUDE.md 2026-09-15e) and is reported alongside rather than re-benchmarked.

Run:  python tools/speech_pipeline_bench.py --n 120
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
from config import Config
from slt.data import normalize_clip
from slt.reference_model import build_model
from slt.metrics import detokenize, corpus_bleu4
from slt.speech import Speaker

PAD, BOS, EOS = 0, 1, 2
POSE_FPS_CPU = 25.4        # detector once per clip, CPU, measured 2026-09-15e


def translate(model, feats, mask, vocab, beam: int, lp: float):
    """Greedy or beam decode, returning the sentence and the mean per-token cross-entropy of the
    model's own output -- the self-confidence signal used by selective prediction."""
    src = torch.from_numpy(feats).unsqueeze(0)
    pad = ~torch.from_numpy(mask).unsqueeze(0)
    with torch.no_grad():
        mem = model.encode(src, pad)
        if beam <= 1:
            ys, lps = torch.tensor([[BOS]]), []
            for _ in range(40):
                step = torch.log_softmax(model.decode_step(ys, mem, pad, None)[:, -1], -1)[0]
                nxt = int(step.argmax()); lps.append(float(step[nxt]))
                ys = torch.cat([ys, torch.tensor([[nxt]])], 1)
                if nxt == EOS: break
            toks = [int(t) for t in ys[0, 1:]]
        else:
            beams = [([BOS], 0.0, [])]
            for _ in range(40):
                nxt = []
                for seq, sc, lps in beams:
                    if seq[-1] == EOS: nxt.append((seq, sc, lps)); continue
                    step = torch.log_softmax(
                        model.decode_step(torch.tensor([seq]), mem, pad, None)[:, -1], -1)[0]
                    top = step.topk(beam)
                    for i, p in zip(top.indices.tolist(), top.values.tolist()):
                        nxt.append((seq + [i], sc + p, lps + [p]))
                beams = sorted(nxt, key=lambda b: b[1] / (len(b[0]) ** lp), reverse=True)[:beam]
                if all(b[0][-1] == EOS for b in beams): break
            seq, _, lps = beams[0]
            toks = seq[1:]
        ce = -float(np.mean(lps)) if lps else float("nan")
    words = [vocab[t] for t in toks if t not in (PAD, BOS, EOS)]
    return " ".join(words), ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "results/ckpts/comm_dwpose_vlp_ctc.pt"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "results/TEST_manifest_official.tsv"))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--length_penalty", type=float, default=0.6)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/speech_pipeline_bench.json"))
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config().to_dict()})
    vocab = ck["vocab"]
    rows = [l.rstrip("\n").split("\t") for l in open(a.manifest) if "\t" in l][:a.n]
    feats0, _ = normalize_clip(*(lambda d: (d["KPS"], d["BBOX"]))(np.load(rows[0][0])), cfg)
    model = build_model(feats0.shape[1], len(vocab), cfg)
    model.load_state_dict(ck["model"]); model.eval()
    speaker = Speaker()
    assert speaker.available_voice(), f"voice {speaker.voice} not installed"
    print(f"{len(rows)} clips | beam {a.beam} | backend {speaker.backend}/{speaker.voice}")

    recs = []
    for i, (path, ref) in enumerate(rows):
        d = np.load(path)
        feats, mask = normalize_clip(d["KPS"], d["BBOX"], cfg)
        t0 = time.perf_counter()
        hyp, ce = translate(model, feats, mask, vocab, a.beam, a.length_penalty)
        translate_ms = (time.perf_counter() - t0) * 1000.0
        u = speaker.synthesise(hyp)
        recs.append(dict(clip=os.path.basename(path)[:-4], ref=ref.strip(), hyp=hyp, ce=ce,
                         frames=int(len(d["KPS"])), translate_ms=translate_ms,
                         synth_ms=u.synth_ms, audio_s=u.audio_seconds))
        if u.audio_path and os.path.exists(u.audio_path):
            os.remove(u.audio_path)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows)}")

    tr = np.array([r["translate_ms"] for r in recs])
    sy = np.array([r["synth_ms"] for r in recs])
    fr = np.array([r["frames"] for r in recs])
    pose_ms = fr / POSE_FPS_CPU * 1000.0
    print("\nLATENCY, per utterance, on this laptop CPU")
    for name, v in [("pose extraction (est., 25.4 fps)", pose_ms), ("translation", tr),
                    ("speech synthesis", sy), ("translation + synthesis", tr + sy),
                    ("whole chain", pose_ms + tr + sy)]:
        print(f"  {name:34s} median {np.median(v):7.0f} ms   p95 {np.percentile(v,95):7.0f} ms")

    ces = np.array([r["ce"] for r in recs])
    refs = [r["ref"] for r in recs]; hyps = [r["hyp"] for r in recs]
    print("\nWHAT THE USER HEARS, as the confidence gate tightens")
    print(f"  {'threshold':>10s} {'voiced':>8s} {'coverage':>9s} {'BLEU-4 of voiced':>18s}")
    table = []
    for q in (1.00, 0.90, 0.75, 0.50, 0.25):
        thr = float(np.quantile(ces, q)) if q < 1.0 else float("inf")
        keep = [j for j in range(len(recs)) if ces[j] <= thr]
        b = corpus_bleu4([detokenize(hyps[j]) for j in keep], [detokenize(refs[j]) for j in keep])
        lab = "none" if q == 1.0 else f"{thr:.2f} nats"
        print(f"  {lab:>10s} {len(keep):8d} {len(keep)/len(recs):8.0%} {b:18.2f}")
        table.append(dict(quantile=q, threshold=None if q == 1.0 else thr,
                          voiced=len(keep), coverage=len(keep)/len(recs), bleu4=b))
    json.dump(dict(n=len(recs), beam=a.beam, records=recs, coverage_table=table,
                   latency_ms={k: dict(median=float(np.median(v)), p95=float(np.percentile(v, 95)))
                               for k, v in [("pose_est", pose_ms), ("translate", tr), ("synth", sy),
                                            ("translate_plus_synth", tr + sy),
                                            ("whole_chain", pose_ms + tr + sy)]}),
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
