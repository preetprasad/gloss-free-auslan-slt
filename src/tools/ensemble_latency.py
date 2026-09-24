"""t3-cost: per-clip translation latency of the reported ensemble at batch 1, with quality from the SAME decodes.

Reuses the ensemble tool's own pieces (ensemble_decode.Member, ensemble_mbr_wer._align_vocabs / ensemble_nbest /
corpus_scores), so a configuration decodes exactly as the headline did: n-best at the given beam, top-1 by
raw / words**LP. Timed = member encoders + n-best decode for one clip; data loading and model loading are not.
Configurations are PREFIXES of the member list in the order given (1, 3, 6 ... members).

Harness gate: the full member list at beam 10 / LP 0.3 must reproduce --expect on this manifest.
"""
import argparse, os, statistics, sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from config import Config                                                  # noqa: E402
from slt.data import Vocab                                                 # noqa: E402
from slt.rgb_fusion import RGBDataset, collate_rgb                        # noqa: E402
from ensemble_decode import Member                                         # noqa: E402
from ensemble_mbr_wer import _align_vocabs, ensemble_nbest, corpus_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="comma-separated, in the recorded ensemble order")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rgb_feat_dir", required=True)
    ap.add_argument("--sizes", default="1,3,6")
    ap.add_argument("--beams", default="5,10")
    ap.add_argument("--lp", type=float, default=0.3)
    ap.add_argument("--expect", type=float, required=True, help="recorded BLEU-4 of the full set at beam 10")
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    cfg = Config(num_keypoints=133, keypoint_subset="all", coord_norm="frame", rgb_feat_dir=a.rgb_feat_dir,
                 rgb_feature_dim=1152, test_manifest=a.manifest, device="cpu")
    dev = torch.device("cpu")
    paths = a.ckpts.split(",")
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    vocab = Vocab(cks[0]["vocab"])
    members = _align_vocabs(cks, [Member(c, cfg, vocab, dev) for c in cks], list(cks[0]["vocab"]), dev, False)
    ds = RGBDataset(a.manifest, vocab, cfg, "")
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_rgb)
    batches = list(dl)
    mx = cfg.decode_max_len or cfg.max_tgt_tokens
    print(f"torch threads {torch.get_num_threads()} | clips {len(batches)} | members {len(members)} "
          f"fusion={[m.fusion for m in members]}", flush=True)

    for beam in (int(b) for b in a.beams.split(",")):
        for k in (int(s) for s in a.sizes.split(",")):
            sub, hyps, refs, ms = members[:k], [], [], []
            for i, (src, src_pad, rgb, rgb_pad, _t, texts) in enumerate(batches):
                t0 = time.perf_counter()
                with torch.no_grad():
                    mems = [m.memory(src, src_pad, rgb, rgb_pad) for m in sub]
                    cands = ensemble_nbest(sub, mems, mx, beam, dev)[0]
                if i >= a.warmup:
                    ms.append((time.perf_counter() - t0) * 1000)
                raw, seq = max(((r, vocab.decode(s)) for r, s in cands),
                               key=lambda c: c[0] / (max(1, len(c[1].split())) ** a.lp))
                hyps.append(seq); refs.append(texts[0])
            bleu, wer, *_ = corpus_scores(hyps, refs)
            q = statistics.quantiles(ms, n=20)
            print(f"beam {beam:2d} | members {k} | BLEU-4 {bleu:6.3f} WER {wer:.4f} | per clip median "
                  f"{statistics.median(ms):6.1f} ms  p95 {q[18]:6.1f} ms  mean {statistics.mean(ms):6.1f} ms", flush=True)
            if beam == 10 and k == len(members):
                ok = abs(bleu - a.expect) < 0.005
                print(f"HARNESS GATE {'PASSED' if ok else 'FAILED'}: {bleu:.3f} vs recorded {a.expect}", flush=True)


if __name__ == "__main__":
    main()
