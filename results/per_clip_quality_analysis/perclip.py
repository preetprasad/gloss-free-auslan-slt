"""Per-clip failure analysis: decode every test clip with both models, join each
prediction to that clip's pose quality + reference, score per-clip, and find what
separates the clips we translate from the clips we fail. Writes per_clip.csv +
perclip_summary.json.
"""
import os, sys, json, csv, collections
import numpy as np, torch

sys.path.insert(0, os.path.expanduser("~/Honours/src"))
from config import Config
from slt.data import AuslanDailyDataset, Vocab, collate
from slt.reference_model import build_model
from slt.engine import evaluate
from torch.utils.data import DataLoader
try:
    from slt.metrics import detokenize
except Exception:
    def detokenize(s):
        import re; return re.sub(r"\s+([.,!?;:])", r"\1", s)
import sacrebleu

BASE = os.path.expanduser("~/Honours")
PB = os.path.join(BASE, "src")
OUT = os.path.join(BASE, "results/per_clip_quality_analysis"); os.makedirs(OUT, exist_ok=True)
TEST = os.path.join(BASE, "comm_run/manifests/test.tsv")
TRAIN = os.path.join(BASE, "comm_run/manifests/train.tsv")
CKPTS = {"control": os.path.join(PB, "comm_scratch_control/best.pt"),
         "transfer": os.path.join(PB, "comm_islr_transfer/best.pt")}
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# manifest (ordered) + train frequency
rows = [ln.rstrip("\n").split("\t") for ln in open(TEST) if ln.strip()]
paths = [r[0] for r in rows]; refs = [r[1] for r in rows]
train_txt = [ln.rstrip("\n").split("\t")[1] for ln in open(TRAIN) if ln.strip()]
train_freq = collections.Counter(train_txt)

def decode_ckpt(name, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    vocab = Vocab(ck["vocab"], cfg.text_tokenization, cfg.sentencepiece_model)
    model = build_model(cfg.feature_dim(), len(vocab), cfg).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    ds = AuslanDailyDataset(TEST, vocab, cfg, "")
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate)
    mx = cfg.decode_max_len or cfg.max_tgt_tokens
    bleu, wer, hyps, rf = evaluate(model, dl, vocab, dev, max_len=mx,
                                   bleu_tokenize=cfg.bleu_tokenize, detok=cfg.detokenize)
    print(f"[{name}] corpus BLEU-4={bleu:.2f} WER={wer:.3f} n={len(hyps)}", flush=True)
    return hyps, rf

preds = {}
order_refs = None
for name, p in CKPTS.items():
    h, rf = decode_ckpt(name, p)
    preds[name] = h
    order_refs = rf   # evaluate returns refs in loader = manifest order

# pose quality per clip
def pose_stats(path):
    try:
        d = np.load(path); kps = d["KPS"].astype(np.float32)
    except Exception:
        return None
    c = kps[..., 2]
    hand = kps[:, 94:136, 2]
    frame_hand = hand.mean(1)
    return dict(T=int(kps.shape[0]),
        body_conf=float(c[:, 0:26].mean()), face_conf=float(c[:, 26:94].mean()),
        hand_conf=float(hand.mean()),
        hand_low_frame_pct=float((frame_hand < 0.1).mean()),
        zero_frame_pct=float((np.abs(kps).reshape(kps.shape[0], -1).sum(1) < 1e-6).mean()))

def sbleu(h, r):
    return sacrebleu.sentence_bleu(detokenize(h), [detokenize(r)], tokenize="13a").score

table = []
for i, (path, ref) in enumerate(zip(paths, refs)):
    ps = pose_stats(path) or {}
    row = dict(clip=os.path.basename(path), ref=ref, ref_len=len(ref.split()),
               ref_in_train=int(ref in train_freq), ref_train_freq=int(train_freq.get(ref, 0)),
               **ps)
    for name in CKPTS:
        hyp = preds[name][i]
        row[name + "_pred"] = hyp
        row[name + "_bleu"] = round(sbleu(hyp, ref), 1)
        row[name + "_exact"] = int(detokenize(hyp).strip() == detokenize(ref).strip())
    table.append(row)

cols = list(table[0].keys())
with open(f"{OUT}/per_clip.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(table)

# ---- aggregate diagnosis ----
def arr(k): return np.array([r.get(k, np.nan) for r in table], float)
S = {"n": len(table)}
for name in CKPTS:
    b = arr(name + "_bleu"); hc = arr("hand_conf")
    exact = arr(name + "_exact")
    hit = b >= 10.0   # "translated" ~ non-trivial per-clip BLEU
    S[name] = {
        "corpus_sentBLEU_mean": round(float(np.nanmean(b)), 2),
        "exact_match": int(np.nansum(exact)),
        "clips_bleu>=10": int(np.nansum(hit)),
        "clips_bleu==0": int(np.nansum(b == 0)),
        "unique_predictions": len(set(r[name + "_pred"] for r in table)),
        "corr_handconf_bleu": round(float(np.corrcoef(hc[~np.isnan(b)], b[~np.isnan(b)])[0, 1]), 3),
        "hit_group_mean_handconf": round(float(np.nanmean(hc[hit])), 3) if hit.any() else None,
        "miss_group_mean_handconf": round(float(np.nanmean(hc[~hit])), 3),
        "hit_group_ref_in_train_pct": round(float(np.nanmean(arr("ref_in_train")[hit])), 3) if hit.any() else None,
        "miss_group_ref_in_train_pct": round(float(np.nanmean(arr("ref_in_train")[~hit])), 3),
        "hit_group_mean_reflen": round(float(np.nanmean(arr("ref_len")[hit])), 2) if hit.any() else None,
        "miss_group_mean_reflen": round(float(np.nanmean(arr("ref_len")[~hit])), 2),
    }
# overall pose vs overlap context
S["pose"] = {"mean_hand_conf": round(float(np.nanmean(arr("hand_conf"))), 3),
             "mean_hand_low_frame_pct": round(float(np.nanmean(arr("hand_low_frame_pct"))), 3)}
S["most_common_predictions"] = {
    name: collections.Counter(r[name + "_pred"] for r in table).most_common(6) for name in CKPTS}
json.dump(S, open(f"{OUT}/perclip_summary.json", "w"), indent=2, default=str)
print(json.dumps(S, indent=2, default=str))
