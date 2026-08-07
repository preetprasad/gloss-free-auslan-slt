"""Deep empirical study of the Auslan-Daily Communication data to explain the
BLEU plateau and the transfer collapse. Writes findings.json + prints a summary.
"""
import os, json, collections, statistics, random
import numpy as np

BASE = os.path.expanduser("~/Honours")
MAN = os.path.join(BASE, "comm_run/manifests")
OUT = os.path.join(BASE, "results/per_clip_quality_analysis"); os.makedirs(OUT, exist_ok=True)

def read_manifest(p):
    rows = []
    for ln in open(p):
        ln = ln.rstrip("\n")
        if not ln:
            continue
        parts = ln.split("\t")
        if len(parts) >= 2:
            rows.append((parts[0], parts[1]))
    return rows

train = read_manifest(f"{MAN}/train.tsv")
val = read_manifest(f"{MAN}/val.tsv")
test = read_manifest(f"{MAN}/test.tsv")
tr_txt = [t for _, t in train]; te_txt = [t for _, t in test]; va_txt = [t for _, t in val]
tok = lambda s: s.split()
F = {}

def text_stats(txts):
    lens = [len(tok(t)) for t in txts]
    vocab = collections.Counter(w for t in txts for w in tok(t))
    sent = collections.Counter(txts)
    return {
        "n": len(txts), "unique_sentences": len(sent),
        "unique_ratio": round(len(sent) / max(1, len(txts)), 3),
        "vocab_size": len(vocab),
        "len_mean": round(statistics.mean(lens), 2), "len_median": statistics.median(lens),
        "len_max": max(lens), "len_p90": int(np.percentile(lens, 90)),
        "top1_sentence_coverage": round(sent.most_common(1)[0][1] / len(txts), 3),
        "top10_sentence_coverage": round(sum(c for _, c in sent.most_common(10)) / len(txts), 3),
        "top5_sentences": [(s, c) for s, c in sent.most_common(5)],
        "top15_tokens": vocab.most_common(15),
        "hapax_pct": round(sum(1 for w, c in vocab.items() if c == 1) / max(1, len(vocab)), 3),
    }

F["train_text"] = text_stats(tr_txt)
F["val_text"] = text_stats(va_txt)
F["test_text"] = text_stats(te_txt)

tr_set = set(tr_txt)
ov = [t for t in te_txt if t in tr_set]
F["test_in_train_verbatim"] = {"count": len(ov), "pct": round(len(ov) / len(te_txt), 3)}

# ---- text-prior / collapse BLEU baselines: how much BLEU is free from the prior?
try:
    import sacrebleu
    refs = [te_txt]
    mf = collections.Counter(tr_txt).most_common(1)[0][0]
    b_mf = sacrebleu.corpus_bleu([mf] * len(te_txt), refs, tokenize="13a").score
    bylen = collections.defaultdict(collections.Counter)
    for t in tr_txt:
        bylen[len(tok(t))][t] += 1
    preds_len = [(bylen[len(tok(r))].most_common(1)[0][0] if bylen[len(tok(r))] else mf) for r in te_txt]
    b_len = sacrebleu.corpus_bleu(preds_len, refs, tokenize="13a").score
    rnd = random.Random(0)
    b_rand = sacrebleu.corpus_bleu([rnd.choice(tr_txt) for _ in te_txt], refs, tokenize="13a").score
    # copy the reference itself (upper bound sanity)
    b_oracle = sacrebleu.corpus_bleu(te_txt, refs, tokenize="13a").score
    F["text_prior_bleu"] = {
        "always_most_frequent_sentence": round(b_mf, 2),
        "most_frequent_by_reflen": round(b_len, 2),
        "random_train_sentence": round(b_rand, 2),
        "oracle_copy_reference": round(b_oracle, 2),
        "most_frequent_sentence_text": mf,
        "note": "control greedy=1.76 beam=2.87 ; transfer greedy=2.22 beam=2.26",
    }
except Exception as e:
    F["text_prior_bleu"] = {"error": repr(e)}

# ---- pose quality on a real sample ----
def find_kps(d):
    for k in d.files:
        a = d[k]
        if a.ndim == 3 and a.shape[1] == 136 and a.shape[2] >= 3:
            return a.astype(np.float32)
    return None

sample = random.Random(1).sample(train, min(500, len(train)))
confs = collections.defaultdict(list); Ts = []; hand_low = []; zero_fr = []; loaded = 0
for path, _ in sample:
    if not os.path.exists(path):
        continue
    try:
        d = np.load(path); kps = find_kps(d)
    except Exception:
        continue
    if kps is None:
        continue
    loaded += 1
    Ts.append(kps.shape[0])
    c = kps[..., 2]
    confs["body"].append(float(c[:, 0:26].mean()))
    confs["face"].append(float(c[:, 26:94].mean()))
    confs["left_hand"].append(float(c[:, 94:115].mean()))
    confs["right_hand"].append(float(c[:, 115:136].mean()))
    hand_low.append(float((kps[:, 94:136, 2] < 0.1).mean()))
    zs = np.abs(kps).reshape(kps.shape[0], -1).sum(1)
    zero_fr.append(float((zs < 1e-6).mean()))

F["pose_quality"] = {
    "clips_analyzed": loaded,
    "T_mean": round(float(np.mean(Ts)), 1), "T_median": float(np.median(Ts)),
    "T_max": int(np.max(Ts)), "T_p90": int(np.percentile(Ts, 90)),
    "mean_confidence_by_region": {k: round(float(np.mean(v)), 3) for k, v in confs.items()},
    "hand_lowconf_frame_pct": round(float(np.mean(hand_low)), 3),
    "zero_frame_pct": round(float(np.mean(zero_fr)), 4),
}

json.dump(F, open(f"{OUT}/findings.json", "w"), indent=2, default=str)
print(json.dumps(F, indent=2, default=str))
