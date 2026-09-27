"""Wave 8 (job 21734) per-arm protocol readouts (PREREG_WAVE8 §3), via tools/arm_readouts.readouts().

Unseen-target BLEU uses the FULL 12,439 train manifest (fixed unseen set for every arm); the seen count uses each arm's
OWN subset manifest (memorisation slope). Self-check first: pa_aug_s42 beam-5 test 11.443, and the recorded WER null
floor 0.9504 for the constant "it is sally ." on the official test references.

  cd src && python ../results/real_data_learning_curve/arm_table.py
"""
import json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PB = os.path.join(HERE, "..", "..", "src")
sys.path.insert(0, PB); sys.path.insert(0, os.path.join(PB, "tools"))
from arm_readouts import readouts, KEY          # noqa: E402
from slt.metrics import corpus_wer, detokenize  # noqa: E402
from stitch_lexicon import STOP                 # noqa: E402

ARMS, SEEDS = ["R025", "R050", "R100", "V050"], [42, 101, 202, 303, 404]


def texts(p):
    return {" ".join(l.rstrip("\n").split("\t", 1)[1].lower().split()) for l in open(p) if "\t" in l}


def dump(arm, s, split, ck):
    j = json.load(open(os.path.join(HERE, "runs", f"{arm}_s{s}", f"beam5_{split}_{ck}.json")))
    return j["beam_hyps"], j["refs"], j["beam_result"]["metrics"][KEY]


# self-check: reproduce recorded numbers before printing anything new
j = json.load(open(os.path.join(PB, "..", "results", "pose_augmentation_decodes", "pa_aug_s42", "beam5_test.json")))
b = readouts(j["beam_hyps"], j["refs"], STOP)["bleu4"]
assert abs(b - 11.443) < 0.002, b
_, refs, _ = dump("R100", 42, "test", "best")
floor = corpus_wer([detokenize("it is sally .")] * len(refs), [detokenize(r) for r in refs])
assert abs(floor - 0.9504) < 0.0005, floor
print(f"self-check OK: pa_aug_s42 beam-5 test {b:.3f}; WER null floor ('it is sally .') {floor:.4f}; silence 1.0000\n")

full = texts(os.path.join(HERE, "train.tsv"))
cols = ["val", "val_soup", "test", "test_soup", "unseen", "seen_own", "wer", "len_ratio", "c_prec", "c_rec", "distinct"]
agg = {a: {c: [] for c in cols} for a in ARMS}
print("arm_seed  " + " ".join(f"{c:>9s}" for c in cols))
for a in ARMS:
    for s in SEEDS:
        own = texts(os.path.join(HERE, "manifests", f"{a}_s{s}.tsv"))
        h, r, rec = dump(a, s, "test", "best")
        f = readouts(h, r, STOP, full); o = readouts(h, r, STOP, own)
        assert abs(f["bleu4"] - rec) < 1e-6 and f["unseen_n"] == 570, (a, s, f)
        row = {"val": dump(a, s, "val", "best")[2], "val_soup": dump(a, s, "val", "soup_last5")[2],
               "test": f["bleu4"], "test_soup": dump(a, s, "test", "soup_last5")[2], "unseen": f["bleu4_unseen"],
               "seen_own": 800 - o["unseen_n"], "wer": f["wer"], "len_ratio": f["len_ratio"],
               "c_prec": f["content_precision"], "c_rec": f["content_recall"], "distinct": f["distinct"]}
        for c in cols:
            agg[a][c].append(row[c])
        print(f"{a}_s{s:<4d}" + " ".join(f"{row[c]:9.3f}" for c in cols))
print("\nmean over 5 seeds (sd)")
for a in ARMS:
    print(f"{a:9s} " + " ".join(f"{st.mean(agg[a][c]):9.3f}" for c in cols))
    print(f"{'  sd':9s} " + " ".join(f"{st.stdev(agg[a][c]):9.3f}" for c in cols))
