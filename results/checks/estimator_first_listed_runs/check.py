"""Early AlphaPose (first-listed person) runs and the DWPose re-extraction: what can be recomputed locally.

Recomputes: the greedy and beam-5 test decodes of the first-listed AlphaPose control (BLEU-4, distinct outputs),
the greedy test decodes of the SignJoey-compatible reconstruction, the mean DWPose hand confidence over 14,041 clips,
the first-listed AlphaPose identification probe (hands top-1), keypoint-set sizes from config.py, the parameter count
and test score of the 25.95 M model, and the bounding-box-corrected generic run. The beam-5 cells of the estimator 2x2
(job 1101250) and the DWPose re-extraction run are on Kaya and are not checked here.
"""
import csv, json, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import detokenize, corpus_bleu4
from config import KP_SUBSETS, NUM_KEYPOINTS, Config


def load(name, hypkey):
    recs = [json.loads(l) for l in open(HERE / name)]
    man = ROOT / "comm_run" / "manifests" / "test.tsv"
    if man.exists():
        m = {Path(a).name[:-4]: b.strip() for a, b in (l.rstrip("\n").split("\t") for l in open(man) if "\t" in l)}
        assert len(m) == 800 and all(m[r["id"]] == r["ref"] for r in recs), "references differ from the test manifest"
        print(f"{name}: references match comm_run/manifests/test.tsv: 800/800")
    hyps = [r[hypkey] for r in recs]; refs = [r["ref"] for r in recs]
    return hyps, refs, corpus_bleu4([detokenize(h) for h in hyps], [detokenize(r) for r in refs])


# --- first-listed AlphaPose control (checkpoint decoded with tools/eval_beam.py: greedy and beam 5, length penalty 0.6)
gh, _, bg = load("control_decodes.jsonl", "hyp_greedy")
bh, _, bb = load("control_decodes.jsonl", "hyp_beam5")
print(f"GATE recorded BLEU-4 of the first-listed AlphaPose control, greedy 1.76 and beam 5 2.87: recomputed {bg:.4f} and {bb:.4f}")
if round(bg, 2) != 1.76 or round(bb, 2) != 2.87:
    sys.exit("GATE FAILED")
print(f"\nFirst-listed AlphaPose control, official test, 800 clips, one run")
print(f"BLEU-4 {bb:.2f} at beam width 5 (length penalty 0.6), 1 run  [c02-006 c09-008]")
print(f"BLEU-4 {bg:.2f} with greedy decoding  [c09-008]")
print(f"{len(set(gh))} distinct sentences for the {len(gh)} test clips with greedy decoding (output collapse)  [c02-007 c08-015 c09-008]")
print(f"(beam width 5 gives {len(set(bh))} distinct sentences)")

# --- SignJoey-compatible reconstruction (25.9 M parameters, raw pose), checkpoint decoded greedily
sh, _, bs = load("signjoey_decodes.jsonl", "hyp_greedy")
print(f"\nSignJoey-compatible reconstruction, greedy, official test: BLEU-4 {bs:.2f}, {len(set(sh))} distinct sentences  [c17-010]")

# --- DWPose hand confidence
hc = np.array([float(r["hand_conf"]) for r in csv.DictReader(open(HERE / "dwpose_hand_conf.csv"))])
print(f"\nDWPose re-extraction: mean hand-keypoint confidence over {len(hc)} clips {hc.mean():.2f} ({hc.mean():.4f})  [c08-020]")

# --- identification probe, first-listed AlphaPose (the wrong person in most clips)
rows = [r for r in csv.DictReader(open(HERE / "pred_firstlisted_alphapose.csv")) if r["variant"] == "hands"]
acc = 100 * sum(int(r["correct"]) for r in rows) / len(rows)
log = open(HERE / "probe_log_firstlisted_alphapose.txt").read()
usable = int(re.search(r"spans usable (\d+)", log).group(1))
print(f"\nFirst-listed AlphaPose identification probe: {usable} annotated spans usable (45 signs) in the saved probe log")
print(f"hand keypoints, nearest-centroid top-1 on {len(rows)} scored spans (the probe scores a span only if its hand "
      f"descriptor exists and its class is predictable in the held-out fold): {acc:.1f}%  [c08-049]")
print(f"({len(rows)} of {usable} usable spans are scored for the hand variant; the quoted 382 is the usable set, 4.2% is over the scored spans)")

# --- keypoint settings from config.py
print(f"\nKeypoint sets (config.py)")
print(f"COCO-WholeBody body+hands subset: {len(KP_SUBSETS['cwb_body_hands'])} keypoints  [c06-030]")
print(f"HALPE body+hands subset: {len(KP_SUBSETS['body_hands'])} keypoints  [c06-030]")
print(f"AlphaPose HALPE layout: {NUM_KEYPOINTS} keypoints  [c06-032 c09-008]")
print(f"DWPose COCO-WholeBody layout: {Config().num_keypoints} keypoints (default)  [c06-033]")

# --- capacity run and the bounding-box-corrected run
cap = json.load(open(HERE / "capacity_accum2_metrics.json"))
npar = json.load(open(HERE / "capacity_params.json"))["n_params"]
print(f"\n1,024-to-512 capacity model: {npar/1e6:.2f} M parameters (pos_enc buffer excluded)  [c17-009]")
print(f"its test BLEU-4 {cap['test']['bleu4']:.2f} on {cap['test']['n']} clips (read from the saved job output capacity_accum2_metrics.json)  [c17-009]")
txt = open(HERE / "bbox_xywh_test_rescore.txt").read()
g = float(re.search(r"tokenize=13a\s+detok=True\s+BLEU-4=\s*([\d.]+)", txt).group(1))
print(f"bounding-box-corrected generic run, greedy test BLEU-4 {g:.2f} (read from the saved job output bbox_xywh_test_rescore.txt)  [c17-011]")
