"""Training recipe, model size and decoding defaults of the reported systems (Methodology, Table 'Training configuration',
Appendix B table convention).

Reads the training configuration stored inside the six checkpoints of the reported ensemble (subset saved in cfg_members.json by
prepare_data.py), the current Config defaults, the training/decoding scripts, and one saved job (pose-augmentation job: log, beam-5
test hypotheses). Gate: the saved hypotheses re-score to the BLEU-4 the job recorded."""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import detokenize, corpus_bleu4
from config import Config

mem = json.load(open(HERE / "cfg_members.json"))
meta = json.load(open(HERE / "run_meta.json"))

# ---- gate: saved beam-5 test hypotheses of one run re-score to the recorded BLEU-4 --------------------------------------
rows = [json.loads(l) for l in open(HERE / "hyps_beam5_pa_aug_s42.jsonl")]
b = corpus_bleu4([detokenize(r["hyp_beam"]) for r in rows], [detokenize(r["ref"]) for r in rows])
print(f"GATE saved beam-5 hypotheses of the pose-augmented seed-42 run: {len(rows)} clips, BLEU-4 {b:.3f} "
      f"(recorded {meta['recorded_bleu4_13a']:.3f}, read from the saved job output)")
if abs(b - meta["recorded_bleu4_13a"]) > 5e-3 or len(rows) != meta["n"]:
    sys.exit("GATE FAILED")
man = ROOT / "comm_run" / "manifests" / "test.tsv"
if man.exists():
    refs = [l.rstrip("\n").split("\t")[1].strip() for l in open(man) if "\t" in l]
    assert refs == [r["ref"] for r in rows], "references differ from the test manifest"
    print("references match comm_run/manifests/test.tsv")

# ---- the six members of the reported ensemble (names from the ensemble job script) --------------------------------------
SIX = ["comm_dwpose_ctc_vlplong80", "comm_dwpose_rgb_fusion", "comm_dwpose_vlp_ctc",
       "comm_dwpose_ctc_s202", "comm_dwpose_fus_s101", "comm_dwpose_fus_s202"]
job = (ROOT / "jobs/slurm/mbr_wer_test2.sbatch").read_text()
assert all(n + "/best.pt" in job for n in SIX), "member list differs from the ensemble job script"
print(f"\nMembers: the {len(SIX)} checkpoints named in the ensemble job script (mbr_wer_test2.sbatch)")
pose_m = [n for n in SIX if not mem[n]["rgb_stream"]]
fus_m = [n for n in SIX if mem[n]["rgb_stream"]]
print(f"{len(pose_m)} pose-only members, {len(fus_m)} RGB-fusion members")


def vals(field, names):
    return sorted({json.dumps(mem[n]["cfg"][field]) for n in names})


def same(field, names=SIX):
    v = vals(field, names)
    assert len(v) == 1, (field, v)
    return json.loads(v[0])


cfg0 = Config()
IDS = {}


def show(label, value, ids, extra=""):
    print(f"{label}: {value}{extra}  [{' '.join(ids)}]")


kp, d, nh, ff = same("num_keypoints"), same("d_model"), same("nhead"), same("dim_feedforward")
ip = {tuple(mem[n]["input_proj_shape"]) for n in SIX}
assert len(ip) == 1
ip = ip.pop()
print("\nTABLE: training configuration (stored in every one of the six members)")
show("keypoints / input feature dimension", f"{kp} COCO-WholeBody / {ip[1]}", ["c07-001", "c06-039"],
     f" (input projection weight {ip[0]} x {ip[1]}; feature dim = {kp} x 3)")
assert ip[1] == kp * 3
show("coordinate normalisation", same("coord_norm"), [])
show("max source frames / target tokens", f"{same('max_src_frames')} / {same('max_tgt_tokens')}", ["c07-002", "c06-025"])
show("d_model / heads / FFN", f"{d} / {nh} / {ff}", ["c07-003", "c06-039"], f" (d_model = model width {d})")
el, dl = same("num_encoder_layers"), same("num_decoder_layers")
show("encoder / decoder layers", f"{el} / {dl}", ["c07-004", "c06-037"],
     f" (weights hold {sorted({mem[n]['encoder_layers_in_weights'] for n in SIX})[0]} encoder and "
     f"{sorted({mem[n]['decoder_layers_in_weights'] for n in SIX})[0]} decoder layers)")
assert {mem[n]["encoder_layers_in_weights"] for n in SIX} == {el} and {mem[n]["decoder_layers_in_weights"] for n in SIX} == {dl}
show("dropout / embedding dropout", f"{same('dropout')} / {same('embedding_dropout')}", ["c07-005"])
show("label smoothing", same("label_smoothing"), ["c07-006"])

# CTC weight: not stored in the checkpoint; the training script default and the job command of the adopted-configuration run
tc = (ROOT / "src/train_ctc.py").read_text()
w = float(re.search(r'--ctc_weight", type=float, default=([0-9.]+)', tc).group(1))
pa = (ROOT / "jobs/slurm/pose_aug.sbatch").read_text()
w2 = float(re.search(r"--ctc_weight ([0-9.]+)", pa).group(1))
assert w == w2
show("auxiliary CTC weight (default of train_ctc.py = value passed by the job script)", w, ["c07-007", "c07-014"])
lr, wd, gc = same("lr"), same("weight_decay"), same("grad_clip")
show("learning rate", f"{lr / 1e-4:g}x10^-4 ({lr})", ["c07-008"])
show("weight decay / gradient clip", f"{wd / 1e-4:g}x10^-4 ({wd}) / {gc}", ["c07-009"])
show("schedule / warmup steps", f"{same('lr_schedule')} / {same('warmup_steps')}", ["c07-010"])
ep = same("epochs")
bp = {vals("batch_size", [n])[0] for n in pose_m}
bf = {vals("batch_size", [n])[0] for n in fus_m}
assert len(bp) == 1 and len(bf) == 1
show("batch size (pose-only members) / epochs", f"{bp.pop()} / {ep}", ["c07-011"])
show("batch size of the RGB-fusion members", bf.pop(), ["c06-040"])

# current Config defaults against the stored values (the table is also what the code now defaults to)
CFG_FIELDS = ["num_keypoints", "d_model", "nhead", "dim_feedforward", "num_encoder_layers", "num_decoder_layers", "dropout",
              "embedding_dropout", "label_smoothing", "lr", "weight_decay", "grad_clip", "lr_schedule", "warmup_steps",
              "max_src_frames", "max_tgt_tokens", "epochs"]
diff = [f for f in CFG_FIELDS if getattr(cfg0, f) != same(f)]
print(f"\nConfig() defaults vs the stored configuration of the members: {len(CFG_FIELDS) - len(diff)} of {len(CFG_FIELDS)} fields agree"
      + (f"; differ: {diff}" if diff else ""))

# ---- sequence limits: uniform subsampling ---------------------------------------------------------------------------------
import numpy as np
src = (ROOT / "src/slt/data.py").read_text()
assert "np.linspace(0, feats.shape[0] - 1, cfg.max_src_frames).round().astype(int)" in src
T, cap = 300, same("max_src_frames")
sel = np.linspace(0, T - 1, cap).round().astype(int)
gaps = np.diff(sel)
print(f"\nSource sequences longer than the {cap}-frame limit are subsampled uniformly (np.linspace in slt/data.py): "
      f"a {T}-frame clip keeps {len(sel)} frames with gaps of {gaps.min()} or {gaps.max()} frames  [c06-025]")
print(f"Targets are limited to {same('max_tgt_tokens')} tokens (stored max_tgt_tokens)  [c06-025]")

# ---- decoding defaults -----------------------------------------------------------------------------------------------------
eb = (ROOT / "src/tools/eval_beam.py").read_text()
beam_d = int(re.search(r'"--beam", type=int, default=(\d+)', eb).group(1))
lp_d = float(re.search(r'"--length_penalty", type=float, default=([0-9.]+)', eb).group(1))
print("\nDECODING defaults")
show("beam width / length penalty (Config)", f"{cfg0.num_beams} / {cfg0.length_penalty}", ["c06-011", "c07-012"])
show("beam width / length penalty (tools/eval_beam.py)", f"{beam_d} / {lp_d}", ["c06-011", "c07-012"])
show("beam width / length penalty (saved beam-5 test job output)", f"{meta['beam']} / {meta['length_penalty']}", ["c06-011", "c07-012"],
     " (read from the saved job output beam5_test.json)")
assert (cfg0.num_beams, cfg0.length_penalty) == (beam_d, lp_d) == (meta["beam"], meta["length_penalty"])

# ---- adopted training configuration ------------------------------------------------------------------------------------------
log = (HERE / "joblog_pose_aug.out").read_text()
m = re.search(r"train (\d+) \((\d+) synthetic\)", log)
tot, syn = int(m.group(1)), int(m.group(2))
print(f"\nAdopted training configuration (job log joblog_pose_aug.out, read from the saved job output): training set {tot} clips, "
      f"of which {syn:,} stitched; real clips {tot - syn:,}  [c07-042]")
m = re.search(r"config pose_aug fields: (\w+) ([0-9.]+) ([0-9.]+) ([0-9.]+)", log)
print(f"pose augmentation switched on in that run (pose_aug {m.group(1)}): speed +-{m.group(2)}, rotation +-{m.group(3)} deg, scale +-{m.group(4)} "
      "(read from the saved job output)")

# ---- seeds of the staged rule -----------------------------------------------------------------------------------------------------
def seeds(path, pat):
    return re.search(pat, (ROOT / path).read_text()).group(1).split()

s1 = seeds("jobs/slurm/wave1_r2.sbatch", r"SEEDS=\(([\d ]+)\)")
s1b = seeds("jobs/slurm/pose_aug.sbatch", r"for S in ([\d ]+); do")
s2 = seeds("jobs/slurm/w6_corpus_s2.sbatch", r"SEEDS=\(([\d ]+)\)")
s2b = seeds("jobs/slurm/w5_train_s2.sbatch", r"SEEDS=\(([\d ]+)\)")
assert s1 == s1b and s2 == s2b
print(f"\nStage 1 job scripts train seeds {', '.join(s1)}; stage 2 job scripts train fresh seeds {', '.join(s2)}  [c21-019]")
print("seeds stored in the six members: " + ", ".join(str(mem[n]["cfg"]["seed"]) for n in SIX) + "  [c21-019]")

# ---- table convention of the appendix ---------------------------------------------------------------------------------------------------
print(f"\nTable convention: the saved test run holds {len(rows)} official Communication test clips decoded at beam width {meta['beam']}  [c17-001 c17-002]")
pb = (ROOT / "src/tools/paired_bootstrap.py").read_text()
ci = float(re.search(r'"--ci", type=float, default=([0-9.]+)', pb).group(1))
half = (100.0 - ci) / 2.0
print(f"paired_bootstrap.py default interval: {ci:g}% (percentiles {half:g} and {100 - half:g})  [c17-003]")
