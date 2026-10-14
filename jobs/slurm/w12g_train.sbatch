#!/bin/bash -l
# WAVE 12g -- training on a signer-disjoint split, with the same clips scored when their signer IS in training.
# Task = (arm, seed), all nine runs in this one array:   arm = task % 3 -> SD | EXPa | EXPb    seed = task / 3 -> 42 | 101 | 202
#   SD    signers 1 and 2 only in training; tested on all 1,087 clips of the 47 other signers
#   EXPa  same size of training set, fold A of the other signers' clips in it; tested on fold B
#   EXPb  same, fold B in training; tested on fold A
# Recipe = wave 11's arm P0 exactly (plain DWPose + CTC, real clips only, 40 epochs, deterministic kernels, vocabulary
# built from the arm's own training manifest), run from the frozen wave-11 code snapshot. No trainer change.
# A MEASUREMENT: nothing is adopted. Manifests come from kaya/w12g_prep.sbatch:
#   P=$(sbatch --parsable kaya/w12g_prep.sbatch)
#   T=$(sbatch --parsable --dependency=afterok:$P --export=ALL,PREP=$P kaya/w12g_train.sbatch)
#   sbatch --dependency=afterany:$T --export=ALL,PREP=$P kaya/w12g_summary.sbatch
# Pre-registration: results/PREREG_WAVE12G_SIGNERSPLIT_2026-10-13.md
#SBATCH --job-name=w12g-train
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w11lib.sh
set -u                                   # after the library: `module` and the venv activation are not nounset-clean
ARMS=(SD EXPa EXPb); SEEDS=(42 101 202)
T=$SLURM_ARRAY_TASK_ID; arm=${ARMS[$((T % 3))]}; s=${SEEDS[$((T / 3))]}
W=$G/wave12g
MAN=$W/manifests_${PREP:?submit with --export=ALL,PREP=<prep job id> --dependency=afterok:<prep job id>}
RUNS=$W/runs; D=$RUNS/${arm}_s${s}; mkdir -p "$RUNS"
TR=$MAN/w12g_${arm}_train.tsv; VA=$MAN/w12g_val.tsv; TE=$MAN/w12g_${arm}_test.tsv
echo "task $T -> arm $arm seed $s | manifests $MAN"
[ "$DET" = "--deterministic True" ] || { echo "FAIL: this wave runs every arm with deterministic kernels"; exit 1; }
# the library takes POSE from the environment if it is set, and --export=ALL passes the submitting shell's variables on
[ "$POSE" = "--num_keypoints 133 --keypoint_subset all --coord_norm frame" ] || { echo "FAIL: POSE is not the P0 setting ($POSE)"; exit 1; }
[ ! -e "$RUNS/DONE_${arm}_s${s}" ] || { echo "FAIL: ${arm}_s${s} already finished; a result is never overwritten"; exit 1; }

# ---- gates: prep passed; manifests unchanged since then; the code is the wave-11 snapshot read on 13 Oct 2026 ----
[ -f "$MAN/BUILD_OK" ] || { echo "FAIL: gate $MAN/BUILD_OK missing"; exit 1; }
( cd "$MAN" && md5sum -c --quiet - <<MD5 ) || { echo "FAIL: a manifest changed after the prep job"; exit 1; }
a22c62a9e74c3ce2a756080f03a1b204  w12g_SD_train.tsv
c8fb5f854a277ed6760f1b6d65acb4a2  w12g_SD_test.tsv
09ee6f3ca47e72fc3a9c715569802b6d  w12g_EXPa_train.tsv
7a287f49776d69e0e66a11b3e5a44af7  w12g_EXPa_test.tsv
a0abd96db602e3df40a380bc1a2d7b72  w12g_EXPb_train.tsv
82133f7f1fd92d2809c41e66c84c6f85  w12g_EXPb_test.tsv
000a54bc2dab309e84da16bf73ca43f1  w12g_val.tsv
MD5
md5sum -c --quiet - <<MD5 || { echo "FAIL: the wave-11 code snapshot changed"; exit 1; }
75783cd2aaddb8d5822f0c5d1a4f10e5  train_ctc.py
1187c9ff21c8de97b6ba98dfedf3897f  slt/data.py
2e54dd6d2a5e3f99572e74bb086c749e  config.py
076d9b1ac5e0008a895b3a5bcfe4f470  slt/model.py
8399ac01b7f856e774b3f31d87dcf385  slt/engine.py
1a2ff663b374e0e4ce5c9dd3badf3b02  slt/metrics.py
6553f52a3f0fbeb57feef6510289d66c  tools/eval_beam.py
2fd20e3989b8e20a9857f6d90106da42  weight_average.py
MD5
pin_pose
assert_group "$TR" "$VA" "$TE"
# leak gate, independent of the split tool: no clip in two parts; no validation video in train or test;
# every training size is 12,024; signers are checked by the prep job (the spreadsheet is not needed here)
python - "$TR" "$VA" "$TE" <<'PY'
import os, re, sys
clips = [[os.path.basename(l.split("\t", 1)[0]) for l in open(p) if "\t" in l] for p in sys.argv[1:4]]
tr, va, te = (set(c) for c in clips)
assert all(len(set(c)) == len(c) for c in clips), "a clip occurs twice in a manifest"
assert not tr & va and not tr & te and not va & te, "a clip is in two of train / validation / test"
vid = lambda cs: {re.match(r"(video_\d+)_", c).group(1) for c in cs}
assert not vid(va) & vid(tr | te), "a validation video also has clips in train or test"
assert (len(tr), len(va)) == (12024, 928) and len(te) in (1087, 543, 544), (len(tr), len(va), len(te))
print(f"leak gate OK: train {len(tr)} | validation {len(va)} ({len(vid(va))} videos) | test {len(te)}")
PY

w4run "$arm" "$RUNS" "$TR" "$VA" "$TE" "$s" 40 -- || { echo "[FAIL] train ${arm}"; exit 1; }

# ---- the vocabulary of the checkpoint is the one built from this arm's training manifest, nothing else ----
python - "$D/best.pt" "$TR" <<'PY'
import sys, torch
from slt.data import Vocab
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
c = ck["cfg"]
txt = [l.rstrip("\n").split("\t")[1] for l in open(sys.argv[2]) if l.strip()]
want = Vocab.build(txt, c["min_freq"], c["text_tokenization"], c["sentencepiece_model"]).itos
assert ck["vocab"] == want, "the checkpoint vocabulary is not the one built from this arm's training manifest"
assert not c.get("vocab_file") and not c.get("init_encoder") and not c.get("resume_checkpoint"), "an outside vocabulary or checkpoint was used"
print(f"vocabulary: {len(want)} entries, built from the training manifest only (OK)")
PY
# w4run only prints a line when a decode fails, so a decode left by an earlier attempt could stand in for it:
# both decodes of best.pt must exist and be newer than the best.pt they decode
for f in beam5_val_best beam5_test_best; do
  if [ ! -s "$D/$f.json" ] || [ ! "$D/$f.json" -nt "$D/best.pt" ]; then echo "FAIL: $f.json missing or older than best.pt"; exit 1; fi
done
touch "$RUNS/DONE_${arm}_s${s}"
echo "W12G_DONE ${arm} seed ${s}"
