#!/bin/bash -l
# WAVE 12i -- confirmatory test of the wave-9 heavy-switch contrast: on the clips where the stored pose often follows a
# bystander, does a model trained and decoded on tracker-v2 pose translate better than one on the stored (detection-0)
# pose, relative to the unaffected clips? Task = (arm, seed): 0-4 C0 | 5-9 V2, seeds 303/404/505/606/707 (none was
# used in wave 9). Trainer, recipe, pose files, manifests, vocabulary and code snapshot are those of wave 9 (job 43506,
# kaya/w9_tracker.sbatch): only the seeds are new. Each arm is trained AND decoded on its own pose.
# Pre-registration: results/PREREG_WAVE12I_TRACKER_HEAVY_2026-10-13.md. Read-out: kaya/w12i_tracker_heavy_summary.sbatch.
# A failed task is resubmitted only after its run directory has been moved aside by hand (see the gate below).
#SBATCH --job-name=w12i-heavy
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --array=0-9
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -euo pipefail
# The wave-5 snapshot is the one wave 9 ran from. `module` and `activate` are not written for nounset.
set +u; source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w5lib.sh; set -u
# The harness library is the file wave 9 sourced (dated 22 Sep on Kaya), and neither of its two variables that the
# submitting shell's environment can override (sbatch exports the environment) was overridden.
echo "c641eb329187d3bf3ca4ce8656e54709  /group/pmc099/pprasad/code/phase2_baseline/kaya/_w5lib.sh" | md5sum -c --quiet - \
    || { echo "FAIL: kaya/_w5lib.sh is not the library wave 9 sourced"; exit 1; }
[ "$DET" = "--deterministic True" ] && [ "$POSE" = "--num_keypoints 133 --keypoint_subset all --coord_norm frame" ] \
    || { echo "FAIL: DET or POSE was overridden from the environment (DET='$DET' POSE='$POSE')"; exit 1; }
pin_pose
T=$SLURM_ARRAY_TASK_ID; SEEDS=(303 404 505 606 707); ARMS=(c0 v2)
arm=${ARMS[$((T / 5))]}; s=${SEEDS[$((T % 5))]}
if [ "$arm" = c0 ]; then other=v2; else other=c0; fi
W9=$G/wave9; MW=$W9/manifests; PW=$G/scratch_backup/dwpose/pose_w9
RUNS=$G/wave12/w12i/runs; DIR=$RUNS/${arm}_s${s}; GATE=$RUNS/_gate
echo "task $T -> arm $arm seed $s"

# Gate 1: the code is byte-identical to what wave 9 ran (first four md5s are printed in the wave-9 logs).
md5sum -c --quiet - <<'EOF' || { echo "FAIL: the wave-5 snapshot is not the code wave 9 ran"; exit 1; }
2e54dd6d2a5e3f99572e74bb086c749e  config.py
1187c9ff21c8de97b6ba98dfedf3897f  slt/data.py
75783cd2aaddb8d5822f0c5d1a4f10e5  train_ctc.py
6553f52a3f0fbeb57feef6510289d66c  tools/eval_beam.py
2fd20e3989b8e20a9857f6d90106da42  weight_average.py
e63a6db76161406c3ab3cb04be0cbd05  slt/__init__.py
8399ac01b7f856e774b3f31d87dcf385  slt/engine.py
1a2ff663b374e0e4ce5c9dd3badf3b02  slt/metrics.py
076d9b1ac5e0008a895b3a5bcfe4f470  slt/model.py
d41145cd1acabb5a89349e6b6a5c51ef  slt/reference_model.py
EOF
echo "code gate OK: 10 files = the wave-9 code"

# Gate 2: the inputs are the files wave 9 read (prep job 31736), unchanged since.
[ -f "$W9/PREP_OK_31736" ] || { echo "FAIL: wave-9 prep marker missing"; exit 1; }
md5sum -c --quiet - <<EOF || { echo "FAIL: a wave-9 manifest, the vocabulary or a pinned pose file changed"; exit 1; }
9c042dcc10e8763b14af9b06cb423727  $W5/manifests/vocab.json
150fecaa4bad312cfe359385e9635ad5  $MW/joint_c0.tsv
100a4516b1b7b295a76ec4c53f470e92  $MW/joint_v2.tsv
df6cf8e8693825a09fa2e7aff8744e65  $MW/c0_val.tsv
f7af237641d9bcf2226789ce170af5ae  $MW/c0_test.tsv
493f59e68a93899728c8278f175e6946  $MW/v2_val.tsv
dd6040a6f9652f4d8ea300f75e5c2491  $MW/v2_test.tsv
17fbbf7b8fecc58afda5e962691043ef  $PW/c0/video_1_0.npz
05355264fedc1cfad2f1eba4c4529ea2  $PW/v2/video_1_0.npz
38d474cb4ac9eed7114be3f95a2e6ad6  $PW/c0/video_89_193.npz
3ede2dee5a11e2e4f523da7b4d404ca8  $PW/v2/video_89_193.npz
EOF
for a in c0 v2; do
  n=$(find "$PW/$a" -type f -name '*.npz' | wc -l); new=$(find "$PW/$a" -type f -newer "$W9/PREP_OK_31736" | wc -l)
  [ "$n" -eq 14039 ] && [ "$new" -eq 0 ] || { echo "FAIL: pose_w9/$a has $n clips, $new changed since the wave-9 prep"; exit 1; }
done
echo "input gate OK: 7 files and 4 pose clips by md5; 14,039 clips per arm, none changed since the wave-9 prep"
python - "$MW" <<'PY' || { echo "FAIL: manifest gate"; exit 1; }
import sys
mw = sys.argv[1]
def rows(n): return [l.rstrip("\n").split("\t", 1) for l in open(f"{mw}/{n}.tsv", encoding="utf-8") if "\t" in l]
name = lambda p: p.rsplit("/", 1)[1]
R = {n: rows(n) for n in ("joint_c0", "joint_v2", "c0_val", "c0_test", "v2_val", "v2_test")}
assert [len(R[n]) for n in R] == [26199, 26199, 800, 800, 800, 800], "row counts"
for a, b in (("joint_c0", "joint_v2"), ("c0_val", "v2_val"), ("c0_test", "v2_test")):
    assert [(name(p), t) for p, t in R[a]] == [(name(p), t) for p, t in R[b]], f"{a} and {b} differ in clip order or English text"
for arm in ("c0", "v2"):
    real = [p for p, _ in R[f"joint_{arm}"][:12439] + R[f"{arm}_val"] + R[f"{arm}_test"]]
    assert all(f"/pose_w9/{arm}/" in p for p in real), f"{arm}: a real clip is not read from pose_w9/{arm}"
    assert all("/stitch_scale2/corpus/clips/" in p for p, _ in R[f"joint_{arm}"][12439:]), f"{arm}: synthetic half"
tr, va, te = ({name(p) for p, _ in R[n]} for n in ("joint_c0", "c0_val", "c0_test"))
assert len(va) == len(te) == 800 and not (tr & va) and not (tr & te) and not (va & te), "train / validation / test clips overlap"
print("manifest gate OK: 26,199 / 800 / 800 rows per arm; clip order and English text identical across arms; "
      "each arm reads its own pose; no clip is shared between training, validation and test")
PY
assert_group "$MW/joint_$arm.tsv" "$MW/c0_val.tsv" "$MW/c0_test.tsv" "$MW/v2_val.tsv" "$MW/v2_test.tsv"

# Gate 3: new seeds only, and nothing is overwritten.
[ ! -e "$W9/runs/${arm}_s${s}" ] || { echo "FAIL: seed $s was already run in wave 9"; exit 1; }
[ ! -e "$DIR" ] || { echo "FAIL: $DIR exists. Move it aside by hand before resubmitting this task."; exit 1; }
mkdir -p "$GATE"

# Gate 4: this node, this snapshot and these pose files reproduce a recorded wave-9 number before anything is trained
# (wave-9 ${arm}_s42 best.pt, beam-5 validation: c0 11.145, v2 11.107).
python tools/eval_beam.py --ckpt "$W9/runs/${arm}_s42/best.pt" --manifest "$MW/${arm}_val.tsv" --beam 5 --device cuda \
    --output_json "$GATE/w9_${arm}_s42_val_task${T}.json" > "$GATE/w9_${arm}_s42_val_task${T}.log" 2>&1 \
    || { echo "FAIL: could not re-decode the wave-9 checkpoint"; tail -5 "$GATE/w9_${arm}_s42_val_task${T}.log"; exit 1; }
python - "$GATE/w9_${arm}_s42_val_task${T}.json" "$W9/runs/${arm}_s42/beam5_val_best.json" <<'PY' || { echo "FAIL: wave-9 reproduction gate"; exit 1; }
import json, sys
new, old = (json.load(open(p)) for p in sys.argv[1:3])
K = "tokenize=13a;detok=True"
a, b = new["beam_result"]["metrics"][K], old["beam_result"]["metrics"][K]
diff = sum(x != y for x, y in zip(new["beam_hyps"], old["beam_hyps"]))
print(f"wave-9 reproduction: beam-5 validation BLEU-4 {a:.3f} now, {b:.3f} recorded; {diff}/{len(old['beam_hyps'])} hypotheses differ")
sys.exit(0 if abs(a - b) <= 0.10 and new["refs"] == old["refs"] and len(new["beam_hyps"]) == 800 else 1)
PY

# Training and decoding: the wave-9 call, unchanged (w4run also checks every passed flag against the saved cfg).
w4run "$arm" "$RUNS" "$MW/joint_$arm.tsv" "$MW/${arm}_val.tsv" "$MW/${arm}_test.tsv" "$s" 40 -- \
    $AUG_DEF --vocab_file "$W5/manifests/vocab.json" || { echo "[FAIL] train ${arm}"; exit 1; }
for f in beam5_val_best beam5_test_best; do [ -s "$DIR/$f.json" ] || { echo "FAIL: $f.json missing"; exit 1; }; done
# The run counts from here: everything the pre-registered statistic reads exists. DONE is written BEFORE the secondary
# decodes, so a node failure or a time-out during them cannot void a finished run and force it to be trained again.
touch "$DIR/DONE"

# Secondary read-out only: the same best.pt decoded on the OTHER arm's pose (separates the pose at decoding time from
# the pose at training time). A failure here does not fail the arm.
for split in val test; do
  python tools/eval_beam.py --ckpt "$DIR/best.pt" --manifest "$MW/${other}_${split}.tsv" --beam 5 --device cuda \
      --output_json "$DIR/beam5_${split}_best_on_${other}.json" > "$DIR/beam5_${split}_best_on_${other}.log" 2>&1 \
      || echo "[WARN] cross-pose decode ${split} failed (secondary read-out only)"
done
echo "W12I_DONE ${arm} seed ${s}"
