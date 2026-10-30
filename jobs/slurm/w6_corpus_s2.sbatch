#!/bin/bash -l
# WAVE 6 STAGE 2 -- survivors of Stage 1 at fresh seeds 303 / 404 with a fresh in-batch control A (PREREG_WAVE6 s4).
# Stage 1 (job 16435): B - A beam-5 val mean +0.010 (3 seeds) -> continues. C is decided when C_s202 lands
# (2 seeds so far: -0.159 / -1.325); if C survived it would get its own Stage-2 submission.
# Task = (arm, seed):  0-1 A | 2-3 B   (seeds 303 / 404). Identical to w6_corpus.sbatch otherwise.
# (Stage-1 header:) JOINT training at the adopted dose, not pre-training.
#   A = adopted corpus (in-batch control, 13,760 @ 1.1:1)   B = new full-dictionary corpus, 13,760 @ 1.1:1
#   C = new corpus, 27,520 @ 2.2:1 (does the ratio optimum move when the corpus is 1.9x wider in signs?)
# Everything else is the adopted recipe: pose-aug ON, 40 epochs, deterministic, one fixed vocabulary for every arm.
# Pre-registration: results/PREREG_WAVE6_CORPUS_2026-09-23.md
#SBATCH --job-name=w6-corpus-s2
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --array=0-3
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w5lib.sh
pin_pose
T=$SLURM_ARRAY_TASK_ID; SEEDS=(303 404); ARMS=(A B)
arm=${ARMS[$((T / 2))]}; s=${SEEDS[$((T % 2))]}
W6=$G/wave6; MW=$W5/manifests; RUNS=$W6/runs; mkdir -p "$W6" "$RUNS"
ADOPTED=$G/stitch_scale2/joint_13760.tsv; V="--vocab_file $MW/vocab.json"
echo "task $T -> arm $arm seed $s"
[ -f "$MW/vocab.json" ] || { echo "FAIL: fixed vocabulary missing"; exit 1; }

# --- build this task's joint manifest (deterministic; per-task file, so array tasks never race) ---
JOINT=$W6/joint_${arm}_s${s}.tsv
case "$arm" in
  A) cp "$ADOPTED" "$JOINT"; WANT=26199 ;;
  B) { head -n 12439 "$ADOPTED"; head -n 13760 "$MW/syn_300000.tsv"; } > "$JOINT"; WANT=26199 ;;
  C) { head -n 12439 "$ADOPTED"; head -n 27520 "$MW/syn_300000.tsv"; } > "$JOINT"; WANT=39959 ;;
esac
GOT=$(grep -c . "$JOINT")
[ "$GOT" = "$WANT" ] || { echo "FAIL: $arm manifest has $GOT rows, expected $WANT"; exit 1; }
# the real part must be byte-identical across arms, and no synthetic row may equal a val/test reference
python - "$JOINT" "$ADOPTED" "$M/val.tsv" "$M/test.tsv" "$arm" <<'PY'
import sys
joint, adopted, val, test, arm = sys.argv[1:6]
rows = [l.rstrip("\n").split("\t", 1) for l in open(joint) if "\t" in l]
real = [l.rstrip("\n") for l in open(adopted) if "\t" in l][:12439]
assert [f"{p}\t{t}" for p, t in rows[:12439]] == real, "real part differs from the adopted corpus"
norm = lambda s: " ".join(s.lower().split())
ref = {norm(l.split("\t", 1)[1]) for f in (val, test) for l in open(f) if "\t" in l}
leak = sorted({norm(t) for _, t in rows[12439:]} & ref)
print(f"[{arm}] rows {len(rows)} | real 12439 identical | synthetic {len(rows)-12439} | "
      f"distinct synthetic sentences {len({t for _, t in rows[12439:]})} | val/test leaks {len(leak)} {leak[:2]}")
if arm != "A":
    assert not leak, f"{arm} leaks {len(leak)} val/test references"   # A's single known test match is disclosed
PY
ALLOW_PREFIX="$W5S/corpora/" assert_group "$JOINT" "$M/val.tsv" "$M/test.tsv"

w4run "$arm" "$RUNS" "$JOINT" "$M/val.tsv" "$M/test.tsv" "$s" 40 -- $AUG_DEF $V \
  || { echo "[FAIL] train ${arm}"; exit 1; }
D=$RUNS/${arm}_s${s}
for f in beam5_val_best beam5_test_best; do [ -s "$D/$f.json" ] || { echo "FAIL: $f.json missing"; exit 1; }; done
touch "$D/DONE"
echo "W6_DONE ${arm} seed ${s}"
