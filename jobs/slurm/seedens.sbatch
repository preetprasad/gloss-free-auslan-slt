#!/bin/bash -l
# SEED-ENSEMBLE REPLICATION -- does ensembling three seeds of ONE recipe beat the validation-selected single seed,
# across independent training jobs? Decode only (CPU), zero training. Pre-registration:
# results/PREREG_SEEDENS_2026-09-24.md.  Task = batch:  0 pa_aug | 1 w1A | 2 w3C | 3 r3_111 | 4 a3ctl | 5 w4A |
# 6 w5C (official split, seeds 42/101/202)  |  7 vdB (video-disjoint split, seeds 42/101/202/303/404).
# Every system (each member alone, and the ensemble) is decoded once per split at beam 10 by the SAME tool, with
# the LP sweep dumped; the analysis picks each system's LP on validation in code, then reads test at that LP.
# Code: a SNAPSHOT of the local tools ($G/seedens/code, with slt/preproc_guard.py), never the shared tree.
#SBATCH --job-name=seedens
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --array=0-7
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
module purge
source /group/pmc099/pprasad/envs/auslan/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
G=/group/pmc099/pprasad; SE=$G/seedens; M=$G/scratch_backup/dwpose/manifests
cd "$SE/code" || { echo "FAIL: snapshot missing"; exit 1; }
[ -f slt/preproc_guard.py ] || { echo "FAIL: snapshot lacks the preprocessing guard"; exit 1; }
echo "node $(hostname)"; md5sum config.py slt/data.py tools/ensemble_mbr_wer.py ensemble_decode.py tools/eval_beam.py
T=$SLURM_ARRAY_TASK_ID
NAMES=(pa_aug w1A w3C r3_111 a3ctl w4A w5C vdB)
DIRS=(pose_aug/runs/pa_aug wave1_r2/runs/w1A wave3_track/runs/w3C wave3_ratioaug/runs/r3_111 \
      wave3_augdose/runs/a3ctl wave4/runs_tq/w4A wave5/runs/C wave3_vd/runs/vdB)
name=${NAMES[$T]}; base=$G/${DIRS[$T]}
if [ "$name" = vdB ]; then SEEDS=(42 101 202 303 404); VAL=$M/vd_val.tsv; TEST=$M/vd_test.tsv; NV=1444; NT=1254
else SEEDS=(42 101 202); VAL=$M/val.tsv; TEST=$M/test.tsv; NV=800; NT=800; fi
[ "$(grep -c . "$VAL")" = "$NV" ] && [ "$(grep -c . "$TEST")" = "$NT" ] || { echo "FAIL: manifest row counts"; exit 1; }
python - "$VAL" "$TEST" <<'PY'
import sys
v = {l.split("\t")[0] for l in open(sys.argv[1]) if "\t" in l}; t = {l.split("\t")[0] for l in open(sys.argv[2]) if "\t" in l}
assert not v & t, f"{len(v & t)} clips in both val and test"; print(f"val {len(v)} / test {len(t)} / overlap 0")
PY
CKS=(); for s in "${SEEDS[@]}"; do f=$base"_s$s/best.pt"; [ -s "$f" ] || { echo "FAIL: missing $f"; exit 1; }; CKS+=("$f"); done
OUT=$SE/out/$name; mkdir -p "$OUT"

# Gate 2 (task 0 only): the single-model harness reproduces a recorded number in this environment.
if [ "$T" = 0 ]; then
  python tools/eval_beam.py --ckpt "${CKS[0]}" --manifest "$TEST" --beam 5 --device cpu \
    --output_json "$OUT/gate_pa_aug_s42_beam5_test.json" > "$OUT/gate.log" 2>&1
  python - "$OUT/gate_pa_aug_s42_beam5_test.json" <<'PY'
import json, sys
b = json.load(open(sys.argv[1]))["beam_result"]["metrics"]["tokenize=13a;detok=True"]
print(f"GATE pa_aug_s42 beam-5 test {b:.3f} (recorded 11.443)"); assert abs(b - 11.443) < 0.01, "harness does not reproduce"
PY
fi

LPS=0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame --max_src_frames 256"
dec () {   # dec <system> <ckpts> <manifest> <split>
  local d="$OUT/$1_$4"
  python tools/ensemble_mbr_wer.py --ckpts "$2" --eval_manifest "$3" $POSE --beam 10 --lps $LPS --device cpu \
    --dump_dir "$d" > "$d.log" 2>&1 || { echo "FAIL: decode $1 $4"; tail -20 "$d.log"; exit 1; }
  grep -q "LP=1.0" "$d.log" || { echo "FAIL: no LP sweep in $d.log"; exit 1; }
  echo "done $1 $4: $(grep 'LP=0.3 ' "$d.log" | head -1)"
}
for split in val test; do
  if [ "$split" = val ]; then MF=$VAL; else MF=$TEST; fi
  for i in "${!SEEDS[@]}"; do dec "s${SEEDS[$i]}" "${CKS[$i]}" "$MF" "$split"; done
  dec "ens${#SEEDS[@]}" "$(IFS=,; echo "${CKS[*]}")" "$MF" "$split"
done
touch "$OUT/DONE"
echo "SEEDENS_DONE $name"
