#!/bin/bash -l
# t2-vdens: does ENSEMBLING replicate on the video-disjoint split? Pre-registration: results/PREREG_WAVE4_2026-09-22.md §5.
# Members = the six VD-CONFIRM checkpoints (job 6596: vdA, vdB x seeds 42/101/202). Every configuration, singles
# included, goes through tools/ensemble_mbr_wer.py (beam 10, --remap_vocab); LP = argmax of a vd_val sweep, chosen in
# code BEFORE any vd_test decode, and only val-selected configurations are decoded on vd_test.
#SBATCH --job-name=t2-vdens
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
G=/group/pmc099/pprasad; cd "$G/code/phase2_baseline"
M=$G/scratch_backup/dwpose/manifests; R=$G/wave3_vd/runs; OUT=$G/t2_vdens; mkdir -p "$OUT"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"
LPS=0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
ck () { local o=""; for m in "$@"; do o="$o,$R/$m/best.pt"; done; echo "${o#,}"; }
declare -A CFG=( [vdA_s42]=$(ck vdA_s42) [vdA_s101]=$(ck vdA_s101) [vdA_s202]=$(ck vdA_s202)
                 [vdB_s42]=$(ck vdB_s42) [vdB_s101]=$(ck vdB_s101) [vdB_s202]=$(ck vdB_s202)
                 [V3A]=$(ck vdA_s42 vdA_s101 vdA_s202) [V3B]=$(ck vdB_s42 vdB_s101 vdB_s202)
                 [V6]=$(ck vdA_s42 vdA_s101 vdA_s202 vdB_s42 vdB_s101 vdB_s202) )
for c in "${!CFG[@]}"; do for f in ${CFG[$c]//,/ }; do [ -f "$f" ] || { echo "FAIL: missing $f"; exit 1; }; done; done
dec () {  # dec <cfg> <split manifest> <lps> <tag>
  python tools/ensemble_mbr_wer.py --ckpts "${CFG[$1]}" $POSE --remap_vocab --device cuda --beam 10 \
    --eval_manifest "$2" --lps "$3" --dump_dir "$OUT/dumps_${1}_$4" > "$OUT/${1}_$4.log" 2>&1 || { echo "[FAIL] $1 $4"; return 1; }
  grep -a "LP=" "$OUT/${1}_$4.log"
}
echo "=== STAGE 1: vd_val LP sweep for every configuration ==="
for c in vdA_s42 vdA_s101 vdA_s202 vdB_s42 vdB_s101 vdB_s202 V3A V3B V6; do echo "--- $c"; dec "$c" "$M/vd_val.tsv" "$LPS" val; done
echo "=== SELECTION (validation only) ==="
python - "$OUT" > "$OUT/selection.txt" <<'PY'
import re, sys, os
out = sys.argv[1]
def best(c):
    rows = [(float(m[1]), float(m[2])) for m in re.finditer(r"LP=([\d.]+)\s+BLEU-4=\s*([\d.]+)", open(f"{out}/{c}_val.log").read())]
    lp, b = max(rows, key=lambda r: r[1]); return lp, b
sel = {c: best(c) for c in ["vdA_s42","vdA_s101","vdA_s202","vdB_s42","vdB_s101","vdB_s202","V3A","V3B","V6"]}
single = lambda fam: max((c for c in sel if c.startswith(fam)), key=lambda c: sel[c][1])
picks = {"V6": "V6", "V3A": "V3A", "V3B": "V3B", "best_single": single("vd"), "best_vdA": single("vdA"), "best_vdB": single("vdB")}
for k, c in picks.items():
    print(f"{k} {c} {sel[c][0]} {sel[c][1]:.3f}")
PY
cat "$OUT/selection.txt"
echo "=== STAGE 2: ONE vd_test decode per val-selected configuration, at its val-argmax LP ==="
awk '{print $2, $3}' "$OUT/selection.txt" | sort -u | while read c lp; do echo "--- $c @ LP $lp"; dec "$c" "$M/vd_test.tsv" "$lp" test; done
echo "T2_VDENS_DONE"
