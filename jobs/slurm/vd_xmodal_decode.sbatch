#!/bin/bash -l
# G1 decode (results/PREREG_VD_XMODAL_2026-09-25.md). CPU. Each new fusion member alone, and VD5 = vdB s42/101/202 +
# fus_s303 + fus_s404, on vd_val then vd_test (beam 10, LP sweep); the analysis reads test at each system's vd_val LP.
#SBATCH --job-name=vd-xmodal
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
module purge; source /group/pmc099/pprasad/envs/auslan/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
G=/group/pmc099/pprasad; M=$G/scratch_backup/dwpose/manifests; X=$G/vd_xmodal; OUT=$X/decode; mkdir -p "$OUT"
cd "$G/seedens/code"; md5sum tools/ensemble_mbr_wer.py slt/preproc_guard.py ensemble_decode.py
for s in 303 404; do [ -f "$X/runs/fus_s$s/DONE" ] || { echo "FAIL: fus_s$s did not pass its gate"; exit 1; }; done
[ "$(grep -c . $M/vd_val.tsv)" = 1444 ] && [ "$(grep -c . $M/vd_test.tsv)" = 1254 ] || { echo "FAIL: vd row counts"; exit 1; }
python - "$M/vd_val.tsv" "$M/vd_test.tsv" <<'PY2'
import sys
v = {l.split("\t")[0] for l in open(sys.argv[1]) if "\t" in l}; t = {l.split("\t")[0] for l in open(sys.argv[2]) if "\t" in l}
assert not v & t, f"{len(v & t)} clips in both vd_val and vd_test"; print(f"vd_val {len(v)} / vd_test {len(t)} / overlap 0")
PY2
F="--num_keypoints 133 --keypoint_subset all --coord_norm frame --max_src_frames 256 --rgb_feat_dir $G/scratch_backup/dwpose/rgb_feats --rgb_feature_dim 1152"
LPS=0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
dec () {   # dec <name> <ckpts> <manifest> <split>
  local d="$OUT/$1_$4"
  python tools/ensemble_mbr_wer.py --ckpts "$2" --eval_manifest "$3" $F --beam 10 --lps $LPS --remap_vocab --device cpu \
    --dump_dir "$d" > "$d.log" 2>&1 || { echo "FAIL: $1 $4"; tail -20 "$d.log"; exit 1; }
  grep -q "LP=1.0" "$d.log" || { echo "FAIL: no LP sweep in $d.log"; exit 1; }
  echo "done $1 $4: $(grep 'rgb_feat_dir=' "$d.log" | head -1)"
}
V=$G/wave3_vd/runs; FU=$X/runs
VD5="$V/vdB_s42/best.pt,$V/vdB_s101/best.pt,$V/vdB_s202/best.pt,$FU/fus_s303/best.pt,$FU/fus_s404/best.pt"
for sp in val test; do
  if [ $sp = val ]; then MF=$M/vd_val.tsv; else MF=$M/vd_test.tsv; fi
  dec fus_s303 "$FU/fus_s303/best.pt" "$MF" $sp
  dec fus_s404 "$FU/fus_s404/best.pt" "$MF" $sp
  dec VD5 "$VD5" "$MF" $sp
done
touch "$OUT/DONE"; echo "VD_XMODAL_DECODE_DONE"
