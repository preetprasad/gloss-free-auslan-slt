#!/bin/bash -l
# ENSEMBLE SEARCH around the NEW best operating point, with the two defects of earlier sweeps fixed.
#
# 2026-09-10 established a verified improvement: F = {dur42, dur101, dur202, rgb_fusion, fus_s101}
# @ beam-10/LP0.7 scores **12.114** on the official test against S6's 11.802 (reproduced exactly).
# Validation 13.246. Two things the previous sweeps got wrong and this job fixes:
#   (1) --rgb_feat_dir was never passed, so every fusion member was fed a single ZERO frame. X1_S6
#       moved 11.707 -> 12.276 from the fix alone.
#   (2) the LP grid started at 0.6, which MISSES S6's own optimum at LP0.3 (val 12.470). Grid here
#       runs 0.2-1.0 so no candidate is scored away from its peak.
#
# WHAT IS UNTESTED AND WORTH TESTING. `ens-ext` covers S6, DUR3, DUR6, S6+DUR3 and a diversity mix,
# but NOTHING combining SIX duration members with fusion members -- even though (a) F won with THREE
# duration + 2 fusion, and (b) pose-only DUR6 beat DUR3 (12.779 vs 12.472). The natural extension of
# the winner is therefore unexplored.
#
# Candidates (validation only; Stage 2 stays manual and single-shot):
#   G1  DUR6 + 2 fusion      -- the direct extension of F
#   G2  DUR6 + 3 fusion      -- does a third fusion member keep paying
#   G3  DUR3 + 3 fusion      -- F plus one fusion member, holding the duration side fixed
#   G4  DUR6 + S6            -- everything
#   G5  DUR3 + 2 fusion (=F) -- IN-BATCH reproduction of the winner, so G1-G4 are compared to F
#                               measured in the same run rather than across runs
# G5 is not redundant: every cross-run comparison this session has needed an in-batch control, and
# two of them turned out to be measuring different things.
#SBATCH --job-name=ens-search
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -e
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
pip install -q jiwer 2>/dev/null || true
G=/group/pmc099/pprasad; B=$G/scratch_backup
cd "$G/code/phase2_baseline"
OUT=$G/ens_search; M="$OUT/manifests"; mkdir -p "$M" "$OUT/dumps"

SRC=/scratch/pmc099/pprasad/dwpose/manifests
for f in val test; do
  if [ -f "$SRC/$f.tsv" ]; then
    sed "s#/scratch/pmc099/pprasad/dwpose/dwpose_clips#$B/dwpose/dwpose_clips#g" "$SRC/$f.tsv" > "$M/$f.tsv"
  else sed "s#/scratch/pmc099/pprasad/dwpose/dwpose_clips#$B/dwpose/dwpose_clips#g" "$B/dwpose/manifests/$f.tsv" > "$M/$f.tsv"; fi
done
python - "$M" <<'PYCHK'
import os, sys
r = sys.argv[1]
g = lambda f: [l.split("\t")[0] for l in open(os.path.join(r, f + ".tsv")) if "\t" in l]
v, t = g("val"), g("test")
ov = len(set(map(os.path.basename, v)) & set(map(os.path.basename, t)))
print(f"val {len(v)} | test {len(t)} | overlap {ov}")
sys.exit(1 if (ov or len(t) != 800 or len(v) != 800) else 0)
PYCHK

R="$B/stitch_meta/stitch_v2/runs"
DUR3="$R/tmp_dur/best.pt,$R/dur_joint_s101/best.pt,$R/dur_joint_s202/best.pt"
DUR6="$DUR3,$R/dur_joint_s303/best.pt,$R/dur_joint_s404/best.pt,$R/dur_joint_s505/best.pt"
FUS2="$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_fus_s101/best.pt"
FUS3="$FUS2,$G/comm_dwpose_fus_s202/best.pt"
S6="$G/comm_dwpose_ctc_vlplong80/best.pt,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt,$G/comm_dwpose_ctc_s202/best.pt,$G/comm_dwpose_fus_s101/best.pt,$G/comm_dwpose_fus_s202/best.pt"
RGB="$B/dwpose/rgb_feats"
for f in $(echo "$DUR6,$FUS3,$S6" | tr "," " "); do [ -f "$f" ] || { echo "FAIL: missing $f"; exit 1; }; done
n=$(find "$RGB" -name "rgb_*.npz" | wc -l); echo "rgb features: $n"; [ "$n" -ge 14000 ] || { echo "FAIL: rgb features missing"; exit 1; }

val () {
  echo "=== CANDIDATE $1 ==="
  python tools/ensemble_mbr_wer.py --ckpts "$2" --beam 10 --device cuda --remap_vocab \
    --lps 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    --rgb_feat_dir "$RGB" --rgb_feature_dim 1152 \
    --eval_manifest "$M/val.tsv" --test_manifest "$M/val.tsv" \
    --num_keypoints 133 --keypoint_subset all --coord_norm frame || echo "  (candidate failed)"
}
val G5_DUR3_FUS2_eqF "$DUR3,$FUS2"
val G1_DUR6_FUS2     "$DUR6,$FUS2"
val G2_DUR6_FUS3     "$DUR6,$FUS3"
val G3_DUR3_FUS3     "$DUR3,$FUS3"
val G4_DUR6_S6       "$DUR6,$S6"

echo
echo "REFERENCE, measured in-environment on 2026-09-10:"
echo "  F (=G5) validation 13.246 @ LP0.7  ->  official TEST 12.114 / WER 0.9526"
echo "  S6      validation 12.470 @ LP0.3  ->  official TEST 11.802 / WER 0.9069"
echo "G5 MUST land near 13.246; if it does not, something in this batch differs and G1-G4 are not"
echo "comparable to the recorded winner. Stage 2 (one locked test) remains manual."
echo "ENS_SEARCH_DONE"
