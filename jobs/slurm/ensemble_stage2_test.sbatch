#!/bin/bash -l
# STAGE 2 — ONE locked official-test evaluation per configuration, each at its OWN validation argmax.
#
# Stage 1b (1196605) swept LP 0.2-1.0 on validation:
#   A_S6                  peak 11.707 @ LP0.6
#   D_S6+dur_x3+vlp       peak 12.770 @ LP0.7
#   F_dur_x3+rgbfus       peak 12.831 @ LP0.7   <- overall validation argmax
#
# TWO configurations are evaluated on test, not one, and the second is NOT a fishing expedition:
#   F at LP0.7 -- the validation winner, the actual claim.
#   A_S6 at LP0.6 -- the IN-BATCH CONTROL, which is mandatory here. Historical S6 is recorded at
#     val 12.470 / test 11.802 (LP0.3), but in THIS run the identical checkpoints peak at 11.707 and
#     score only 11.047 at LP0.3. A ~0.76 validation gap on unchanged checkpoints means the historical
#     test number is not a safe comparator, and this project's standing rule is to compare against the
#     in-batch control rather than across batches. Without A_S6 on test, F's test number cannot be
#     interpreted at all.
# Each LP is pinned with --lps (no sweeping on test). Hypotheses are dumped so the operating point can
# finally be error-analysed -- the project log has recorded since 2026-09-05 that it cannot be.
#
# A local pre-check already showed validation parity does NOT imply test parity for this family:
# E_dur_x3_only matched S6 on validation (12.472 vs 12.470) and lost by 0.27 on test (11.537 vs 11.802).
# So F's +1.12 validation margin over the in-batch S6 is a hypothesis to test, not a result.
#SBATCH --job-name=ens-s2test
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
OUT=$G/ens_stage2; M="$OUT/manifests"; mkdir -p "$OUT/dumps"

# manifests were built and clip-verified by 1196605; rebuild defensively and re-assert
SRC=/scratch/pmc099/pprasad/dwpose/manifests
for f in val test; do
  if [ -f "$SRC/$f.tsv" ]; then
    sed "s#/scratch/pmc099/pprasad/dwpose/dwpose_clips#$B/dwpose/dwpose_clips#g" "$SRC/$f.tsv" > "$M/$f.tsv"
  else sed "s#/scratch/pmc099/pprasad/dwpose/dwpose_clips#$B/dwpose/dwpose_clips#g" "$B/dwpose/manifests/$f.tsv" > "$M/$f.tsv"; fi
done
# assert test is the TEST split and disjoint from val -- a manifest that silently held val clips would
# report a validation score as an official test score, the worst error available here.
python - "$M" <<'PYCHK'
import os, sys
r = sys.argv[1]
g = lambda f: [l.split("\t")[0] for l in open(os.path.join(r, f + ".tsv")) if "\t" in l]
v, t = g("val"), g("test")
ov = len(set(map(os.path.basename, v)) & set(map(os.path.basename, t)))
miss = sum(1 for p in t if not os.path.exists(p))
print(f"val {len(v)} | test {len(t)} | overlap {ov} | missing test clips {miss}")
sys.exit(1 if (ov or miss or len(t) != 800) else 0)
PYCHK

R="$B/stitch_meta/stitch_v2/runs"
D42="$R/tmp_dur/best.pt"; D101="$R/dur_joint_s101/best.pt"; D202="$R/dur_joint_s202/best.pt"
S6="$G/comm_dwpose_ctc_vlplong80/best.pt,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt,$G/comm_dwpose_ctc_s202/best.pt,$G/comm_dwpose_fus_s101/best.pt,$G/comm_dwpose_fus_s202/best.pt"
F_SET="$D42,$D101,$D202,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_fus_s101/best.pt"
POSE="--num_keypoints 133 --keypoint_subset all --coord_norm frame"

echo "########## STAGE 2 — locked official test ##########"
echo "=== TEST A_S6 @ LP 0.6 (in-batch control, its own validation argmax 11.707) ==="
python tools/ensemble_mbr_wer.py --ckpts "$S6" --beam 10 --device cuda --remap_vocab \
  --lps 0.6 --eval_manifest "$M/test.tsv" --test_manifest "$M/test.tsv" $POSE \
  --dump_dir "$OUT/dumps/A_S6_test"

echo "=== TEST F_dur_x3_plus_rgbfus @ LP 0.7 (validation argmax 12.831) ==="
python tools/ensemble_mbr_wer.py --ckpts "$F_SET" --beam 10 --device cuda --remap_vocab \
  --lps 0.7 --eval_manifest "$M/test.tsv" --test_manifest "$M/test.tsv" $POSE \
  --dump_dir "$OUT/dumps/F_test"

echo
echo "READ AS: F beats the IN-BATCH A_S6 -> a real ensemble gain from duration members."
echo "         F <= in-batch A_S6        -> the validation margin did not convert (as E already warned)."
echo "historical context only, NOT the comparator: S6 beam-10/LP0.3 = 11.802 / WER 0.907"
echo "local pre-check: E_dur_x3_only val 12.472 -> test 11.537 (val parity did NOT imply test parity)"
echo "ENS_S2TEST_DONE"
