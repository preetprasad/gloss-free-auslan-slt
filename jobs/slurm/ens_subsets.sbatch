#!/bin/bash -l
# FULL-COVERAGE ABLATION of the reported six-model ensemble: all 63 non-empty member subsets, VALIDATION ONLY,
# decoded exactly as the headline (ensemble_mbr_wer n-best at beam 10, top-1 by raw / words**0.3). CPU, no training,
# no test decode. Pre-registration: results/PREREG_ENSEMBLE_ABLATION_2026-10-10.md (addendum).
#   decode:  sbatch --array=0-3 kaya/ens_subsets.sbatch
#   report:  sbatch --array=0 --dependency=afterok:<decode job> --export=ALL,MODE=report kaya/ens_subsets.sbatch
# Code: a SNAPSHOT of the local tools ($G/ens_ablation/code), never the shared tree (which lacks slt/preproc_guard.py).
#SBATCH --job-name=ens-subsets
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
module purge
source /group/pmc099/pprasad/envs/auslan/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
G=/group/pmc099/pprasad; E=$G/ens_ablation; M=$G/scratch_backup/dwpose/manifests; OUT=$E/out
NSH=4
cd "$E/code" || { echo "FAIL: snapshot missing"; exit 1; }
echo "node $(hostname)  mode ${MODE:-decode}  task $SLURM_ARRAY_TASK_ID"
md5sum -c "$E/code.md5" || { echo "FAIL: snapshot differs from the files that were verified locally"; exit 1; }
python tools/ensemble_subsets.py self_test

if [ "${MODE:-decode}" = report ]; then
  [ "$(ls "$OUT"/shard*of$NSH.json | wc -l)" = "$NSH" ] || { echo "FAIL: not all $NSH shards present"; exit 1; }
  python -u tools/ensemble_subsets.py report --out "$OUT" --names VLP80,RGBfus,VLP40,VLP40s202,fus101,fus202 \
      --types pose,fusion,pose,pose,fusion,fusion --expect 12.470 --expect_wer 0.9164 --boot 500
  echo "ENS_SUBSETS_REPORT_DONE"; exit 0
fi

VAL=$M/val.tsv; TEST=$M/test.tsv
[ "$(grep -c . "$VAL")" = 800 ] || { echo "FAIL: validation manifest row count"; exit 1; }
python - "$VAL" "$TEST" <<'PY'
import os, sys
v = [l.split("\t")[0] for l in open(sys.argv[1]) if "\t" in l]; t = {l.split("\t")[0] for l in open(sys.argv[2]) if "\t" in l}
assert not set(v) & t, f"{len(set(v) & t)} clips in both val and test"
assert all(p.startswith("/group/") and os.path.exists(p) for p in v), "validation clips must exist on /group"
print(f"val {len(v)} clips, all on /group, overlap with test 0")
PY
RGB=$G/scratch_backup/dwpose/rgb_feats
[ "$(find "$RGB" -maxdepth 1 -name '*.npz' | wc -l)" = 14041 ] || { echo "FAIL: RGB feature count"; exit 1; }
CK="$G/comm_dwpose_ctc_vlplong80/best.pt,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt,$G/comm_dwpose_ctc_s202/best.pt,$G/comm_dwpose_fus_s101/best.pt,$G/comm_dwpose_fus_s202/best.pt"
# the six reported members, byte-identical to the local copies (md5 checked 10 Oct 2026)
md5sum ${CK//,/ } | awk '{print $1}' | tr '\n' ' ' | grep -q "^3ea55e3474eaa9d1e093d6da300ad157 9b068067a1c4b90db5dcfd5b46d102b5 634d760e6e2eb09823992a03fa3b9c10 7146145c437014895a548e018888805c ccb09833497b81d541147103c2c382fd 2b3644b1b2c26b06641f0e6dbce1660b $" \
  || { echo "FAIL: checkpoint md5 mismatch"; exit 1; }
mkdir -p "$OUT"
python -u tools/ensemble_subsets.py decode --ckpts "$CK" --manifest "$VAL" --rgb_feat_dir "$RGB" --out "$OUT" \
    --beam 10 --lp 0.3 --shard "$SLURM_ARRAY_TASK_ID" --nshards $NSH
echo "ENS_SUBSETS_DECODE_DONE task $SLURM_ARRAY_TASK_ID"
