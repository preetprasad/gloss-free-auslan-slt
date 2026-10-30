#!/bin/bash -l
# WAVE 12e, step 2 of 2: the reported six-member ensemble (beam 10, LP 0.3) on people who are not signing.
# Pre-registration: results/PREREG_WAVE12E_NONSIGNER_ENSEMBLE_2026-10-13.md
# Order is the design: (1) code, checkpoint and input gates, (2) feature gates incl. G-RGB, (3) the ensemble must
# reproduce its recorded VALIDATION (12.470 / 0.9164, row by row against job 8001's dump) and TEST (11.802 / 0.9069)
# decodes, and ONLY THEN (4) the 1,678 non-signer crops and (5) the tracked-pose signers (arm ii) are decoded.
# Decoding uses the frozen snapshot of job 120153 ($G/ens_ablation/code, md5-checked), tools/ensemble_mbr_wer.py
# unmodified, with the command line of kaya/t1_enssel_s6val.sbatch. No statistic of the negatives is computed here:
# the analysis runs locally (tools/w12e_nonsigner_s6.py build, analyse) and refuses to run unless every gate passed.
# CPU by default (the recorded test decode, kaya/mbr_wer_test2.sbatch, was a CPU job; a CPU re-decode on the Mac
# reproduced the cuda validation dump of job 8001 on 800 of 800 rows). The pre-registration allows either device:
#   cd /group/pmc099/pprasad/code/phase2_baseline
#   sbatch --dependency=afterok:<w12e-extract job> kaya/w12e_decode.sbatch
#   on a GPU instead:  add  --partition=gpu --gres=gpu:1 --export=ALL,DEV=cuda
#   steps 1-2 only (minutes; nothing is decoded):  add  --export=ALL,GATES_ONLY=1
# A run that stops (time limit, node failure) is resubmitted unchanged: finished decode sets carry a DONE marker.
# The marker holds the device. A resubmission on the OTHER device decodes every set again, so the five sets of one
# result always come from one device (e.g. after a harness-gate failure on the CPU: resubmit with DEV=cuda).
#SBATCH --job-name=w12e-decode
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -euo pipefail
DEV=${DEV:-cpu}
module purge; if [ "$DEV" = cuda ]; then module load cuda/12.6.3; else module load Anaconda3/2024.06; fi   # as job 8001 / as mbr-wer-test2
source /group/pmc099/pprasad/envs/auslan/bin/activate
G=/group/pmc099/pprasad; B=$G/scratch_backup/dwpose; M=$B/manifests; MT=$B/manifests_tracked; W12=$G/wave12; W=$W12/w12e
TOOL=$W12/code/tools/w12e_nonsigner_s6.py; TOOL_MD5=5e934edae55f20e0142016332cbc7b8b   # the file tools/w12e_dryrun.py tested
DEC=$G/ens_ablation/code
GATES=$W/gates; DUMPS=$W/dumps; mkdir -p "$GATES" "$DUMPS"
RGBP=$B/rgb_feats; RGBN=$W/rgb_feats_nonsigner; RGBT=$W/rgb_feats_tracked
export W12E_CODE=$DEC OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "node $(hostname)  device $DEV  cpus $SLURM_CPUS_PER_TASK"

# ---- (1) code, checkpoints, pose
cd "$DEC" || { echo "FAIL: decode snapshot missing"; exit 1; }
md5sum -c --quiet "$G/ens_ablation/code.md5" || { echo "FAIL: decode snapshot differs from its md5 list"; exit 1; }
[ "$(md5sum "$TOOL" | cut -d' ' -f1)" = "$TOOL_MD5" ] || { echo "FAIL: $TOOL is missing or is not the reviewed file"; exit 1; }
CK="$G/comm_dwpose_ctc_vlplong80/best.pt,$G/comm_dwpose_rgb_fusion/best.pt,$G/comm_dwpose_vlp_ctc/best.pt,$G/comm_dwpose_ctc_s202/best.pt,$G/comm_dwpose_fus_s101/best.pt,$G/comm_dwpose_fus_s202/best.pt"
# the six reported members, in the recorded order (kaya/mbr_wer_test.sbatch), byte-identical to the local copies
md5sum ${CK//,/ } | awk '{print $1}' | tr '\n' ' ' | grep -q "^3ea55e3474eaa9d1e093d6da300ad157 9b068067a1c4b90db5dcfd5b46d102b5 634d760e6e2eb09823992a03fa3b9c10 7146145c437014895a548e018888805c ccb09833497b81d541147103c2c382fd 2b3644b1b2c26b06641f0e6dbce1660b $" \
  || { echo "FAIL: checkpoint md5 mismatch"; exit 1; }
[ "$(md5sum "$B/dwpose_clips/video_1_0.npz" | cut -d' ' -f1)" = 582483e5214cb1c0f7c6b40315fab625 ] \
  || { echo "FAIL: pinned production DWPose clip changed"; exit 1; }
[ "$(md5sum "$G/enssel/dumps_S6_val/nbest.jsonl" | cut -d' ' -f1)" = 13352e40a0927846fc787006bfa64091 ] \
  || { echo "FAIL: the saved validation dump of job 8001 changed"; exit 1; }
[ "$(find "$RGBP" -maxdepth 1 -name 'rgb_*.npz' | wc -l)" -eq 14041 ] || { echo "FAIL: stored RGB feature count"; exit 1; }
python "$TOOL" selftest
for k in 0 1 2 3 4 5 6 7; do
  [ -e "$W/DONE_extract_$k" ] || { echo "FAIL: extraction shard $k has no DONE marker"; exit 1; }
done

# ---- manifest of the negatives (text = crop id), split and leak checks, md5 of every pose file that is decoded
python "$TOOL" manifest --expected "$B/nonsigner_expected.tsv" --neg_dir "$B/dwpose_nonsigner" \
  --val_manifest "$M/val.tsv" --test_manifest "$M/test.tsv" --trk_val "$MT/val.tsv" --trk_test "$MT/test.tsv" \
  --neg_mp4_dir "$B/rgb_nonsigner" --sig_mp4_dir "$B/rgb_signer" --out_dir "$W"
[ "$(grep -c . "$W/negatives.tsv")" = 1678 ] || { echo "FAIL: negatives manifest row count"; exit 1; }

# ---- (2) feature gates: a missing or malformed file would silently become ONE ZERO FRAME inside the dataset
python "$TOOL" featcheck --manifest "$M/val.tsv"        --rgb_dir "$RGBP" --n 800  --name feat_sig_val  --gates "$GATES"
python "$TOOL" featcheck --manifest "$M/test.tsv"       --rgb_dir "$RGBP" --n 800  --name feat_sig_test --gates "$GATES"
python "$TOOL" featcheck --manifest "$W/negatives.tsv"  --rgb_dir "$RGBN" --n 1678 --name feat_neg      --gates "$GATES"
python "$TOOL" featcheck --manifest "$MT/val.tsv"       --rgb_dir "$RGBT" --n 800  --name feat_trk_val  --gates "$GATES"
python "$TOOL" featcheck --manifest "$MT/test.tsv"      --rgb_dir "$RGBT" --n 800  --name feat_trk_test --gates "$GATES"
# (rgb_<clip>.writing.npz is the extractor's temporary name: a task killed mid-write leaves one behind; the loader never reads it)
[ "$(find "$RGBN" -maxdepth 1 -name 'rgb_*.npz' ! -name '*.writing.npz' | wc -l)" -eq 1678 ] \
  && [ "$(find "$RGBT" -maxdepth 1 -name 'rgb_*.npz' ! -name '*.writing.npz' | wc -l)" -eq 1600 ] \
  || { echo "FAIL: unexpected files in the new feature folders"; exit 1; }
# G-RGB: today's extraction of 100 signer clips on the production pose must equal the stored August features
python "$TOOL" rgbgate --list "$W/ext_gate.tsv" --new "$W/rgb_gate" --stored "$RGBP" --gates "$GATES"
if [ "${GATES_ONLY:-0}" = 1 ]; then echo "W12E_GATES_DONE (code, input and feature gates passed; nothing was decoded)"; exit 0; fi
if [ "$DEV" = cuda ]; then
  python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "FAIL: DEV=cuda but this node has no GPU"; exit 1; }
fi

decode () {   # <name> <manifest> <rgb feature dir> <rows>
  local d=$DUMPS/$1
  if [ "$(cat "$d/DONE" 2>/dev/null)" = "$DEV" ]; then echo "decode $1: already done on $DEV"; return 0; fi
  mkdir -p "$d"; rm -f "$d/DONE"
  python tools/ensemble_mbr_wer.py --ckpts "$CK" --num_keypoints 133 --keypoint_subset all --coord_norm frame \
    --rgb_feat_dir "$3" --rgb_feature_dim 1152 --device "$DEV" --beam 10 \
    --eval_manifest "$2" --lps 0.3 --dump_dir "$d" > "$d/decode.log" 2>&1 \
    || { tail -n 25 "$d/decode.log"; echo "FAIL: decode $1"; exit 1; }
  grep -E "^\[(mbr|preproc)\]" "$d/decode.log" | grep -v "===" || true
  [ "$(grep -c . "$d/nbest.jsonl")" = "$4" ] || { echo "FAIL: decode $1 wrote the wrong number of rows"; exit 1; }
  echo "$DEV" > "$d/DONE"
}

# ---- (3) harness gate: the recorded validation and test decodes, before any non-signer crop is decoded
decode sig_val "$M/val.tsv" "$RGBP" 800
grep "  LP=0.3 " "$DUMPS/sig_val/decode.log"
python "$TOOL" dumpgate --new "$DUMPS/sig_val" --split val --saved "$G/enssel/dumps_S6_val" --name dump_sig_val --gates "$GATES"
decode sig_test "$M/test.tsv" "$RGBP" 800
grep "  LP=0.3 " "$DUMPS/sig_test/decode.log"
SAVED_TEST=${SAVED_TEST:-$W/saved/dumps_S6_test}     # optional copy of the local results/dumps_S6_test
if [ -s "$SAVED_TEST/nbest.jsonl" ]; then
  python "$TOOL" dumpgate --new "$DUMPS/sig_test" --split test --saved "$SAVED_TEST" --name dump_sig_test --gates "$GATES"
else
  echo "no saved test dump on Kaya: BLEU-4 and WER are gated here, the row-by-row comparison is made by the local build stage"
  python "$TOOL" dumpgate --new "$DUMPS/sig_test" --split test --name dump_sig_test --gates "$GATES"
fi

# ---- (4) the non-signer crops. Their references are placeholders (the crop id), so the BLEU / WER lines of the
#          decode log mean nothing and are not printed (amendment B13 of the September pre-registration).
decode neg "$W/negatives.tsv" "$RGBN" 1678
# ---- (5) arm (ii): the same signer clips on the person-tracked pose with features cropped on that pose
decode trk_val  "$MT/val.tsv"  "$RGBT" 800
decode trk_test "$MT/test.tsv" "$RGBT" 800

ls "$GATES"
touch "$W/DONE_decode"
echo "W12E_DECODE_DONE  (next: copy $W/{dumps,gates,negatives.tsv,negatives_meta.json,pose_facts.json} to the Mac, then build and analyse)"
