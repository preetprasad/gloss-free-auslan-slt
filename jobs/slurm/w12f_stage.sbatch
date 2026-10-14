#!/bin/bash -l
# WAVE 12f, step 1 of 3 (CPU, no GPU): stage Auslan-Daily's isolated-sign clips and find where each was cut from.
#   1. fetch New_Gloss_RGB_Data.zip (1.47 GB), its index and Sign_Spotting.xlsx from Preet's own Drive mirror
#      (rclone, as kaya/stage_nonsigner.sbatch); sha256 of each file is checked. A file already in $RAW with the
#      right sha256 is not fetched again, so the three files may also be copied there by hand.
#   2. unzip, require 3,209 clips
#   3. tools/w12f_locate.py: for every isolated clip, the Communication signer clip and frame range with the same pixels
#   4. tools/w12f_probe.py spans: the leak-safe spotting spans (Communication TRAIN clips only)
#   5. tools/w12f_probe.py facts: the isolated clips of the spotting glosses must be the spotting spans (gate)
# No pose is read and no accuracy is computed here. Pre-registration: results/PREREG_WAVE12F_POSE_MEASUREMENTS_2026-10-13.md
#   sbatch kaya/w12f_stage.sbatch
#SBATCH --job-name=w12f-stage
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w12lib.sh   # modules, environment, cd into the snapshot
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w12flib.sh      # paths, code pins, gates (the snapshot holds no job scripts)
set -u                                                              # only now: module and activate read unset variables
w12f_code_gate
mkdir -p "$RAW" "$OUT" "$W12F/rgb_isolated"
rm -f "$OUT/DONE_stage"

# ---- data gates on what is already on /group ----
for s in train:12439 val:800 test:800; do
  [ "$(grep -c . "$M/${s%%:*}.tsv")" = "${s##*:}" ] || fail "manifest ${s%%:*}.tsv row count"
done
[ "$(find "$B/rgb_signer" -name '*_signer.mp4' | wc -l)" = "$N_SIGNER" ] || fail "signer video count"
python - "$M" "$B/rgb_signer" <<'PY' || fail "a manifest clip has no signer video, or the splits overlap"
import os, sys
m, rgb = sys.argv[1:]
ids = {s: [os.path.basename(l.split("\t", 1)[0])[:-4] for l in open(f"{m}/{s}.tsv") if "\t" in l] for s in ("train", "val", "test")}
have = {f[:-len("_signer.mp4")] for _, _, fs in os.walk(rgb) for f in fs if f.endswith("_signer.mp4")}
allc = sum(ids.values(), [])
assert len(allc) == len(set(allc)) == 14039, "a clip is in two splits"
miss = [c for c in allc if c not in have]
print(f"manifest clips {len(allc)} | signer videos {len(have)} | manifest clips without a video {len(miss)}")
assert not miss
PY

# ---- 1. fetch (Preet's own Drive mirror; checksums from data/integrity/cloud_hash_ledger_rclone.json and the local copies) ----
R="/usr/bin/rclone --config $G/code/rclone.conf"
MIR="auslan_drive:Honours_Datasets_Official_Mirror_2026/Auslan-Daily/Dataset Split Table"
fetch () {   # fetch <path under the mirror> <local file> <sha256>
  if [ -f "$2" ] && echo "$3  $2" | sha256sum --quiet -c - 2>/dev/null; then echo "present with the right sha256: $2"; return 0; fi
  echo "fetching $1"
  $R copyto "$MIR/$1" "$2" --stats 60s --stats-one-line || fail "rclone could not fetch $1"
  echo "$3  $2" | sha256sum -c - || fail "sha256 of $2"
}
fetch "Isolated Sign Language Recognition/Isolated_Sign_Language_Recognition.xlsx" "$RAW/Isolated_Sign_Language_Recognition.xlsx" \
      0868cdf0c8625e8393ae11fa83cb20476518db3df411d955ee88855f56074af6
fetch "Sign Spotting/Sign_Spotting.xlsx" "$RAW/Sign_Spotting.xlsx" \
      d38b297d86dc15f7af73e5383883601cf45caa29f2cf0b71a37a949f228ced33

# ---- 2. the clips ----
if [ "$(find "$ISO" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)" != "$N_ISO" ]; then
  fetch "Isolated Sign Language Recognition/New_Gloss_RGB_Data.zip" "$RAW/New_Gloss_RGB_Data.zip" \
        60bdbf954139ee654366fb2d55e58e44354c9271f479b0aa498df400ddc0915a
  export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
  unzip -q -o "$RAW/New_Gloss_RGB_Data.zip" -d "$W12F/rgb_isolated"
fi
[ "$(find "$ISO" -maxdepth 1 -name '*.mp4' | wc -l)" = "$N_ISO" ] || fail "isolated clip count is not $N_ISO"
rm -f "$RAW/New_Gloss_RGB_Data.zip"                 # 1.47 GB; the clips are on /group now and the zip stays in Drive

# ---- 3. where was each isolated clip cut from? (pixels only) ----
W=$SLURM_CPUS_PER_TASK
python -u tools/w12f_locate.py thumbs --videos "$ISO" --out "$OUT/bank_thumbs.npz" --workers "$W"
python -u tools/w12f_locate.py thumbs --videos "$B/rgb_signer" --suffix _signer.mp4 --only_sizes "$OUT/bank_thumbs.npz" \
    --out "$OUT/signer_thumbs.npz" --workers "$W"
python -u tools/w12f_locate.py match --bank "$OUT/bank_thumbs.npz" --signer "$OUT/signer_thumbs.npz" \
    --xlsx "$RAW/Isolated_Sign_Language_Recognition.xlsx" --train "$M/train.tsv" --val "$M/val.tsv" --test "$M/test.tsv" \
    --out "$OUT/located.csv" | tee "$OUT/located.txt"
[ "$(($(wc -l < "$OUT/located.csv") - 1))" = "$N_ISO" ] || fail "located.csv does not have one row per isolated clip"

# ---- 4. the leak-safe spotting spans, and the clips measurement 2 needs ----
python -u tools/w12f_probe.py spans --xlsx "$RAW/Sign_Spotting.xlsx" --train "$M/train.tsv" --out "$OUT/spans_safe.csv"
python - "$OUT/spans_safe.csv" "$M" "$OUT/pose3d_ids.txt" "$N_SPAN_CLIPS" <<'PY' || fail "span clips"
import csv, os, re, sys
spans, m, out, n = sys.argv[1:]
ids = sorted({re.sub(r"_signer\.mp4$", "", r["Video_Name_Clip"]) for r in csv.DictReader(open(spans))})
held = {os.path.basename(l.split("\t", 1)[0])[:-4] for s in ("val", "test") for l in open(f"{m}/{s}.tsv") if "\t" in l}
assert len(ids) == int(n) and not set(ids) & held, f"{len(ids)} span clips, {len(set(ids) & held)} of them in validation or test"
open(out, "w").write("\n".join(ids) + "\n")
print(f"span clips {len(ids)}, none in validation or test -> {out}")
PY

# ---- 5. what the located table says (no pose): stop here, before the GPU step, if the isolated clips are not the spans ----
#         and gate H: with the production poses on /group the probe reproduces the recorded run on this cluster
pin_pose
python -u tools/w12f_probe.py facts --located "$OUT/located.csv" --spans "$OUT/spans_safe.csv" --out "$OUT/facts.json" \
    --prod_dir "$B/dwpose_clips" | tee "$OUT/facts.txt"
touch "$OUT/DONE_stage"
echo "W12F_STAGE_DONE"
