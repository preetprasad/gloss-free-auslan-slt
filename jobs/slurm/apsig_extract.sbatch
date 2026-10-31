#!/bin/bash -l
# Re-extract the released AlphaPose keypoints taking Annotation['Signer'] (the old extraction took the first person,
# a non-signer in 88.4% of Communication clips; found 25 Oct 2026, the project log 2026-10-25g). Writes new clips and
# manifests beside the old ones; nothing old is changed.
#SBATCH --job-name=apsig-extract
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
module purge
source /group/pmc099/pprasad/envs/auslan/bin/activate
G=/group/pmc099/pprasad; A=$G/scratch_backup/alphapose; N=$G/scratch_backup/alphapose_signer
mkdir -p "$N/clips" "$N/manifests"
python -I $G/code/phase2_baseline/tools/extract_clips.py --pose_root "$A/pt" --out_dir "$N/clips"
for s in train val test; do
  sed "s#$A/clips/#$N/clips/#" "$A/manifests/ap_$s.tsv" > "$N/manifests/ap_$s.tsv"
done
python -I - "$N" "$A" <<'EOF'
import sys, hashlib, numpy as np
N, A = sys.argv[1], sys.argv[2]
for s in ("train", "val", "test"):
    old = [l.split("\t") for l in open(f"{A}/manifests/ap_{s}.tsv").read().splitlines() if l]
    new = [l.split("\t") for l in open(f"{N}/manifests/ap_{s}.tsv").read().splitlines() if l]
    assert len(old) == len(new) and all(o[1:] == n[1:] for o, n in zip(old, new)), s
    assert all(n[0].startswith(f"{N}/clips/") for n in new), s
    miss = [n[0] for n in new if not __import__("os").path.exists(n[0])]
    assert not miss, (s, len(miss), miss[:3])
    print(s, len(new), "rows, texts unchanged, all clips present")
h = hashlib.md5(np.load(f"{N}/clips/video_1_4.npz")["KPS"].tobytes()).hexdigest()
assert h == "a4a85a6da74c39305c9a82ba3f551621", h          # = the local re-extraction on the Mac
o = np.load(f"{A}/clips/video_1_4.npz")["KPS"]; n = np.load(f"{N}/clips/video_1_4.npz")["KPS"]
assert o.shape == n.shape and not np.array_equal(o, n)
print("APSIG_EXTRACT_OK")
EOF
