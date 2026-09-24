#!/bin/bash -l
# t3-docfix item 5: per-clip detection counts from the TRACKED re-extraction (NDET = people detected in the signer
# crop per frame; CHOSEN = which detection the tracker kept). Joined locally with Signer_Detection.xlsx, it tells
# whether a scene-level bystander label (88.4% of clips) actually means a second person INSIDE the crop.
#SBATCH --job-name=t3-ndet
#SBATCH --partition=work
#SBATCH --account=pmc099
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%j.out
set -eo pipefail
source /group/pmc099/pprasad/envs/auslan/bin/activate
python - <<'PY'
import glob, os
import numpy as np
from multiprocessing import Pool
D = "/group/pmc099/pprasad/scratch_backup/dwpose/dwpose_tracked"
def row(p):
    z = np.load(p); n, c = z["NDET"], z["CHOSEN"]
    return f"{os.path.basename(p)[:-4]},{len(n)},{int(n.max())},{(n >= 2).mean():.4f},{(c != 0).mean():.4f},{int((c[1:] != c[:-1]).sum())}\n"
files = sorted(glob.glob(f"{D}/*.npz"))
with Pool(8) as pool: rows = pool.map(row, files, chunksize=64)
out = "/group/pmc099/pprasad/wave3_track/ndet_per_clip.csv"
open(out, "w").writelines(["clip,frames,max_ndet,frac_multi,frac_chosen_nonzero,chosen_changes\n"] + rows)
print(len(rows), "clips ->", out)
PY
