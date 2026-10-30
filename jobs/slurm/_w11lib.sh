# Wave-11 harness (validation-only selection, 9 Oct 2026). Identical to _w4lib.sh except for the code snapshot:
# $G/wave11/code = a copy of the wave-5 snapshot plus the seeded pretrain_vlp.py.
# Code runs from a FROZEN SNAPSHOT ($G/wave4/code, train_ctc.py = the warn_only=False determinism fix),
# so wave-3 jobs still importing the shared tree are untouched and every wave-4 arm runs identical code.
module purge; module load cuda/12.6.3
source /group/pmc099/pprasad/envs/auslan/bin/activate
G=/group/pmc099/pprasad; W11=$G/wave11
cd "$W11/code" || { echo "FAIL: wave-11 code snapshot missing"; exit 1; }
grep -q "warn_only=False" train_ctc.py || { echo "FAIL: snapshot lacks the determinism fix"; exit 1; }
grep -q "vlp-seed" pretrain_vlp.py || { echo "FAIL: snapshot lacks the seeded pretrain_vlp.py"; exit 1; }
M="$G/scratch_backup/dwpose/manifests"
POSE=${POSE:-"--num_keypoints 133 --keypoint_subset all --coord_norm frame"}
DET=${DET-"--deterministic True"}   # all-or-nothing within a wave (R2 rule 8); A/A job 6952: 0/132 tensors differ
AUG_DEF="--pose_aug True --aug_speed 0.2 --aug_rotate_deg 10.0 --aug_scale 0.1"
REC="--batch_size 64 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
     --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8"
echo "node $(hostname) | DET='${DET}'"; md5sum config.py slt/data.py train_ctc.py pretrain_vlp.py tools/eval_beam.py

# S6 gate: the untracked DWPose clips every arm reads must be the originals, not job 6590's tracked pose.
pin_pose () {
  local got; got=$(md5sum "$G/scratch_backup/dwpose/dwpose_clips/video_1_0.npz" | cut -d' ' -f1)
  [ "$got" = 582483e5214cb1c0f7c6b40315fab625 ] || { echo "FAIL: pinned DWPose clip changed ($got)"; exit 1; }
  echo "pinned DWPose clip video_1_0.npz md5 OK"
}

# assert_group <manifest...>: every clip must exist AND live on /group. /scratch purges files nobody has
# read for the purge window (by access time, 2026-09-22j), so a /scratch path is a time bomb in a multi-day wave.
assert_group () {
  python - "$@" <<'PY'
import os, sys
bad = 0
for m in sys.argv[1:]:
    rows = [l.split("\t", 1)[0] for l in open(m) if "\t" in l]
    off = sum(not p.startswith("/group/") for p in rows)
    miss = sum(not os.path.exists(p) for p in rows)
    print(f"  {os.path.basename(m)}: {len(rows)} rows | off-/group {off} | missing {miss}")
    bad += off + miss
sys.exit(1 if bad else 0)
PY
}

# w4run <arm> <runs> <train> <val> <test> <seed> <epochs> -- <extra train flags...>
# Trains, then asserts the checkpoint's saved cfg carries EVERY --flag value passed (POSE, REC, DET, extras),
# so a flag that is parsed but ignored fails the arm instead of producing a plausible number.
w4run () {
  local arm=$1 runs=$2 train=$3 mval=$4 mtest=$5 s=$6 ep=$7; shift 8
  local d="$runs/${arm}_s${s}"
  echo "=== ARM ${arm} (seed $s, ${ep} ep) ==="
  local flags="--train_manifest $train --val_manifest $mval --test_manifest $mtest $POSE $REC $DET --epochs $ep --seed $s $*"
  python train_ctc.py $flags --swa_last_k 5 --out_dir "$d" || { echo "[FAIL] train ${arm}"; return 1; }
  python - "$d/best.pt" "$flags" <<'PYCFG' || { echo "[FAIL] cfg ${arm}"; return 1; }
import shlex, sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["cfg"]
t = shlex.split(sys.argv[2]); bad = {}
for i in range(0, len(t) - 1):
    if t[i].startswith("--") and not t[i + 1].startswith("--"):
        k, v = t[i][2:], t[i + 1]
        if k in c and str(c[k]) != v and not (isinstance(c[k], float) and abs(c[k] - float(v)) < 1e-9):
            bad[k] = (c[k], v)
print("checkpoint cfg", "OK (every passed flag verified)" if not bad else f"MISMATCH (saved, passed): {bad}")
sys.exit(1 if bad else 0)
PYCFG
  local swa; swa=$(ls "$d"/swa_ep*.pt | paste -sd, -)
  python weight_average.py --ckpts "$swa" --out "$d/soup_last5.pt" || echo "[WARN] soup ${arm}"
  local mm
  for ck in best soup_last5; do for split in val test; do
      if [ "$split" = test ]; then mm="$mtest"; else mm="$mval"; fi
      python tools/eval_beam.py --ckpt "$d/$ck.pt" --manifest "$mm" --beam 5 --device cuda \
        --output_json "$d/beam5_${split}_${ck}.json" > "$d/beam5_${split}_${ck}.log" 2>&1 \
        || echo "[FAIL] beam5 ${split} ${ck} ${arm}"
  done; done
}

# w4summary <runs> <seed> <control_arm> <arm...>: readouts only; decisions need every seed.
w4summary () {
  python - "$@" <<'PYSUM'
import json, os, sys
runs, s, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
KEY = "tokenize=13a;detok=True"
def b5(d, split, ck, what="bleu"):
    try: j = json.load(open(os.path.join(d, f"beam5_{split}_{ck}.json")))
    except Exception: return float("nan")
    return j["beam_result"]["metrics"][KEY] if what == "bleu" else len(set(j["beam_hyps"]))
rows = {}
for arm in arms:
    d = os.path.join(runs, f"{arm}_s{s}")
    try: p = json.load(open(os.path.join(d, "progress.json")))
    except Exception as e: print(f"{arm}: NO RESULT ({e})"); continue
    rows[arm] = dict(val_best=p["best"]["bleu4"], best_epoch=p["best"]["epoch"],
                     greedy_test=p.get("test", {}).get("bleu4", float("nan")),
                     beam5_val=b5(d, "val", "best"), beam5_test=b5(d, "test", "best"),
                     beam5_test_unique=b5(d, "test", "best", "uniq"),
                     soup_beam5_val=b5(d, "val", "soup_last5"), soup_beam5_test=b5(d, "test", "soup_last5"))
    print(arm, json.dumps(rows[arm]))
ctrl = arms[0]
for arm in arms[1:]:
    if arm in rows and ctrl in rows:
        print(f"{arm} - {ctrl} (seed {s}): beam-5 val {rows[arm]['beam5_val'] - rows[ctrl]['beam5_val']:+.3f} | "
              f"beam-5 test {rows[arm]['beam5_test'] - rows[ctrl]['beam5_test']:+.3f}")
json.dump(rows, open(os.path.join(runs, f"summary_{'_'.join(arms)}_s{s}.json"), "w"), indent=1)
PYSUM
}
