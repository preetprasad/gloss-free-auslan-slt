#!/bin/bash -l
# W12A-NEWS: decomposition of the adopted News bundle (stitched clips at 1.1:1 + pose augmentation).
#   nA  = News real clips only (9,649)                                    the arm of w4-news and of w11 nF512
#   nS  = News + the first 10,614 stitched clips, NO pose augmentation    the arm that was never run
#   nSA = News + the first 10,614 stitched clips + pose augmentation      the adopted bundle (w4-news)
#   nRA = News real clips + pose augmentation    OPTIONAL, tasks 9-11, only if approved BEFORE submission
# One arm per array task, seeds 42/101/202, 150 epochs, --max_src_frames 512, deterministic kernels. The trainer
# command of nA and of nSA is the one kaya/w4_news.sbatch issued; nS is the nSA command without $AUG_DEF.
# Every arm is in this ONE array job:   task = 3 * arm + seed index   0-2 nA | 3-5 nS | 6-8 nSA | (9-11 nRA)
# Code: the frozen wave-11 snapshot ($G/wave11/code), md5-gated below; it ran jobs 116619 and 116620.
# Pre-registration: results/PREREG_WAVE12A_NEWSDECOMP_2026-10-13.md
#   real run:  sbatch kaya/w12a_news_decomp.sbatch
#   smoke   :  sbatch --array=0,3,6 --export=ALL,W12A_SMOKE=1 kaya/w12a_news_decomp.sbatch   (one task per arm; 24 h as always)
#              The smoke trains on the first 256 rows of the manifest, which are real clips: it tests the trainer, the flags
#              and the arm check, not the loading of the stitched clips (assert_group has checked that they exist).
#   one rerun of a task that died of a node fault before its first epoch (pre-registration, section 9):
#              sbatch --array=<k> --export=ALL,W12A_RUNS_ID=<the original array job id> kaya/w12a_news_decomp.sbatch
#SBATCH --job-name=w12a-news
#SBATCH --partition=gpu
#SBATCH --account=pmc099
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --exclude=k035,k036
#SBATCH --array=0-8
#SBATCH --output=/group/pmc099/pprasad/code/logs/%x-%A_%a.out
set -eo pipefail
source /group/pmc099/pprasad/code/phase2_baseline/kaya/_w11lib.sh   # module and venv set-up are not nounset-safe
set -u

# --- gate 0: the recipe is the pre-registered one. POSE and DET can be overridden from the submitting shell's
# environment (sbatch exports it), and the library sits in the shared tree, so the values are asserted, not assumed.
[ "$POSE" = "--num_keypoints 133 --keypoint_subset all --coord_norm frame" ] && [ "$DET" = "--deterministic True" ] \
  && [ "$AUG_DEF" = "--pose_aug True --aug_speed 0.2 --aug_rotate_deg 10.0 --aug_scale 0.1" ] \
  && [ "$(echo $REC)" = "--batch_size 64 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8" ] \
  || { echo "FAIL: POSE / DET / AUG_DEF / REC differ from the pre-registered recipe (an exported variable or an edited library)"; exit 1; }
echo "gate 0 OK: recipe variables"

T=$SLURM_ARRAY_TASK_ID
ARMS=(nA nS nSA nRA); SEEDS=(42 101 202)
if [ "$T" -lt 0 ] || [ "$T" -gt 11 ]; then echo "FAIL: task $T is outside 0-11"; exit 1; fi
arm=${ARMS[$((T / 3))]}; s=${SEEDS[$((T % 3))]}
RID=${W12A_RUNS_ID:-$SLURM_ARRAY_JOB_ID}
EP=150; LIM=""; KIND=runs
if [ "${W12A_SMOKE:-0}" = 1 ]; then
  EP=6; LIM="--limit_samples 256"; KIND=smoke
  echo "##### SMOKE MODE: 6 epochs on 256 training clips. Nothing written here is an experiment result. #####"
fi
RUNS=$G/wave12/a_news/${KIND}_$RID; SA=$G/wave4/manifests/news_sa_train.tsv; d=$RUNS/${arm}_s$s

# --- gate 1: the code is the wave-11 snapshot, byte for byte (cwd is $G/wave11/code, set by the library)
md5sum -c --quiet - <<'MD5' || { echo "FAIL: the snapshot differs from the code that ran jobs 116619 / 116620"; exit 1; }
2e54dd6d2a5e3f99572e74bb086c749e  config.py
75783cd2aaddb8d5822f0c5d1a4f10e5  train_ctc.py
2fd20e3989b8e20a9857f6d90106da42  weight_average.py
6553f52a3f0fbeb57feef6510289d66c  tools/eval_beam.py
e63a6db76161406c3ab3cb04be0cbd05  slt/__init__.py
1187c9ff21c8de97b6ba98dfedf3897f  slt/data.py
8399ac01b7f856e774b3f31d87dcf385  slt/engine.py
1a2ff663b374e0e4ce5c9dd3badf3b02  slt/metrics.py
076d9b1ac5e0008a895b3a5bcfe4f470  slt/model.py
d41145cd1acabb5a89349e6b6a5c51ef  slt/reference_model.py
MD5
echo "gate 1 OK: code = wave-11 snapshot"

# --- gate 2: the inputs are the files that jobs 7950, 16398 and 116620 read
pin_pose
md5sum -c --quiet - <<MD5 || { echo "FAIL: a News manifest changed"; exit 1; }
48c2b247159ed5eb9c3a595f6b8fedcb  $M/news_train.tsv
fd31d79eef3a5d6dde5b9f0839d05dbd  $M/news_val.tsv
c11da128f6c46844ff6ce4f3352d894f  $M/news_test.tsv
80eb65dd8fc036afc453ca47e0dc1afd  $SA
MD5
# pin_pose pins a Communication clip that no News manifest reads. These two are inputs of THIS job: the first real
# News training clip and the first stitched clip (its md5 is the one recorded on 10 Sep 2026, CLAUDE.md 09-10ab).
md5sum -c --quiet - <<MD5 || { echo "FAIL: a pinned News or stitched clip changed"; exit 1; }
057c38a5d21ae29e4318c4f746799ca0  $G/scratch_backup/news/dwpose_clips/16_10_2022_0.npz
a765ab81fd219c7c5a123e284c793dff  $G/stitch_scale2/corpus/clips/stitch2_000000.npz
MD5
assert_group "$M/news_train.tsv" "$SA" "$M/news_val.tsv" "$M/news_test.tsv"
echo "gate 2 OK: manifests pinned, every clip on /group"

# --- gate 3: structure of the stitched manifest and leaks, measured again (the md5 pins already imply them)
python - "$M/news_train.tsv" "$SA" "$M/news_val.tsv" "$M/news_test.tsv" <<'PYGATE' || { echo "FAIL: manifest structure or leak gate"; exit 1; }
import sys
rd = lambda f: [l.rstrip("\n").split("\t", 1) for l in open(f, encoding="utf-8") if "\t" in l]
tr, sa, va, te = (rd(f) for f in sys.argv[1:5])
norm = lambda t: " ".join(t.lower().split())
syn = sa[len(tr):]
checks = {
    "rows: 9,649 real train, 20,263 real + stitched, 700 validation, 699 test":
        (len(tr), len(sa), len(va), len(te)) == (9649, 20263, 700, 699),
    "the first 9,649 rows of news_sa_train are news_train, in order": sa[:len(tr)] == tr,
    "the other 10,614 rows are stitch_scale2 clips 000000-010613, in order":
        [p.rsplit("/", 1)[-1] for p, _ in syn] == ["stitch2_%06d.npz" % i for i in range(10614)]
        and all("/stitch_scale2/corpus/clips/" in p for p, _ in syn),
    "no clip occurs twice in news_sa_train": len({p for p, _ in sa}) == len(sa),
    "no training clip is a validation or test clip": not ({p for p, _ in sa} & {p for p, _ in va + te}),
    "no validation clip is a test clip": not ({p for p, _ in va} & {p for p, _ in te}),
    "no stitched sentence equals a validation or test reference":
        not ({norm(t) for _, t in syn} & {norm(t) for _, t in va + te}),
}
for name, ok in checks.items():
    print("  [%s] %s" % ("ok" if ok else "FAIL", name))
sys.exit(0 if all(checks.values()) else 1)
PYGATE
echo "gate 3 OK: structure and leaks"

# --- never overwrite a run
mkdir -p "$d"
if [ -e "$d/progress.json" ] || [ -e "$d/best.pt" ] || [ -e "$RUNS/DONE_${arm}_s$s" ]; then
  echo "FAIL: $d already holds a run. Nothing is overwritten; move a dead attempt aside by hand."; exit 1
fi
echo "${SLURM_ARRAY_JOB_ID}_$T $(hostname) $(date +%FT%T) smoke=${W12A_SMOKE:-0}" >> "$d/slurm_job.txt"   # last line = the run

case $arm in   # expected vocabulary sizes: logs of job 7950 ("[ctc] vocab=12407" for nA, "vocab=12667" for nSA)
  nA)  TRAIN=$M/news_train.tsv; AUG="";       WANT_AUG=False; NVOC=12407 ;;
  nS)  TRAIN=$SA;               AUG="";       WANT_AUG=False; NVOC=12667 ;;
  nSA) TRAIN=$SA;               AUG=$AUG_DEF; WANT_AUG=True;  NVOC=12667 ;;
  nRA) TRAIN=$M/news_train.tsv; AUG=$AUG_DEF; WANT_AUG=True;  NVOC=12407 ;;
esac
echo "=== task $T: $arm, seed $s, $EP epochs, train $(basename "$TRAIN"), pose_aug $WANT_AUG, runs $RUNS ==="
w4run "$arm" "$RUNS" "$TRAIN" "$M/news_val.tsv" "$M/news_test.tsv" "$s" "$EP" -- --max_src_frames 512 $AUG $LIM

# --- the arm is what its name says: saved cfg, vocabulary size, finished training (w4run checks only passed flags)
python - "$d" "$WANT_AUG" "$NVOC" "$TRAIN" "$EP" "$s" <<'PYARM' || { echo "FAIL: arm check $arm s$s"; exit 1; }
import json, sys, torch
d, aug, nvoc, train, ep, seed = sys.argv[1], sys.argv[2] == "True", int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
ck = torch.load(d + "/best.pt", map_location="cpu", weights_only=False)
c, p = ck["cfg"], json.load(open(d + "/progress.json"))
want = dict(pose_aug=aug, max_src_frames=512, epochs=ep, seed=seed, deterministic=True, train_manifest=train,
            vocab_file="", init_encoder="", resume_checkpoint="", rgb_feat_dir="", gloss_map="", paraphrase_prob=0.0,
            temporal_pool=1, hand_local_norm=False, aug_hand_drop=0.0, aug_hand_span=0.0, eval_every=1)
bad = {k: (c.get(k, "<absent>"), v) for k, v in want.items() if c.get(k, "<absent>") != v}
if len(ck["vocab"]) != nvoc:
    bad["vocabulary size"] = (len(ck["vocab"]), nvoc)
if p.get("status") != "done" or p.get("epoch") != ep or len(p.get("history", [])) != ep:
    bad["progress.json"] = (p.get("status"), p.get("epoch"), len(p.get("history", [])))
print("arm check", "OK: cfg, vocabulary size and finished training" if not bad else f"MISMATCH (found, expected): {bad}")
sys.exit(1 if bad else 0)
PYARM
for f in beam5_val_best beam5_test_best; do
  [ -s "$d/$f.json" ] || { echo "FAIL: $d/$f.json is missing (the criterion decode)"; exit 1; }
done
for f in beam5_val_soup_last5 beam5_test_soup_last5; do
  [ -s "$d/$f.json" ] || echo "WARN: $d/$f.json is missing (the soup is never a criterion)"
done
touch "$RUNS/DONE_${arm}_s$s"
echo "W12A_NEWS_DONE $arm s$s"
