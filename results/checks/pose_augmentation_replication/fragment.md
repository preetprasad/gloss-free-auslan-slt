### AX — Pose augmentation replication, recipe and the 1.1:1 baseline

**Claim.** Pose augmentation (temporal resampling by a factor from 0.8 to 1.2 with nearest-index frame selection, in-plane rotation within 10 degrees, x/y scaling 0.9 to 1.1; Methodology) was run on the 1.1:1 stitched corpus with three paired seeds and concurrent controls. Test BLEU-4 at beam width 5 rose from 10.34 to 11.44, 10.93 to 11.38 and 10.66 to 11.19 (means 10.64 to 11.34, +0.70); validation and greedy test also rose in all three seeds; the 95% t-interval over seeds, -0.18 to +1.57, includes zero; 11.34 is compared with 11.14 for 1.1:1 without augmentation (Results, ladder table, ratio and pose-augmentation paragraphs; Discussion). Two numbers do not reproduce: the second seed scores 11.37, not 11.38 (11.3749), and the pooled content-word precision rise (method chapter's 71-word content list) is about +2.38, +1.72, +1.73 points, not the quoted 1.6 to 2.3 (the 75-word training-script list is printed as a labelled second line).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/pose_augmentation_replication/check.py
```

```text
GATE recomputed aug s42 11.4435 vs recorded 11.4435 (read from the saved job output beam5_test.json)

Beam-5 official test BLEU-4, aug vs its same-seed control (recomputed from the saved hypotheses, 800 clips)
  seed 42: 10.34 -> 11.44  (+1.10)
  seed 101: 10.93 -> 11.37  (+0.45)
  seed 202: 10.66 -> 11.19  (+0.54)
Test BLEU-4 at beam width 5 rose in all 3 (three) seeds, aug above its same-seed control in each  [c09-074]
seed 42 from 10.34 to 11.44; seed 202 from 10.66 to 11.19; seed 101 from 10.93 to 11.37 (unrounded 11.3749; the dissertation says 11.38: DISAGREES, a rounding difference of 0.01)  [c09-075]
means 10.64 to 11.34, +0.70 (unrounded +0.695); control mean 10.64, augmented mean 11.34  [c09-018 c09-075 c09-076]
+ pose augmentation (3 seeds) 11.34, +0.70 over its concurrent controls (10.64), 3 of 3 seeds  [c09-018]
seed-level 95% t-interval of the paired differences: -0.18 to +1.57  (t=4.303, df=2; includes zero: True)  [c09-077 c14-026]
95% t-interval over the seeds -0.18 to +1.57  [c14-026]
validation BLEU-4 aug minus ctrl per seed (read from the saved job output progress.json): +0.188 +0.796 +0.190; rose in 3 of 3  [c09-076]
greedy test BLEU-4 aug minus ctrl per seed (read from the saved job output beam5_test.json): +0.349 +0.389 +0.423; rose in 3 of 3  [c09-076]
beam-5 test rose in 3 of 3 seeds: the pre-registered three-seed rule is met: True  [c09-074 c09-076 c09-077]
Pooled content-word precision of the beam-5 test outputs (stitch_lexicon.STOP, 71 words): seed 42 22.37 -> 24.75; seed 101 23.25 -> 24.97; seed 202 24.02 -> 25.75
Content-word precision rise per seed (points): +2.38 +1.72 +1.73; in every seed: True
range of the rise +1.72 to +2.38 points under the 71-word list; the dissertation says 1.6 to 2.3: DISAGREES (c09-078 not covered)
second line, labelled: under the 75-word train_ctc.STOP list the rise per seed is +2.26 +1.66 +1.55 points

Recipe (read from src/config.py defaults and the job script)
temporal resampling factor from [0.8, 1.2] (aug_speed 0.2; job 1215231 passed the same value)  [c06-034]
in-plane rotation from [-10, +10] degrees (aug_rotate_deg 10.0)  [c06-035]
x and y scaling from [0.9, 1.1] (aug_scale 0.1)  [c06-036]

Synthetic-to-real ratio 1.1:1 without augmentation, beam-5 test BLEU-4 (read from the saved job outputs stitch_ratio_job_logs.txt: stitch-scale 1196567 seed 42, stitch-scl2 1199393 seeds 101, 202)
  sc_r111_s101 (stitch-scl2-1199393.out): 11.03
  sc_r111_s202 (stitch-scl2-1199393.out): 10.87
  sc_r111_s42 (stitch-scale-1196567.out): 11.52
mean of the three logged aug-free 1.1:1 scores 11.14  [c09-052]
the higher means at 1.1:1 (11.14, and 11.34 with pose augmentation)  [c09-052]

Recipe checked against the augmentation code slt/data.py (augment_pose), run on synthetic keypoints
temporal resampling picks frame indices with linspace(0, T-1, new_T).round(), i.e. nearest-index frame selection (source line checked)  [c06-034]
temporal factor over 2000 draws: min 0.80, max 1.20, inside [0.8, 1.2]: True  [c06-034]
in-plane rotation over 2000 draws: min -10.0, max +10.0 degrees (limit +-10)  [c06-035]
x and y scale factors over 4000 draws: min 0.90, max 1.10 (limit [0.9, 1.1])  [c06-036]
```

**Scope.** BLEU-4 is recomputed from the saved beam-5 hypotheses of job 1215231 (800 official test clips); validation and greedy scores are read from the saved job outputs (`progress.json`, `beam5_test.json`); the interval is computed here from the three paired differences. The 1.1:1 baseline mean is parsed from the saved logs of jobs 1196567 and 1199393. The recipe ranges are measured by running the repository's `augment_pose` on synthetic keypoints. Content-word precision uses `STOP` from `src/tools/stitch_lexicon.py` (71 words); the 75-word `STOP` of `src/train_ctc.py` gives a labelled second line. Training is not re-run. Not covered: the dose sweep (job 6612; its script is not saved and its summaries are on the cluster) the 11.38 versus 11.37 difference and the 1.6 to 2.3 precision range.
