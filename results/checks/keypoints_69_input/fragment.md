### AX — 69-keypoint input, graph front ends and the tracked-keypoint check
**Claim.** The dissertation (Results, "Negative Results"; Appendix B, `tab:app_additional`, the 69-keypoint rows) states that a 69-keypoint input scaled per body part gave an unresolved small positive effect on the translation score. In the first submission it gave validation +0.35 (-0.85 to +1.56) and test +0.86 (-2.75 to +4.48) over three seeds; in a second submission with its own control, validation +0.64 (-0.10 to +1.37) and test +0.02 (-1.79 to +1.82). The same configuration scored up to 1.66 apart on test between the two submissions. Over five seeds the gain was +0.94 on validation (-0.28 to +2.16) and +0.94 on test (-0.35 to +2.23), with a one-sided test p of 0.057, and about 0.35 of the test gain came from longer output. A 133-keypoint input with median shoulder-width scaling lowered validation by 0.63, and two graph-convolution front ends on the 69 keypoints (15.7 M against 10.6 M parameters) scored 1.50 and 1.48 lower on validation than the plain linear input layer. The gain of the 69-keypoint input was not shown to differ between tracked and first-detection keypoints (difference of gains +0.09 on validation), and the check could not detect switches because the two keypoint sets changed the body scale by more than 10% in only 14 of 200 sampled clips.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/keypoints_69_input/check.py
```
```text
GATE PASSED: 183 of 183 values reproduced (BLEU-4 of every saved decode = the value its job recorded, to 1e-6; sweep at LP 0.6 = GPU decode; F69B seed 42 = F69 seed 42)
references: validation 800 clips, test 800; unseen-target clips 549 (validation), 570 (test)

=== Settings ===
input width per frame (x, y, confidence): 69-keypoint arms 207 = 3 x 69 keypoints; 133-keypoint arms 399 = 3 x 133
  every 69-keypoint run has width 207; every 133-keypoint run 399
first submission: job set 168265: arms ctrl, C133, F69, seeds 42, 101, 202 (3 seeds each)
second submission: job set 168266: arms ctrlB, F69B, P69TC, SG, seeds 42, 101, 202 (3 seeds each, with its own control ctrlB)
parameters: linear front end on 69 keypoints (F69) 10.60 M, control 10.65 M; graph front ends (P69TC, SG) 15.74 and 15.74 M -> 15.7 M against 10.6 M

=== First submission (array A): 69-keypoint input and 133-keypoint median scaling against the control ===
  ctrl  beam-5 val/test s42 11.307/11.711  s101 11.939/11.366  s202 11.199/10.492 | means 11.482/11.190
  C133  beam-5 val/test s42 10.516/10.924  s101 11.451/11.427  s202 10.584/11.698 | means 10.850/11.350
  F69   beam-5 val/test s42 11.856/11.109  s101 11.739/12.250  s202 11.909/12.800 | means 11.835/12.053
-- F69 - ctrl  (beam 5, best.pt, length penalty 0.6; job set 168265)
  validation                         per seed +0.549 -0.199 +0.711 | mean +0.35 95% t [-0.85, +1.56] | higher in 2 of 3
  test                               per seed -0.602 +0.884 +2.309 | mean +0.86 95% t [-2.75, +4.48] | higher in 2 of 3
  unseen-target validation           per seed +0.948 +0.695 +0.792 | mean +0.81 95% t [+0.49, +1.13] | higher in 3 of 3
  length matching (each arm at its own validation-best length penalty: F69 1.0, ctrl 1.0):
  validation, F69* - ctrl*           per seed +0.892 +0.218 +0.677 | mean +0.60 95% t [-0.26, +1.45] | higher in 3 of 3
-- C133 - ctrl  (beam 5, best.pt, length penalty 0.6; job set 168265)
  validation                         per seed -0.792 -0.488 -0.614 | mean -0.63 95% t [-1.01, -0.25] | higher in 0 of 3
  test                               per seed -0.787 +0.061 +1.207 | mean +0.16 95% t [-2.33, +2.65] | higher in 2 of 3

=== Second submission (array B): the same 69-keypoint configuration against its own control ===
  ctrlB beam-5 val/test s42 10.902/11.826  s101 11.939/11.366  s202 11.338/10.411 | means 11.393/11.201
  F69B  beam-5 val/test s42 11.856/11.109  s101 12.306/11.399  s202 11.927/11.144 | means 12.029/11.218
  P69TC beam-5 val/test s42 10.832/10.569  s101 10.709/9.941  s202 10.062/9.542 | means 10.534/10.018
  SG    beam-5 val/test s42 10.434/9.794  s101 10.789/10.085  s202 10.440/10.598 | means 10.554/10.159
-- F69B - ctrlB  (beam 5, best.pt, length penalty 0.6; job set 168266)
  validation                         per seed +0.954 +0.367 +0.589 | mean +0.64 95% t [-0.10, +1.37] | higher in 3 of 3
  test                               per seed -0.717 +0.033 +0.733 | mean +0.02 95% t [-1.79, +1.82] | higher in 2 of 3
  unseen-target validation           per seed +1.125 +1.063 +1.223 | mean +1.14 95% t [+0.94, +1.34] | higher in 3 of 3
  length matching (F69B at 1.0, ctrlB at 0.8):
  validation, F69B* - ctrlB*         per seed +0.925 +0.278 +1.017 | mean +0.74 95% t [-0.26, +1.74] | higher in 3 of 3
-- same configuration in the two submissions (F69B - F69, raw, not a contrast)
  test per seed +0.000 -0.851 -1.656 | largest absolute difference 1.66 | validation per seed +0.000 +0.566 +0.017

=== Graph-convolution front ends on the 69 keypoints (array B) ===
-- P69TC against the plain linear input layer on the same input (F69B)
  validation                         per seed -1.024 -1.596 -1.865 | mean -1.50 95% t [-2.56, -0.43] | higher in 0 of 3
  test                               per seed -0.540 -1.458 -1.602 | mean -1.20 95% t [-2.63, +0.23] | higher in 0 of 3
-- P69TC against the 133-keypoint control (ctrlB)
  validation                         per seed -0.070 -1.229 -1.276 | mean -0.86 95% t [-2.56, +0.84] | higher in 0 of 3
  test                               per seed -1.257 -1.425 -0.869 | mean -1.18 95% t [-1.89, -0.48] | higher in 0 of 3
-- SG against the plain linear input layer on the same input (F69B)
  validation                         per seed -1.422 -1.516 -1.487 | mean -1.48 95% t [-1.60, -1.36] | higher in 0 of 3
  test                               per seed -1.315 -1.314 -0.546 | mean -1.06 95% t [-2.16, +0.04] | higher in 0 of 3
-- SG against the 133-keypoint control (ctrlB)
  validation                         per seed -0.468 -1.149 -0.898 | mean -0.84 95% t [-1.69, +0.02] | higher in 0 of 3
  test                               per seed -2.032 -1.281 +0.187 | mean -1.04 95% t [-3.85, +1.76] | higher in 1 of 3
-- skeleton graph over identity graph (SG - P69TC)
  validation                         per seed -0.398 +0.080 +0.378 | mean +0.02 95% t [-0.95, +0.99] | higher in 2 of 3

=== Five seeds of the 69-keypoint input against the control ===
seeds 42, 101, 202, 303, 404; job sets 42: 168265, 101: 168265, 202: 168265, 303: 172447, 404: 172447 (the first three from the first submission, the fresh seeds 303 and 404 from the later one)
  validation                         per seed +0.549 -0.199 +0.711 +2.455 +1.168 | mean +0.94 95% t [-0.28, +2.15] | higher in 4 of 5
  test                               per seed -0.602 +0.884 +2.309 +1.194 +0.895 | mean +0.94 95% t [-0.35, +2.23] | higher in 4 of 5
  fresh seeds only (303, 404): validation +2.46 +1.17, test +1.19 +0.90: positive on both splits in 4 of 4
  unseen-target validation mean +1.18, test mean +1.29
  one-sided t test on test: t = 2.014, p = 0.057
  test gap at the control's brevity penalty +0.588 of the raw +0.936 -> longer output accounts for 0.35 of the 0.94 test gain

=== Tracked keypoints (second tracker) against first-detection keypoints, 69-keypoint input ===
  gain on first-detection (validation) per seed +1.354 +0.452 -0.251 | mean +0.52 95% t [-1.48, +2.52] | higher in 2 of 3
  gain on tracked (validation)       per seed +0.405 +0.161 +1.267 | mean +0.61 95% t [-0.83, +2.05] | higher in 3 of 3
  difference of the two gains        per seed -0.950 -0.291 +1.518 | mean +0.09 95% t [-3.08, +3.27] | higher in 1 of 3
  body scale differs by more than 10% between the two keypoint sets in 14 of 200 sampled clips (read from the saved job summary trackpose_pose_gate.json: the keypoint arrays are not shipped)
```
**Scope.** Recomputed: the beam-5 BLEU-4 of every saved decode (best checkpoint, length penalty 0.6, validation and official test) of the 25 training runs of the three job sets (and 12 further runs of the tracked-keypoint comparison, validation only), re-scored against the references restored by `make data` with the project's own scoring (`slt/metrics.py`, detokenised, 13a); the per-seed differences, their seed-level 95% t-intervals and sign counts; unseen-target BLEU (references absent from the training text); each arm at its own validation-best length penalty from the saved length-penalty sweeps (CPU re-decodes of the first and second submission's controls and 69-keypoint arms, 9 penalties each); the brevity share of the five-seed test gain; the one-sided test; and the difference of the two gains on tracked and first-detection keypoints from 12 saved validation decodes. The gate first reproduces the BLEU-4 each original job recorded (183 values to 1e-6). Original jobs: first submission 168265 (training array; sweeps 171802/171804), second submission 168266 (sweeps 171803/171805), the five-seed continuation 172447 (seeds 303 and 404; sweep 172448), the tracked-keypoint comparison 179416 (pose check 179415). Not recomputed (read from saved job files, as the output states): the parameter counts (the training script's smoke test, job 168264), the input widths (the first lines of each training log) and the 14 of 200 clip count (the pose check's saved summary, because the keypoint arrays are not shipped). No training or inference is repeated; the layout of the 69 keypoints is defined in training code that is not part of this repository.
