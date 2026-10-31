### AX — Person tracking and the heavy-switch clips

**Claim.** The tracked re-extraction of all 14,041 Communication clips shows an identity switch in 23.8% of training clips (5.7% of frames), 77% of evaluation clips have none, and a blind two-rater audit finds the tracker on the signer in 87 of 96 decisive frames (91%). Training on tracked keypoints at five seeds changes test BLEU-4 by -0.06 (-0.66 to +0.55) and tracker v2 (wave 9) is stopped at stage 1; but on the 64 test clips with more than 30% switched frames, tracker v2 raises BLEU-4 by 3.52 more than on clips without switches (+1.44 to +5.61, five new seeds), with +0.80 (-0.54 to +2.13) on the whole test set (Results, person tracking; Discussion; appendix tables on tracked keypoints and tracker v2).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/tracker_v2_heavy_switch/check.py
```

```text
Tracked re-extraction covers 14,041 Communication clips (read from the saved job output tracking_prevalence_agg.json) [c12-062]
Training clips with at least one switched frame 23.8% (2955/12439); switched frames 5.7% (52996/935938) [c12-063]
Evaluation clips (validation+test) with no switched frame 77% (1227/1600) [c12-069]
  (the switched-clip numerators are recounted from the per-clip file and match the job totals)

Blind audit: 160 frames, each labelled by 2 raters [c12-064]
Tracker on the signer and the original extraction not: 87 of 96 decisive frames (91%) [c12-065]

Wave 3 (job 6592 + 9444): index-0 pose (w3C) vs person-tracked pose (w3T), seeds 42/101/202/303/404
GATE: all 20 beam-5 BLEU-4 values re-score to the recorded job values
Test (800 clips) tracked - original, per seed +0.70 -0.50 -0.27 -0.35 +0.13: mean -0.06 (-0.057) 95% t [-0.66, +0.54] (-0.658, +0.545), 2/5 seeds positive [c12-066]
  (the dissertation states the upper bound +0.55; the five-seed t-interval here is +0.545 rounded down to +0.54: a rounding difference, not forced)
Stage-2 seeds (303, 404): tracked scores lower in 1 of 2 (informational for c12-068)
Heavy bin = clips on which the tracker replaced more than 30% of the frames (frozen rule: share > 0.3): 64 test clips [c12-071]
Heavy-switch clips, validation, rounded to one decimal: +4.4 on 55 clips [c17-015]
Heavy-switch clips, test, rounded to one decimal: +0.6 on 64 clips [c17-016]
Heavy-switch clips, validation (n=55) tracked - original: mean +4.36 (+4.359) 95% t [+2.27, +6.45] (+2.273, +6.445), 5/5 seeds positive [c17-015]
Heavy-switch clips, test (n=64): mean +0.59 (+0.587) 95% t [-2.52, +3.69] (-2.521, +3.694), 3/5 seeds positive; negative in 2 seeds [c17-016]

Wave 9 (job 43506): tracker v2 - original, seeds 42/101/202
Validation mean -0.02 (-0.023): stage 1 stops (mean <= 0) [c17-017]
Test mean -0.04 (-0.038) 95% t [-1.79, +1.71] (-1.785, +1.710), 2/3 seeds positive [c17-018]

Wave 12i (jobs 138391/138392): tracker v2 vs original on the heavy-switch bin, five new seeds 303/404/505/606/707
Seeds: 5 new seeds [c12-070]
Heavy bin (test n=64) DiD tracker v2 - original, minus clips without switches: mean +3.52 (+3.523) 95% t [+1.44, +5.61] (+1.435, +5.611), 5/5 seeds positive [c12-071 c12-072 c17-019]
Heavy bin validation DiD: mean +4.74 (+4.738) [c12-072 c17-020]
Clip-sampling interval (test, 1000 resamples within bins, seed 0) [-0.13, +8.86] [c17-020]
Whole test set, tracker v2 - original: mean +0.80 (+0.798) 95% t [-0.54, +2.13] [c12-073 c17-021]
Original-trained model decoded on tracked keypoints, heavy-bin DiD vs decoding on its own pose: mean +4.49 (+4.490) [c17-022]

Hand speed on 591 training clips: first-detection mean 0.391, tracked mean 0.143 per frame
The citation-form comparison value is a recorded mean of 0.044 per frame (MM-WLAuslan clips are not held locally), recorded 9 Sep.
Ratio to the recorded citation value, per frame 3.2, per second 2.7  (informational, untagged)

Pose-jump census over locally stored clips with shoulders tracked on at least 90% of frames: 2,128 clips [c17-012]
Clips with at least one jump 25.0% (531); frame transitions with a jump 2.39% (3116/130353)
  (the dissertation states 25.2% and 2.31%; this definition of a jump, from pose/tracker_v2/research/alphapose_identity_check.py, gives the values above: not forced to agree)
```

**Scope.** Beam-5 hypotheses of every run are re-scored through `slt/metrics.py` and must equal the recorded BLEU-4 before any difference is printed; references come from the official manifests. Means, t-intervals and the 1,000-resample clip bootstrap (seed 0) are recomputed. Switch counts (job 6591) are read from the saved job output. Hand speed is recomputed on tracked keypoints only (the citation-form value is not held), the 552-event cause split has no saved output, and the pose-jump rate under this repository's jump definition differs slightly from the dissertation (25.0% and 2.39% against 25.2% and 2.31%). No training or decoding is repeated. Jobs: 6590/6591/6592/9444 (wave 3), 43506 (wave 9), 138391/138392 (wave 12i).
