### AX — DWPose front end: counts, hand confidence and hand speed

**Claim.** All 14,041 Communication clips were re-extracted with DWPose, giving a 399-dimensional frame vector (133 keypoints of x, y, confidence); the DWPose training list holds 12,439 of the 12,441 training clips (Methodology, Results). On four frames of one training clip the distributed hand keypoints have mean confidence 0.02 to 0.12 and the DWPose re-extraction 0.72 to 0.94 (`fig:res_pose_estimators`). Not recomputable here: the hand-speed ratios to citation form and the hand-confidence pair for correct versus wrong spans (see Scope).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/hand_speed_dwpose_frontend/check.py
```

```text
Communication clips with DWPose keypoints: 14,041  [c06-029 c08-019]
Training clips in the official list: 12,441; in the DWPose training list: 12,439  [c06-017 c08-002]
Official training clips absent from the DWPose list: 2 (video_12_62, video_68_179)
Keypoints per frame 133, values per keypoint 3 (x, y, confidence): frame vector of 399 dimensions  [c06-028]
Figure clip, frames [10, 28, 43, 59] (15/40/62/85% of the frames with DWPose hand confidence above 0.5)
Distributed hand keypoints, mean confidence per frame: 0.07, 0.05, 0.12, 0.02  range 0.02 to 0.12  [c08-023]
DWPose re-extraction, mean confidence per frame: 0.94, 0.72, 0.87, 0.84  range 0.72 to 0.94  [c08-023]

GATE hand speed per frame on first-detection keypoints, 591 clips: mean 0.391 (recorded 0.391)
Tracked keypoints, same 591 clips: mean 0.143; medians 0.097 (first-detection) and 0.086 (tracked)
First-detection speed is 2.7 times the tracked speed on the same clips (inflation of the first-detection figure)

Probe variant 'hands': 334 scored spans; mean hand confidence of correctly identified spans 0.817, of wrongly identified 0.814, correlation +0.019
The two means differ by 0.003 (hand confidence does not separate correct from wrong spans); the dissertation's particular pair of values is not reproduced by the saved probe output
```

**Scope.** Recomputed from saved per-clip files: row count of the DWPose confidence table, the training list against the official manifest, the keypoint tensor shape, per-frame hand confidence of both keypoint sets on the figure clip, and hand speed (`hand_stats`, unchanged) on 591 training clips held in first-detection and tracked form (gate: first-detection mean 0.391 as recorded 10 October 2026). Per-span hand confidence against probe correctness is recomputed and does not reproduce the dissertation's pair of values (c14-042, listed as disagreeing). Not recomputed: speed relative to MM-WLAuslan citation clips and News (9.86, 1.10, "about three times"): their inputs are no longer saved. No training or inference was run.
