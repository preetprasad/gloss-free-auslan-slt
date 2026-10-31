### AX — Real-data learning curve (wave 8)

**Claim.** Doubling the real training pairs from 6,241 to 12,439 raises BLEU-4 on the official Communication test set from 7.04 to 9.90 (+2.86) and on the 570 unseen-target clips from 3.15 to 5.00 (+1.84), positive in all five seeds, with seed-level 95% t-intervals; the table of nested subsets (Results, learning-curve table), the held-versus-kept episode contrast (appendix, -0.78 and -0.03) and the count-only prediction of content recall (appendix, 16.56 against 16.47) are recomputed too.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/learning_curve_with_real_data/check.py
```
```text
unseen flags match comm_run/manifests/train.tsv
GATE: all 40 decodes re-score to the BLEU-4 recorded by the original job (<1e-6); unseen clips = 570 of 800
Official test set: 800 clips  [c10-017]; unseen-target clips: 570  [c10-018]

Setting: five seeds, beam 5, best checkpoint; unseen = 570 of 800 test clips  [c10-010 c10-011 c10-021 c14-049]
Subset sizes (rows of the saved training manifests, seed 42): R025 3,141, R050 6,241, R100 12,439, V050 6,241  [c10-010 c10-016 c14-049]

Table (five-seed means): pairs / epochs / val / test / unseen / content recall
25% of each episode    3,141  320  6.80  6.03  2.27  8.8%  [c10-012]
50% of each episode    6,241  160  7.89  7.04  3.15  11.0%  [c10-013]
All training pairs     12,439  80  10.51  9.90  5.00  16.5%  [c10-014]
Whole episodes, 50%    6,241  160  8.08  7.63  3.53  11.6%  [c10-015]
  (epochs read from the saved progress.json of each run)

Unseen-target test BLEU-4, R050 3.15 -> R100 5.00 (mean of five seeds)  [c01-019 c03-015 c10-018 c14-049]
R100 - R050, official test (800 clips): per seed +2.897 +1.635 +2.849 +4.087 +2.821
  mean +2.86 (+2.858), 95% t-interval +1.78 to +3.93, positive in 5 of 5 seeds  [c10-017 c10-020 c14-003 c14-027 c10-019]
R100 - R050, unseen-target test (570 clips): per seed +2.071 +0.391 +1.823 +3.176 +1.752
  mean +1.84 (+1.842), 95% t-interval +0.61 to +3.08, positive in 5 of 5 seeds  [c01-020 c10-018 c10-020 c14-049 c10-019]
Test 6,241 -> 12,439 pairs: 7.04 -> 9.90  [c10-017]
Doubling from 6,241 to 12,439 pairs adds 2.86 on test and 1.84 on unseen targets  [c10-020 c14-003 c14-027 c14-049]

Episode familiarity: (V050 - R050) on clips of held-out episodes minus the same difference on kept episodes, at equal size (five seeds; difference between two differences; the one cut episode dropped)  [c17-111]
  all affected test clips: -0.78 (-2.06 to +0.49)  [c17-112]
  unseen-target clips: -0.03 (-0.98 to +0.93)  [c17-113]

Count-only curve (logistic on log(1+training count), fitted on R025 and R050): predicts R100 content recall 16.56% against 16.47% observed  [c18-091]
```
**Scope.** Recomputed from the saved beam-5 decodes (best checkpoint, test and validation, 20 runs each) of the original job 21734: BLEU-4 through `slt/metrics.py`, content recall through `tools/arm_readouts.py`, intervals from five seeds. Subset sizes, epochs and per-token training counts are derived once by `prepare_data.py` from the saved manifests and progress files; the unseen flags are re-checked against `comm_run/manifests/train.tsv`. Nothing is trained.
