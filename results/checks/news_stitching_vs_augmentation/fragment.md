### AX — News: stitched data and pose augmentation separated (wave 12a)
**Claim.** On the News subset, the adopted bundle (stitched clips plus pose augmentation) gains +0.53 BLEU-4 over the real-only control on the official News test (-0.40 to +1.45, seed-level 95% t-interval, three seeds). Separating the two parts gives stitched data alone +0.22 (-0.84 to +1.27) and augmentation added on top of it +0.31 (-0.50 to +1.12); augmentation alone (a later arm, three seeds, descriptive) gives +0.19 (-0.50 to +0.88), not lower in any seed, and its seeds 101 and 202 are compared with controls from an earlier job submission. No part is attributed (Appendix B, supplementary results table).

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/news_stitching_vs_augmentation/check.py
```
```text
gate: all 24 beam-5 BLEU-4 (13a, detokenised) recomputed from per-sentence n-gram counts equal the values recorded by the jobs
control nA seed 42 re-run in job 172449_0: 700/700 identical hypotheses on val, BLEU 2.00944 vs 2.00944
control nA seed 42 re-run in job 172449_0: 699/699 identical hypotheses on test, BLEU 1.87272 vs 1.87272
nRA jobs: s42 172449_9, s101 172449_10, s202 172449_11 | controls nA jobs: s42 138377_0, s101 138377_1, s202 138377_2
nRA seeds 101 and 202 are compared with controls from an earlier job submission (138377); seed 42's control was reproduced above
seeds per arm: {'nA': 3, 'nS': 3, 'nSA': 3, 'nRA': 3}
nA   beam-5 BLEU-4 mean  val 1.97  test 1.70
nS   beam-5 BLEU-4 mean  val 2.26  test 1.92
nSA  beam-5 BLEU-4 mean  val 2.15  test 2.23
nRA  beam-5 BLEU-4 mean  val 2.16  test 1.89
both together (B) = nSA - nA  test: per seed +0.23 +0.95 +0.40  positive 3/3  mean +0.53 (+0.525) [-0.40, +1.45]
both together (B) = nSA - nA  val: per seed -0.19 +0.35 +0.35  positive 2/3  mean +0.17 (+0.172) [-0.60, +0.95]
stitched alone (S) = nS - nA  test: per seed +0.25 +0.62 -0.23  positive 2/3  mean +0.21 (+0.215) [-0.84, +1.27]
stitched alone (S) = nS - nA  val: per seed +0.39 +0.47 -0.00  positive 2/3  mean +0.29 (+0.286) [-0.35, +0.92]
augmentation added (A) = nSA - nS  test: per seed -0.02 +0.33 +0.63  positive 2/3  mean +0.31 (+0.310) [-0.50, +1.12]
augmentation added (A) = nSA - nS  val: per seed -0.58 -0.12 +0.36  positive 1/3  mean -0.11 (-0.114) [-1.28, +1.05]
augmentation alone (R) = nRA - nA  test: per seed +0.00 +0.51 +0.07  positive 3/3  mean +0.19 (+0.193) [-0.50, +0.88]
augmentation alone (R) = nRA - nA  val: per seed +0.29 +0.11 +0.16  positive 3/3  mean +0.19 (+0.186) [-0.05, +0.42]
```
**Scope.** Recomputes the 24 beam-5 BLEU-4 scores (sacreBLEU 13a on detokenised text, best checkpoint, validation and test) from per-sentence n-gram match counts and lengths, requires each to equal the value recorded by its job, and then recomputes the seed-level contrasts and t-intervals (df = 2, t = 4.303, as `tools/seed_decision.py`). News reference sentences are not part of the spreadsheet that `make data` restores, so the saved files hold the per-sentence counts instead of the references; the hypotheses are in clear. Original jobs: 138377 (arms nA, nS, nSA, seeds 42/101/202; read out by 138378) and 172449 (nRA, plus a re-run of nA seed 42 whose 700/700 validation and 699/699 test hypotheses equal the original run). Not recomputed: training. Note: the stitched-alone test mean is +0.2150 (+0.21497 before rounding), which two-decimal rounding writes as +0.21; the dissertation prints +0.22 (rounded from the three-decimal value +0.215).
