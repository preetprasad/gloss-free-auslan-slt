### AX — Three training ideas (sign targets, re-timing, preference stage)
**Claim.** Appendix B (Table of further results, row "Three training ideas") states that three ideas were each trained with three seeds in one job: CTC targets set to the signs shown (sign targets), every stitched sign re-timed by a random factor each epoch (re-timing), and a ten-epoch preference stage against a ten-epoch continuation. Against their in-job controls none is resolvable: sign targets validation +0.28 (-0.65 to +1.20), test -0.31 (-1.80 to +1.18), and with five seeds validation +0.19 and test +0.33, not adopted; re-timing validation +0.26, test +0.15, test content-word recall +1.27 points, and with five seeds validation +0.21 and test +0.24, not adopted (one fresh seed negative on test); preference stage validation -0.31, test -0.15, recall -0.34 points, not positive. The adopted CTC head places 47.3% of 603 annotated sign spans correctly against 26.2% for random placement, so the alignment gate (50% and twice random) failed.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/three_training_ideas/check.py
```
```text
GATE: all 42 decodes re-score to the BLEU-4 stored by their jobs; A seed 42 test = 11.715239 (recorded 11.715239)

Setup: 15 Stage-1 runs (5 arms x 3 seeds) all marked done by job ['144831']; E1/ST/P seeds [42, 101, 202] -> three seeds each, one job
Epochs: P (preference stage) 10/10/10 ; AC (continuation control) 10/10/10 -> ten-epoch stage and ten-epoch continuation

Stage 1, beam-5, seed-level 95% t-interval over seeds 42/101/202 (df 2), paired against the in-job control
Sign targets      validation +0.28 [-0.65, +1.20] (2/3 seeds higher) | test -0.31 [-1.80, +1.18] (1/3) | test content-word recall +0.82 [-1.56, +3.19] points | output length ratio val +0.013 (2/3 longer)
Re-timing         validation +0.26 [-1.81, +2.33] (1/3 seeds higher) | test +0.15 [-0.61, +0.92] (2/3) | test content-word recall +1.27 [-1.16, +3.70] points | output length ratio val +0.026 (3/3 longer)
Preference stage  validation -0.31 [-2.07, +1.44] (1/3 seeds higher) | test -0.15 [-1.36, +1.06] (1/3) | test content-word recall -0.34 [-2.47, +1.79] points | output length ratio val -0.000 (2/3 longer)
Sign targets: validation mean +0.28 > 0 -> passes the futility screen (stopped only if <= 0)
Re-timing: exploratory rule (validation, test and test recall means all > 0): POSITIVE  (val +0.26, test +0.15, recall +1.27)
Preference stage: exploratory rule (validation, test and test recall means all > 0): NOT POSITIVE  (val -0.31, test -0.15, recall -0.34)

Stage 2, five seeds 42/101/202/303/404 (df 4), paired against A
Sign targets  with five seeds, validation +0.19 [-0.18, +0.56], test +0.33 [-1.02, +1.68]; fresh seeds 303/404 test +0.60 +1.98 (2/2 positive)
Re-timing     with five seeds, validation +0.21 [-0.56, +0.98], test +0.24 [-0.48, +0.97]; fresh seeds 303/404 test -0.37 +1.12 (1/2 positive)
E1: one-sided t(4) on the five test differences = 0.681 (p = 0.267); Holm level 0.0100 needs t >= 3.747 -> not rejected, not adopted  [levels from the pre-registration: E1 0.01, ST 0.05/6]
ST: one-sided t(4) on the five test differences = 0.933 (p = 0.202); Holm level 0.0083 needs t >= 3.961 -> not rejected, not adopted  [levels from the pre-registration: E1 0.01, ST 0.05/6]

Alignment of the adopted CTC head (read from the saved job summary g1_alignment_summary.json; needs the checkpoints): 603 annotated sign spans: 47.3% hit against 26.2% for random placement
Gate rule: hit >= 0.50 and hit >= 2 x random and hit > shuffled, on both sets, means over checkpoints; hit >= 50%: False; hit >= 2 x random: False -> gate failed (saved verdict FAIL)
```
**Scope.** Recomputed: BLEU-4 (sacreBLEU 13a on detokenised text, `slt/metrics.py`) and content-word recall from the saved beam-5 decodes of the best checkpoint (validation and test, 5 arms x seeds 42/101/202 from job 144831, plus A/E1/ST x seeds 303/404 from the five-seed follow-up job 152915), each first checked against the BLEU the job itself stored; then the paired differences, seed-level t-intervals, seed counts and the five-seed one-sided t statistics. Not recomputed: training, the epoch counts (read from the saved per-run progress files), the 0.7 to 1.3 re-timing range (it lives in unpublished training code), and the alignment hit rates, which are read from the saved job summary (`g1_alignment_summary.json`) because recomputing them needs the model checkpoints and the sign-spotting annotation. The Holm levels are the ones fixed in the pre-registration, not recomputed from the other experiments of that family.
