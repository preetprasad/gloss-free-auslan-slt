### AX — Input-side predictors of translation quality
**Claim.** Results ("Training-free measurements") states that neither eleven clip-level pose descriptors nor video identity predicts translation quality, and that each descriptor explains at most about 1% of variance.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/input_side_predictors/check.py
```
```text
references match comm_run/manifests/test.tsv
GATE: 800 clips, 13 descriptors x 4 outcomes, null bands and AUCs reproduce the recorded run (tolerance 1e-6)

clips 800; descriptors 13; constant on all clips: subsampled, invalid_frac
Non-degenerate clip-level pose descriptors: 11  [c10-047]
descriptor x outcome pairs exceeding the shuffled-outcome null band: 0 of 44  [c10-047]
null band (95th percentile of max|r| over shuffles): content_recall 0.099, any_content 0.101, chrf 0.103, exact 0.105
largest |r| = 0.103 (body_conf, exact)  [c10-048]
largest variance explained r^2 = 1.1% (about 1%)  [c10-048]
joint logistic model, video-grouped 5-fold AUC 0.518 vs shuffled-outcome control 0.519  [c10-047]
positive control (reference side, not available at inference): n_content_words vs any_content r = 0.150, exceeds band 0.101: True

video identity: 82 videos with >= 5 scored clips hold 760 of 786 scored clips; per-video success rates run 0.056 to 0.750
between-video sd of success rate 0.1275 vs shuffled null mean 0.1287 (95th percentile 0.1449), permutation p = 0.54 (1000 shuffles, seed 0)  [c10-047]
observed spread is within the shuffled null (p > 0.05): True  [c10-047]
```
**Scope.** Recomputed from per-clip data (`per_clip.jsonl`: 13 pose descriptors of the 800 official test clips, references, and beam-5 hypotheses of three single models): the correlations of each descriptor with four per-clip outcomes, the family-wise shuffled-outcome null band (500 permutations, seed 0, the tool's own random-number order), the video-grouped cross-validated AUC, the descriptor count, the exceedances, the largest r^2, and the video-identity permutation test. The gate requires the recomputed correlations, null bands and AUCs to equal the recorded 2026-09-09 run of `tools/translation_predictors.py` (local run, no cluster job). The video-identity test was not saved by that run, so it is re-implemented here (videos with at least 5 scored clips, 1000 shuffles); its p-value (0.54) differs slightly from the 0.526 in the write-up because the shuffle count and seed are this check's own, and the conclusion (p above 0.05) is the same. The pose descriptors were computed once by `prepare_data.py` from the pose arrays, which are not saved. Nothing is retrained.
