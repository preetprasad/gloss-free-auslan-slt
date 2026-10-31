### AX — Paired bootstrap, development ladder and the final ensemble against the published system

**Claim.** On the 800-clip official test set the final six-member ensemble scores 11.80 BLEU-4, 1.47 above its RGB-fusion member (95% paired-bootstrap interval +0.40 to +2.56, 2,000 resamples), 1.00 above its strongest member (not resolvable) and 1.85 (18.6%) above the published GASLT score of 9.95; its own interval (9.87 to 13.89) includes 9.95. Only the ensemble-over-RGB-fusion step is resolvable (Results, Table res_bootstrap; ladder Table res_ladder; Discussion; appendix adequacy note).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/paired_bootstrap_ladder/check.py
```

```text
references match comm_run/manifests/test.tsv: 800/800
GATE final ensemble re-scores to 11.802 (recorded 11.802)
GATE baseline 9.522 (recorded 9.522), 40-epoch VLP 10.272 (recorded 10.272), RGB fusion 10.335 (recorded 10.335)
GATE stitched-data ensembles 12.114 (recorded 12.114), 11.562 (recorded 11.562), 12.836 (recorded 12.836)

DEVELOPMENT LADDER, official test, 800 clips
+ VLP pretraining, 40 epochs: greedy 8.75, beam 5 10.27  [c09-012]
+ RGB fusion: greedy 8.82, beam 5 10.33  [c09-013 c09-005 c14-031 c21-007]
RGB fusion adds +0.06 over the 40-epoch pre-trained model  [c09-013]
Final ensemble: beam 10, length penalty 0.3, BLEU-4 11.80 (11.802)  [c09-005 c14-005 c14-031 c21-007 c21-025 c11-028]
Final ensemble WER 0.907 on the 800-clip official test set  [c21-025]
Official test set size: 800 clips  [c21-025 c12-001 c19-017 c12-012 c02-025 c12-003 c14-031 c21-007]

Ensemble member list, read from jobs/slurm/mbr_wer_test2.sbatch (the job whose --ckpts decoded the final test ensemble at length penalty 0.3): 6 translators averaged  [c11-088]
the 6 listed checkpoints are exactly the 6 member dumps used below: True  [c11-088]
Members decoded alone with the ensemble decoder settings (6 dumps)  [c21-007]
  member comm_dwpose_ctc_s202: 10.46
  member comm_dwpose_ctc_vlplong80: 10.80
  member comm_dwpose_fus_s101: 10.62
  member comm_dwpose_fus_s202: 10.32
  member comm_dwpose_rgb_fusion: 10.01
  member comm_dwpose_vlp_ctc: 10.34
strongest member (comm_dwpose_ctc_vlplong80) 10.80; ensemble 11.80; ensemble minus strongest member +1.00  [c21-007]

GASLT published score 9.95 (published value, Auslan-Daily benchmark paper); ensemble minus 9.95 = +1.85  [c02-015 c11-004 c11-028 c13-053 c14-033 c21-007 c21-025]
relative to GASLT published: 18.6% (published value 9.95)  [c11-004]

Settings read from the source of src/tools/paired_bootstrap.py (the tool that produced Table res_bootstrap): default 2,000 resamples, seed 0, 95% interval, one shared draw matrix of sentence indices over the 800 sentences; this check uses the same convention  [c07-064 c07-066 c02-025 c12-001 c12-003 c19-017]

PAIRED BOOTSTRAP of BLEU-4 differences (BLEU-4, 4-gram): 2,000 resamples with replacement of the 800 test sentences, seed 0; interval = 2.5th and 97.5th percentiles of the difference = 95% interval  [c07-064 c07-066 c02-025 c12-001 c12-003 c12-012 c19-017]
GATE every interval below equals the recorded output of tools/paired_bootstrap.py (saved in recorded_bootstrap_tool_output.txt) to 0.0015
S6 minus RGB: +1.47 (observed +1.467), 95% interval [+0.40, +2.56], P(>0) 99.4%, resolvable, half-width 1.08  [c09-005 c14-005 c14-031 c21-007 c12-006 c07-058]
S6 minus M: +1.00 (observed +1.004), 95% interval [-0.13, +2.11], P(>0) 96.0%, not resolvable, half-width 1.12  [c21-007]
RGB minus CTC: +0.81 (observed +0.812), 95% interval [-0.34, +1.97], P(>0) 91.5%, not resolvable, half-width 1.15  [c12-006]
VLP minus CTC: +0.75 (observed +0.749), 95% interval [-0.22, +1.71], P(>0) 93.2%, not resolvable, half-width 0.96  [c12-007]
F minus S6: +0.31 (observed +0.313), 95% interval [-0.89, +1.50], P(>0) 69.3%, not resolvable, half-width 1.19  [c12-012]
Fp minus S6: +1.03 (observed +1.034), 95% interval [-0.10, +2.15], P(>0) 96.2%, not resolvable, half-width 1.13  [c12-012]
(M = comm_dwpose_ctc_vlplong80, same decoder settings, 10.80; table row 'its strongest member')  [c21-007]
RGB fusion (10.33) vs baseline (9.52): P(>0) 91.5%  [c12-006]
40-epoch pre-trained model (10.27) vs baseline (9.52): P(>0) 93.2%  [c12-007]
Figure res_bootstrap rows 5 and 7, which are not in Table res_bootstrap (stitched-data ensembles):  [c12-012]
G1 minus S6: -0.24 (observed -0.240), 95% interval [-1.34, +0.76], P(>0) 31.7%, not resolvable, half-width 1.05  [c12-012]
Fp minus F: +0.72 (observed +0.722), 95% interval [-0.30, +1.81], P(>0) 91.1%, not resolvable, half-width 1.06  [c12-012]
Stitched-data ensemble scores: 0.55:1 12.11, 1.1:1 12.84, six-model 11.56; final ensemble 11.80  [c12-012]
RGB-fusion checkpoint decoded alone with the ensemble decoder settings 10.01; its standalone beam-5 decode (the ladder row) 10.33
95% half-widths, the 6 rows of Table res_bootstrap: 1.08, 1.12, 1.15, 0.96, 1.19, 1.13; range 0.96 to 1.19, mean 1.11
95% half-widths, those 6 rows plus the two extra rows of Figure res_bootstrap (8 comparisons): range 0.96 to 1.19, mean 1.09; largest 1.19, median 1.10
(single-system interval, not a paired difference) half-width of the 800-sentence interval on the ensemble alone: 2.0
intervals excluding zero among the 8 compared pairs: ['S6-RGB']  [c07-058 c12-012]

Final ensemble's own 95% bootstrap interval over test sentences: 9.87 to 13.89  [c11-029 c21-023]
interval includes GASLT's published 9.95: True  [c21-023]

Judge adequacy (0-4), read from the saved per-clip grades of job 10288: final ensemble 1.311, RGB-fusion member 1.259  [c20-010]
difference +0.052, 95% interval -0.01 to +0.11 (paired bootstrap over 800 clips, 2,000 resamples, seed 0)  [c20-010]

src/tools/seed_decision.py: T975[df] (two-sided 95%) matches Student's t quantile for every df 1..6, df = n - 1 [GATE]
seed_decision.interval on 3 paired differences uses T975[2] = 4.303 (Student's t, n - 1 = 2 degrees of freedom): half-width 1.242 equals t(0.975, 2) x s / sqrt(3) = 1.242  [c07-057]
seed_decision.interval on 5 paired differences uses T975[4] = 2.776 (Student's t, n - 1 = 4 degrees of freedom): half-width 0.491 equals t(0.975, 4) x s / sqrt(5) = 0.491  [c07-057]
Stage 1 seeds, read from jobs/slurm/wave1_r2.sbatch line 33: SEEDS=(42 101 202); S=${SEEDS[$SLURM_ARRAY_TASK_ID]}  (three seeds: 42, 101, 202)  [c07-052 c07-054]
Stage 2 seeds, read from jobs/slurm/w4_train.sbatch line 19: S=(42 101 202 303 404); s=${S[$SLURM_ARRAY_TASK_ID]}  (303 and 404 follow 42, 101, 202)  [c07-055]
```

**Scope.** Recomputed from the saved per-sentence test decodes (baseline, 40-epoch pre-trained model, RGB-fusion model, six ensemble members, final ensemble, and the three stitched-data ensembles of rows 5 and 6 of Table res_bootstrap and rows 4 to 7 of Figure res_bootstrap) through `slt/metrics.py`; bootstrap with seed 0 and one shared draw matrix as in `tools/paired_bootstrap.py`, whose defaults (2,000 resamples, seed 0, 95%) are read from its source. The run first checks that the faster scorer equals the project scorer, re-scores the recorded BLEU-4 of every system, and requires every interval to equal the original tool run (a saved output, `recorded_bootstrap_tool_output.txt`). Adequacy grades are read from the saved per-clip output of the judge job 10288. The seed-level t quantiles are checked against the table `T975` in `tools/seed_decision.py`, which the staged decision rule uses; seed lists are read from the job scripts. The settings claims (2,000 resamples, seed 0, percentiles, 800 sentences) confirm that this convention produced these rows; they do not re-derive other intervals in the dissertation. Not done: training or decoding; the three-seed means, the 80-epoch single model's beam-5 decode and the first-listed-keypoint run are on Kaya or only written down (see claims.json). The dissertation's paired-bootstrap half-width of "about 1.2" is only partly reproduced: across the six rows of Table res_bootstrap the half-widths are 0.96 to 1.19 (mean 1.11; with the two extra rows of the figure, mean 1.09), so 1.2 is the upper end and "about 1.1" would be the typical value; the claim is listed in claims.json as disagreeing and its lines are deliberately untagged. The six-member count is read from the member list of the sbatch script that produced the test ensemble. The RGB-fusion checkpoint decoded alone with the ensemble decoder settings scores 10.01; the ladder row is its standalone beam-5 decode (10.33), while the strongest member (10.80) is compared under the ensemble decoder settings, as in the dissertation.
