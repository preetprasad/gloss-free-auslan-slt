### AX — PoseStitch side experiments (sign order, wider vocabulary, function signs, letters, wrong-sense signs)
**Claim.** The dissertation reports five negative or null variations of the stitched-data recipe, each as a paired comparison with its in-batch control: a shuffled sign order costs about 0.7 BLEU-4 on validation unless the CTC targets follow the signs (Appendix B, wave 10); a wider-vocabulary corpus (B) and a doubled amount (C) give no gain (wave 6); not signing function words (32% of stitched signs dropped) gives no resolvable gain over five seeds (wave 7); the fingerspelled-letter signs (8.7% of stitched signs) can be removed with no measurable change (wave 2); and 3.45% of the signs are wrong-sense glosses (wave 12b corpus facts and its three-seed comparison).

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/stitching_side_experiments/check.py
```
```text
saved test references = comm_run/manifests/test.tsv (as a set of 800)
GATE 34 runs: BLEU-4 recomputed from the saved beam-5 hypotheses equals the job summary (expected 0 mismatches, found 0)

WAVE 10 shuffled sign order (3 seeds, beam-5 best.pt; difference to the English-order control A)
S - A validation per seed -0.456 -0.704 -0.917; mean -0.69  [c17-063]
S - A test per seed -1.550 +0.181 -0.662; mean -0.68  [c17-063]
SC - A validation per seed +0.791 +0.293 +0.266; mean +0.45  [c17-064]
SC - A test per seed -0.800 +0.357 +0.071; mean -0.12  [c17-064]

WAVE 6 wider-vocabulary stitched corpus (B: 1.1:1; C: twice the amount)
B - A validation -0.19 (2/5 positive)  [c16-032]
B - A test +0.09 [-0.45, +0.63] (95% t-interval over 5 seeds)  [c16-032]
C - A validation -0.80 (0/3 positive; stage 1 uses the first three seeds)  [c16-032]
seeds used for B vs A: 5; for C vs A: 3 (counted from the saved job summaries)

WAVE 7 function-word signs dropped (D vs A, 5 seeds; DL = length-matched arm)
D - A validation +0.18 (4/5 positive)  [c16-034]
D - A test -0.09 [-0.76, +0.59] (2/5 positive; fresh seeds 303 and 404: -0.444, -0.515)  [c16-034]
DL (length-matched) arm run at 2 seeds [303, 404]  [c16-034]
DL - A test per seed -0.072 -0.606 (descriptive)
content-word precision, D - A, test: mean -0.95 (0/5 positive) (read from the saved job output wave7_readout_output.txt; the tool that wrote it is src/tools/arm_readouts.py)  [c16-034]
function-word signs dropped: 30247 of 94428 spans = 32% (read from the saved job output wave7_prep_66643.out)  [c16-033]
seeds compared: 5  [c16-033]
constrained generation pool: 20608 sentences, 20608 unique (read from the saved corpus-audit output stitch_corpus_audit.json; the 6,880 unconstrained sentences are not saved locally)

ADOPTED STITCHED CORPUS (13,760 clips; segment list of the corpus audit)
GATE signs 94428 (recorded in the corpus-facts output: 94428); wrong-sense signs 3257 (recorded: 3257)
single-letter (fingerspelled alphabet) signs: 8187 of 94428 = 8.7% of stitched signs  [c17-079]
wrong-sense signs (the 11 sense-disambiguated glosses): 3257 of 94428 = 3.45% of the signs  [c17-048]

WAVE 12b wrong-sense signs removed (W - A, 3 seeds, beam-5 best.pt; per-run values read from the saved job output wave12b_summary_138381.out)
GATE per-seed W - A differences recomputed from the six run records equal the job's printed lines (3 seeds)
W - A validation per seed -0.657 +0.627 +0.374; mean +0.1146 = +0.115 to 3 decimals, which the dissertation rounds again to +0.12 (a single rounding gives +0.11); 95% t-interval [-1.57, +1.80] over 3 seeds  [c17-049 c17-048]
W - A test per seed -0.643 +0.525 +0.034; mean -0.03 [-1.48, +1.43] (95% t-interval over 3 seeds)  [c17-049 c17-048]
GATE mean and 95% t-interval equal the job's own seed_decision.py lines (validation and test)

WAVE 2 letter-fixed corpus (w2LF vs w2A, beam-5 best.pt, read from the saved job summaries)
w2LF - w2A validation per seed -1.230 +0.611 +0.306; mean -0.10  [c17-080]
w2LF - w2A test per seed -0.243 +0.296 +0.026; mean +0.03  [c17-080]
seeds: 3  [c17-079]
```
**Scope.** Recomputed from the saved beam-5 hypotheses of 34 runs (jobs 49135, 31220, 66644, 16435 and 25863) through `slt/metrics.py`, with the official references; every run first reproduces the BLEU-4 value in its job summary (the gate). Differences, signs, counts and the t-interval are computed here. The wave-2 differences are read from the saved job summaries (job 5403), the content-word precision and the 32% from the saved job outputs (66644, 66643), the 20,608 constrained sentences from the saved corpus audit, and the sign counts from the adopted corpus's segment list. The wave 12b differences (job 138380, summary 138381) are formed here from the six per-run records (beam-5 validation and test BLEU-4 of arms A and W) in the saved job output, gated against the job's own printed per-seed lines, mean and t-interval; the 3.45% comes from the adopted corpus's segment list. The 20,608 is printed untagged because its claim also states a half that is not saved locally (6,880 unconstrained sentences). Finding: the validation mean is +0.1147, which the job prints as +0.115 and the dissertation gives as +0.12; a single rounding gives +0.11 (immaterial; interval unchanged). Not done: training, the scrambled-sign control (job 1193231), the seed replication (job 1173725), wave-5 pre-training (job 16397) whose outputs are on Kaya.
