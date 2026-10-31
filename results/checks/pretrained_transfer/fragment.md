### AX — Transfer from a pre-trained sign-language model
**Claim.** The dissertation (abstract, introduction, Results "Transfer from a Pre-trained Sign-Language Model", Appendix `app_transfer`, conclusion) states that fine-tuning the released How2Sign checkpoint on Auslan-Daily Communication gives a five-seed mean test BLEU-4 of 18.21 (validation 17.37; 13.73 on the 570 unseen-target clips), against 10.37 for the smaller translator of this thesis trained from scratch in the same jobs (+7.84, 95% seed-level interval +7.04 to +8.64, five of five seeds; +5.79 after adjusting for output length), 6.60 with random pose-side weights (one run collapsed to 30 distinct outputs), and 20.60 for the five-model ensemble. The port reproduces the checkpoint's How2Sign output at 14.37 against 14.50.
**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/pretrained_transfer/check.py
```
```text
GATE: all 30 saved decodes re-score to the BLEU-4 and distinct-output count recorded by their jobs (tolerance 1e-6)

== Five-seed mean beam-5 BLEU-4 (seeds [42, 101, 202, 303, 404]); unseen = 570 test clips whose reference is absent from training ==
arm          validation     test  unseen (test)
ctrl_small   val 10.52 test 10.37 unseen 5.47 over 5 seeds  [c11-046 c21-002 c21-012 c19-020 c11-048]
u_scratch    val 7.06 test 6.60 unseen 3.21 over 5 seeds  [c11-049 c19-021]
u_asl        val 17.37 test 18.21 unseen 13.73 over 5 seeds  [c01-022 c03-008 c11-014 c11-045 c21-006 c21-012 c21-005 c13-055 c19-019 c11-048]
570 unseen-target test clips  [c11-048 c19-047]
per-seed test BLEU-4 of the fine-tuned arm u_asl: 17.80 18.43 18.16 18.77 17.87
BLEU-4: fine-tuned test mean 18.21 (five seeds) against published best 9.95 (published value, cited)  [c03-008 c11-045]
from scratch test mean 10.37; random pose-side weights 6.60  [c21-012 c11-046]

== Paired comparisons, seed-level 95% t-interval (df = 4) ==
95% t-interval over 5 seeds (df = 4)  [c03-010 c11-047]
test fine-tuned minus from scratch: +7.84 [+7.04, +8.64]  positive in 5 of 5 seeds  [c03-010 c11-047 c19-029 c19-030 c11-047]
test fine-tuned minus random pose-side weights: +11.61 [+10.09, +13.13]  positive in 5 of 5 seeds  [c19-031 c19-030 c11-047]
val  fine-tuned minus from scratch: +6.85 [+6.67, +7.02]  positive in 5 of 5 seeds  [c19-029 c19-030 c11-047]
val  fine-tuned minus random pose-side weights: +10.31 [+9.06, +11.56]  positive in 5 of 5 seeds  [c19-031 c19-030 c11-047]
test fine-tuned minus from scratch, adjusted for output length (brevity penalty): +5.79 [+4.82, +6.76] (unrounded 5.7917 [4.8187, 6.7648])  [c19-030]

== Distinct outputs (of 800 clips) ==
val  u_scratch  s42:204 s101:274 s202:242 s303:30 s404:159
val  u_asl      s42:568 s101:586 s202:587 s303:544 s404:585
test u_scratch  s42:193 s101:267 s202:216 s303:31 s404:170
test u_asl      s42:576 s101:594 s202:601 s303:543 s404:588
random-weight runs with fewer than 100 distinct validation outputs (the collapse guard of 100): 1 -> seed 303 has 30 distinct outputs  [c11-049 c19-032]

== Episode specificity of the per-clip gain (3-seed mean sentence-BLEU lift, 89 videos with at least 3 clips) ==
between-video permutation test, 2,000 permutations: p = 0.094  [c19-049]

GATE: ensemble re-scores to the BLEU-4 recorded by its job (20.6044)
five-model ensemble (beam 10, length penalty 0.8) test BLEU-4 20.60 (20.604)  [c13-055 c21-005 c21-006]

How2Sign port parity (read from the saved job output gate_how2sign_port.json, 2349 clips, beam 4): port 14.37 (14.372), released output 14.50  [c19-009]
GATE target 14.5 +- 0.5: passed
```
**Scope.** Recomputed from the saved beam-5 hypotheses of 3 arms x 5 seeds x 2 splits (Kaya jobs 167035/167036, seeds 42/101/202, and 175088/175089, seeds 303/404) and the saved ensemble decode (job 195361, beam 10, length penalty 0.8), scored against the official references; every decode must re-score to the BLEU-4 recorded by its job. The How2Sign parity numbers are read from the saved job output of the gate. Not recomputed: training, decoding, the YouTube-ASL identifier lookup (10,327 / 769) and the overlap with external label text (13 clips; 4.37 / 13.75), whose inputs were not saved. The upper bound of the length-adjusted interval comes out as +6.76 (+6.7648), the dissertation prints +6.77.
