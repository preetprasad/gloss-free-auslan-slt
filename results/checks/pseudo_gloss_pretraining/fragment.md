### AX — CTC-only pseudo-gloss first stage against a compute-matched control (wave 23)

**Claim.** Appendix B (supplementary results table, the row on CTC-only pseudo-gloss pre-training, wave 23) states that a first stage of 40 epochs with only the pseudo-gloss CTC loss, then the 40-epoch training from its encoder, was compared with 40 further epochs of the full loss with the same transfer, over three seeds: validation +0.39 (-0.81 to +1.60; higher in two of three seeds), test +0.38 (-0.34 to +1.10; all three); the 40 further epochs alone against no first stage: test +0.32 (-0.10 to +0.75); the CTC-only stage against no first stage: test +0.70 (+0.10 to +1.31); and that the control differed from an earlier job by 0.62 at one seed.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/pseudo_gloss_pretraining/check.py
```
```text
val references match comm_run/manifests/val.tsv: 800/800
test references match comm_run/manifests/test.tsv: 800/800
GATE: all 18 recomputed beam-5 BLEU-4 scores (validation and test) equal the job's recorded values to 1e-6
GATE: the 6 recomputed beam-5 test scores of the earlier jobs (wave 16 and wave 6 controls) equal those jobs' recorded values to 1e-6

Setting: beam 5, best.pt by validation, 800 clips per split, seeds 42/101/202; X = CTC-only first stage then 40-epoch training, Y = 40 further full-loss epochs then 40-epoch training, C = neither

Beam-5 BLEU-4 per seed (42 / 101 / 202) and mean
  C val: 11.307 / 11.939 / 11.338   mean 11.528
  C test: 11.711 / 11.366 / 10.411   mean 11.163
  X val: 11.845 / 12.162 / 12.453   mean 12.153
  X test: 12.185 / 12.045 / 11.372   mean 11.867
  Y val: 11.879 / 11.871 / 11.534   mean 11.761
  Y test: 12.136 / 11.491 / 10.827   mean 11.485

Paired differences, 95% t interval over three seeds (df = 2)
  X - Y val: mean +0.39 (-0.81 to +1.60); higher in 2 of 3 seeds; per seed -0.034 +0.291 +0.919  [c17-102]
  X - Y validation: higher in 2 of 3 seeds, interval -0.81 to +1.60  [c17-102]
  X - Y test: mean +0.38 (-0.34 to +1.10); higher in 3 of 3 seeds; per seed +0.049 +0.555 +0.545  [c17-103]
  X - Y test: higher in 3 of 3 seeds  [c17-103]
  Y - C test: mean +0.32 (-0.10 to +0.75); higher in 3 of 3 seeds; per seed +0.425 +0.124 +0.416  [c17-104]
  X - C test: mean +0.70 (+0.10 to +1.31); higher in 3 of 3 seeds; per seed +0.474 +0.679 +0.960  [c17-105]
  Y - C val: mean +0.23 (-0.56 to +1.03); higher in 2 of 3 seeds; per seed +0.571 -0.068 +0.196
  X - C val: mean +0.63 (-0.50 to +1.75); higher in 3 of 3 seeds; per seed +0.537 +0.223 +1.115

Control against earlier jobs of the same recipe: beam-5 test of control C (this job) minus the earlier control, per seed
  wave 16 control A: seed 42: this job 11.711, earlier 11.711, difference +0.000; seed 101: this job 11.366, earlier 10.745, difference +0.621; seed 202: this job 10.411, earlier 10.411, difference +0.000
  wave 6 control A: seed 42: this job 11.711, earlier 11.711, difference +0.000; seed 101: this job 11.366, earlier 11.366, difference +0.000; seed 202: this job 10.411, earlier 10.839, difference -0.428
  the control differed from the wave 16 control (same scores as the wave-5 control the read-out compares with) by 0.62 at seed 101, and by 0.00 and 0.00 at seeds 42 and 202  [c17-106]
  difference at seed 101 against wave 16 control A: +0.621; against wave 6 control A: +0.000 (this job's control reproduces wave 6's at this seed)
  read from the saved job output summary_179457.txt: differences from wave 5's control (seed: got - control) {42: 0.0, 101: 0.621, 202: 0.0}
GATE: the differences from wave 16 control A (recomputed here) equal the read-out job's recorded differences from wave 5's control at all three seeds to 1e-3, so the wave 16 control carries wave 5's scores  [c17-106]

Training length and seeds (read from the saved job output progress.json of each run)
  all 9 runs (C, X, Y x seeds 42/101/202) completed 40 training epochs of the full loss (progress.json)
  seeds per arm: 3  [c17-101]
```

**Scope.** Beam-5 BLEU-4 of best.pt is recomputed from the saved hypotheses and references of all 18 runs (arms C, X, Y; seeds 42/101/202; validation and official test) with the project's metric code, gated against the scores the job itself recorded (to 1e-6); differences, sign counts and 95% t intervals (df = 2) are computed here. The "earlier job" is not named in the dissertation. The read-out compares the control with wave 5's control, whose per-item output is on Kaya and not local; its saved difference line (read from the job output `summary_179457.txt`) is {42: 0.0, 101: 0.621, 202: 0.0}. The local control of wave 16 (same recipe, same scores) is recomputed from its saved hypotheses and gives the same three differences (seed 101: 11.366 against 10.745, +0.621), and wave 6's control, also recomputed, equals this job's control at seed 101. Training length (40 epochs) is read from progress.json. Not covered: that the "40 further epochs" of arm Y are a continuation of C's training (c17-100) and the 40-epoch CTC-only first stage itself (c17-099): both are stated only by the job script and the pre-registration, not by any saved output. Not done: training and decoding. Original jobs: training 179456, read-out 179457 (smoke 179455); earlier controls: wave 16 (174916), wave 6 (16435).
