### AX — Extra pre-training cycle with fresh seeds
**Claim.** Appendix B (the supplementary-results table) reports a pre-registered re-test of an extra first training cycle: 92 epochs on the adopted corpus before the usual 40-epoch training, on five new seeds. Against training without it, beam-5 test BLEU-4 was lower by 0.56 (interval -1.41 to +0.28), higher in one seed of five; validation was higher by 0.26 (-0.12 to +0.65); it was not adopted (one-sided p = 0.93). A first cycle of the same number of steps on the 13,760 synthetic clips alone scored -0.07 (-0.59 to +0.45) against no first cycle and +0.49 (-0.26 to +1.25) against the joint one; these are bounds only.
**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/extra_pretraining_cycle/check.py
```
```text
GATE: 30 decodes re-scored; max |recomputed - BLEU recorded by the jobs| = 0.00e+00
GATE: PJ seed 808 beam-5 test = 10.080669 (recorded in the read-out: 10.080669)
first cycle on the adopted corpus (PJ): [92] epochs; then main training: [40] epochs; seeds: 5 ([505, 606, 707, 808, 909])
first cycle on the synthetic clips alone (PS): [175] epochs
read from the job script header: PJ = pre-train 92 epochs on the adopted joint corpus (12,439 real + 13,760 synthetic clips = 26,199;
read from the job script header: PS = pre-train 175 epochs on the 13,760 synthetic clips of that corpus only (2,408,000 clip-passes,
read from the job script header: N_TRAIN=12439; N_SYN=13760; N_VAL=800; N_TEST=800

PJ - C, beam-5 test (primary): -0.56 (-1.41 to +0.28)  [-0.562 [-1.407, +0.284]]  higher in 1 of 5 seeds
   per seed: -1.230 / +0.466 / -0.456 / -1.136 / -0.452
PJ - C, beam-5 validation: +0.26 (-0.12 to +0.65)  [+0.265 [-0.122, +0.651]]  higher in 3 of 5 seeds
one-sided paired t on the five test differences: t = -1.844, p = 0.93 (0.931); Holm level 0.05 (m = 1): retain
both fresh seeds (808, 909) positive on test: False

PS - C, beam-5 test (synthetic first cycle vs none): -0.07 (-0.59 to +0.45)  [-0.069 [-0.589, +0.452]]  higher in 3 of 5 seeds
PS - PJ, beam-5 test (synthetic vs joint first cycle): +0.49 (-0.26 to +1.25)  [+0.493 [-0.264, +1.250]]  higher in 4 of 5 seeds

five-seed means, beam-5 val / test: C 11.385 / 11.336; PJ 11.650 / 10.775; PS 11.234 / 11.268
```
**Scope.** Re-scores the 30 saved beam-5 decodes of the best checkpoints (arms: no first cycle, joint first cycle, synthetic-only first cycle; seeds 505, 606, 707, 808, 909; validation and test; length penalty 0.6) through `slt/metrics.py` (detokenised, sacreBLEU 13a). The gate requires every recomputed score to equal the BLEU the original jobs recorded (maximum deviation 0) and the seed-808 joint-arm test score recorded in the read-out. Differences, 95% t-intervals (exact quantile, 4 degrees of freedom) and the one-sided paired t-test are then recomputed from the per-seed scores. The epoch counts come from the saved per-run progress files; the 13,760 synthetic clips are read from the header of the job script (not re-counted from clips, which are not shipped). Training is not re-run. Original jobs: Kaya 138389 (seeds 505-707, summary 138390) and 140946 (all seeds, summary 140947); the decodes were copied from `results/wave12h_140946/`.
