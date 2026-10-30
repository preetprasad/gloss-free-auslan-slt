### AX — Ensemble averaging rule (power mean)
**Claim.** The final ensemble combines its six members by the geometric mean of their probabilities. In a pre-registered comparison on validation (Results, ensemble-ablation paragraph; Appendix B, further-results table), no power mean of the members' probabilities with exponent -1, 0.5, 1 or 2 was distinguishable from the geometric mean: the BLEU-4 differences, each rule at its own length penalty, are -0.68, 0.00, +0.13 and +0.23 with 98.75% intervals that all include zero, and the nominal gain of exponents 1 and 2 is carried by output length (brevity penalty 0.80 for the geometric mean against 0.84 and 0.92) while n-gram precision falls.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/ensemble_averaging_rule/check.py
```
```text
GATE PASSED: control (geometric mean) at LP 0.3: BLEU-4 12.47 (12.470), WER 0.9164; hypotheses identical to job 120153: 800/800
GATE PASSED: recomputed point differences and 98.75% bootstrap intervals equal those saved by job 138383

Rules compared: geometric mean and power means of the members' probabilities with exponents -1, 0.5, 1, 2 (800 validation clips, six members, beam 10; no test clip)
Interval level: 98.75% per comparison (Bonferroni over 4 comparisons), 2000 paired resamples, length penalty re-chosen inside each resample

Each rule at its validation-chosen length penalty:
                          rule   LP  BLEU-4     BP n-gram prec.
   geometric mean (exponent 0)  0.3   12.47  0.797        15.65
       power mean, exponent -1  0.4   11.79  0.838        14.07
      power mean, exponent 0.5  0.4   12.47  0.812        15.35
        power mean, exponent 1  0.5   12.60  0.843        14.96
        power mean, exponent 2  0.7   12.70  0.919        13.82

BLEU-4 difference to the geometric mean, each rule at its own length penalty:
  exponent   -1: -0.68 (-1.79 to +0.10)   [-0.677; 98.75% interval [-1.790, +0.100]]   not distinguishable from the geometric mean
  exponent  0.5: 0.00 (-0.59 to +0.60)   [-0.001; 98.75% interval [-0.591, +0.602]]   not distinguishable from the geometric mean
  exponent    1: +0.13 (-0.53 to +0.74)   [+0.133; 98.75% interval [-0.530, +0.740]]   not distinguishable from the geometric mean
  exponent    2: +0.23 (-0.60 to +1.05)   [+0.230; 98.75% interval [-0.601, +1.050]]   not distinguishable from the geometric mean

Brevity penalty (dissertation rounding): geometric mean 0.80, exponent 1 0.84, exponent 2 0.92; n-gram precision falls from 15.65 to 14.96 (exponent 1) and 13.82 (exponent 2)

Rules measurably different from the geometric mean: 0 of 4
```
**Scope.** Re-chooses the candidate at every length penalty from the saved n-best lists (beam 10, six members, 800 validation clips; Kaya decode job 138382, one list per rule), re-scores them with `slt/metrics.py` (13a on detokenised text), and recomputes the paired bootstrap (2,000 resamples, seed 0, length penalty re-chosen in each resample, Bonferroni level 98.75%) with the original read-out's own functions (`tools/w12d_powermean.py`). Gates: the control reproduces the reported ensemble (12.470, WER 0.9164, 800/800 hypotheses equal to job 120153's six-member decode in `results/ensemble_member_subsets_ablation/`), and the recomputed differences and intervals equal those saved by the read-out job 138383. The decoding itself (running the six checkpoints with each averaging rule) is not repeated here. The n-best lists were copied from `results/wave12d_138382/out/` and split in halves to keep each file small; the saved job summary is used only for the second gate.
