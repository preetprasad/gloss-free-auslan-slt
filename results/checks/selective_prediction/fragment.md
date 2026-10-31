### AX — Selective prediction: answering only the confident clips

**Claim.** If the system answers only the clips it is most confident about (confidence = the model's own cross-entropy on its output, plus self-reliance), BLEU-4 of the answered clips rises. For the single DWPose+CTC model on the 800 official test clips it goes from 9.52 at full coverage to 17.42 at 50% coverage (16.13 with the cross-entropy term alone), on validation from 9.64 to 18.42, and on the 570 unseen-target clips from 4.88 to 7.92. Keeping the longest outputs instead gives only 10.48 at 50% and 7.22 at 25%; at 25% coverage 98% of the answered outputs are verbatim training sentences. With the six-member ensemble and thresholds fixed on validation, 51.9% of test clips are answered at 19.07, +7.30 above random abstention at the same coverage (Results, selective prediction; Appendix tables of the selective analysis).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/selective_prediction/check.py
```

```text
GATE full-set beam-5 BLEU-4: test 9.52 (recorded 9.52), validation 9.64 (recorded 9.64)

Setting: single model, beam 5 hypotheses (field hyp_beam of the saved decodes), n test 800, n validation 800, unseen-target test clips 570 (validation 549)  [c16-044]
Coverage = clips answered / clips, from 0% to 100%; here 100% = all 800 answered  [c06-010]
At 50% coverage the system answers 400 of 800 test clips (its more confident half, coverage 0.5)  [c01-029 c02-029 c18-036]
Coverages evaluated: 90%, 75%, 50% and 25% (plus 100%)  [c07-073]

Retained-set BLEU-4 by coverage (ranking: self-CE + self-reliance, z-sum)
                           100%      90%      75%      50%      25%  [c16-046]
Test                       9.52    10.37    12.05    17.42    30.82  [c16-046]
Validation                 9.64    10.37    12.44    18.42    30.51  [c16-047]
Test, self-CE alone        9.52    10.28    11.97    16.13    29.24
Val, self-CE alone         9.64    10.45    12.15    17.06    32.32

Test, all 800 clips, 100% -> 50%: 9.52 -> 17.42  [c01-030 c02-030 c13-001 c13-003 c13-008]
Validation 100% -> 50%: 9.64 -> 18.42  [c13-005]
Self-CE term alone at 50% coverage: test 16.13, validation 17.06; two-term score 17.42 and 18.42  [c13-002 c18-035]
Beam width 5 (beam-5 decodes; the full-set score matches the recorded beam-5 value above)  [c13-004]

Unseen-target test clips (570): all 4.88; the more confident half of the 800 clips, unseen ones only (238): 7.92  [c13-007 c16-048 c02-030]
Unseen-target validation: all 4.27, confident half 7.24

Control, keep the longest outputs: 50% coverage 10.48  [c13-006]; 25% coverage 7.22  [c18-037]
Baseline model, 25% coverage: 98.0% of the answered outputs are verbatim training sentences (all 800: 80.8%)  [c13-018]
Signer repeats 50% of utterances at 50% coverage, i.e. one in two  [c14-092]

S6 ensemble, beam-10 n-best, length penalty 0.3; thresholds from validation, applied unchanged to test
GATE ensemble test BLEU-4 11.802 (recorded 11.802)
all    cov 0.90: test  12.90 @ 89.4% (715 clips); selected - random at matched coverage +1.10 [+0.39, +1.94]
all    cov 0.75: test  14.61 @ 75.6% (605 clips); selected - random at matched coverage +2.80 [+1.53, +4.29]
all    cov 0.50: test  19.07 @ 51.9% (415 clips); selected - random at matched coverage +7.30 [+4.55, +10.17]  [c13-012 c18-038]
unseen cov 0.90: test   7.18 @ 88.8% (506 clips); selected - random at matched coverage +0.59 [+0.05, +1.43]
unseen cov 0.75: test   7.71 @ 76.7% (437 clips); selected - random at matched coverage +1.12 [+0.14, +2.33]
unseen cov 0.50: test   8.95 @ 55.1% (314 clips); selected - random at matched coverage +2.36 [+0.47, +4.42]  [c18-039]
Ensemble answers 51.9% of the 800 official test clips with the validation threshold for 50% coverage  [c13-012]
```

**Scope.** Recomputed from per-clip saved outputs: the single model's beam-5 hypotheses, references, self-CE and self-reliance (800 test and 800 validation clips, saved by `tools/self_reliance.py`), re-scored through `slt/metrics.py`, and the ensemble's saved n-best lists (jobs 1101821 / 1125067), with the 2,000-resample bootstrap of `tools/selective_ci.py` (seed 0). Gates: the full-set scores 9.52 / 9.64 and 11.802 and the ensemble's three recorded coverage points must reproduce or the check exits non-zero. Not done: decoding or computing self-CE and self-reliance (model inference, run once locally). The 570/238-clip unseen figure ranks all 800 clips and then keeps the unseen ones among the answered half.
