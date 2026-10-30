### AX — Published systems run with their own code
**Claim.** The dissertation (Results section on the published systems' code, Table `tab:res_published` and the text below it; Appendix B, Table `tab:app_ownccode`, blocks 1 and 2) states that the authors' own SL-Transformer and GASLT code, run here with three seeds on the signer's AlphaPose keypoints (56 points, 200 epochs at a constant learning rate, batch size 64), scores 8.77 and 9.16 test BLEU-4 against the published 9.20 and 9.95, within the pre-registered tolerance of 1.0 (the intervals of GASLT and of the standardised SL-Transformer cell cross the tolerance edge, so they are only consistent with the published score). On 53 DWPose points the same code scores below this project's translator trained for 40 epochs in the same jobs (differences with 95% intervals, three and five seeds), and DWPose minus AlphaPose inside their code is not resolved.
**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/published_systems_own_code/check.py
```
```text
Gate passed: all 24 authors'-code runs (and the 10 further DWPose runs of the five-seed comparison) re-score to the
  recorded project and authors' BLEU-4 (1e-6); the 3 + 3 + 2 translator control runs re-score to their recorded values;
  references identical across the 24 runs (800 clips).
Training length of every authors'-code run (8 cells x 3 seeds): [(200, 'Training ended after 200 epochs.')]
Seeds per cell: 3 (42, 101, 202)

Block 1: the authors' own code on the signer's AlphaPose keypoints (test BLEU-4, authors' script, 3-seed mean, 95% t-interval)
  SL-Transformer  ap56_raw    8.77  [8.25, 9.29]  published 9.20; |diff| 0.43 <= 1.0: True; interval inside the band: True
  SL-Transformer  ap56_frame  8.87  [7.50, 10.23]  published 9.20; |diff| 0.33 <= 1.0: True; interval inside the band: False  (interval crosses the tolerance edge: consistent with, not shown)
  GASLT           ap56_raw    9.16  [7.36, 10.96]  published 9.95; |diff| 0.79 <= 1.0: True; interval inside the band: False  (interval crosses the tolerance edge: consistent with, not shown)
  GASLT           ap56_frame  9.67  [9.23, 10.11]  published 9.95; |diff| 0.28 <= 1.0: True; interval inside the band: True
  project scoring function, paper-literal cell (ap56_raw), SL-Transformer: 8.66
  project scoring function, paper-literal cell (ap56_raw), GASLT: 8.69

Block 2: authors' code on 53 DWPose points minus this project's translator (40 epochs, beam 5), test BLEU-4, project function
  SL-Transformer  3 seeds (8.44 against 10.30): -1.86  [-2.71, -1.02]
  GASLT           3 seeds (9.70 against 10.30): -0.60  [-1.44, +0.24]
  SL-Transformer  5 seeds (8.63 against 10.43): -1.80  [-2.27, -1.34]
  GASLT           5 seeds (9.58 against 10.43): -0.85  [-1.32, -0.39]
  Authors' code only: DWPose minus the signer's AlphaPose cut to the same 53 points (3 seeds, project function):
    SL-Transformer +0.07 [-0.86, +0.99];  GASLT +0.49 [-1.35, +2.34]
```
**Scope.** Recomputed from the saved per-clip test hypotheses of every run: the authors' scoring (sacreBLEU on the tokenised text, `tokenize='none'`) and this project's scoring (`slt/metrics.py`: detokenise, sacreBLEU 13a) are both re-applied and must equal the values the jobs recorded (the gate, 1e-6); the seed means, t-intervals and pooled-variance differences are computed here. Not recomputed: training and decoding (the runs were made with the authors' code, which is not part of this repository, on a cluster: jobs 197330 to 197332 for the three-seed cells and the in-job translator control; jobs 167758 and 179411 for the five-seed comparison, of which seeds 42, 101 and 202 come from the first and seeds 303 and 404 from the second); the published scores 9.20 and 9.95 are literature values entered as constants; the learning rate, batch size, number of keypoints, the 40-epoch length of the translator and the 56/53-point input layouts are job settings that the saved hypotheses do not carry (the check prints only the 200 epochs the scores files record). The earlier runs of this code on the first-listed AlphaPose person are not used.
