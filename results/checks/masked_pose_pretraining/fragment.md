### AX — Masked keypoint pre-training (wave 28): checks that ran, no scores quoted

**Claim.** The appendix table of supplementary results (and the departures table) state that nine masked-keypoint pre-training runs (three arms, three seeds) all met their pre-registered training check, that the decision criterion is validation, that the pre-training used 16,800 updates on 44,010 unlabelled dictionary clips (with the Auslan-Daily training clips added in one arm), and that one control seed failed the pre-registered reproduction check, so no comparison is quoted.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/masked_pose_pretraining/check.py
```
```text
read from the saved job output summary_239450.out (job 239450)
pre-training runs reported: 9; TRAINED under the pre-registered check: 9  [c17-110]
nine pre-training runs, all nine met the check: True  [c17-110]
final held-out loss / first evaluation: 0.179 to 0.449 (line 0.5)  [c17-110]
arm MD: pre-training seeds [42, 101, 202] = 3 seeds  [c17-109]
arm MDA: pre-training seeds [42, 101, 202] = 3 seeds  [c17-109]
arm MDR: pre-training seeds [42, 101, 202] = 3 seeds  [c17-109]
latest update at which a held-out loss was evaluated: 16800 (no evaluation beyond it)
(not checked: the 44,010 dictionary-clip count and the pre-training pool sizes are not printed by any saved job output; see claims.json)
arm S: finished recipe runs for seeds ['42', '101']  [c03-021]
arm MD: finished recipe runs for seeds ['42', '101', '202']  [c03-021]
arm MDA: finished recipe runs for seeds ['42', '101', '202']  [c03-021]
arm MDR: finished recipe runs for seeds ['42', '101', '202']  [c03-021]
control S finished for seeds [42, 101] of the three needed: 2 of 3  [c03-021]
reproduction check, control seed 202: epoch-1 ctc loss 24.340 here vs 24.341 recorded for wave 5 (third decimal differs)  [c03-021 c17-110]
number of control seeds failing the reproduction check: 1  [c03-021]
every contrast with the control is labelled NO VERDICT in the summary: 36 lines; contrasts without it carry 'HARNESS NOT REPRODUCED'  [c03-021]
decision rule of the summary reads the 3-seed mean beam-5 VALIDATION difference (2 contrasts: STOPPED, CONTINUES); the criterion is validation, not test  [c17-109]
```

**Scope.** Parsed from two saved job outputs (summary job 239450 and the training log of array 239449, task 2); no number is a score and none is quoted. Nothing is retrained. Checked: the nine pre-training runs and their check, the three seeds per pre-training arm, validation as the decision rule, the failed reproduction check of control seed 202. Not checked (no saved output prints them; listed in `claims.json`): the 44,010 dictionary-clip count, the 12,439 clips added to the pre-training pool (the 12,439 in the training log belongs to the fine-tuning manifest, not to the pre-training data), and the total of 16,800 updates (only the last evaluation point, update 16800, is seen).
