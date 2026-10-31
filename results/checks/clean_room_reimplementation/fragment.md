### AX — Clean-room code and own-code fine-tuning of the pre-trained model
**Claim.** Appendix B and the transfer section of the Results state that an independent implementation, written from the paper and a behavioural description, reproduces the private port's output logits on the released checkpoint to within about 2e-5 on 20 validation clips; that re-decoding the five port-fine-tuned models with our own code gives a test mean of 17.86 (validation 17.35, unseen 13.49) against the port's 18.21 (17.37, 13.73), with most outputs identical; that fine-tuning with our own code gives 17.72 validation, 18.27 test and 13.49 unseen over three seeds, level with the port (+0.35 validation, +0.06 test); and that adding stitched data and pose augmentation (KSA) lowers validation BLEU-4 from 17.72 to 16.84 (mean difference -0.88, -2.99 to +1.23; test -0.64, -3.16 to +1.89; one of three seeds positive on validation) and is stopped by the futility rule (Table `tab:app_transfer`, `tab:app_transfer_diff`).

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/clean_room_reimplementation/check.py
```
```text
gates passed: every run re-scores to the BLEU-4 recorded by its own job (tolerance 1e-6)

[1] Independent implementation vs the private port, released checkpoint (read from the saved job summary gk_job186422.json)
  clips compared: 20; all in the validation manifest: True
  max |logit difference| = 1.9e-05  (rounds to 2e-05; tolerance 1e-03)
  exponent form: 1.9 x 10^-5; tolerance exponent -3
  max relative embedding difference = 5.3e-07; passed = True

[2] The five fine-tuned models decoded with our own code (beam 5, LP 1.0) vs the port's decodes
  seed | split | port BLEU-4 | our code | identical hypotheses
    42 | val   | 17.19 | 17.65 | 678/800 = 84.8%
   101 | val   | 17.68 | 16.98 | 654/800 = 81.8%
   202 | val   | 17.29 | 17.25 | 668/800 = 83.5%
   303 | val   | 17.24 | 17.03 | 680/800 = 85.0%
   404 | val   | 17.44 | 17.84 | 659/800 = 82.4%
    42 | test  | 17.80 | 17.23 | 645/800 = 80.6%
   101 | test  | 18.43 | 18.58 | 664/800 = 83.0%
   202 | test  | 18.16 | 18.10 | 665/800 = 83.1%
   303 | test  | 18.77 | 17.97 | 682/800 = 85.2%
   404 | test  | 17.87 | 17.43 | 658/800 = 82.2%
  val: port mean 17.37 / our code 17.35; unseen-570 11.95 / 11.92
  test: port mean 18.21 / our code 17.86; unseen-570 13.73 / 13.49
  identical hypotheses over the ten seed x split comparisons: 80.6% to 85.2%
  rounded as in the dissertation: 81 to 85% of outputs identical

[3] Fine-tuning with our own code, 3 seeds (42/101/202): K = plain, KSA = + stitched data and pose augmentation
  system | split | mean BLEU-4 | unseen-570 (test only)
  K   | val   | 17.72
  K   | test  | 18.27 | unseen 13.49
  KSA | val   | 16.84
  KSA | test  | 17.63 | unseen 13.21
  KSA minus K, per seed (seed-level 95% t-interval):
  val: -1.50, -1.22, +0.09; mean -0.88 (-2.99 to +1.23); positive in 1 of 3 seeds
  test: -1.20, +0.54, -1.24; mean -0.64 (-3.16 to +1.89); positive in 1 of 3 seeds
  futility rule (3-seed mean validation difference <= 0): STOP

[4] Our code minus the port, fine-tuned models (K 3 seeds minus port 5 seeds)
  val: 17.72 - 17.37 = +0.35
  test: 18.27 - 18.21 = +0.06
```
**Scope.** Re-scores 32 saved decodes (hypotheses against the official references, BLEU-4 with the project's sacreBLEU 13a convention; unseen = the 570 test references absent from training, from `results/train_target_hashes.txt`) and first requires every one to reproduce the BLEU-4 its own job recorded. The five port-fine-tuned models (training job 175088 and its first stage 167035; seeds 42, 101, 202, 303, 404), their clean re-decodes (job 193342) and the three-seed plain and KSA fine-tunes (job 195640) were trained and decoded on the cluster; the check does not train or decode. The seed-level t-intervals and the identical-hypothesis counts are recomputed from the saved decodes. The logit-parity figure (job 186422) is read from the saved job summary: its per-item dumps are derived from the checkpoint and were not kept. The 20 clip names it used are checked against the validation manifest. Not recomputed: the ensemble rows, the ten-point behavioural description, and the finding that the cause of the re-decode difference is unconfirmed. The identical-hypothesis range in the saved data is 80.6% to 85.2%, wider than the 82 to 85% in the text.
