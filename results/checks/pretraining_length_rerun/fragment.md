### AX — Pre-training length and News frame limit chosen on validation alone

**Claim.** The pose-text pre-training length and the News input-frame limit were re-chosen on validation alone (Method; Results, pre-training and News paragraphs; Discussion limitations). Five pre-training settings (none, 40, 80, 128 and 256 epochs) were each trained with three seeds; the pre-registered rule selects 40 epochs, and no pre-trained length beats no pre-training (40 epochs minus none: validation −0.258 [−1.57, +1.06], test −0.890 [−2.14, +0.36]). On News, mean beam-5 validation BLEU-4 is 2.07 at 256 frames and 2.08 at 512 (+0.01, interval −0.37 to +0.39), so the cheaper 256 frames is selected.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/pretraining_length_rerun/check.py
```
```text
GATE P0 mean beam5_val 10.450 (recorded 10.450)
GATE P40 mean beam5_val 10.192 (recorded 10.192)
GATE P0 mean beam5_test 10.122 (recorded 10.122)
GATE P40 mean beam5_test 9.232 (recorded 9.232)
pre-training lengths in the job script w11_vlpepochs.sbatch: [40, 80, 128, 256] epochs (0 = no pre-training); seeds [42, 101, 202]  [c07-020 c07-021]
pre-training epochs completed per arm (saved progress files): P40: 40, P80: 80, P128: 128, P256: 256  [c07-020]
seeds per setting: 3 (P0: 3; P40: 3; P80: 3; P128: 3; P256: 3) = 3 seeds  [c07-021 c09-029 c09-034]

PRE-TRAINING RE-RUN (beam-5 BLEU-4, mean of three seeds)
  P0    validation 10.450  test 10.122
  P40   validation 10.192  test 9.232
  P80   validation 10.001  test 9.810
  P128  validation 9.827  test 9.697
  P256  validation 9.907  test 9.398
pre-registered rule (tie 0.10, as in the recorded read-out command): cheapest length within 0.10 of the best mean validation 10.192
SELECTED ON VALIDATION: P40 = 40 pre-training epochs
  P40 - P0 validation -0.258 [-1.57, +1.06]; test -0.890 [-2.14, +0.36]
  P80 - P0 validation -0.450 [-1.09, +0.19]; test -0.311 [-0.66, +0.04]
  P128 - P0 validation -0.624 [-2.37, +1.12]; test -0.425 [-1.58, +0.73]
  P256 - P0 validation -0.544 [-2.97, +1.88]; test -0.723 [-1.45, +0.01]
among the pre-trained lengths the highest mean is P40 on validation and P80 on test

NEWS FRAME LIMIT (150 epochs, News real clips only, three seeds)
  256 frames: mean beam-5 validation BLEU-4 2.07 (2.065); test 1.875  [c11-032]
  512 frames: mean beam-5 validation BLEU-4 2.08 (2.077); test 1.815  [c11-032]
  512 - 256 validation +0.01 (+0.012); 95% t-interval over the 3 seeds -0.37 to +0.39  [c11-033]
  rule (cheapest within 0.05 of the best mean validation): 256 frames selected  [c11-034 c06-026]
  job script w11_newsframes.sbatch (the wave-11 re-run, not the earlier runs): frame limits ['256', '512'], 150 epochs, seeds ['42', '101', '202']
  frame limit in the earlier News-only job scripts: w4_news.sbatch 512, news_bpe.sbatch 512, w12a_news_decomp.sbatch 512; selected on validation in wave 11: 256  [c06-026]
  news_long.sbatch (earlier single-run job, 150 epochs, outputs not saved locally) compares: 256-frame against 512-frame  [c18-069]
```

**Scope.** Recomputed from the stored beam-5 BLEU-4 of each of the 21 runs (cluster jobs 116619, pre-training arms, and 116620, News arms; read from the saved job outputs), with the selection rule and 95% t-intervals computed here. No training or decoding is repeated. The frame limits of the earlier News-only runs (c06-026) and of the earlier single-run comparison (c18-069) are read from the recorded job scripts (`w4_news.sbatch`, `news_bpe.sbatch`, `w12a_news_decomp.sbatch`, `news_long.sbatch`); the scores of that earlier comparison (job 1196181, c18-070), like the single-run 128- and 256-epoch scores, are on the cluster only and are not checked here.
