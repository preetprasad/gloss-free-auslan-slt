### AX — Input-blind constant baseline

**Claim.** One five-word training sentence, output for every clip regardless of the video, scores 4.41 BLEU-4 on the 800-clip official test set and 1.09 on the 570 unseen-target clips; 18 test references are exactly that sentence. The trained systems score 4.88 to 6.60 on the unseen clips (Abstract, Introduction, Results null-model paragraph, Discussion).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/input_blind_baselines/check.py
```

```text
references match comm_run/manifests/test.tsv: 800/800
GATE S6 11.802 (recorded 11.802), DWPose+CTC 9.52 (recorded 9.52)

Setting: 800 official test clips, one constant training sentence for every clip, BLEU-4 13a on detokenised text
constant sentence has 6 whitespace tokens incl. final full stop = 5 words
The null model outputs one sentence of 5 words for every clip  [c07-063 c11-079]
Constant, all 800 test clips: BLEU-4 4.41 (4.406)  [c01-025 c02-024 c11-079 c14-011 c14-077]
Test clips: 800 (800)  [c02-024 c11-079]
Constant, 570 unseen-target clips: BLEU-4 1.09 (1.093)  [c01-025 c02-023 c11-081 c14-011 c14-077]
Unseen-target clips: 570 (570)  [c02-023]
Constant, 230 seen-target clips: BLEU-4 10.65
Test references exactly equal to the constant: 18  [c11-079]
Constant: one sentence, the same for all clips (1 distinct output)  [c14-077 c02-023]

Trained systems, unseen-target BLEU-4 (beam 5; S6 = ensemble beam 10, LP 0.3):
DWPose+CTC single model 4.88  [c11-081]
S6 ensemble 6.60  [c11-081]
Constant 1.09 vs trained 4.88 to 6.60; ratio of the lowest trained score to the constant 4.5  [c11-081]
constant / published SL-Transformer (9.2 published) = 0.48  [c14-077]
constant / published GASLT (9.95 published) = 0.44  [c14-077]
```

**Scope.** Recomputed from the saved beam-5 test decodes (semantic-grading dump, job 1075705) and the S6 ensemble dump (job 1125067), the training-target digests and the test manifest; the constant sentence is named by its SHA-1 digest. The gate re-scores the recorded S6 (11.802) and DWPose+CTC (9.52) scores. The published baseline values (9.20, 9.95) are cited, not measured here. No training or inference.
