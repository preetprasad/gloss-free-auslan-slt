### AX — Seen/unseen decomposition, split definition and per-signer gap

**Claim.** The official Communication split has 12,441 training, 800 validation and 800 test clips (14,041 in all); all 90 test videos occur in training, 230 of the 800 test references (28.75%, over a quarter) occur verbatim in training, and the two most frequent signers supply 92% of the test clips. Three single models score about 30 BLEU-4 on the 230 seen-target clips and about 5 on the 570 unseen-target clips (5.2 to 6.1 times), and 72.6% of the pre-trained model's outputs are verbatim training sentences (Results, Discussion, Methodology, Appendix). The appendix per-signer row states the signer 1 minus signer 2 gap as +2.2 to +4.1 (p 0.02 to 0.29) and the unseen-target gap as +1.1 to -0.3; this check finds small differences (see Scope).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/seen_unseen_decomposition/check.py
```

```text
Split sizes: train 12,441, validation 800, test 800; total 14,041  [c06-016 c21-016 c09-001 c09-007 c20-011]
Test references occurring verbatim in training: 230 of 800 (28.75%); unseen 570  [c14-010 c14-074 c21-016 c01-026]
Share of test references seen in training 0.29: over a quarter (> 0.25)  [c01-026]
Test videos 90; of these in training 90 of 90  [c07-060]
Test clips by signer: signer 1 389, signer 2 350; two most frequent signers 739 of 800 = 92.4%  [c05-053]
GATE S6 ensemble (beam 10, LP 0.3) official test BLEU-4 11.802 (recorded 11.802)

Setting: all 800 official test clips, beam width 5, three single models
DWPose+CTC  all   9.52 (n=800)  seen  29.72 (n=230)  unseen   4.88 (n=570)  ratio 6.09  [c11-059 c11-066 c11-067 c14-075 c09-024]
VLP-40      all  10.27 (n=800)  seen  29.91 (n=230)  unseen   5.31 (n=570)  ratio 5.63  [c11-059 c11-066 c11-067 c14-075 c09-024]
RGB-fusion  all  10.33 (n=800)  seen  30.22 (n=230)  unseen   5.81 (n=570)  ratio 5.20  [c11-059 c11-066 c11-067 c14-075 c09-024]
GATE recorded beam-5 test scores 9.52 / 10.27 / 10.33 for DWPose+CTC / VLP-40 / RGB-fusion
Seen-target range 29.7 to 30.2; unseen-target range 4.88 to 5.81; ratio 5.2 to 6.1  [c11-059 c14-075 c11-066]
Seen about 30 (mean 29.9), unseen about 5 (mean 5.3); 230 seen / 570 unseen clips; 3 models  [c11-066 c14-075]
Outputs that are verbatim training sentences, DWPose+CTC: 80.8%  [c10-038]
Outputs that are verbatim training sentences, VLP-40: 72.6%  [c10-038]
Outputs that are verbatim training sentences, RGB-fusion: 70.8%  [c10-038]
Outputs that are verbatim training sentences, S6: 84.8%  [c10-038]

Signer 1 minus signer 2, official test, beam 5 (within-video permutation, 2000 draws, seed 0)
DWPose+CTC  signer1-signer2 +4.08  perm p 0.029  unseen-target only +0.61
VLP-40      signer1-signer2 +3.91  perm p 0.039  unseen-target only +0.42
RGB-fusion  signer1-signer2 +2.15  perm p 0.297  unseen-target only -0.28
S6 ensemble (not a single model) signer1-signer2 +4.42, unseen-target only +1.11
Single-model gap range +2.147 to +4.085 (to one decimal: +2.1 to +4.1); permutation p between 0.03 and 0.30  [c18-023]
  (recorded p 0.02 to 0.29 came from an earlier 300-draw run (seed not kept); p agrees only within permutation noise)  [c18-023]
Unseen-target gap, single models: +0.61 to -0.28; with the S6 ensemble the upper end is +1.11  [c18-024]
```

**Scope.** Recomputed from saved per-clip decodes (hypotheses and references), the training-target digests and the spreadsheet; gate is the reported ensemble's 11.802 and the three single models' recorded 9.52, 10.27 and 10.33. The permutation test shuffles Signer_ID within each video (2000 draws, seed 0); the original 300-draw run is not recoverable, so p differs slightly. The signer 1 minus signer 2 gap of the RGB-fusion model is 2.147 (dissertation +2.2) and the +1.1 upper end of the unseen-target gap belongs to the S6 ensemble, not a single model. No training or inference is done.
