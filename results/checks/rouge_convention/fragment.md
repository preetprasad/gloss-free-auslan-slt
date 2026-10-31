### AX — ROUGE-L tokenisation convention

**Claim.** The ROUGE-L of one set of saved outputs (the baseline model on the 800 official test clips) is 16.56 when hypotheses and references are detokenised and 36.22 when they are whitespace-tokenised (both at beta = 1.0), so tokenisation alone moves it by more than 19 points (abstract, Results, Discussion, Conclusion). The appendix table gives the same for three single models, adds a beta = 1.2 column, and the appendix note gives the ROUGE-L to BLEU-1 ratio of the three models (1.119 to 1.151) beside the published systems.
**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/rouge_convention/check.py
```
```text
GATE Baseline model (DWPose + CTC): beam-hypothesis BLEU-4 9.52 = recorded 9.52 (greedy hypotheses give 8.90, so these are the beam outputs)
GATE 40-epoch pre-trained model: beam-hypothesis BLEU-4 10.27 = recorded 10.27 (greedy hypotheses give 8.75, so these are the beam outputs)
GATE RGB-fusion model: beam-hypothesis BLEU-4 10.33 = recorded 10.33 (greedy hypotheses give 8.82, so these are the beam outputs)

Official Communication test set; saved outputs, one run per model; ROUGE-L x100
clips scored per model: 800 of 800; one saved run per model; beam hypotheses (the beam-5 outputs, see GATE lines)  [c16-038]
ROUGE-L per sentence = F_beta of the longest common subsequence, averaged over the 800 sentences, multiplied by 100  [c06-003]
  Baseline model (DWPose + CTC)  detokenised beta=1.0: 16.56  whitespace-tokenised beta=1.0: 36.22  whitespace-tokenised beta=1.2: 35.70
  40-epoch pre-trained model     detokenised beta=1.0: 17.74  whitespace-tokenised beta=1.0: 36.68  whitespace-tokenised beta=1.2: 36.27
  RGB-fusion model               detokenised beta=1.0: 17.84  whitespace-tokenised beta=1.0: 37.67  whitespace-tokenised beta=1.2: 37.08
beta used per column: 1.0 / 1.0 / 1.2  [c16-043]
Baseline model outputs: ROUGE-L 16.56 detokenised and 36.22 whitespace-tokenised, both at beta = 1.0, on 800 clips  [c12-057]
Baseline model (DWPose + CTC): 16.56 / 36.22 / 35.70  [c16-040]
ROUGE-L of one set of outputs is 16.56 or 36.22  [c14-081 c21-004]
40-epoch pre-trained model: 17.74 / 36.68 / 36.27  [c16-041]
RGB-fusion model: 17.84 / 37.67 / 37.08
  swing Baseline model (DWPose + CTC): +19.65 points
  swing 40-epoch pre-trained model: +18.94 points
  swing RGB-fusion model: +19.83 points
tokenisation alone moves the baseline outputs by 19.65 points, more than 19: True  [c03-017 c12-058]

ROUGE-L (beta 1.2) divided by BLEU-1
  published SL-Luong: 37.27 / 30.15 = 1.236
  published SL-Transformer: 35.65 / 31.31 = 1.139
  published GASLT: 35.74 / 28.19 = 1.268
  Baseline model (DWPose + CTC): ROUGE-L 35.70 / BLEU-1 31.18 = 1.145
  40-epoch pre-trained model: ROUGE-L 36.27 / BLEU-1 32.41 = 1.119
  RGB-fusion model: ROUGE-L 37.08 / BLEU-1 32.22 = 1.151
ratio for the three single models: 1.119 to 1.151  [c18-017]
```
**Scope.** Recomputed from the saved greedy and beam hypotheses with references of three single models (`results/semantic_adequacy_grading/hyps_*.jsonl`) through `metric_suite.py` and `slt/metrics.py`. Gate: the beam hypotheses reproduce the recorded beam-5 test BLEU-4 (9.52, 10.27, 10.33), while the greedy hypotheses do not, which identifies the saved outputs as beam-5. The published systems' ROUGE-L and BLEU-1 are cited values; only their division is computed. No training or inference is run.
