### AX — WER null floor, brevity trade and error decomposition

**Claim.** The Results chapter, Methodology and appendix state that WER is uninformative on this corpus: an empty output scores 1.000, a constant sentence 0.950 against the RGB-fusion model's 0.950 and the final ensemble's 0.907, and cutting the RGB-fusion output to three words lowers WER to 0.900 (0.8999) while BLEU-4 falls from 10.33 to 4.87. The appendix adds that substitutions are 56 to 58% of word edits (0.53 of reference length), that output is 0.79 to 0.86 of reference length, that mean output length correlates +0.990 with WER over six sets of outputs, and that a per-sentence oracle reaches 0.807. The Methodology defines BLEU-4 (sacreBLEU, 13a, detokenised) and ROUGE-L (beta 1.2) and the worked BLEU/WER example of clip 104.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/wer_bleu_null_models/check.py
```

```text
references match comm_run/manifests/test.tsv: 800/800
GATE RGB-fusion beam 5: BLEU-4 10.335 WER 0.9497 (recorded 10.335 / 0.9497)
GATE six-member ensemble: BLEU-4 11.802 WER 0.9069 (recorded 11.802 / 0.9069)

Test set: 800 official Communication clips (references only, no training data used)  [c12-050 c12-053]
BLEU-4 computed with sacreBLEU|4-gram|tokenize=13a|detok=True|smooth=exp  [c06-002 c06-007]
detokenised text, sacreBLEU 13a tokeniser, scale 0 to 100: scores below lie between 0 and 100 for 3 single models x 2 decodings, 4-gram  [c06-001 c06-007 c10-022]

NULL MODELS (no video used), WER and BLEU-4 on the official test set
3 null models scored for WER: 1 empty output and 2 constant sentences  [c07-067]
empty output                   WER 1.0000  BLEU-4 0.00
constant A (sha1 0edb4308)     WER 0.9988  BLEU-4 0.00
  constant A (sha1 0edb4308): WER 0.9988 BLEU-4 0.00  [c12-045]
constant B (sha1 d49b14b4)     WER 0.9504  BLEU-4 0.38
  constant B (sha1 d49b14b4): WER 0.9504 BLEU-4 0.38  [c12-046 c12-053 c12-054 c02-016]
empty output WER 1.000; lowest null-model WER (floor) 0.950; RGB-fusion 0.950; ensemble 0.907  [c14-079 c12-051]
constant B WER 0.9504, RGB-fusion WER 0.9497, difference -0.0007 -> 0.0007  [c12-053]
constant B BLEU-4 0.38, RGB-fusion BLEU-4 10.33  [c12-054]
constant WER 0.950, RGB-fusion WER 0.950, final ensemble 0.907  [c12-051 c02-016]

BREVITY: RGB-fusion output cut to its first three words
truncated to three words: WER 0.8999 (0.900) BLEU-4 4.87  [c07-068 c12-051]
RGB-fusion WER 0.9497 -> 0.8999, below the ensemble's 0.9069; BLEU-4 10.33 -> 4.87  [c12-055 c21-003]
one model's WER 0.9497 -> 0.8999, BLEU-4 10.33 -> 4.87, truncated to three words  [c21-003]

WORKED EXAMPLE: clip 104 of the 40-epoch pre-trained pose-only model (beam width 5)
model 40-epoch pre-trained, beam 5 (corpus BLEU-4 10.27, the recorded 10.27)  [c04-005]
output tokens 5, matching reference tokens 4 (full stop included)  [c04-004]
matching 4-token runs 0; sentence BLEU-4 0.00  [c04-004]
token WER 0.43 (tokenised, 7 reference tokens)  [c04-004]

ROUGE-L: whitespace tokens, punctuation kept, beta = 1.2 (read from metric_suite.py)  [c06-008]

ERROR DECOMPOSITION and length, six sets of outputs (3 single models x greedy / beam 5)
system                           WER  S/edits    S/N len ratio mean len
DWPose+CTC greedy             1.0678    57.7%  0.616     0.981     5.04
  DWPose+CTC greedy: insertions 889 exceed matched reference words 610, WER 1.0678 above 1.0  [c06-006]
DWPose+CTC beam 5             0.9810    56.7%  0.556     0.824     4.24
40-ep pre-trained greedy      1.0783    57.9%  0.625     1.016     5.22
  40-ep pre-trained greedy: insertions 965 exceed matched reference words 643, WER 1.0783 above 1.0  [c06-006]
40-ep pre-trained beam 5      0.9837    58.1%  0.571     0.861     4.43
RGB-fusion greedy             1.0296    58.2%  0.599     0.936     4.81
  RGB-fusion greedy: insertions 753 exceed matched reference words 631, WER 1.0296 above 1.0  [c06-006]
RGB-fusion beam 5             0.9497    56.0%  0.532     0.792     4.07
RGB-fusion beam 5: S 2188 D 1288 I 432 of N 4115 reference words; substitutions are 0.53 of reference length  [c18-013]
substitutions as % of word edits, six sets of outputs: 58, 57, 58, 58, 58, 56  (range 56-58 %)  [c18-013]
output / reference length ratio, three beam-5 systems: 0.82, 0.86, 0.79  (range 0.79-0.86)  [c18-015]
correlation of mean output length with WER across the six sets of outputs: +0.990  [c18-088]
per-sentence oracle WER over the six sets of outputs: 0.807  [c18-014]

sentence bootstrap (2000 resamples, seed 0) of the ensemble's BLEU-4 11.80: 95% interval [9.87, 13.89], half-width 2.0 points, wider than a difference of about one BLEU-4 point (one point lies within the variation)  [c14-009]
```

**Scope.** Recomputed from the saved per-clip test decodes (greedy and beam 5 of three single models; ensemble beam 10, length penalty 0.3) through `slt/metrics.py` and jiwer; gates reproduce the recorded 10.335 / 0.9497 and 11.802 / 0.9069. The two constants are named by digest. The identification simulation (recall 0.15 to 0.66; WER 0.8685, BLEU-4 12.45) has no saved output and is not covered. The ROUGE beta is read from `metric_suite.py`. No training or inference is done.
