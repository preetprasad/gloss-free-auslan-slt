### AX — Ensemble of the fine-tuned pre-trained sign-language model
**Claim.** Five models fine-tuned from the released How2Sign checkpoint on the Auslan-Daily Communication training clips score 17.86 on average on the official test set (best on validation 17.43). Their ensemble scores 20.60 (E*: beam 10, length penalty 0.8, chosen on validation) and 20.20 (E0: beam 5, length penalty 1.0) against the best published 9.95, which is +2.75 and +3.18 above the members under simultaneous (max-T, family of four) 95% intervals that exclude zero, with a predicted power of 0.99 and 0.72 (0.63 for E0), so the pre-registered claim rule is met. The three-model ensemble of models fine-tuned with our own code scores 19.77 (members 18.27, best 18.69); its second interval includes zero, so it is not claimed. On the 570 test clips whose reference is absent from training the five-model ensemble scores 16.39 against 6.60 for the 11.80 system (Results, Section on transfer, Discussion, Appendix table `tab:app_transfer` and `tab:app_transfer_diff`).

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/pretrained_ensemble/check.py
```
```text
=== Five fine-tuned models and their ensemble (n = 800 clips per split) ===
validation grid, five-model ensemble BLEU-4 (beam, length penalty):
  beam  5 lp 0.6: 19.73
  beam  5 lp 0.8: 19.70
  beam  5 lp 1.0: 19.54
  beam  5 lp 1.2: 17.82
  beam  5 lp 1.5: 16.15
  beam 10 lp 0.6: 19.69
  beam 10 lp 0.8: 19.88   <- selected on validation
  beam 10 lp 1.0: 19.25
  beam 10 lp 1.2: 17.32
  beam 10 lp 1.5: 13.02
E0 = beam 5, length penalty 1.0 (fixed in advance); E* = beam 10, length penalty 0.8 (validation argmax)
single models, beam 5 length penalty 1.0, BLEU-4 validation / test:
  seed  42: 17.65 / 17.23
  seed 101: 16.98 / 18.58
  seed 202: 17.25 / 18.10
  seed 303: 17.03 / 17.97
  seed 404: 17.84 / 17.43   <- best on validation
member mean: validation 17.35, test 17.86; best-on-validation member: test 17.43
five-model ensemble E0: validation 19.54, test 20.20
five-model ensemble E*: validation 19.88, test 20.60
test decodes of the five-model ensemble: two (E0: beam 5, length penalty 1.0; E*: beam 10, length penalty 0.8)
--- predicted power from the validation decodes (plug-in: test effect = validation effect) ---
  E0-meanS  validation effect +2.19, predicted power 0.99
  E0-bestS  validation effect +1.70, predicted power 0.63
  E*-meanS  validation effect +2.53, predicted power 0.99
  E*-bestS  validation effect +2.04, predicted power 0.72
--- test reading ---
max-T over the family of four (B = 2000, alpha 0.05), q = 2.379; observed on test:
  E0-meanS  +2.34  max-T 95% [+1.17, +3.52]  (paired-bootstrap 95% [+1.34, +3.30])
  E0-bestS  +2.78  max-T 95% [+1.08, +4.47]  (paired-bootstrap 95% [+1.36, +4.11])
  E*-meanS  +2.75  max-T 95% [+1.54, +3.95]  (paired-bootstrap 95% [+1.81, +3.80])
  E*-bestS  +3.18  max-T 95% [+1.43, +4.92]  (paired-bootstrap 95% [+1.72, +4.59])
  member mean 17.86; member best on validation 17.43
claim rule E0: both max-T intervals exclude 0 = True; predicted power 0.63 (>= 0.50: True) -> CLAIM
claim rule E*: both max-T intervals exclude 0 = True; predicted power 0.72 (>= 0.50: True) -> CLAIM
descriptive, outside the family: E* minus E0 +0.40 [-0.18, +1.13]
unseen-570 test BLEU-4 (references absent from training, n = 570):
  member mean unseen 13.49
  five-model E0 unseen 16.11, E* unseen 16.39
  E* length ratio 0.937, brevity penalty 0.935; verbatim-train outputs E* 0.585
  11.80-system (S6, beam 10, lp 0.3): test 11.80, unseen 6.60
  best published score on this subset (GASLT): 9.95 (literature value, not recomputed here)

=== Three models fine-tuned with our own code and their ensemble ===
three single models, test BLEU-4: seed 42 18.69, seed 101 17.44, seed 202 18.67; mean 18.27
best on validation: seed 42 (validation 18.14, test 18.69)
test decodes of the three-model ensemble: two (E0: beam 5, length penalty 1.0; E*: beam 10, length penalty 0.8)
three-model ensemble: E0 test 19.65, E* test 19.77
max-T over the family of four (B = 2000, alpha 0.05), q = 2.310; observed on test:
  E0-meanS  +1.39  max-T 95% [+0.29, +2.48]  (paired-bootstrap 95% [+0.51, +2.36])
  E0-bestS  +0.96  max-T 95% [-0.66, +2.57]  (paired-bootstrap 95% [-0.40, +2.40])
  E*-meanS  +1.50  max-T 95% [+0.38, +2.62]  (paired-bootstrap 95% [+0.59, +2.44])
  E*-bestS  +1.07  max-T 95% [-0.55, +2.69]  (paired-bootstrap 95% [-0.26, +2.51])
E*: both max-T intervals exclude 0 = False -> NO CLAIM: bounded estimate only
  three-model E0 unseen-570 test BLEU-4 14.81
  three-model E* unseen-570 test BLEU-4 14.71
  three-model member mean unseen 13.49
gates passed: 32 saved decodes re-scored equal to the BLEU-4 recorded by the original jobs and to slt.metrics
CHECK_DONE
```
**Scope.** Recomputed from the saved decodes: BLEU-4 of every member, ensemble and grid setting (sacreBLEU 13a on detokenised text, each re-scored value asserted equal to the BLEU-4 the original job recorded and to `slt.metrics`, 32 decodes), the validation selection of E*, the paired bootstrap (2,000 sentence resamples, seed 0, shared by all systems), the simultaneous max-T intervals, the predicted power from the validation decodes, the claim rule, and the unseen/seen, length and verbatim-train summaries; the 11.80 system is re-scored from the published dump `results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl`. References come from `comm_run/manifests` (written by `make data`). Not recomputed: training, fine-tuning and the beam-search decoding itself (the checkpoints are licensed CC BY-NC and are not part of the repository), the 9.95 published score (literature value), and the validation grid of the three-model ensemble (its selection is not stated as a number in the text). The five members were fine-tuned with a private port and the three members with our own code; the ensemble decoder is ours. Source jobs: five-model ensemble Kaya jobs 195359 (grid), 195360 (selection), 195361 (test decodes), 195362 (summary) and the clean-code single-model re-decodes of job 193342 (copied from `results/` of the working tree); three-model ensemble jobs 238134 to 238139. Saved files are in `results/checks/pretrained_ensemble/data/` (hypotheses only); see `SOURCES.md` there.
