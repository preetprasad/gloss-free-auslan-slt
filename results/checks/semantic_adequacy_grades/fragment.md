### AX — Semantic adequacy grading
**Claim.** Appendix B (automatic adequacy rating, supplementary table row) states that a small instruction-tuned model (Qwen2.5-3B-Instruct) graded every output of four systems on the official test set for adequacy on a 0–4 scale, that mean adequacy rises from 1.155 for the baseline model to 1.311 for the final ensemble, and that the final ensemble was graded higher than the RGB-fusion model on 148 clips and lower on 134.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/semantic_adequacy_grades/check.py
```
```text
GATE dwpose_ctc: recomputed mean 1.155 vs recorded 1.155 (read from the saved job log dwpose_ctc.log)
GATE vlp: recomputed mean 1.232 vs recorded 1.232 (read from the saved job log vlp.log)
GATE rgbfusion: recomputed mean 1.259 vs recorded 1.259 (read from the saved job log rgbfusion.log)
GATE s6: recomputed mean 1.311 vs recorded 1.311 (read from the saved job log s6.log)

Judge model (read from the saved job logs): Qwen2.5-3B-Instruct  [c20-008]
Scale: grades take values 0 to 4, i.e. 0-4  [c20-008]
Systems graded: 4, each on 800 official test clips  [c20-008]

Mean adequacy
  dwpose_ctc  1.155
  vlp         1.232
  rgbfusion   1.259
  s6          1.311
Baseline model (DWPose+CTC) mean adequacy 1.155  [c20-009]
Final ensemble (S6) mean adequacy 1.311  [c20-009]

Final ensemble vs RGB-fusion model, per clip: graded higher on 148 clips and lower on 134 (equal on 518)  [c18-022]
```
**Scope.** Recomputed from the saved per-clip grades (800 per system) of the re-grading job 10288; the gate compares each recomputed mean with the mean printed in that job's saved log. The judge model name is read from the logs. The grading itself (running the language model) is not repeated.
