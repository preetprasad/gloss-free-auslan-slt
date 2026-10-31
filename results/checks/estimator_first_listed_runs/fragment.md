### AX — Early AlphaPose runs and the DWPose re-extraction

**Claim.** The Introduction and Results (Section on the pose estimator, the table of the early attempts, Appendix B) state that a translator trained on the first-listed person's AlphaPose keypoints scored BLEU-4 2.87 at beam width 5 and 1.76 with greedy decoding on the 800-clip official test set, with only 53 distinct greedy sentences (output collapse); that DWPose re-extraction raised the mean hand-keypoint confidence to 0.82; that the same keypoints identified signs from the hands at 4.2% in the identification probe; the keypoint-set sizes (59, 68, 136, 133); that the SignJoey-compatible reconstruction scored 1.03 with 9 distinct sentences, the 25.95 M-parameter model 2.29, and the bounding-box-corrected generic run 2.13.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/estimator_first_listed_runs/check.py
```
```text
control_decodes.jsonl: references match comm_run/manifests/test.tsv: 800/800
control_decodes.jsonl: references match comm_run/manifests/test.tsv: 800/800
GATE recorded BLEU-4 of the first-listed AlphaPose control, greedy 1.76 and beam 5 2.87: recomputed 1.7604 and 2.8673

First-listed AlphaPose control, official test, 800 clips, one run
BLEU-4 2.87 at beam width 5 (length penalty 0.6), 1 run  [c02-006 c09-008]
BLEU-4 1.76 with greedy decoding  [c09-008]
53 distinct sentences for the 800 test clips with greedy decoding (output collapse)  [c02-007 c08-015 c09-008]
(beam width 5 gives 26 distinct sentences)
signjoey_decodes.jsonl: references match comm_run/manifests/test.tsv: 800/800

SignJoey-compatible reconstruction, greedy, official test: BLEU-4 1.03, 9 distinct sentences  [c17-010]

DWPose re-extraction: mean hand-keypoint confidence over 14041 clips 0.82 (0.8218)  [c08-020]

First-listed AlphaPose identification probe: 382 annotated spans usable (45 signs) in the saved probe log
hand keypoints, nearest-centroid top-1 on 333 scored spans (the probe scores a span only if its hand descriptor exists and its class is predictable in the held-out fold): 4.2%  [c08-049]
(333 of 382 usable spans are scored for the hand variant; the quoted 382 is the usable set, 4.2% is over the scored spans)

Keypoint sets (config.py)
COCO-WholeBody body+hands subset: 59 keypoints  [c06-030]
HALPE body+hands subset: 68 keypoints  [c06-030]
AlphaPose HALPE layout: 136 keypoints  [c06-032 c09-008]
DWPose COCO-WholeBody layout: 133 keypoints (default)  [c06-033]

1,024-to-512 capacity model: 25.95 M parameters (pos_enc buffer excluded)  [c17-009]
its test BLEU-4 2.29 on 800 clips (read from the saved job output capacity_accum2_metrics.json)  [c17-009]
bounding-box-corrected generic run, greedy test BLEU-4 2.13 (read from the saved job output bbox_xywh_test_rescore.txt)  [c17-011]
```
**Scope.** The control and the SignJoey-compatible reconstruction are decoded from their local checkpoints with `tools/eval_beam.py` (CPU, once, by `prepare_data.py`'s inputs; see SOURCES.md) and the saved hypotheses are scored through `slt/metrics.py`; the gate is the recorded 1.76 (greedy) and 2.87 (beam 5) of the control. The hand confidence, the probe top-1 and `config.py` are recomputed from per-clip and per-span files. The 4.2% is over the 333 spans the probe scores for the hand variant; the quoted 382 is the usable-span count printed in the same probe log. The 2.29 and 2.13 scores and the parameter count are read from saved job outputs and a checkpoint. Not checked: the beam-5 cells of the estimator 2x2 (job 1101250: 4.60, 5.58, 1.45, 10.03, the single collapsed output), the beam width and keypoint subset of those cells, and the DWPose re-extraction run (8.78, 7.92, 309), all on Kaya; the design labels of the 2x2 and the "as released, first-listed person" / "re-extracted" labels of its table are not checked. No training is done.
