### AX — Sign-identification probes and isolated-sign references
**Claim.** The Results, Discussion and Appendix state that on the 45 signs shared by MM-WLAuslan and Auslan-Daily (382 annotated conversational spans, 337 scored, chance 2.2%), DWPose hand keypoints identify the sign at 18.1% top-1 in conversational recordings (Figures on the identification probe, `tab:app_additional`). The Discussion and Appendix add that the 45 shared signs are an atypical subset, that using Auslan-Daily's own isolated clips (3,209 segments over 610 glosses; 379 usable for these signs, 7.5 same-sign references per query) as references gives 19.9% against 18.1% (p = 0.33), that the 630 MM-WLAuslan citation recordings (14 per sign) give 4.5%, and that estimated hand depth gives 5.9% alone (null 3.9%) and lowers x/y accuracy from 16.3% to 11.3% (p = 0.014).

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/sign_identification_probes/check.py
```
```text
GATE per-span file hands correct 61/337; recorded in the saved probe job output 61/337 (expected 61/337)

CONVERSATIONAL SPANS (the recorded probe, hands descriptor, clip-disjoint folds)
annotated spans 382 of 45 signs; chance 1/45 = 2.2%  [c08-046 c10-051]
signs 45  [c08-046 c10-051 c14-071 c14-070 c10-041 c14-047]
chance 2.2%  [c01-018 c02-021 c08-046 c10-040 c14-071]
spans 382  [c08-046 c10-051]
scored spans 337  [c14-070]
hands top-1 18.1% (61/337)  [c01-018 c02-021 c10-040 c14-072 c18-077]

AUSLAN-DAILY ISOLATED CLIPS: 3,209 segments over 610 glosses  [c04-039]
isolated clips of the 45 signs on a TRAIN clip with at least 3 frames: 379  [c14-070]
same-sign references per query (own clip excluded): mean 7.5, min 2, max 16  [c14-070]

Spotting glosses 120; also MM-WLAuslan glosses: 45 shared signs  [c10-041 c14-047]
MM-WLAuslan citation recordings of the 45 signs: 630  [c14-070]
recordings per sign: 14 (min) to 14 (max), i.e. 14 per sign  [c14-070]

ISOLATED CLIPS AS REFERENCES (read from the saved job output job_probe_a2_a5_depth.out)
isolated clips as references, top-1 19.9% (67/337) against conversational spans 18.1%  [c14-052 c14-072 c18-077]
citation-form recordings as references, top-1 4.5% (15/337)  [c14-052 c14-072 c18-078]
isolated clips (A2) minus spans: only first 16, only second 10; exact McNemar p recomputed 0.33 (log 0.33)  [c18-077]

ESTIMATED HAND DEPTH (read from the saved job output)
x/y/z top-1 11.3% against x/y 16.3% (same 3-D model)  [c18-079]
only first 13, only second 30; exact McNemar p recomputed 0.014 (log 0.014)  [c18-079]
depth alone 5.9% above a permutation null 95th percentile 3.9%  [c18-080]
```
**Scope.** Recomputed from per-item files: the 337 per-span hand-probe outcomes (61 correct; the gate matches the recorded run), the located table of the isolated clips (3,209; 379 references; 7.5 per query), and the MM-WLAuslan label counts (630 recordings, 14 per sign, 45 shared glosses). Read from the saved job output of Kaya probe job 140892 (wave 12f): the isolated-reference, citation-reference and depth accuracies; the McNemar p-values are recomputed from the printed discordant counts. Not recomputed: the 42.8% citation-form top-1 and the hand-confidence figures (only written up), and the SignCL density and spotting probes (outputs on Kaya, jobs 1174747 and 1171035). No training or pose extraction is done here.
