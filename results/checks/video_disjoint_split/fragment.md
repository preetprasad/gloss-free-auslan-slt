### AX — Video-disjoint split and VD-CONFIRM
**Claim.** The official split shares videos between train and test, so a video-disjoint split (11,341 / 1,444 / 1,254 clips from 73 / 9 / 8 videos) was built (Methodology, Results "Video-disjoint"). On it the adopted recipe raises five-seed mean beam-5 test BLEU-4 from 6.14 to 7.08 (+0.94), 29.3% of test references occur verbatim in training (886 do not), and a six-model ensemble (7.12) does not exceed the best single model (7.29; -0.16, interval -0.86 to +0.57). The best single model is the adopted-recipe model with the highest video-disjoint validation score.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/video_disjoint_split/check.py
```
```text
video-disjoint split: train 11,341 clips from 73 videos, val 1,444 from 9, test 1,254 from 8; no video shared  [c07-061 c11-070]
GATE recorded job read-out (seed_decision_5seeds.txt) contains 'mean +0.939 | 95% t [+0.155, +1.722]': True

VD-CONFIRM: 5 seeds, read from the saved job outputs summary_s*.json (original job 6596 + 7965)  [c18-018 c07-062 c11-073]
beam-5 test BLEU-4, mean over 5 seeds: plain baseline 6.14 (6.144), adopted recipe 7.08 (7.082)  [c11-073]
difference +0.94 (+0.939); positive in 5 of 5 seeds  [c11-073]
seed-level 95% t-interval +0.16 to +1.72  [c11-073]
decode references equal the split's test manifest in order: 1254/1254

references occurring verbatim in the video-disjoint training text: 368 of 1254 test clips = 29.3%  [c11-075]
test clips whose reference never occurs in that training text: 886  [c11-076]
GATE recorded 7.123 (ensemble) / 7.287 (best single): 7.123 / 7.287
ensemble members (read from the saved read-out line v6_members_line.txt): 6 (six)  [c14-035 c11-078]
ensemble test BLEU-4 7.12 (7.123); best single model 7.29 (7.287)  [c14-035 c11-078]
difference -0.16 (-0.164); 95% paired-bootstrap interval -0.855 to +0.570, i.e. -0.86 to +0.57 (2,000 resamples, seed 0)  [c11-078]
GATE recorded vdB_s42 at LP 1.0: validation 6.600, test 7.407; recomputed 6.600 / 7.407
GATE recorded vdB_s101 at LP 0.9: validation 6.718, test 7.287; recomputed 6.718 / 7.287
GATE recorded vdB_s202 at LP 0.7: validation 6.373, test 7.149; recomputed 6.373 / 7.149
the best single model is the one with the highest video-disjoint validation BLEU-4 among the three adopted-recipe models: vdB_s101 (6.72 validation, 7.29 test)  [c14-035 c11-078]
scope of 'best single model': the highest-validation model among the three adopted-recipe (vdB) singles that make up the ensemble comparison; the plain-baseline (vdA) singles are not considered  [c14-035 c11-078]
(not the best on test: vdB_s42 scores 7.41 on test; the single model is selected on validation, so its test score is not a test-selected maximum)  [c14-035 c11-078]
```
**Scope.** Recomputed: split sizes and video counts from the clip identifiers, reference overlap against the training text, BLEU-4 and the 2,000-resample paired bootstrap (seed 0) from the saved test decodes of the ensemble and the best single model. Read from saved job outputs (not recomputed): the per-seed beam-5 summaries (jobs 6596 and 7965). The best single model is shown to be validation-selected: three adopted-recipe singles are re-scored on validation and test from their saved decodes, gated on the recorded read-out values (the test-best of the three differs). Not done: training; that the adopted-recipe arm adds 13,760 duration-modelled stitched clips (its own joint manifest is on the cluster, job 6596; not claimed by this check); the 'best single model' comparison covers the three adopted-recipe singles only (the plain-baseline singles are not considered); and the per-seed 886-clip subset scores, content precision/recall and WER, whose per-seed decodes are on the cluster. Gates: the recorded seed-level read-out and the recorded 7.123 / 7.287.
