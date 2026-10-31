### AX — Ensemble member-set ablation and member-set selection
**Claim.** The dissertation (Results, Discussion, Appendix B, abstract) reports an ablation of the final six-member ensemble over all 63 member subsets on validation (kinds of member differ by +0.56, 95% interval -0.57 to +1.97), that validation and test BLEU-4 of four ensemble member sets are uncorrelated (r = -0.014), that three ensembles scored above 11.80 on test (12.11, 12.84 and 12.73), that the third, rebuilt from three pose-augmented models of the adopted configuration and two RGB-fusion models, was ahead in 91.5% of 1,000 validation halvings, three test-selected configurations (11.15, 11.51, 11.70), the video-disjoint ensemble (7.84 against 7.29 for the validation-selected single model), and that paired intervals include zero for most improvements, so the 12.11 gain was withdrawn.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/ensemble_member_subsets/check.py
```
```text
GATE six-member set, validation beam 10 / LP 0.3: BLEU-4 12.470 WER 0.9164 (recorded 12.470 / 0.9164, job 120153)

Decoded and scored 63 non-empty subsets of the 6 members (VLP80, RGBfus, VLP40, VLP40s202, fus101, fus202) on 800 validation clips  [c12-017]
Number of members 6, number of subsets 63  [c12-017]
Members added in the recorded order, validation BLEU-4: 9.78, 11.08, 11.47, 11.57, 11.78, 12.47  [c18-058]
Shapley sums: fusion members 6.51, pose members 5.96  [c12-021]
Fusion minus pose Shapley sum +0.56, 95% interval -0.57 to +1.97 (500 resamples)  [c12-021]
read from the saved job output ensemble_member_subsets_ablation/subsets_scores.json for comparison: type_diff [-0.5656478072422426, 1.9651782390706263]; recomputed matches: True
GATE S6 official test BLEU-4 11.802 (recorded 11.802)
GATE E_dur_x3 official test BLEU-4 11.537 (recorded 11.537)
GATE F official test BLEU-4 12.114 (recorded 12.114)
GATE G1 official test BLEU-4 11.562 (recorded 11.562)
GATE F-prime official test BLEU-4 12.836 (recorded 12.836)
GATE F-triple-prime official test BLEU-4 12.727 (recorded 12.727)

Member set | validation BLEU-4 (read from the saved job output for E, F, G1; S6 from the subset decode) | official test BLEU-4 (recomputed from the dump)
  S6                                  12.470   11.802
  three stitched-data models          12.472   11.537
  F                                   13.246   12.114
  six stitched-data models plus two   13.969   11.562
Four member sets scored on both splits  [c12-038]
Pearson r between validation and test BLEU-4 -0.014  [c12-038]
Larger variant (six stitched-data models plus two RGB-fusion): validation 13.97, test 11.56  [c12-037]
Pose-only ensemble of three stitched-data models: validation 12.472 against 12.470 for the final ensemble; test 11.54 against 11.80  [c18-097]
Ensembles that scored above 11.80 on test: F 12.11, F-prime 12.84, rebuilt 12.73; count above = 3  [c14-084]
Rebuilt ensemble (saved decode log): 5 members, 3 without RGB fusion, 2 RGB-fusion; test BLEU-4 12.73  [c12-040 c18-006]
Member kinds in the order of the decode (3 pose-augmented checkpoints, then 2 RGB-fusion): from the checkpoints' saved settings [False, False, False, True, True]; from the decode log [False, False, False, True, True]; equal: True  [c12-040 c18-006]
Pose augmentation switched on in the saved settings of 3 of the 3 pose-augmented checkpoints (seeds 42, 101, 202)  [c18-006]
Pose augmentation switched on in the 0 of 3 duration-stitch members of the earlier 1.1:1 ensemble (settings field present: 0)  [c18-006]
Settings of the three checkpoints: training manifest joint_13760.tsv (13,760 stitched clips added to the real training clips), 133 keypoints, frame normalisation, 40 epochs, learning rate 0.0005, cosine schedule, batch 64, pose augmentation on, no RGB input; all three agree: True  [c12-040]
Stitched to real training clips: 13,760 to 12,441 = 1.1:1  [c12-040]
The earlier 1.1:1 members of seeds 101 and 202 used the same training manifest and no pose augmentation: True  [c18-006]
(The decode log does not name its checkpoint files; which checkpoints formed the rebuilt ensemble is recorded in the pre-registration, not in a saved output, so the settings above are those of the checkpoints it names. The log's member kinds and the remapped vocabulary of the last two members agree.)
Video-disjoint ensemble (saved decode log): 5 members, 3 pose-only, 2 RGB-fusion  [c18-004]
Pose-only duration ensemble (saved job output ens-ext 1197079, X2_DUR3): 3 members, 0 with RGB fusion  [c18-096]

Exploratory configurations chosen with test scores in view, against the validation-selected configuration of the same job (read from the saved job outputs):
  beam/length-penalty re-tune of the two-member ensemble (tier1 1082033): validation-selected beam=5 lp=1.0 scored 10.71 on test; beam 10 / LP 1.0 scored 11.15 (+0.43)
  SWA ensemble candidates (swa-ens 1094775): validation-selected C:SV+SF+V0+C2+F1+F2 (C) scored 11.26 on test; candidate D scored 11.51 (+0.25)
  weighted ensembles (ens-max 1091645): validation-selected S6-weighted scored 11.41 on test; S9-weighted scored 11.70 (+0.28)
Test-selected configurations scoring above the validation-selected one of their job: 3 of 3; their BLEU-4 11.15, 11.51 and 11.70  [c08-009]

Rebuilt ensemble minus final ensemble, official test, 2,000 resamples: +0.93, 95% interval -0.28 to +2.11  [c18-007]
Test BLEU-4 12.73 against 11.80  [c18-007]
F minus final ensemble, official test: +0.31, 95% interval -0.89 to +1.50; includes zero: True  [c01-028]
F scored 12.11 on test against 11.80, having scored 13.25 against 12.47 on validation (validation read from the saved job output ens-search / the subset decode): an apparent gain of +0.31 whose interval includes zero, the condition under which the dissertation withdrew it (the withdrawal itself is the author's decision, not a computed number)  [c01-028]
GATE CTC official test BLEU-4 9.522 (recorded 9.522)
GATE VLP official test BLEU-4 10.272 (recorded 10.272)
GATE RGB official test BLEU-4 10.335 (recorded 10.335)
GATE Fp official test BLEU-4 12.836 (recorded 12.836)
  S6 minus RGB: +1.47, 95% interval +0.40 to +2.56, excludes zero: True
  S6 minus member ctc_vlplong80: +1.00, 95% interval -0.13 to +2.11, excludes zero: False
  RGB minus CTC: +0.81, 95% interval -0.34 to +1.97, excludes zero: False
  VLP minus CTC: +0.75, 95% interval -0.22 to +1.71, excludes zero: False
  F minus S6: +0.31, 95% interval -0.89 to +1.50, excludes zero: False
  Fp minus S6: +1.03, 95% interval -0.10 to +2.15, excludes zero: False
Of the 6 paired comparisons of Table res_bootstrap, 5 have an interval that includes zero and 1 excludes it; the intervals' half-widths are about 1.2 BLEU-4  [c01-028]
Rebuilt ensemble, unseen-target subset (570 clips): BLEU-4 7.89  [c18-008]

GATE rebuilt ensemble validation BLEU-4 at LP 0.8 13.472 (recorded 13.472); reported ensemble at LP 0.3 12.470 (recorded 12.470)
Rebuilt minus reported on the held-out half over 1,000 halvings: mean +0.71, 2.5th to 97.5th percentile -0.29 to +1.74  [c12-041]
Rebuilt ahead on the other half in 91.5% of 1,000 halvings (fixed threshold 95%)  [c12-041]
GATE vdB_s101 video-disjoint test BLEU-4 7.287 (recorded 7.287)
GATE VD5 video-disjoint test BLEU-4 7.841 (recorded 7.841)

Single models among VD5's members, validation (video-disjoint) / test BLEU-4, read from the saved job output g1_readout.txt: vdB_s42 6.60/7.41; vdB_s101 6.72/7.29; vdB_s202 6.37/7.15; fus_s303 6.00/6.14; fus_s404 6.03/6.82
Highest validation score among them: vdB_s101; the baseline named in the job output: vdB_s101; the baseline used here: vdB_s101; all equal: True  [c18-005]
(Highest test score among them: vdB_s42 7.41; the baseline is the validation-selected single model, not the test-best one.)

Video-disjoint ensemble VD5 7.84 against 7.29 for the validation-selected single model vdB_s101  [c18-005]
Difference +0.55; simultaneous 95% interval (max-T over 5 looks, 2,000 resamples) -0.23 to +1.34  [c18-005]
```
**Scope.** Subset decodes (shards of Kaya jobs 120153/120154), the test dumps of the final ensemble, F, F-prime, the pose-only duration ensemble, the six-plus-two variant, the rebuilt ensemble and the video-disjoint systems are re-scored with the repository scoring code; the bootstraps (500 resamples for the kind difference, 2,000 for the paired and video-disjoint intervals, 1,000 halvings; original seeds), Shapley values and the six Table res_bootstrap intervals are recomputed (sufficient-statistics BLEU, identical to corpus BLEU-4). Validation scores of three other member sets, the validation-selected configurations and the test-selected configurations are read from the saved job outputs (ens-ext 1197079, ens-search 1197719, tier1 1082033, swa-ens 1094775, ens-max 1091645), matched by section and configuration label rather than by line order. The single model used as the video-disjoint baseline is the one with the highest validation score among the ensemble's single members (read from the saved job output of 31205), not the highest on test. Member kinds are read from the saved decode logs; the pose augmentation, seeds, training manifest and learning settings of the rebuilt ensemble's checkpoints are read from the settings saved inside the checkpoints (`data/member_cfgs.json`, extracted once by `prepare_data.py`). The decode log does not name its checkpoint files, so which checkpoints formed the rebuilt ensemble is recorded only in the pre-registration, and the check cannot show that the decode used them; it shows that the checkpoints so named are pose-augmented, share one adopted configuration, and match the decode's member kinds. The withdrawal of the 12.11 gain is the author's decision; the check recomputes the interval that motivated it. No decoding or training is done; the 0.55:1 and 1.1:1 ratios of the older stitched-data ensembles are not recomputable from saved outputs.
