### AX — Rest-cut stitching (wave 16)
**Claim.** Stitched training clips drop both hands to rest between signs far more often than real clips do (4.68 times per clip against 0.25 with tracked keypoints and 0.67 as extracted). A pre-registered test cut those rest frames from the edges of each stitched sign at two depths (12.6% and 8.8% of frames removed, leaving 0.28 and 1.91 pauses per clip). Over five seeds the deeper cut raised test BLEU-4 by 0.29 (+0.13 to +0.45) and the shallower by 0.49 (-0.03 to +1.02), higher in all five seeds for both, with validation gains of +0.19 and +0.18 whose intervals include zero and length-adjusted test gains of +0.20 and +0.34 (Results, appendix table of additional results). Neither cut was added to the adopted configuration.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/rest_cut_stitching/check.py
```
```text
gate 1 OK: all 30 beam-5 decodes (3 arms x 5 seeds x val/test) re-score to the BLEU-4 recorded by the training jobs
gate 2 OK: control (A) test BLEU-4 per seed = [11.711, 10.745, 10.411, 11.305, 10.869] = wave-5 control

=== Five-seed paired differences against the control A (beam 5, best checkpoint) ===
R08 - A  test      mean +0.29 (+0.13 to +0.45), higher in 5 of 5 seeds, one-sided t 4.935, p 0.0039
R08 - A  validation mean +0.19 (-0.36 to +0.74), higher in 4 of 5 seeds; Stage-1 (seeds 42/101/202) mean +0.18
R08 - A  length-adjusted test gain +0.20 (per seed +0.40, -0.10, +0.10, +0.54, +0.07); fresh seeds 303/404 test +0.22, +0.27
R08 mean test/val 11.30/11.31  A 11.01/11.12
R10 - A  test      mean +0.49 (-0.03 to +1.02), higher in 5 of 5 seeds, one-sided t 2.610, p 0.0297
R10 - A  validation mean +0.18 (-0.54 to +0.90), higher in 3 of 5 seeds; Stage-1 (seeds 42/101/202) mean +0.11
R10 - A  length-adjusted test gain +0.34 (per seed +0.27, -1.00, +1.05, +0.82, +0.53); fresh seeds 303/404 test +0.69, +0.90
R10 mean test/val 11.50/11.30  A 11.01/11.12
Holm (m = 2): R08 p 0.0039 vs 0.025 -> reject; R10 p 0.0297 vs 0.05 -> reject

gate 3 OK: the audit's bootstrap interval of the stitched mean is reproduced: [4.596, 4.777]
=== Interior rest runs (both hands down between signs) per clip, 2,000 clips per group ===
stitched clips 4.68 (exact 4.6845); real clips, tracked keypoints 0.25 (exact 0.2450); real clips as extracted 0.67 (exact 0.6660)

=== Depth of the cut (per-segment table of the cut arms; 94,428 stitched signs) ===
R08: frames removed 12.6% as recorded by the build check (cut / (cut + trimmed length); matches its saved value exactly); cut / trimmed length alone: 16.1%; mean frames cut per sign: start 5.08, end 3.57
R08: interior rest runs left per clip (500 clips; read from the saved job summary PREP_OK_R08.json) 0.28
R10: frames removed 8.8% as recorded by the build check (cut / (cut + trimmed length); matches its saved value exactly); cut / trimmed length alone: 10.6%; mean frames cut per sign: start 3.84, end 1.98
R10: interior rest runs left per clip (500 clips; read from the saved job summary PREP_OK_R10.json) 1.91
gate 4 OK: the uncut arm on the same 500 clips gives 4.722 rest runs per clip (the saved build check: 4.722)
```
**Scope.** BLEU-4 is re-scored from the saved beam-5 decodes (best checkpoint, 800 validation and 800 test clips, 13a, detokenised) of the 15 training runs of Kaya job 174916 (arms A, R08, R10; seeds 42, 101, 202, 303, 404), against the restored official references, and must equal the scores recorded by the jobs; the paired differences, t intervals, one-sided t tests, Holm decision and brevity-penalty length adjustment are recomputed from them. The rest-run counts are recomputed from the per-clip table of the audit job 174914 (2,000 clips per group; the audit's bootstrap interval is reproduced). The cut depths are recomputed from the per-sign cut table taken from the build audit of job 174915 (94,428 signs). Not recomputed: training, and the number of rest runs left after the cut (0.28 and 1.91), which was measured on 500 cut clips whose pose arrays are not saved and is read from the saved job summaries `PREP_OK_R08.json` and `PREP_OK_R10.json` (the same 500 clips of the uncut arm are recomputed, 4.722). The "frames removed" percentages are the build check's statistic, cut frames divided by cut plus trimmed-exemplar frames; as a share of the trimmed exemplar alone the cut is 16.1% and 10.6%, printed beside them.
