### AX — Figure settings and speed verdict
**Claim.** The effect-size figure (Results) uses the 800-clip official Communication test set with 95% intervals (seed-level t, paired bootstrap for the marked row); the qualitative figure shows outputs of the 40-epoch pre-trained pose-only model at beam width 5; the Discussion states that the pose-estimation stage, not the translator, limits speed.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/figure_settings/check.py
```
```text
GATE saved beam-5 decodes: BLEU-4 10.2716 (recorded 10.27 for the 40-epoch pre-trained model, 10.71 for the 80-epoch one)
official Communication test set: 800 clips (manifest rows 800, decoded 800) [c10-008]
95% two-sided t critical value, 3 seeds (2 degrees of freedom): 4.303; figure script uses 4.303 for the re-run row [c10-009]
figure axis label states 95% interval [c10-009]
qualitative figure: 5 of 5 example outputs equal the saved beam-5 decodes of the model scoring 10.27 (40-epoch pre-trained, pose-only) [c10-029]
decoding setting: beam width 5, 40-epoch pre-trained model (BLEU-4 10.27 = recorded 40-epoch score) [c10-029]
read from the saved job output: pose-only rate 25.4 frames/s
median pose estimation: 1969 ms (71.7% of the median chain) [c14-085]
median translation: 92 ms (3.3% of the median chain) [c14-085]
median synthesis: 661 ms (24.1% of the median chain) [c14-085]
pose estimation is the largest stage; translation is 4.7% of its median time [c14-085]
```
**Scope.** Recomputed: BLEU-4 of the saved beam-5 decodes (gate, 10.27), the match of the five example outputs to those decodes, stage shares from the saved per-clip timings. The interval constant and axis label are read from the figure script. The values of the intervals themselves belong to other checks. Timing is read from the saved benchmark output (M3 Max CPU, no Kaya job); nothing is trained or re-timed.
