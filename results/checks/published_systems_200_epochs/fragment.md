### AX — Published systems' own code and the project's translator with the published recipe
**Claim.** The Results chapter (Section on the published systems, Table `tab:app_ownccode` in the appendix) states that the authors' SL-Transformer and GASLT code, run for 200 epochs on 53 DWPose points, scored 8.63 and 9.58 (five seeds) against 10.43 for the project's translator trained for 40 epochs in the same jobs (differences -1.80 and -0.85); that the project's translator given the authors' recipe scored 9.91 (O53), +1.48 above their SL-Transformer (8.44) and not resolvably above their GASLT (9.47); and that on its usual 133-point input at beam width 5 it scored 9.63 against 10.20 for its own recipe. The appendix also states that runs on the first-listed AlphaPose person scored 2 to 5 BLEU-4 and gives the scores of older single runs (2.08, 1.03, 7.11, 7.92). One appendix statement (60 epochs at learning rate 3e-4 for the larger model) is not reproduced; see Scope.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/published_systems_200_epochs/check.py
```
```text
GATE 1: 79 saved test decodes re-scored through slt.metrics; 79 equal the score recorded by the job that wrote them (1e-6)
GATE 2: the three-seed Rule B differences of the 12l2 summary (SL-T -1.91, -2.67 to -1.15; GASLT -0.86, -1.77 to +0.05) are reproduced: yes

== 1. The authors' code (200 epochs) on 53 DWPose points against this project's translator, five seeds (jobs 167758 + 179411), test BLEU-4, project scoring function
  SL-Transformer, dw53 cells, 5 seeds  mean 8.63   per seed [7.99, 8.74, 8.57, 9.01, 8.83]  [c11-022]
  GASLT, dw53 cells            mean 9.58   per seed [9.86, 9.69, 8.91, 9.57, 9.87]  [c11-022]
  translator of this thesis (Edw, 40 epochs, beam 5) mean 10.43   per seed [10.44, 10.05, 10.55, 10.6, 10.52]  [c11-023]
  Edw epochs completed / planned: [[40, 40], [40, 40], [40, 40], [40, 40], [40, 40]]  [c11-023]
  Rule B (authors' code minus translator, pooled t, 8 df):
  SL-Transformer - Edw                           -1.80 [-2.27, -1.34]  [c11-024]
  GASLT - Edw                                    -0.85 [-1.32, -0.39]  [c11-024]

== 2. This project's translator with the published recipe (200 epochs, constant 5e-5), job 239443, three seeds
  O53  (our layers, 53 DWPose points)  authors' decoding protocol: mean 9.91   beam 5: mean 9.80  [c11-025 c19-073 c19-078]
  Ocwb (our layers, usual input)       beam 5: mean 9.63  [c11-027 c19-076]
  Edw  (our recipe, 40 epochs)         beam 5: mean 10.20  [c11-027 c19-076]
  authors' SL-Transformer, retrained in this job: mean 8.44;  authors' GASLT: mean 9.47  [c11-025 c11-026 c19-073 c19-075]
  O53 (authors' protocol) minus the authors' code retrained in the same job:
  O53 - SL-Transformer                           +1.48 [+0.63, +2.33]  seeds positive 3 of 3  [c11-025 c19-073 c19-074]
  O53 - GASLT                                    +0.44 [-0.16, +1.05]  seeds positive 2 of 3  [c11-026 c19-075]
  Recipe effect on this project's translator, beam 5:
  Ocwb - Edw                                     -0.57 [-1.55, +0.42]  seeds positive 0 of 3  [c11-027 c19-076 c19-077]
  Input format at the authors' recipe, beam 5:
  Ocwb - O53 (O53 beam-5 mean 9.80)              -0.17 [-1.14, +0.80]  seeds positive 2 of 3  [c19-078]
  Ocwb settings: arrays of 133 keypoints, subset cwb_body_hands, confidence True, frame normalisation; 200 epochs, constant learning rate 5e-05, batch 64; decoded at beam width 5  [c19-062]

== 3. Earlier runs of the authors' code on the first-listed AlphaPose person (not the signer), test BLEU-4
  12l (stopped early)        gaslt__ap56_bbox       mean 2.18  (n=3)
  12l (stopped early)        gaslt__ap56_frame      mean 2.12  (n=3)
  12l (stopped early)        gaslt__ap56_raw        mean 2.46  (n=3)
  12l (stopped early)        slt__ap56_bbox         mean 4.37  (n=3)
  12l (stopped early)        slt__ap56_frame        mean 3.66  (n=3)
  12l (stopped early)        slt__ap56_raw          mean 3.33  (n=3)
  12l2 (200 epochs)          gaslt__ap53_frame      mean 4.49  (n=3)
  12l2 (200 epochs)          gaslt__ap56_frame      mean 4.33  (n=3)
  12l2 (200 epochs)          gaslt__ap56_raw        mean 4.82  (n=3)
  12l2 (200 epochs)          slt__ap53_frame        mean 4.24  (n=3)
  12l2 (200 epochs)          slt__ap56_frame        mean 4.63  (n=3)
  12l2 (200 epochs)          slt__ap56_raw          mean 4.27  (n=3)
  w22 (200 epochs, 2 seeds)  gaslt__ap53_frame      mean 4.08  (n=2)
  w22 (200 epochs, 2 seeds)  slt__ap53_frame        mean 3.79  (n=2)
  cell means range 2.12 to 4.82; single runs range 1.75 to 5.26 (40 runs, 14 cells)  [c19-056]

== 4. Older single runs. No per-item outputs (hypotheses) exist for them in this working tree: the folders of these runs hold a training log, a metrics.json or progress.json and a checkpoint only, so the scores are read, not recomputed
  exact SL-Transformer re-implementation (25,315,840 parameters), distributed keypoints, official test: beam 5 2.08, greedy 1.55  (read from the saved run summary exact_slt_run_summary.json; its training log ends at epoch 25 of 60 and exact_slt_progress.json says stopped_at_epoch_25_gate)  [c08-016]
  25.9 M SignJoey-compatible model (25,945,600 parameters), first-listed AlphaPose keypoints: greedy test 1.03  (read from the saved run summary signjoey_run_summary.json; its training log ends at epoch 10 of 60)  [c16-021]
  d=512, 6+6 layers on DWPose: greedy test 7.11  (d_model 512, 6+6 layers; read from the metrics.json that the job wrote, best epoch 36)  [c16-021]
  default size on DWPose: greedy test 7.92  (d_model 256, 4+4 layers; read from the metrics.json that the job wrote, best epoch 28)  [c16-021]
  recorded training settings of the runs in this row of the appendix table (c16-022 compares them):
    d512 DWPose run, test 7.11 (metrics.json of the job): 40 epochs, cosine learning rate 0.0005, batch 32 x 1 accumulation
    full DWPose run, test 7.92 (metrics.json of the job): 40 epochs, cosine learning rate 0.0005, batch 64 x 1 accumulation
    25.9 M model scored 1.03 (checkpoint settings): 60 epochs planned (ran 10), plateau learning rate 0.001, batch 16 x 2
    another 25.9 M-class model (512-d, 3+3 layers, 1024-d adapter, first-listed AlphaPose; not the 1.03 run): greedy test 2.29 (metrics.json of the job), 60 epochs, cosine learning rate 0.0003, batch 16 x 2
  c16-022 CHECK: the 7.11 run recorded 40 epochs at 0.0005 (same as the default-size run: 40 at 0.0005); '60 epochs at 3e-4' is recorded only for the 2.29 run above: DISAGREES with the dissertation sentence for the 7.11 run
```
**Scope.** Sections 1 to 3 re-score 79 saved test decodes (800 clips each) through `slt/metrics.py`; the gate requires each to equal the score its job recorded, and the three-seed differences printed by the 12l2 summary are reproduced. Means, differences and pooled-variance 95% t-intervals are computed here. Training is not repeated; original jobs 138567, 167758, 179411 (the authors' code and the project's translator, on the cluster) and 239443. Section 4: no hypothesis file exists for the four older runs (their folders hold a training log, a metrics.json or progress.json, and a checkpoint; `results/legacy_runs/` was searched too), so their scores are read from what the jobs saved, not recomputed. The 7.11 and 7.92 scores are in the metrics.json that the jobs wrote; the 2.08 and 1.03 scores exist only in run summaries written on 5 Aug 2026 (the training logs, copied, confirm where each run stopped). The appendix statement that the larger model used 60 epochs at learning rate 3e-4 against 40 at 5e-4 disagrees with the metrics.json of the 7.11 run (40 epochs at 5e-4): 60 epochs at 3e-4 is recorded for a different 25.9 M-class run that scored 2.29, and a submit script with those settings has no saved output. This is reported as a finding, not covered.
