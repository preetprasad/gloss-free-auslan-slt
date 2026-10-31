### AX — Run-to-run noise bands

**Claim.** The dissertation states (Introduction, Discussion, Appendix B "Reproducibility" and the seed-spread figure) that a difference of about one BLEU-4 point lies within the measurement variation: single-run differences have a 95% range of about +-0.6 within one job submission and +-1.2 across job submissions, from standard deviations of 0.22 (six runs, one job) and 0.42 (pooled, adopted configuration across five jobs). The seed-spread figure (`fig:seed_spread`) uses the 800-clip official test set at beam width 5 and plots three-seed means and ranges; its second and third rows are one configuration in two job submissions.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/run_to_run_noise_bands/check.py
```

```text
GATE six replicates sd 0.216, recorded 0.216 (read from the saved job output null_band_output.txt)
Baseline configuration, one seed, six runs in one job submission: n=6, test BLEU-4 sd 0.22 [95% 0.13, 0.53]
Within-job standard deviation of six identical-seed runs: 0.22
Paired single-run 95% band within one job submission: +-0.6 BLEU-4 (1.96 x sd x sqrt 2)  [c02-010 c14-029 c14-073]
seed 42: five job submissions 10.091-11.443, range 1.353
seed 101: five job submissions 11.223-12.118, range 0.894
seed 202: five job submissions 10.486-11.483, range 0.997
GATE pooled within-seed sd 0.422, recorded 0.422 (read from the saved job output adopted_crossbatch_output.txt)
Pooled within-seed standard deviation of the adopted configuration across 5 job submissions: 0.42 (df 12)  [c02-010 c14-073]
Paired single-run 95% band across job submissions: +-1.2 BLEU-4  [c02-010 c14-029 c14-073]
Differences below about 1 BLEU-4 point lie within the measurement variation (bands +-0.6 within a job, +-1.2 across jobs)  [c02-010 c14-029 c14-073]
same seed, same configuration, different job submission (beam-5 test BLEU-4, read from the saved job output for 1.1:1 runs):
  seed 42: 11.52 vs 10.34, gap 1.18
  seed 101: 11.03 vs 10.93, gap 0.10
  seed 202: 10.87 vs 10.66, gap 0.21
Largest gap between two submissions of one configuration at one seed: 1.18
Six same-configuration runs (three seeds x two submissions): mean 10.891, standard deviation 0.39
Figure setup: 800 official Communication test clips, beam width 5 (saved decode of job 1215231)  [c20-033]
second row (control of the later job, wave1 w1A): 3 seeds 10.64, 11.30, 11.35; three-seed mean 11.09, range 10.64-11.35  [c20-035]
third row (first job, pa_aug): 3 seeds 11.44, 11.37, 11.19; three-seed mean 11.34, range 11.19-11.44  [c20-035]
Second and third rows are one configuration in 2 job submissions of 3 seeds each; per-seed gaps 0.81, 0.08, 0.15  [c20-036]
deterministic-mode repeats of one configuration, two submissions (wave 6 and wave 16 controls; read from the saved job outputs):
  seed 42: 11.711 vs 11.711, gap 0.000
  seed 101: 11.366 vs 10.745, gap 0.621
  seed 202: 10.839 vs 10.411, gap 0.428
Of 3 pairs, 1 bit-identical score and 2 differ, by 0.62 and 0.43 BLEU-4
```

**Scope.** Recomputed from saved per-run beam-5 test BLEU-4 scores: the six same-seed replicates (job 7952), the adopted configuration across five job submissions (jobs 1215231, 1216919 and three wave-3 jobs), 1.1:1 runs against the control of job 1215231, and deterministic-mode repeats (waves 6 and 16). Scores are read from saved job outputs, not recomputed from hypotheses. The configuration identity of the repeated runs was checked on the cluster and is not rerun here. Not covered: the determinism A/A (job 6952), whose weight comparison is on the cluster. The figure rows for the real-only and August pre-trained runs are not saved locally.
