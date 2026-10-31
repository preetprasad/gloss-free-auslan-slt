### AX — Wave 4: stitched data without boundary trim, and two realisations of half the sentences
**Claim.** Appendix B (negative results) states that stitched data built without the 15% boundary trim scored -0.44 (-0.90 to +0.01) on the beam-5 test set against the adopted corpus, lower in all three seeds, and that two stitched realisations of each of half as many sentences (five seeds) scored -0.29 (-1.15 to +0.57). The Methodology and Appendix B also state that the citation clips are trimmed by 15% at each end.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/stitch_trim_and_realisations/check.py
```
```text
seeds with summaries: [42, 101, 202, 303, 404] (read from the saved job output summary_*.json, job 7949)
GATE T-A beam-5 validation mean +0.256 (recorded +0.256 in the pre-registration outcome)

T (no trim) - A, beam-5 test, seeds [42, 101, 202]: per seed -0.54 -0.56 -0.23
T - A test mean -0.44 (95% t -0.90 to +0.01; unrounded -0.895 to +0.011) [c17-051]
T - A test lower in 3 of 3 seeds (all three seeds lower: True) [c17-051]
number of seeds 3 [c17-050]

Q (two realisations, half the sentences) - A, beam-5 test, seeds [42, 101, 202, 303, 404]: per seed +0.46 +0.10 -0.31 -0.32 -1.39
Q - A test mean -0.29 (95% t -1.15 to +0.57; unrounded -1.152 to +0.568) [c17-059]
number of seeds 5 [c17-060]

builder default --trim_frac 15% of each isolated clip's frames at each end (arm A uses the default) [c07-035 c15-016 c17-050]
arm T passes --trim_frac 0.0: True [c17-050]
arm Q passes --realizations 2 (A: 1) with the first 6,880 of A's sentences: True [c17-060]
```
**Scope.** Parses the five per-seed summary files of the original training job (7949, 3 seeds for T, 5 for Q), computes per-seed differences, means and two-sided 95% t intervals (the convention of `tools/seed_decision.py`), and reads the trim setting from the corpus builder and the job script. The gate is the validation mean recorded when the job was read out. No training is repeated. Not covered: the motion-aware resampling run (job 1197043) and the pose-reliance probe (job 1174944), whose outputs are on the cluster only.
