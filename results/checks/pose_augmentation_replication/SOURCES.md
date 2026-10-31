# Sources
Saved here:
- `stitch_ratio_job_logs.txt`: byte copy of the local file `results/beam5_validation_selector_check/kaya_beam5_test_logs.txt`, the extracted lines (beam-5 test BLEU-4 per arm) of the cluster logs `stitch-scale-1196567.out` and `stitch-scl2-1199393.out`. It supplies the 1.1:1 (`sc_r111`) scores for seeds 42 (job 1196567), 101 and 202 (job 1199393).

Read in place (already published, byte copies of cluster outputs of job 1215231, `pose-aug`):
- `results/pose_augmentation_decodes/pa_{ctrl,aug}_s{42,101,202}/hyps_beam5.jsonl`, `progress.json`, `beam5_test.json`
- `src/tools/stitch_lexicon.py` (71-word STOP), `src/train_ctc.py` (75-word STOP, second line), `src/config.py`, `jobs/slurm/pose_aug.sbatch`, `src/slt/data.py` (augmentation code, executed on synthetic keypoints)

Not available locally: wave-3 augmentation-dose job 6612 (`w3-augdose`); no script or output was found.

Dose sweep (job 6612, `w3-augdose`; claims c17-081, c17-082): searched the whole tree; no script and no output are saved locally (summaries are on Kaya under `/group/pmc099/pprasad/wave3_augdose/runs/`).
