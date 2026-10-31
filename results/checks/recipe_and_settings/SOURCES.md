# Sources

- `joblog_pose_aug.out` - byte copy of `results/pose_augmentation_decodes/pose-aug-1215231.out` (Kaya job 1215231, pose-augmentation run; training-set composition and augmentation fields).
- `hyps_beam5_pa_aug_s42.jsonl` - byte copy of `results/pose_augmentation_decodes/pa_aug_s42/hyps_beam5.jsonl` (beam-5 test hypotheses and references of the seed-42 adopted-configuration run, job 1215231).
- `run_meta.json` - programmatic subset (beam, length penalty, n, recorded BLEU-4) of `results/pose_augmentation_decodes/pa_aug_s42/beam5_test.json`, made by `prepare_data.py`.
- `cfg_members.json` - programmatic subset of the training configuration, layer counts, input-projection shape and vocabulary size stored inside `results/ckpts/*.pt` (the 13 checkpoints copied from Kaya `/group/.../comm_dwpose_*`, `dur_*`; checkpoints themselves are not saved here), made by `prepare_data.py`. The six reported members: jobs of 11-23 Aug 2026 (named in `mbr_wer_test2.sbatch`).
- Read directly from the repository by `check.py`: `src/config.py`, `train_ctc.py`, `slt/data.py`, `tools/eval_beam.py`, `tools/paired_bootstrap.py`, and the job scripts `jobs/slurm/mbr_wer_test2.sbatch`, `pose_aug.sbatch`, `wave1_r2.sbatch`, `w5_train_s2.sbatch`, `w6_corpus_s2.sbatch`.
