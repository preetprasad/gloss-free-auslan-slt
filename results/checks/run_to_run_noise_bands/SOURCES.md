# Sources
Saved here:
- `stitch_ratio_job_logs.txt`: byte copy of `results/beam5_validation_selector_check/kaya_beam5_test_logs.txt` (extracted lines of the cluster logs of jobs 1196567 and 1199393; supplies the 1.1:1 beam-5 test scores).
- `det_pairs.json`: programmatic subset (by `prepare_data.py`, run once) of `results/wider_corpus_joint_training/summary_s{42,101,202}.json` (key A, beam5_test; job 16435) and `results/wave16_174916/runs/A_s{42,101,202}/beam5_test_best.json` (job 174916).

Read in place (already published):
- `results/run_to_run_noise/{null_metrics.json,null_band_output.txt,adopted_values.json,adopted_crossbatch_output.txt}` (jobs 7952; adopted-recipe runs of jobs 1215231, 1216919, wave 3 jobs 6592/6612/6613).
- `results/pose_augmentation_decodes/pa_ctrl_s{42,101,202}/beam5_test.json` (job 1215231).

Not available locally: job 6952 (determinism A/A) outputs.
