# Sources (byte copies; nothing retyped)

- `pred_dwpose.csv`, `pred_ap_clips_signer.csv`, `probe.json`, `log_dwpose.txt`, `log_ap_clips_signer.txt`: copied from `results/probe_signerfix_2026-10-25/` (signer-keypoint identification probe, run locally 25 Oct 2026; `--dump_pred` per-span outputs and logs).
- `signer_check_results.json`: copy of `results/dwpose_signer_tracking_check/results.json` (per-clip check of which person each keypoint array follows, run locally 25 Oct 2026).
- `w5-estsig-196407_{0,1,2}.out`: copies of the job logs of Kaya job 196407 (`results/estimator_signer_keypoints_rerun/logs/`); kept as provenance of the per-seed summaries, not read by check.py.
- Read in place (published files): the six beam-5 test dumps `results/estimator_signer_keypoints_rerun/runs/{Eap,Edw}_s{42,101,202}/beam5_test_best.json` (Kaya job 196407, estimator comparison on the signer's keypoints) and `results/train_target_hashes.txt`.
