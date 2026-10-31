# Sources of the saved files

- `pred_dwpose.csv`, `spans45.csv`, `log_dwpose.txt`: byte copies from the local `results/probe_signerfix_2026-10-25/` (DWPose hands/body/face probe on the 382 annotated spans of 45 signs, `tools/continuous_id_probe.py --dump_pred`, run on the Mac, 25 Oct 2026; same settings as the September run: 8 exemplars per class, seed 42).
- `spans_safe.csv`: byte copy of `results/wave12f_out/spans_safe.csv` (leak-safe Sign_Spotting spans on Communication TRAIN clips, 1,008 rows, 120 glosses).
- `located.csv`: byte copy of `results/wave12f_out/located.csv` (output of `tools/w12f_locate.py`, Kaya job 140890, wave 12f): the 3,209 isolated clips and the source clip each was cut from.
- `job_probe_a2_a5_depth.out`: byte copy of the job log `results/wave12_build_2026-10-13/readout_logs/w12f-probe-140892.out` (Kaya probe job 140892; stage 140890, extraction 140891). Numbers taken from it are only read and re-derived (McNemar p from the printed counts).
- `mmwl_gloss_counts.json`: written by `prepare_data.py` from the label files `Train.json` and `Valid.json` of MM-WLAuslan (gloss name -> number of recordings, lower case). Run once by hand.
- `check.py`, `claims.json`, `fragment.md`: written for this check.
