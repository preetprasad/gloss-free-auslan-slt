# Sources (all files made by prepare_data.py, run once; not run by make)

- control_decodes.jsonl: greedy and beam-5 (length penalty 0.6) test decodes of the first-listed AlphaPose control checkpoint (src/comm_scratch_control/best.pt, trained Aug 2026 on a local MPS run; this is the control whose greedy decode scored 1.76), produced on 31 Oct 2026 by `python tools/eval_beam.py --ckpt comm_scratch_control/best.pt --manifest ../comm_run/manifests/test.tsv --beam 5 --length_penalty 0.6 --device cpu --output_json <file>` and converted to one JSON object per clip by prepare_data.py (ids from comm_run/manifests/test.tsv, which fix the order of eval_beam's hypotheses). References under "ref". The checkpoint is not stored here.
- signjoey_decodes.jsonl: greedy test decodes of the SignJoey-compatible 25.9 M-parameter reconstruction (comm_slt_signjoey_reference/out/best.pt, local MPS run, 5 Aug 2026; trained on raw first-listed pose with the legacy xyxy box rule), produced on 31 Oct 2026 by `python tools/eval_beam.py --ckpt ../comm_slt_signjoey_reference/out/best.pt --manifest ../comm_run/manifests/test.tsv --beam 1 --device cpu --legacy_abs_bbox_validity --output_json <file>`; prepare_data.py keeps the greedy hypotheses. References under "ref".
- dwpose_hand_conf.csv: pose/extraction/dwpose_conf.csv (hand column only), written by compute_dwpose_conf.py on Kaya from the 14,041 re-extracted DWPose clips.
- pred_firstlisted_alphapose.csv: byte copy of results/probe_signerfix_2026-10-25/pred_ap_clips.csv (identification probe on the first-listed AlphaPose clips, local run, 25 Oct 2026).
- probe_log_firstlisted_alphapose.txt: byte copy of results/probe_signerfix_2026-10-25/log_ap_clips.txt (log of the same probe run; gives the 382 usable spans and the n of each variant).
- bbox_xywh_test_rescore.txt: byte copy of comm_bbox_xywh_baseline/test_rescore.txt (local MPS run, 5 Aug 2026).
- capacity_accum2_metrics.json: byte copy of results/legacy_runs/comm_slt_1024_512_3x3_accum2.metrics.json (local MPS run, 5 Aug 2026).
- capacity_params.json: parameter count (all tensors of comm_slt_1024_512_3x3_accum2/out/best.pt except the fixed pos_enc buffer), by prepare_data.py.

Not available locally (Kaya): comm_dwpose_full (DWPose re-extraction run) and estimator 2x2 job 1101250.
