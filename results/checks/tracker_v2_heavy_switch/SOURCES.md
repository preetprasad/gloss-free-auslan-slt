# Sources

All files in `data/` are produced by `prepare_data.py` (run once, not by `make`) from saved outputs; `data/w12i_bins.json` is a byte copy of `src/tools/w12i_bins.json`.

| file | local source | job |
|---|---|---|
| `w12i_{val,test}_{c0,v2}.jsonl` | `results/wave12i_138391/<arm>_s<seed>/beam5_<split>_best.json` (hypotheses and recorded BLEU-4 only) and the cross-pose decodes `beam5_<split>_best_on_<other>.json` copied from the cluster | 138391 / 138392 |
| `w9_{val,test}.jsonl` | `results/tracker_v2_translation_test/runs/<arm>_s<seed>/beam5_<split>_best.json` | 43506 |
| `w3_{val,test}.jsonl` | `results/switch_video_89_193/w3_val/<arm>_s<seed>[_test].json` | 6592 / 9444 |
| `clip_order.json` | `results/tracker_v2_translation_test/manifests/c0_{val,test}.tsv` (clip names; checked equal to `comm_run/manifests`) | 43506 |
| `w12i_bins.json` | `src/tools/w12i_bins.json` (heavy/none bins from `results/ndet_per_clip.csv`) | 6590 |
| `tracking_switched_clips.csv`, `tracking_prevalence_agg.json` | `results/switch_video_89_193/w3_val/tracking_prevalence.json` | 6591 |
| `tracker_audit_labels.json` | `results/switch_video_89_193/audit/labels_scored.json` | blind audit, 25 Sep |
| `hand_speed_per_clip.csv` | `results/nmm_probe/tracked_train/*.npz` and `pose/extraction/dwpose_clips/*.npz` via `tools/hand_visibility_cv.py::hand_stats` | local |
| `pose_jump_census.csv` | `pose/extraction/dwpose_clips`, `test_clips_dwpose`, `val_clips_dwpose` | local |
