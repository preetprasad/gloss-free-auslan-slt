# Sources
All files in data/ were written by prepare_data.py (run once, not by make).

| file | derived from | note |
|---|---|---|
| dwpose_conf.csv | pose/extraction/dwpose_conf.csv | byte copy; one row per DWPose-extracted Communication clip (compute_dwpose_conf.py, Kaya DWPose extraction, August 2026) |
| dwpose_train_clip_ids.txt | results/checks/dataset_statistics/data/dwpose_train_clip_ids.txt | byte copy; clip ids of the DWPose training list |
| pred_dwpose.csv | results/probe_signerfix_2026-10-25/pred_dwpose.csv | byte copy; per-span top-1 correctness of the continuous identification probe (DWPose) |
| spans45.csv | results/probe_signerfix_2026-10-25/spans45.csv | byte copy; the annotated sign-spotting spans |
| hand_speed_591.csv | pose/extraction/dwpose_clips and results/nmm_probe/tracked_train | per-clip hand_speed from tools/hand_visibility_cv.py::hand_stats (unchanged) on the 591 training clips held in both versions |
| span_hand_conf.csv | results/probe_signerfix_2026-10-25/dwpose_clips + spans45.csv | per-span hand_conf from hand_stats |
| figure_clip_frame_hand_conf.csv | full_run/clips/video_11_6.npz, pose/extraction/dwpose_clips/video_11_6.npz | per-frame mean hand confidence of both keypoint sets (confidences only; frame count = video frames, as in figures/make_new_figures.py) |
| kps_shape.txt | the same DWPose npz | keypoint count and values per keypoint |

Training manifest (12,441 rows) is read from comm_run/manifests/train.tsv (written by `make data`).
