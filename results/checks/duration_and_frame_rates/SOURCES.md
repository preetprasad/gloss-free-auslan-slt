# Sources
All CSVs are produced once by prepare_data.py (numbers only, no text); it is not run by make.
- clip_lengths.csv: clip, split, frame count of each per-clip pose tensor in full_run/clips_signer (14,041 clips, extracted from the released per-video pose files, signer entry), and the End-Begin span of the same clip in the alignment table (join on video and subtitle over pairs unique in both tables, 12,438 clips).
- alignment_frames.csv: Begin/End frame and time columns of the Auslan-Daily Communication alignment table (data/raw/Auslan-Daily/Dataset Split Table/Sign Language Alignment).
- spotting_frames.csv: Begin/End frame columns of Sign_Spotting.xlsx (1,069 instances).
- mmwl_lengths.csv: frames per clip of the isolated-sign dictionary pose file (mmwl_pose/pose_cam1_trainval.pkl, 45,010 clips).
- mmwl_video_frames.csv: frame count of each dictionary recording (8,581) as extracted from the videos, taken from the segment sidecar results/stitched_corpus_audit/audit_A.jsonl (field source_frames_original; Kaya job 1216905, stitch audit of the adopted 13,760-sentence corpus).
The gate also reads results/stitched_corpus_audit/stitch_corpus_audit.json (published).
No Kaya job output for jobs 1175390, 1175508, 1175598 is local; none is used.
