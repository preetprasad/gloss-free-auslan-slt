# Sources
- prepare_data.py: written here; run once in the private tree (needs raw data), not run by make.
- spotting.csv: data/raw/Auslan-Daily/Dataset Split Table/Sign Spotting/Sign_Spotting.xlsx (clip, gloss, begin, end, sheet split) plus the official split of each clip from the Communication translation spreadsheet (Split column). Dataset annotation, no subtitles.
- alignment_frames_seconds.csv: Sign Language Alignment/Auslan_Daily_Communication.xlsx (Begin/End frame, begin/end time in seconds).
- auslan_daily_clip_frames.csv: frame counts (T of KPS) of the 14,041 per-clip tensors in full_run/clips_signer (extracted from the released SP_Annotation_V1 pose files, signer entry), plus official split.
- mmwl_clip_frames.csv: frame counts and gloss labels of the 45,010 clips in mmwl_pose/pose_cam1_trainval.pkl with Train.json/Valid.json (MM-WLAuslan Train+Valid, camera Kinect_F).
- output.txt: exact output of check.py. No job ids: all inputs are released dataset files.
