### AX — Real-clip lengths and frame rates behind the temporal-fidelity argument

**Claim.** The appendix table on clip length (Appendix B, temporal fidelity) states that Auslan-Daily is recorded at 25 fps while MM-WLAuslan is 30 fps, that not resampling lengthens every synthetic clip 1.2 times, that a real Auslan-Daily clip averages 73.1 frames and a real continuous sign 22.5 frames, and that an MM-WLAuslan dictionary clip averages 79.4 frames. The duration-modelling scores (10.97 against 9.49, +1.48 over three seeds; the one-seed temporal arms) are not recomputable here.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/duration_and_frame_rates/check.py
```
```text
GATE alignment-table mean End-Begin 72.7 frames, recorded 72.7: ok
GATE per-clip tensor length vs its alignment span over 12438 unique (video, subtitle) pairs: 99.6% within 2 frames, r 0.9996; recorded: all within 2 frames, r 1.0000: ok
GATE sign-spotting table 1069 instances, median 20, p10 10, p90 37; recorded 1069, 20, 10, 37 (comment of the stitching builder): ok
GATE dictionary pose file 45010 clips; its frame counts agree within 1 frame with the counts extracted from the videos for 8581 recordings (100.0%); recorded 45010 clips, 8581 recordings: ok
Auslan-Daily Communication frame rate 25 fps (frames / seconds over 14041 alignment rows)  [c15-017 c15-018]
Real Auslan-Daily clip (a whole sentence, 25 fps): mean 73.1 frames over 14041 shipped tensors  [c15-017]
Real continuous sign, per sign (25 fps): mean 22.5 frames over 1069 sign-spotting instances (median 20)  [c15-018]
Not recomputed: the 30 fps of MM-WLAuslan and the 1.2x it implies need the frame rate of the dictionary videos, which are not in this repository
DISAGREES: MM-WLAuslan isolated-sign clip as shipped pose: mean 80.3 frames over 45010 clips (dissertation: 79.4); the video-extracted counts of 8581 recordings give mean 78.66 against 78.74 for the same recordings in the pose file
```
**Scope.** Recomputed from per-clip frame counts: the shipped per-clip tensors (clip_lengths.csv), the alignment table (alignment_frames.csv, frame rate = frames / seconds) and the sign-spotting table (spotting_frames.csv). Four gates tie each input to an independent record: the alignment-table mean of 72.7 frames, the per-clip tensor length against its alignment span, the recorded sign-spotting distribution, and the dictionary pose file against the frame counts extracted from the videos by the stitching builder. The 25 fps half of the frame-rate statement is measured; the MM-WLAuslan 30 fps and the 1.2 times it implies need the dictionary videos, which are not in this repository, so they are not claimed as reproduced. The dictionary clip length measured on the pose file is 80.3, which disagrees with the 79.4 in the dissertation; it is reported, not forced. The training scores of Kaya jobs 1175390, 1175508 and 1175598 and the synthetic-corpus lengths (246.1, 205.9, 108.4) are not stored locally and are not recomputed.
