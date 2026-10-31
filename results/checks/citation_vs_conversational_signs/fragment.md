### AX — Citation versus conversational signs: clip lengths and durations
**Claim.** The Methodology (method.tex, stitching and duration paragraphs) and Results (clip-length table) state that a real Auslan-Daily sentence clip averages 73.1 frames, that a MM-WLAuslan citation clip averages 79.4, that the 1,069 Sign_Spotting instances (61 on validation and test clips) average 22.49 frames (median 20) at 25 fps, that signs are 72% shorter in conversation over 45 shared glosses, and that GOAL lasts 6.4 times as long in citation form (median 3.08 s) as in conversation (median 0.48 s).

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/citation_vs_conversational_signs/check.py
```
```text
GATE train-split mean clip length 75.2 frames (recorded 75.2)

clips 14041; mean 73.1 frames (train 75.2, dev 57.1, test 56.9)  [c09-037 c09-045 c07-031]
Auslan-Daily frames per second from the alignment table: 25.00 (25 expected)  [c10-044]
MM-WLAuslan pose clips (Train+Valid): 45010; mean 80.27 frames; dissertation says 79.4  [c07-031]
Sign_Spotting: 1069 instances, 61 on official validation and test clips, mean 22.49 and median 20 frames at 25 fps (End - Begin)  [c07-034]
GOAL citation form: n=14, median 92.5 frames = 3.08 s at 30 fps  [c07-038]
GOAL conversation:  n=10, median 12 frames = 0.48 s at 25 fps  [c07-038]
ratio 6.4 times (dissertation: 6.4, medians 3.08 s and 0.48 s, n=12 and n=6)  [c07-038]
conversation is 65% shorter in seconds (2.89 times); dissertation says 72% (3.60 times)  [c07-032]
```
**Scope.** Recomputed from the released Sign_Spotting and alignment tables, the frame counts of the shipped Communication tensors and of the MM-WLAuslan pose clips. Reproduced: 73.1, 25 fps, 1,069/61/22.49/20, the GOAL medians and ratio. Not reproduced: 79.4 (saved data give 80.27), 72% shorter (65%), and the GOAL sample sizes (n=14 and n=10, not 12 and 6). The MM-WLAuslan 30 fps and the within-sign variability (0.6319, 0.0872) have no saved output. No training or inference.
