### AX — Dataset statistics

**Claim.** The dissertation (Review, Table `tab:datasets` and the dataset subsection; Methodology annotation inventory; Results opening; Appendix B inventory) states the sizes of Auslan-Daily and MM-WLAuslan: the Communication subset has 90 videos, 12,441 / 800 / 800 clips, 49 / 12 / 9 signers and a 3,064-word training vocabulary; the training list used for the DWPose input has 12,439 clips; the release carries 1,069 sign-spotting instances (120 glosses), 3,209 isolated gloss segments (610 glosses), 2,004 fingerspelling spans (1,343 on Communication clips) and a 14,041-row alignment file; MM-WLAuslan has 45,010 clips of 3,215 glosses (14 per gloss). The Results also say that clip extraction is exact to two frames and that about 1% of clips are affected by overlapping subtitles. The News figures (9,665 training clips, 18/17/17 signers, vocabulary 12,346) and the totals 25,106 / 22,106 are the Auslan-Daily paper's; the released spreadsheets give different numbers, which this check reports.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/dataset_statistics/check.py
```

```text
GATE Communication rows 14041 (expected 14041), split 12441/800/800 (expected 12441/800/800)

AUSLAN-DAILY COMMUNICATION (translation spreadsheet)
Videos: 90  [c03-016]
Clips train/dev/test: 12,441 / 800 / 800  [c04-010 c04-022]
Signers train/dev/test: 49/12/9  [c04-010 c04-022]
Training vocabulary (lower-cased whitespace tokens): 3,064  [c04-010 c04-022]
Total clips (Communication): 14,041  [c15-013]
Training list used for DWPose input and CTC target: 12,439 of 12,441 training clips (2 of the official list absent: ['video_12_62', 'video_68_179'])  [c08-001]
Test clips 800  [c08-001]

AUSLAN-DAILY NEWS (counts saved from the News translation spreadsheet; read from the saved file news_counts.json)
Clips train/dev/test: 9,669 / 700 / 700; rows 11,069  [c04-011 c04-023]
Signers train/dev/test (identifiers stripped of spaces): 22/21/21  [c04-011 c04-023]
Training vocabulary (lower-cased whitespace tokens): 18,893  [c04-011 c04-023]
Total clips, both subsets: 25,110; training clips both subsets: 22,110  [c04-025 c04-037 c05-028]
published (Auslan-Daily paper, Table 2): News 9,665 / 700 / 700 clips, 18/17/17 signers, vocabulary 12,346; 25,106 clips in all
DISAGREES with the published News figures: signers 22/21/21 vs published 18/17/17, vocabulary 18,893 vs published 12,346  [c04-011 c04-023]
DISAGREES with the published News figures: News train 9,669 vs 9,665 (+4); total 25,110 vs 25,106 (+4); training clips 22,110 vs 22,106 (+4)  [c04-011 c04-023 c04-025 c04-037 c05-028]

ANNOTATION INVENTORY (release tables, counts saved by prepare_data.py)
Sign-spotting instances: 1,069 occurrences of 120 glosses  [c06-014 c15-009]
Isolated gloss segments: 3,209 (610 glosses)  [c06-015 c15-010]
Fingerspelling spans: 2,004, of which 1,343 on Communication clips  [c06-015 c15-011]
Subtitle alignment file (Communication): 14,041 rows  [c15-012]
English sentence per Communication clip: 14,041 clips  [c15-013]

MM-WLAuslan (Train.json + Valid.json labels, per-gloss clip counts)
Front-view clips 45,010, glosses 3,215, exemplars per gloss 14.0  [c06-027]

ALIGNMENT AUDIT
Extraction vs alignment span on 587 locally held clips: identical 57.9%, within 2 frames 100.0%, largest difference 1 frames, correlation 1.0000  [c10-049]
Consecutive subtitle pairs 13,951; overlapping 15.98% (recorded by the audit: 15.98%)
Overlapping pairs with both signers known: 1,704; different signer 64.3%, same signer 35.7%  [c10-050]
Same signer and more than 25% of the later clip overlapped: 144 clips, 1.16% of the 12,438 matched clips (about 1%; the audit's own join gave 131 clips, 1.05%)  [c10-050]
```

**Scope.** Counts are recomputed from the Communication translation spreadsheet that `make data` restores and from counts-only files saved from the release's other tables (News translation table, annotation tables, MM-WLAuslan label files, alignment table, frame counts of 657 locally held DWPose clips); `prepare_data.py` made them and `SOURCES.md` names each. The gate requires the Communication spreadsheet to have the 14,041 rows and the 12,441 / 800 / 800 split and the alignment audit's overlap share to equal the 15.98% it recorded (13,951 consecutive pairs). The overlap share of same-signer clips is recomputed with this check's own definition (overlap above 25% of the later clip); it gives 144 clips (1.16%), where the original audit's slightly different join gave 131 (1.05%), both "about 1%". The extraction check uses 587 clips (the audit used 586). Not done: the News figures cannot be confirmed: the News translation table has 9,669 training clips, 22/21/21 signers and a vocabulary of 18,893, so the dissertation's News counts and the totals 25,106 / 22,106 (25,110 / 22,110 here) are reported as disagreeing, not forced. The dissertation's vocabulary and signer figures for News are the paper's.
