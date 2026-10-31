### AX — Extraction-error audit: signer entry, keypoint layouts, registration, bounding boxes
**Claim.** The released Auslan-Daily pose files store HALPE-136 keypoints, shape (T, 136, 3) (26 body, 42 hand and 68 face keypoints), and list every detected person with one entry marked as the signer. In 12,411 of the 14,041 Communication clips (88.4%) the file lists several people, and the project's first extraction kept the first-listed entry, a non-signer (Method, "Signer selection"; Fig. signer selection). The DWPose arrays hold 133 keypoints per frame; for 12,409 of the 12,411 multi-person clips a DWPose file exists. In the pose-estimator figure the distributed keypoints align to the crop with a median residual of 5.3 pixels, and the extractor probe compares input sizes 384x288 and 256x192 (Appendix). The bounding-box audit (Method, Table "Bounding-box format audit") found 200 Communication clips for which the boxes read as corners contain 14.4% of the confident joints and the boxes read as corner plus width and height 88.9%.
**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/extraction_error_audit/check.py
```
```text
GATE clips in the released pose files 14,041 (expected 14,041): ok
GATE clip ids of the manifests (make data) equal the pose-file ids: ok

AlphaPose HALPE layout: every entry of every clip stores keypoints of shape (T, 136, 3): 14,041 of 14,041 clips (shape counts {(136, 3): 14041})  [c05-007 c06-018 c08-012 c15-003]
keypoint count per frame in the layout: 136 (HALPE-136)  [c05-005 c05-007 c06-018 c08-012 c15-003]
every clip has one array per entry with 4 box columns; keypoint columns x, y, confidence = 3  [c05-007]
HALPE-136 groups, from the definition in the re-indexing tool (not measured; the ranges partition all 136 measured keypoint indices): 26 body (incl. head/neck/hip and feet), 42 hand (21 per hand), 68 face; total 136  [c05-005 c05-007]
DWPose (COCO-WholeBody) stored arrays: 14,039 of 14,039 files hold 133 keypoints per person and frame (body, hands, face)  [c02-002]
GATE multi-person clips recorded as 12,411 (88.4%), old extraction = first-listed entry in all and the signer entry in none: ok

clips whose file lists several people: 12,411 of 14,041 (88.4%)  [c06-019]
  of these, the entry marked 'Signer' is listed first in 0; every clip has a 'Signer' entry: True  [c06-019]
the old extraction (first array found in the entry order) equals the FIRST-LISTED entry in 12,411 of 12,411 multi-person clips and the 'Signer' entry in 0 -> the program kept a non-signer in 12,411 clips (88.4% of 14,041)  [c06-019]
single-person clips (1,630): old extraction = the 'Signer' entry in 1,630  [c06-019]

multi-person clips with a DWPose file: 12,409 of 12,411; without: 2 (video_12_62, video_68_179)  [c21-015]

video_11_6: scale 0.997; median residual of the 11 shared body keypoints after alignment 5.3 pixels (770 keypoint pairs)  [c08-026]
in that clip the first-listed array equals the signer array: True

extraction library mode table (input size height x width; the checkpoint file names agree): 'performance' 384x288, 'balanced' 256x192; the production extraction script defaults to 'balanced'  [c17-005]
GATE audit sample recorded as 200 clips, 15,231 frames, 14,910 (xywh) and 6,271 (xyxy) valid boxes: ok
GATE audited clips = first 200 of the seed-42 shuffle of the Communication training manifest: ok

bounding-box audit on 200 Communication training clips (15,231 frames): confident joints inside the implied box: box read as xywh 88.9%, read as xyxy 14.4%  [c06-022]
  valid (positive-size) boxes: xywh 14,910 of 15,231, xyxy 6,271 of 15,231  [c06-022]
  (not the audited table's definition) x/y normalised by the box, all 136 keypoints, frames with a positive box, read as xywh: std 0.28, extrema -1.0 / 1.9
  (not the audited table's definition) x/y normalised by the box, all 136 keypoints, frames with a positive box, read as xyxy: std 52.41, extrema -45.5 / 6252.2
```
**Scope.** Recomputed from `pt_clip_table.csv` (one row per released clip: entries, shapes, equality of the old extraction with each entry), the stored DWPose layout of all 14,039 local files, the 11 shared body keypoints of one clip, the extraction library's mode table (copied from its source; the checkpoint file names agree with the input sizes) and per-clip counts of the bounding-box audit. The headline figures (12,411 multi-person clips, 88.4%, no clip whose old extraction equals the signer entry) and the audit sample (200 clips, 15,231 frames, 14,910 and 6,271 valid boxes) are gates: the check exits non-zero if they change. The 26/42/68 split is NOT measured: the keypoint arrays carry no group labels, so the split is read from the HALPE-136 definition in the project's re-indexing tool (whose own assertions are executed) and shown to partition the measured 136 indices; only the totals 136 and 133 are measured. The extractor input sizes show what the library's two modes are and that the production extraction script defaults to `balanced`; that the resolution probe used these two modes (Mac re-extraction, 22 Sep) is not recomputed. The audited bounding-box sample is identified here: the first 200 clips of the seed-42 shuffle of the Communication training manifest (via `tools/audit_bbox_format.py`'s rule, confidence above 0.1) reproduces the recorded counts. Not reproduced: the coordinate standard deviations and extrema of the audit table (87 and -5,036 / +12,912 for corners, 0.60 and -2.1 / +2.6 for corner plus size): the script that defined them is lost, and the natural definition gives other values (printed untagged); these are listed as disagreements. Not recomputed: the 384x288 vs 256x192 hand-probe scores (re-extraction not kept) and the HRNet-vs-DWPose frame check (Kaya job 1100622). No training or inference. Local measurements; no Kaya jobs.
