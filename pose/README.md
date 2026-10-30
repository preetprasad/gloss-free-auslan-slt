# Pose front end (`pose/`)

This directory holds the code for the pose front end: keypoint extraction with DWPose, the person-tracking work, the signing-activity detector and the per-session person labels used by the live demonstrator. These scripts were written beside the outputs they produced, so some paths inside them point to output folders that are not distributed.

## `extraction/`

| File | Description |
|---|---|
| [`validate_dwpose.py`](extraction/validate_dwpose.py) | Runs an RTMPose/RTMW whole-body estimator through rtmlib on one sample signer video and compares hand-keypoint confidence with the shipped AlphaPose tensor. |
| [`validate_multi.py`](extraction/validate_multi.py) | Runs RTMW-DW-x on a random sample of extracted signer clips and compares hand-keypoint confidence with the shipped AlphaPose tensors for those clips. |
| [`agreement_rtmlib.py`](extraction/agreement_rtmlib.py) | Runs several rtmlib whole-body models on sampled clips and measures hand-keypoint disagreement, normalised by hand span, as a proxy for accuracy without ground truth. |
| [`dwpose_fps.py`](extraction/dwpose_fps.py) | Times the DWPose (RTMW whole-body) estimator per frame on a set of clips and reports milliseconds per frame and frames per second. |
| [`dwpose_extract_worker.py`](extraction/dwpose_extract_worker.py) | Shardable, resumable DWPose extraction worker with capped onnxruntime threads that writes COCO-WholeBody pose and a dummy bounding box per clip. |
| [`dwpose_extract_worker_tracked.py`](extraction/dwpose_extract_worker_tracked.py) | DWPose extraction that follows one person across a clip by seeding on the likely signer, matching shoulder midpoints, and writing a real bounding box. |
| [`dwpose_extract_worker_alldet.py`](extraction/dwpose_extract_worker_alldet.py) | DWPose extraction that keeps every detected person's keypoints, scores and detector boxes alongside the tracked output, so that trackers can be compared offline. |
| [`test_tracking.py`](extraction/test_tracking.py) | Tests the tracked worker's seeding and tracking functions on synthetic two-person frames whose detection order flips from frame to frame. |
| [`run_dwpose_pipeline.sh`](extraction/run_dwpose_pipeline.sh) | Unattended, resumable shell pipeline that streams the signer clip archive with rclone, extracts the videos, re-extracts DWPose pose and retrains. |
| [`run_dwpose_v2.sh`](extraction/run_dwpose_v2.sh) | Runs parallel thread-capped DWPose extraction workers, rebuilds the manifests, then runs retraining, beam decoding and a CTC experiment in sequence. |
| [`build_dwpose_manifests.py`](extraction/build_dwpose_manifests.py) | Rewrites the official Communication train, validation and test manifests to point at the DWPose files, keeping only successfully re-extracted clips. |
| [`compute_dwpose_conf.py`](extraction/compute_dwpose_conf.py) | Computes per-clip mean DWPose hand and body keypoint confidence from the re-extracted files and writes them to a CSV table. |
| [`mmwl_qc.py`](extraction/mmwl_qc.py) | Applies a per-clip quality gate to MM-WLAuslan DWPose files using hand confidence, failed-frame fraction and frame count, and writes `qc.csv` and `keep.txt`. |
| [`classical_cv_worker.py`](extraction/classical_cv_worker.py) | Shardable, resumable classical computer-vision analysis of raw signer videos that measures exposure, sharpness, motion, skin regions and faces, and sets difficulty flags. |
| [`join_pose_confidence.py`](extraction/join_pose_confidence.py) | Joins the classical computer-vision per-clip table with shipped AlphaPose hand confidence and analyses which properties predict pose failure through correlations and buckets. |
| [`extract_rgb_feats.py`](extraction/extract_rgb_feats.py) | Extracts per-clip RGB features by encoding signer and hand crops, located with DWPose keypoints, using a frozen DINOv2-small backbone. |

## `person_switch_audit/`

| File | Description |
|---|---|
| [`render_overlay.py`](person_switch_audit/render_overlay.py) | Draws the stored first-detection skeleton and the person-tracked DWPose skeleton over the frames of one validation clip, writing the images outside the repository. |
| [`decode_counterfactual.py`](person_switch_audit/decode_counterfactual.py) | Decodes one walkthrough clip with the walkthrough's checkpoint and decoding settings under stored, person-tracked and truncated person-tracked pose inputs. |
| [`render_audit.py`](person_switch_audit/render_audit.py) | Renders stratified panels with randomly coloured stored and tracked skeletons on validation and test clips, so that labellers can audit the tracker blind. |
| [`score_audit.py`](person_switch_audit/score_audit.py) | Scores the two raters' blind-audit labels against the hidden colour key, reporting inter-rater agreement and consensus counts per stratum. |
| [`tracking_dose_response.py`](person_switch_audit/tracking_dose_response.py) | Exploratory difference-in-differences comparison of models trained on first-detection and person-tracked pose, binned by tracker-override fraction, against a rule written beforehand. |

## `tracker_v2/`

| File | Description |
|---|---|
| [`pick_gt_clips.py`](tracker_v2/pick_gt_clips.py) | Samples ground-truth clips from the official validation and test splits, stratified by tracker-override fraction, and divides each stratum into development and held-out parts. |
| [`render_gt.py`](tracker_v2/render_gt.py) | Draws every detected person on sampled frames with randomly ordered letters, plus context frames, so that raters can identify the signer blind. |
| [`build_labels.py`](tracker_v2/build_labels.py) | Combines two raters' panel labels with a third adjudicating rater into final ground-truth labels, marking unresolved panels as unclear. |
| [`tracker_v2.py`](tracker_v2/tracker_v2.py) | Offline tracker that merges duplicate detections, links them into tracklets, scores tracklets over the whole clip and masks frames without the signer. |
| [`score_gt.py`](tracker_v2/score_gt.py) | Scores trackers against the ground-truth panel labels as hit, wrong, miss or correct mask, with wrong-person rate as the primary metric. |
| [`materialise_arms.py`](tracker_v2/materialise_arms.py) | Builds first-detection, frame-by-frame-tracker and offline-tracker pose arms and their manifests from one all-detections extraction for a training comparison. |

## `tracker_v2/port/`

| File | Description |
|---|---|
| [`make_tracklets_fixture.py`](tracker_v2/port/make_tracklets_fixture.py) | Generates seeded random edge-case clips and the Python tracker's choices on them as a parity fixture for the JavaScript tracklets test. |
| [`export_boxes.py`](tracker_v2/port/export_boxes.py) | Writes detector boxes for the ground-truth clips beside links to the existing flat keypoint and score files, checking byte-level consistency. |
| [`dump_v2_reference.py`](tracker_v2/port/dump_v2_reference.py) | Records the Python tracker's per-frame choices under tuned and default weights on the ground-truth clips, without reading labels, as a port reference. |
| [`eval_port.mjs`](tracker_v2/port/eval_port.mjs) | Compares the demonstrator's JavaScript tracklets module with the Python tracker on every frame of the ground-truth clips and exits non-zero on any mismatch. |
| [`eval_variant.mjs`](tracker_v2/port/eval_variant.mjs) | Runs the demonstrator's tracklets-plus-activity JavaScript variant on stored all-detections clips, writes per-frame choices and checks path grouping against the tested tracker. |
| [`score_variant.py`](tracker_v2/port/score_variant.py) | Scores the tracklets-plus-activity variant against the other trackers and the roster-activity rule on the ground truth, under an acceptance rule stated beforehand. |

## `tracker_v2/research/`

| File | Description |
|---|---|
| [`residuals.py`](tracker_v2/research/residuals.py) | Analyses residual defects of the tuned tracker on the ground-truth clips: masking causes, masked-gap lengths, hand quality and development-only variant selection. |
| [`verify_residuals.py`](tracker_v2/research/verify_residuals.py) | Independently recomputes the residual-defect analysis from raw all-detections files and labels, describing the held-out part without scoring variants on it. |
| [`seed_selection.py`](tracker_v2/research/seed_selection.py) | Examines the seed tracklet behind long masked gaps, accounts masked frames by their labels, and scores two seed rules on development panels only. |
| [`cuts.py`](tracker_v2/research/cuts.py) | Measures scene change at tracklet boundary joins from consecutive greyscale frame differences in signer-crop videos, and tests a per-shot tracker on development clips. |
| [`shipped_anchor.py`](tracker_v2/research/shipped_anchor.py) | Builds an anchor tracker that selects the DWPose detection matching the shipped annotated AlphaPose signer after an estimated crop offset, and scores it. |
| [`shipped_track_vs_gt.py`](tracker_v2/research/shipped_track_vs_gt.py) | Maps the shipped AlphaPose signer track onto DWPose detections through fitted per-clip transforms and measures how often it lies on the signer. |
| [`alphapose_identity_check.py`](tracker_v2/research/alphapose_identity_check.py) | Compares rates of switch-sized shoulder jumps in shipped AlphaPose and stored DWPose tracks, binned by override fraction, as a frame-invariant identity check. |
| [`switch_lens.py`](tracker_v2/research/switch_lens.py) | Measures where person switches concentrate and regresses per-clip translation quality on switch exposure, with intervals from a video-cluster bootstrap. |
| [`switch_lens_controls.py`](tracker_v2/research/switch_lens_controls.py) | Adds bystander, signer-group and crowding controls to the switch-exposure regression, and restricts it to clips in which a switch was possible. |
| [`verify_switch_lens.py`](tracker_v2/research/verify_switch_lens.py) | Independently re-implements the switch-exposure analysis and adds cluster-robust intervals, permutation tests and leave-one-video-out influence checks. |
| [`pose_cleaning_headroom.py`](tracker_v2/research/pose_cleaning_headroom.py) | Describes how much confidence thresholding, gap interpolation and Savitzky-Golay smoothing would alter in the project's DWPose input, without training a model. |
| [`verify_pose_cleaning_tail.py`](tracker_v2/research/verify_pose_cleaning_tail.py) | Checks what share of large hand smoothing residuals in validation DWPose clips lie near switch-sized jumps, also split by hand speed. |
| [`verify_numpy_elision_hazard.py`](tracker_v2/research/verify_numpy_elision_hazard.py) | Minimal reproduction that tests whether numpy temporary elision modifies a named local array in place under the Python interpreter in use. |

## `signing_activity/`

| File | Description |
|---|---|
| [`pose_activity_probe.py`](signing_activity/pose_activity_probe.py) | Exploratory probe of a detector based on shoulder-normalised keypoint displacement that separates signers from non-signer crops, fitted on development data and scored once on test. |
| [`signing_activity.py`](signing_activity/signing_activity.py) | Fits a logistic regression on interpretable per-clip pose features to separate signers from non-signer crops, with cross-validation grouped by video. |
| [`signing_activity_checks.py`](signing_activity/signing_activity_checks.py) | Runs robustness checks on the detector: a reproduction gate, an identity confound by signer, a per-clip top-ranking analysis and a filter sensitivity test. |
| [`verify_signing_activity.py`](signing_activity/verify_signing_activity.py) | Independently re-implements the signing-activity analysis with its own features, folds and bootstrap, and adds orientation-confound and per-category checks. |

## `person_labels/`

| File | Description |
|---|---|
| [`fit_activity_model.py`](person_labels/fit_activity_model.py) | Fits the frozen signing-activity logistic model for the demonstrator on development data, excluding ground-truth clips, and writes it as JSON and a JavaScript module. |
| [`export_alldet.py`](person_labels/export_alldet.py) | Exports all-detections files to JSON metadata and raw binary keypoint and score arrays for evaluation in Node. |
| [`eval_roster.mjs`](person_labels/eval_roster.mjs) | Runs the demonstrator's roster and activity code on stored all-detections clips and writes the detection index each rule follows on every frame. |
| [`score_person_labels.py`](person_labels/score_person_labels.py) | Scores the roster and activity rule, alongside comparison trackers, against the two-rater ground truth on development panels or, once only, held-out panels. |