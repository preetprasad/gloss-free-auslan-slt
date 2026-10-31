# Results: saved outputs and read-out scripts

This directory holds two things.

The first is the saved outputs from which the reported numbers are re-derived without a GPU: the decoded hypotheses of the reported systems (`final_ensemble_*_decodes/`, the `ensemble_*_decodes/` folders of the unclaimed ensembles, and `semantic_adequacy_grading/`), the per-seed summary files of the multi-seed experiments, and `train_target_hashes.txt`, which holds SHA-1 digests of the training targets. `REPRODUCING.md` at the repository root states which check reads which file. The reference sentences belong to Auslan-Daily and are not stored here: each is replaced by its SHA-1 digest until `make data` restores it from the dataset's spreadsheet, and `SHA256SUMS` lists the checksums that the restored files must have.

The second is the read-out and verification scripts of each experiment, listed below. A directory name describes its content; the table "Where each result lives" in the root README gives, for each directory, the dissertation section and the cluster job numbers. Saved outputs are included only for the checks in `REPRODUCING.md`, so the read-out scripts of the other experiments document how their numbers were computed and do not run stand-alone. The scripts that draw the dissertation's figures are in `../figures/` and are listed at the end.

## Data and failure analysis

### `per_clip_quality_analysis/`

| File | Description |
|---|---|
| [`analyze.py`](per_clip_quality_analysis/analyze.py) | Studies the Auslan-Daily Communication manifests empirically to investigate the BLEU plateau and the transfer outcome, writing findings to a JSON file. |
| [`perclip.py`](per_clip_quality_analysis/perclip.py) | Decodes every test clip with both models, joins each prediction to its pose quality and reference, and scores clips individually. |
| [`ollama_adequacy.py`](per_clip_quality_analysis/ollama_adequacy.py) | Asks a local Ollama language model to grade, for sampled clips, how well each model's output conveys the reference meaning. |

### `semantic_adequacy_grading/`

| File | Description |
|---|---|
| [`dump_hyps.py`](semantic_adequacy_grading/dump_hyps.py) | Decodes the official test split from a pose or RGB-fusion checkpoint and writes per-clip reference and hypothesis pairs as JSON Lines. |
| [`grade_semantic.py`](semantic_adequacy_grading/grade_semantic.py) | Grades each reference and hypothesis pair for semantic adequacy on an ordinal rubric, using a local language model served through Ollama. |
| [`grade_semantic_hf.py`](semantic_adequacy_grading/grade_semantic_hf.py) | Applies the rubric of `grade_semantic.py` with a local HuggingFace instruction-tuned model as judge, writing graded rows to a CSV file. |

## Evaluation protocol and run-level noise

### `run_to_run_noise/`

| File | Description |
|---|---|
| [`null_band.py`](run_to_run_noise/null_band.py) | Estimates run-level noise from replicate runs at one seed, reporting the standard deviation, its chi-square confidence interval and the paired single-run band. |
| [`adopted_identity.py`](run_to_run_noise/adopted_identity.py) | Checks whether adopted-recipe runs from different jobs share one configuration by comparing checkpoint configurations and training-manifest content hashes per seed. |
| [`adopted_crossbatch.py`](run_to_run_noise/adopted_crossbatch.py) | Estimates cross-batch noise of the adopted recipe as a pooled within-seed standard deviation of beam-search test BLEU with a chi-square interval. |

### `beam5_validation_selector_check/`

| File | Description |
|---|---|
| [`analyse_e4.py`](beam5_validation_selector_check/analyse_e4.py) | Compares beam-search and greedy validation as arm selectors under pre-stated agreement and Spearman-correlation rules, as an exploratory post-hoc analysis. |

## Training-data experiments

### `pose_augmentation_decodes/`

| File | Description |
|---|---|
| [`pooled_bootstrap.py`](pose_augmentation_decodes/pooled_bootstrap.py) | Computes a pooled paired bootstrap across seeds of the BLEU difference between pose-augmented and control arms, resampling official test sentences jointly. |

### `hand_masking_and_longer_training/`

| File | Description |
|---|---|
| [`stage1_decision.py`](hand_masking_and_longer_training/stage1_decision.py) | Applies the pre-registered first-stage futility rule, comparing each arm's beam-search validation BLEU with the control arm's per seed from saved decodes. |

### `stitch_trim_proxies/`

| File | Description |
|---|---|
| [`trim_proxies_readout.py`](stitch_trim_proxies/trim_proxies_readout.py) | Reads the trim-proxies job log for two corpora, checks the audit tool's hash, and computes the synthetic-join to real-interior gap ratio. |
| [`verify_trim_proxies.py`](stitch_trim_proxies/verify_trim_proxies.py) | Independently re-checks the trim-proxies readout from the raw log, the audit-tool source, the earlier motion-fidelity record and the training summaries. |

### `pretraining_selection_audit/`

| File | Description |
|---|---|
| [`pj_audit.py`](pretraining_selection_audit/pj_audit.py) | Purpose not stated in the file header. |
| [`pj_seen.py`](pretraining_selection_audit/pj_seen.py) | Purpose not stated in the file header. |

### `wider_corpus_joint_training/`

| File | Description |
|---|---|
| [`analyse_wave6.py`](wider_corpus_joint_training/analyse_wave6.py) | Applies the pre-registered corpus decision rule after gating recorded values and re-scoring stored BLEU, using the project's decision and readout tools. |
| [`verify_wave6.py`](wider_corpus_joint_training/verify_wave6.py) | Independently re-scores every raw beam-search decode, cross-checks decode logs and recorded first-stage values, then reapplies the pre-registered decision rule. |

### `sign_density_stage1/`

| File | Description |
|---|---|
| [`w7_summaries.py`](sign_density_stage1/w7_summaries.py) | Builds per-seed summaries for the sign-density arms after asserting run completion, BLEU re-scoring from saved hypotheses and reference agreement with manifests. |
| [`w7_readout.py`](sign_density_stage1/w7_readout.py) | Computes paired per-seed differences, sign counts and seed-level t-intervals for the density arm, and applies the pre-registered first-stage futility rule. |

### `sign_density_stage2/`

| File | Description |
|---|---|
| [`w7s2_summaries.py`](sign_density_stage2/w7s2_summaries.py) | Builds gated per-seed summaries for the second-stage sign-density runs, including the length-rescaled arm, using the same checks as the first stage. |
| [`w7s2_readout.py`](sign_density_stage2/w7s2_readout.py) | Applies the pre-registered adoption rule to the density arm across first- and second-stage seeds, and reports the length-rescaled arm descriptively. |

### `real_data_learning_curve/`

| File | Description |
|---|---|
| [`arm_table.py`](real_data_learning_curve/arm_table.py) | Produces per-arm protocol readouts, including unseen-target BLEU and seen counts, after self-checks against a recorded control value and WER null floor. |
| [`q2_single_diffs.py`](real_data_learning_curve/q2_single_diffs.py) | Summarises per-seed held and kept test-cell differences from the learning-curve tool's printed values as seed-level t-intervals, after a self-check. |
| [`verify_wave8.py`](real_data_learning_curve/verify_wave8.py) | Independently recomputes the readout from raw decodes, progress files and manifests, regenerating training subsets with the pinned builder to check integrity. |

### `tracker_v2_translation_test/`

| File | Description |
|---|---|
| [`w9_summaries.py`](tracker_v2_translation_test/w9_summaries.py) | Builds per-seed summaries for the tracker arms after re-scoring stored BLEU from saved hypotheses and checking references against manifests. |
| [`w9_secondaries.py`](tracker_v2_translation_test/w9_secondaries.py) | Computes pre-registered descriptive secondaries: paired unseen-target and content-word deltas, a heavy-switch-bin difference-in-differences, and a frame-retention sensitivity analysis. |

## Cross-domain and split experiments

### `news_stage2_decision/`

| File | Description |
|---|---|
| [`news_stage2_decision.py`](news_stage2_decision/news_stage2_decision.py) | Applies the pre-registered second-stage News adoption rule: a one-sided paired t-test, a validation-mean condition and fresh-seed signs, after a reproduction gate. |

### `news_decoding_length_check/`

| File | Description |
|---|---|
| [`bp_readout.py`](news_decoding_length_check/bp_readout.py) | Reports brevity penalty and system-to-reference length ratio before and after the revised News decoding settings, reusing the pre-written analysis tool's selection. |
| [`verify_news_decode.py`](news_decoding_length_check/verify_news_decode.py) | Independently re-scores raw News decode dumps and logs through the project metric module, reproducing recorded numbers before evaluating the decoding test. |

### `video_disjoint_cross_modal_ensemble/`

| File | Description |
|---|---|
| [`g1_readout.py`](video_disjoint_cross_modal_ensemble/g1_readout.py) | Computes cross-modal ensembling readouts on the video-disjoint split from saved decodes: integrity gates, length-penalty selection, max-T intervals, paired bootstrap and secondaries. |

## Signer-keypoint checks

Saved outputs are in `estimator_signer_keypoints_rerun/` (the corrected estimator comparison: `readout.json`, per-seed summaries and beam-5 hypothesis dumps with the references replaced by digests; REPRODUCING.md, A32).

### `dwpose_signer_tracking_check/`

| File | Description |
|---|---|
| [`signer_check.py`](dwpose_signer_tracking_check/signer_check.py) | Measures for every Communication clip whether the stored DWPose keypoints follow the person marked as the signer in the released AlphaPose file. |
| [`trainclips_check.py`](dwpose_signer_tracking_check/trainclips_check.py) | Identifies which of two stored paths (first detection per frame, or the tracked path) the training clips follow. |
| [`panels.py`](dwpose_signer_tracking_check/panels.py) | Draws evidence panels of the clips where the stored keypoints and the signer disagree. |

### `dwpose_signer_guided_selection/`

| File | Description |
|---|---|
| [`wG_measure.py`](dwpose_signer_guided_selection/wG_measure.py) | Measures, per clip, how well signer-guided DWPose (the detection matching the annotated signer track) agrees with the signer, against first-detection and tracked pose. |
| [`wG_panels.py`](dwpose_signer_guided_selection/wG_panels.py) | Draws evidence panels for the signer-guided arm. |

## `checks/`: one folder per verified claim

Each folder holds the check script (`check.py`), the saved data it reads (`data/` or JSON files; model hypotheses, never reference sentences), `claims.json` (identifiers of the dissertation claims that the check covers, from the author's claim inventory), `SOURCES.md` (where each saved file came from) and `fragment.md` (the section that appears in `REPRODUCING.md` as A36 to A47). `make reproduce` runs the `check.py` of each folder through `REPRODUCING.md`. The few preparation scripts were run once in the private working tree to shrink the saved outputs to what the check needs; they cannot be run from this repository.

| File | Description |
|---|---|
| [`checks/pretrained_ensemble/check.py`](checks/pretrained_ensemble/check.py) | A36. Re-scores the saved decodes of the five fine-tuned pre-trained models and their ensembles and recomputes the max-T intervals and predicted power. |
| [`checks/clean_room_reimplementation/check.py`](checks/clean_room_reimplementation/check.py) | A37. Compares the independent implementation's logits and decodes with the saved ones and recomputes the own-code fine-tuning comparison. |
| [`checks/published_systems_own_code/check.py`](checks/published_systems_own_code/check.py) | A38. Re-scores the saved decodes of the authors' own SL-Transformer and GASLT code on AlphaPose and DWPose keypoints and applies the pre-registered reading rules. |
| [`checks/rest_cut_stitching/check.py`](checks/rest_cut_stitching/check.py) | A39. Recomputes the rest-run counts, the cut depths and the five-seed paired differences of the rest-cut stitching test. |
| [`checks/keypoints_69_input/check.py`](checks/keypoints_69_input/check.py) | A40. Recomputes the 69-keypoint input and front-end comparisons from saved decodes and parameter counts. |
| [`checks/keypoints_69_input/prepare_data.py`](checks/keypoints_69_input/prepare_data.py) | Shrinks the private result folders to the hypotheses and summary numbers the check reads (run once in the private tree). |
| [`checks/three_training_ideas/check.py`](checks/three_training_ideas/check.py) | A41. Recomputes the paired differences of the three training ideas and the span-placement statistic of the CTC head. |
| [`checks/extra_pretraining_cycle/check.py`](checks/extra_pretraining_cycle/check.py) | A42. Recomputes the extra-pre-training comparison over five new seeds from saved decodes and epoch logs. |
| [`checks/news_stitching_vs_augmentation/check.py`](checks/news_stitching_vs_augmentation/check.py) | A43. Recomputes the News decomposition from per-sentence BLEU sufficient statistics. |
| [`checks/news_stitching_vs_augmentation/make_stats.py`](checks/news_stitching_vs_augmentation/make_stats.py) | Turns saved News decodes into per-sentence n-gram counts, because News references are not shipped (run once in the private tree). |
| [`checks/ensemble_averaging_rule/check.py`](checks/ensemble_averaging_rule/check.py) | A44. Rescores the saved n-best lists under each power mean and recomputes the paired differences with their intervals. |
| [`checks/nonsigner_confidence_gate/check.py`](checks/nonsigner_confidence_gate/check.py) | A45. Recomputes the voiced shares, ratios, areas under the curve and grounding comparison on non-signing crops. |
| [`checks/nonsigner_confidence_gate/prep_data.py`](checks/nonsigner_confidence_gate/prep_data.py) | Shrinks the saved per-clip decodes and grounding scores to the fields the check needs (run once in the private tree). |
| [`checks/nonsigner_confidence_gate/prep_features.py`](checks/nonsigner_confidence_gate/prep_features.py) | Derives the per-clip wrist speed of the "hardly moves" stratum from pose arrays that are not shipped (run once in the private tree). |
| [`checks/speech_chain/check.py`](checks/speech_chain/check.py) | A46. Recomputes the latency by stage and the BLEU-4 of the voiced output by coverage from the saved per-clip records. |
| [`checks/rouge_convention/check.py`](checks/rouge_convention/check.py) | A47. Rebuilds the ROUGE-L tokenisation table from the saved hypotheses of three models. |
| [`checks/citation_vs_conversational_signs/check.py`](checks/citation_vs_conversational_signs/check.py) | A48. Citation vs conversational signs: clip lengths, frame rates, Sign_Spotting durations, GOAL durations. |
| [`checks/citation_vs_conversational_signs/prepare_data.py`](checks/citation_vs_conversational_signs/prepare_data.py) | Run once in the private tree (needs the raw datasets); writes the small csv files check.py reads. Not run by make. |
| [`checks/dataset_statistics/check.py`](checks/dataset_statistics/check.py) | A49. Dataset statistics of Auslan-Daily and MM-WLAuslan (dissertation Review, Methodology, Results, Appendix B). |
| [`checks/dataset_statistics/prepare_data.py`](checks/dataset_statistics/prepare_data.py) | Prep (run once in the private tree, not by make): reduce the dataset's annotation tables to counts-only files. No subtitle text is written. Usage: python prepare_data.py   (run from the repository root or anywhere). |
| [`checks/duration_and_frame_rates/check.py`](checks/duration_and_frame_rates/check.py) | A50. Real-clip length and frame-rate facts behind the temporal-fidelity argument (appendix table of clip lengths). Recomputed from per-clip frame counts of the shipped tensors, the alignment table and the sign-spotting table (see SOURCES.md). |
| [`checks/duration_and_frame_rates/prepare_data.py`](checks/duration_and_frame_rates/prepare_data.py) | Run once (not by make): reduce saved raw outputs to per-clip frame counts (numbers only, no text). Reads: full_run/clips_signer/*.npz (shipped per-clip pose tensors), the alignment and sign-spotting tables, the pose file of the isolated-sign ... |
| [`checks/ensemble_member_subsets/check.py`](checks/ensemble_member_subsets/check.py) | A51. Ensemble member-set ablation and member-set selection (dissertation Results, Discussion, Appendix B). |
| [`checks/ensemble_member_subsets/prepare_data.py`](checks/ensemble_member_subsets/prepare_data.py) | Byte-copy the saved outputs this check needs that are not already published (run once by hand, not by make). |
| [`checks/ensemble_cost_and_latency/check.py`](checks/ensemble_cost_and_latency/check.py) | A52. Ensemble / translator / pose-front-end cost and latency (Results 'latency', Discussion, appendix latency table, Method set-up). Parses the saved timing outputs: ensemble_latency_log.txt (tools/ensemble_latency.py, M3 Max CPU, 800 validation clips), ... |
| [`checks/ensemble_cost_and_latency/prepare_data.py`](checks/ensemble_cost_and_latency/prepare_data.py) | Run once (not by make): writes the two small data files of this check from local saved outputs.   pose_throughput_subset.json  <- results/pose_throughput_m3max_2026-09-15.json (tools/pose_throughput_bench.py)   test_clip_frames.json        <- ... |
| [`checks/ensemble_mechanism/check.py`](checks/ensemble_mechanism/check.py) | A53. Why ensembling works: S6 ensemble vs its members, mechanism, oracle, MBR, seed ensembles (dissertation Results, Appendix B). |
| [`checks/ensemble_mechanism/prepare_data.py`](checks/ensemble_mechanism/prepare_data.py) | Run once by hand (not by make): read the six S6 ensemble checkpoints and save a small text summary of what they say about themselves. |
| [`checks/estimator_first_listed_runs/check.py`](checks/estimator_first_listed_runs/check.py) | A54. Early AlphaPose (first-listed person) runs and the DWPose re-extraction: what can be recomputed locally. |
| [`checks/estimator_first_listed_runs/prepare_data.py`](checks/estimator_first_listed_runs/prepare_data.py) | Run once (not by make): makes the small data files of this check from saved outputs. See SOURCES.md. |
| [`checks/estimator_signer_rerun/check.py`](checks/estimator_signer_rerun/check.py) | A55. Estimator comparison on the signer's keypoints (job 196407) and the signer-keypoint probes. |
| [`checks/extraction_error_audit/check.py`](checks/extraction_error_audit/check.py) | A56. Extraction-error audit: layout of the released keypoint files, the signer entry, registration, extractor input size, bounding-box format. |
| [`checks/extraction_error_audit/prepare_data.py`](checks/extraction_error_audit/prepare_data.py) | One-off data preparation (run by hand, NOT by make). Reads the released Auslan-Daily pose files and the local extraction outputs and writes small text tables used by check.py. Needs ~/Downloads/SP_Annotation_V1 (the released ... |
| [`checks/content_word_failure_analysis/check.py`](checks/content_word_failure_analysis/check.py) | A57. Content-word failure analysis of the saved test decodes (dissertation 'Error Analysis', Discussion, appendix retrieval row). |
| [`checks/figure_settings/check.py`](checks/figure_settings/check.py) | A58. Figure settings and verdict sentences: effect-size figure (test set, 95% intervals), qualitative figure (model, beam), speed verdict. |
| [`checks/figure_settings/prepare_data.py`](checks/figure_settings/prepare_data.py) | Run once (not by make): programmatic subsets of saved outputs. No reference sentences are copied. |
| [`checks/hand_speed_dwpose_frontend/check.py`](checks/hand_speed_dwpose_frontend/check.py) | A59. DWPose front end: extraction and training-list counts, input vector size, example-frame hand confidence, hand speed on first-detection versus tracked keypoints, and hand confidence versus identification. |
| [`checks/hand_speed_dwpose_frontend/prepare_data.py`](checks/hand_speed_dwpose_frontend/prepare_data.py) | One-off: derive the data files of this package from local saved outputs (not run by make). Per-clip hand speed uses tools/hand_visibility_cv.py::hand_stats unchanged. |
| [`checks/sign_identification_probes/check.py`](checks/sign_identification_probes/check.py) | A60. Sign-identification probes (dissertation: Results, Discussion, Appendix 'isolated-sign references'). |
| [`checks/sign_identification_probes/prepare_data.py`](checks/sign_identification_probes/prepare_data.py) | Run once by hand (not by make): derives mmwl_gloss_counts.json from the MM-WLAuslan label files. Only gloss-name counts are kept (no clips, no text). |
| [`checks/input_blind_baselines/check.py`](checks/input_blind_baselines/check.py) | A61. Input-blind constant baseline: official test vs unseen targets (Results 'null model', Introduction, Discussion). |
| [`checks/input_side_predictors/check.py`](checks/input_side_predictors/check.py) | A62. Input-side predictors of translation quality (dissertation Results, 'Training-free measurements'). |
| [`checks/input_side_predictors/prepare_data.py`](checks/input_side_predictors/prepare_data.py) | One-off (not run by make): per-clip data behind tools/translation_predictors.py. |
| [`checks/learning_curve_with_real_data/check.py`](checks/learning_curve_with_real_data/check.py) | A63. Wave 8 real-data learning curve (job 21734) and its count-only mechanism check. |
| [`checks/learning_curve_with_real_data/prepare_data.py`](checks/learning_curve_with_real_data/prepare_data.py) | One-off preparation (run once by hand, not by make): copies the compact saved outputs of job 21734 and derives per-item numbers that need training text, so check.py never reads training text. Run from src: python ... |
| [`checks/news_subset_results/check.py`](checks/news_subset_results/check.py) | A64. Claims check: News wave 4 (nA vs nSA, five seeds), validation-LP decoding, signable-word recall, ensemble gate. Run: cd src && python3 ../results/checks/news_subset_results/check.py. |
| [`checks/news_subset_results/make_extra.py`](checks/news_subset_results/make_extra.py) | Prep (run once in the private tree, not by make): small numeric extras for the News check. data/news_clip_frames.csv  one column 'frames' = End_Frame - Begin_Frame of every row of the News alignment spreadsheet (numbers only, no text) ... |
| [`checks/news_subset_results/make_stats.py`](checks/news_subset_results/make_stats.py) | Prep (run once in the private tree, not by make): News decodes -> per-sentence BLEU sufficient statistics. News references are not restored by `make data`, so none are shipped; n-gram match/total counts and lengths carry the corpus BLEU. Reads ... |
| [`checks/paired_bootstrap_ladder/check.py`](checks/paired_bootstrap_ladder/check.py) | A65. Paired-bootstrap comparisons, development ladder and S6 against the published systems. |
| [`checks/paired_bootstrap_ladder/prepare_data.py`](checks/paired_bootstrap_ladder/prepare_data.py) | Run once by hand: subset results/semantic_regrade_10288/{s6,rgbfusion}.graded_hf.csv to idx + judge grade (no sentences) -> adequacy_grades.json. Source: saved per-clip output of job 10288. |
| [`checks/pose_augmentation_replication/check.py`](checks/pose_augmentation_replication/check.py) | A66. Pose augmentation replication (job 1215231): 3 paired seeds, aug vs same-seed control, beam-5 official test. Recomputes BLEU-4 from the saved hypotheses, validation/greedy from saved job outputs, seed-level t-interval, content precision (stop list = ... |
| [`checks/stitching_side_experiments/check.py`](checks/stitching_side_experiments/check.py) | A67. PoseStitch v2 side experiments: shuffled sign order (wave 10), wider-vocabulary corpus (wave 6), dropping function-word signs (wave 7), fingerspelled-letter signs (wave 2), wrong-sense signs (wave 12b corpus facts). |
| [`checks/stitching_side_experiments/prepare_data.py`](checks/stitching_side_experiments/prepare_data.py) | Run once in the private tree (not by make): subsets of saved job outputs -> data/. Per-run beam-5 hypotheses (+ references, stored once per split under "refs") from the run folders, byte copies of the small summary / log files, and gloss counts of ... |
| [`checks/pretrained_transfer/check.py`](checks/pretrained_transfer/check.py) | A68. Recompute the scores of the fine-tuned pre-trained sign-language model (How2Sign checkpoint) on Auslan-Daily from saved decodes. |
| [`checks/pretrained_transfer/prepare_data.py`](checks/pretrained_transfer/prepare_data.py) | Run once by hand (not by make): cut the hypotheses out of the saved job outputs. References are NOT copied (they come from make data). |
| [`checks/pretraining_length_rerun/check.py`](checks/pretraining_length_rerun/check.py) | A69. Pre-training length (wave 11 re-run) and News frame limit, selected on validation alone. |
| [`checks/pretraining_length_rerun/prepare_data.py`](checks/pretraining_length_rerun/prepare_data.py) | Subset of the saved wave-11 job outputs (cluster job 116619 pre-training arms, 116620 News arms). Keeps only scores and run settings (no hypotheses, no reference text). Run once by the author, not by make. |
| [`checks/pseudo_gloss_pretraining/check.py`](checks/pseudo_gloss_pretraining/check.py) | A70. Wave 23: CTC-only pseudo-gloss first stage (X) against a compute-matched full-loss first stage (Y) and no first stage (C). |
| [`checks/pseudo_gloss_pretraining/prepare_data.py`](checks/pseudo_gloss_pretraining/prepare_data.py) | Run once by hand: programmatic subset of the saved wave-23 job output (job set 179456). Keeps per run and split the beam-5 hypotheses and references (key "ref") and copies progress.json and summary_w23.json byte for byte. |
| [`checks/published_systems_200_epochs/check.py`](checks/published_systems_200_epochs/check.py) | A71. Published systems' own code (200 epochs) and this project's encoder with the published recipe. |
| [`checks/published_systems_200_epochs/prepare_data.py`](checks/published_systems_200_epochs/prepare_data.py) | Builds data/old/ from the saved outputs of the older single runs. Run once by hand, not by make. |
| [`checks/recipe_and_settings/check.py`](checks/recipe_and_settings/check.py) | A72. Training recipe, model size and decoding defaults of the reported systems (Methodology, Table 'Training configuration', Appendix B table convention). |
| [`checks/recipe_and_settings/prepare_data.py`](checks/recipe_and_settings/prepare_data.py) | One-off (not run by make): subset the saved training checkpoints into cfg_members.json. Reads results/ckpts/*.pt (torch needed; the checkpoints are not published) and keeps only the saved training configuration, the tensor shapes that show the ... |
| [`checks/run_to_run_noise_bands/check.py`](checks/run_to_run_noise_bands/check.py) | A73. Run-to-run noise bands: recomputed from saved per-run beam-5 test BLEU-4 scores (800 official test clips). Gates: pooled sd and null-band sd must equal the values the original read-out tools recorded. |
| [`checks/run_to_run_noise_bands/prepare_data.py`](checks/run_to_run_noise_bands/prepare_data.py) | Run once (not by make): programmatic subset of saved outputs -> det_pairs.json. Beam-5 test BLEU-4 of the baseline-with-augmentation control of wave 6 (summary_s*.json, key A) and of wave 16 (runs/A_s*/beam5_test_best.json). |
| [`checks/seen_unseen_decomposition/check.py`](checks/seen_unseen_decomposition/check.py) | A74. Seen/unseen decomposition, verbatim overlap, split definition and per-signer breakdown (official Communication test split). |
| [`checks/selective_prediction/check.py`](checks/selective_prediction/check.py) | A75. Selective prediction: single model (DWPose+CTC, beam 5) and the S6 ensemble with validation-fixed thresholds. |
| [`checks/selective_prediction/prepare_data.py`](checks/selective_prediction/prepare_data.py) | One-off subset builder (not run by make): joins each saved beam-5 decode with its saved self-CE / self-reliance vectors (tools/self_reliance.py output), keeping only id, hypothesis, reference and the two scores. |
| [`checks/semantic_adequacy_grades/check.py`](checks/semantic_adequacy_grades/check.py) | A76. Semantic adequacy grading (Qwen2.5-3B-Instruct judge, 0-4): means of four systems and the paired comparison. |
| [`checks/semantic_adequacy_grades/prepare_data.py`](checks/semantic_adequacy_grades/prepare_data.py) | One-off: reduce the saved per-clip grade CSVs to grades.jsonl (system, idx, adequacy); no text is kept. |
| [`checks/signer_disjoint_split/check.py`](checks/signer_disjoint_split/check.py) | A77. Signer-disjoint split (wave 12g): rebuilds the split from the dataset spreadsheet, re-scores the saved beam-5 decodes of the three arms (three seeds) and recomputes every number the dissertation quotes for it. Run from src/:  python3 ... |
| [`checks/signer_disjoint_split/prepare_data.py`](checks/signer_disjoint_split/prepare_data.py) | Make data/<arm>_s<seed>/beam5_{val,test}.json from the saved beam-5 decodes of the signer-disjoint experiment (wave 12g). Run once in the private tree (it needs the saved decodes); make does not run it. Each source file ... |
| [`checks/speech_round_trip/check.py`](checks/speech_round_trip/check.py) | A78. Speech round trip (Whisper small.en): round-trip WER of voiced output vs human and scrambled references. |
| [`checks/speech_round_trip/prepare_data.py`](checks/speech_round_trip/prepare_data.py) | One-off: copy the minimal saved speech round-trip data into this folder (private tree only). Run from the repository root of the private tree:  python3 results/checks/speech_round_trip/prepare_data.py Transcripts of spoken REFERENCE text (arms ... |
| [`checks/tracker_v2_heavy_switch/check.py`](checks/tracker_v2_heavy_switch/check.py) | A79. Person tracking: switch prevalence, blind audit, wave 3 (five seeds), wave 9 tracker v2, wave 12i heavy-switch bin, hand speed, pose jumps. |
| [`checks/tracker_v2_heavy_switch/prepare_data.py`](checks/tracker_v2_heavy_switch/prepare_data.py) | One-off: build the compact saved inputs of check.py from the private working tree (not run by `make reproduce`). |
| [`checks/video_disjoint_split/check.py`](checks/video_disjoint_split/check.py) | A80. Video-disjoint split and VD-CONFIRM (dissertation Methodology, Results 'Video-disjoint', Discussion, Appendix). |
| [`checks/video_disjoint_split/prepare_data.py`](checks/video_disjoint_split/prepare_data.py) | Run once by hand (not by make): copy the saved outputs this check reads into data/. |
| [`checks/masked_pose_pretraining/check.py`](checks/masked_pose_pretraining/check.py) | A81. Wave 28 (masked keypoint pre-training): what the dissertation rows state, checked against the saved job output. |
| [`checks/masked_pose_pretraining/prepare_data.py`](checks/masked_pose_pretraining/prepare_data.py) | Byte copies of two saved job outputs (wave 28), with banned words replaced. Run once by hand; not run by make. |
| [`checks/stitch_trim_and_realisations/check.py`](checks/stitch_trim_and_realisations/check.py) | A82. Wave 4 (job 7949): stitched data without the boundary trim (T) and two realisations of half as many sentences (Q) against the adopted corpus (A). Recomputed from the five per-seed summary files. |
| [`checks/wer_bleu_null_models/check.py`](checks/wer_bleu_null_models/check.py) | A83. WER null floor, brevity trade, error decomposition and worked BLEU/WER example (Results 'WER', Methodology, appendix). |

## `../figures/`

| File | Description |
|---|---|
| [`figstyle.py`](../figures/figstyle.py) | Defines the shared figure style, sizing and colour assignments imported by the concept and thesis figure scripts. |
| [`make_concept_figures.py`](../figures/make_concept_figures.py) | Draws the conceptual figures for the Introduction, Literature Review and Methodology chapters, building data panels from project files. |
| [`make_thesis_figures.py`](../figures/make_thesis_figures.py) | Draws the Results-chapter figures from the experiment ledger, the results logbook and the failure-analysis record. |
| [`make_uncertainty_figure.py`](../figures/make_uncertainty_figure.py) | Plots every recipe with more than one run, showing individual runs grouped by batch, from the ledger and saved decode files. |
| [`make_extra_figures.py`](../figures/make_extra_figures.py) | Draws seven further Results-chapter figures (pretraining re-run, learning curve, latency by stage, effect sizes with intervals, ensemble cost, non-signer control, score by signer) from run files and from values asserted against the Results text. |
| [`make_new_figures.py`](../figures/make_new_figures.py) | Draws the clip-length and reference-length figure from the released alignment tables, the published-against-reimplemented comparison, and the pose-estimator comparison on video frames. |
| [`make_signerfix_figures.py`](../figures/make_signerfix_figures.py) | Draws the corrected estimator comparison, the effect-size figure and the results ladder after the keypoint-extraction error; it supersedes the estimator panel of `make_thesis_figures.py` and the effect-size panel of `make_extra_figures.py`. Reads `results/estimator_signer_keypoints_rerun/readout.json`. |
| [`make_failure_figure.py`](../figures/make_failure_figure.py) | Draws a failure taxonomy classifying all official test clips into mutually exclusive modes from saved decodes and the training manifest. |
| [`make_internals_figure.py`](../figures/make_internals_figure.py) | Plots per-frame auxiliary CTC label probabilities from a trained checkpoint on one test clip, after reproducing the recorded greedy hypothesis. |
| [`extract_paper_figures.py`](../figures/extract_paper_figures.py) | Extracts captioned figures from reference-paper PDFs at print resolution, recording paper, page, figure number and caption for attributed reproduction. |