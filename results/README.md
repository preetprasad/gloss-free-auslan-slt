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