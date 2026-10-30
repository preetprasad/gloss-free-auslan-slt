# `src/`

This folder was called `phase2_baseline/` on the UWA Kaya cluster; the job scripts in `../jobs/slurm/` still use that name in their paths.

This directory holds the translation system. `config.py` defines every setting; `slt/` is the library (data loading, model, losses, metrics, decoding); `train_ctc.py` trains every model reported in the dissertation and `pretrain_vlp.py` and `ensemble_decode.py` provide pose-text pretraining and ensemble decoding. The other trainers at this level implement alternatives that were tested and not adopted. Scripts expect to be run from this directory.

## Core files

| File | Description |
| --- | --- |
| [`config.py`](config.py) | Defines the pipeline configuration as plain dataclass fields, including the HALPE keypoint layout, without importing torch. |
| [`train_ctc.py`](train_ctc.py) | Trains the translator with decoder cross-entropy plus a weighted encoder CTC head supervised by rule-based content-word pseudo-glosses. |
| [`pretrain_vlp.py`](pretrain_vlp.py) | Pretrains the pose encoder to align with subtitle text through a CLIP-style contrastive loss and saves a translator-compatible checkpoint. |
| [`ensemble_decode.py`](ensemble_decode.py) | Averages per-step output log-probabilities of checkpoints sharing one vocabulary, including pose and RGB-fusion members, then decodes with beam or greedy search. |
| [`gen_pseudo_gloss.py`](gen_pseudo_gloss.py) | Prompts an instruction-tuned language model with examples to turn training subtitles into ordered pseudo-gloss sequences, written as a TSV for `train_ctc.py`. |
| [`weight_average.py`](weight_average.py) | Averages the parameters of several checkpoints with identical architecture and vocabulary into one model for single-pass inference. |
| [`metric_suite.py`](metric_suite.py) | Scores a hypothesis dump with BLEU, chrF and ROUGE-L, and splits test clips by whether each reference appears verbatim among training targets. |
| [`bench_latency.py`](bench_latency.py) | Measures single-clip encode and greedy-decode latency and throughput of the pose-to-text translator, excluding pose extraction and speech synthesis. |
| [`smoke_test.py`](smoke_test.py) | Runs the pipeline end to end on synthetic CPU data: normalisation, batching, brief training, greedy decoding, metrics and decision-gate logic. |
| [`make_synthetic.py`](make_synthetic.py) | Generates small synthetic HALPE-layout pose clips and manifests with templated target sentences for the smoke test, using no real data. |
| [`test_reference_arch.py`](test_reference_arch.py) | Runs fast architecture and parity checks for the SignJoey SL-Transformer and GASLT model modes. |
| [`test_pose_mae.py`](test_pose_mae.py) | Runs structural, masking and device tests for the multi-stream pose masked autoencoder. |

## Library modules (`slt/`)

| File | Description |
| --- | --- |
| [`__init__.py`](slt/__init__.py) | Marks `slt/` as a Python package; its docstring names the gloss-free Auslan-Daily translation baseline. |
| [`cli.py`](slt/cli.py) | Provides shared argparse helpers, including a strict boolean parser that rejects any value other than true or false. |
| [`data.py`](slt/data.py) | Loads per-clip pose tensors, masks all-zero frames, normalises coordinates by bounding box, builds the vocabulary and batches with padding. |
| [`model.py`](slt/model.py) | Defines the original pose-sequence Transformer encoder-decoder with optional feature adapter, positional encoding and a vocabulary generator. |
| [`reference_model.py`](slt/reference_model.py) | Implements SignJoey-compatible SL-Transformer and GASLT architectures following pinned public commits, with an explicit pose adapter where input widths differ. |
| [`engine.py`](slt/engine.py) | Provides the training and evaluation loops. |
| [`metrics.py`](slt/metrics.py) | Computes BLEU with sacreBLEU and word error rate with jiwer. |
| [`preproc_guard.py`](slt/preproc_guard.py) | Checks that decode-time preprocessing settings match those stored in every member checkpoint before the ensemble tools decode. |
| [`pseudo_labels.py`](slt/pseudo_labels.py) | Derives conservative English content-token pseudo-labels from translation text in one place; these are not Auslan gloss annotations. |
| [`vlp.py`](slt/vlp.py) | Implements CLIP-style contrastive alignment of the pose encoder with text, treating each clip and its own subtitle as a positive pair. |
| [`rgb_fusion.py`](slt/rgb_fusion.py) | Defines a two-stream translator concatenating DWPose and precomputed DINO crop-feature memories in time, with pseudo-gloss CTC on the pose stream. |
| [`rgb_vlp.py`](slt/rgb_vlp.py) | Contrastively aligns precomputed RGB crop features with text, using module names that let the RGB-fusion model load the weights directly. |
| [`temporal_pyramid.py`](slt/temporal_pyramid.py) | Defines a feature-level temporal-motion adapter of depthwise temporal convolutions at several receptive fields over cached frozen RGB vectors. |
| [`temporal_rgb_fusion.py`](slt/temporal_rgb_fusion.py) | Defines RGB-fusion model variants that add an explicit feature-level temporal adapter. |
| [`gated_temporal_rgb_fusion.py`](slt/gated_temporal_rgb_fusion.py) | Lets pose and RGB memories exchange a gated residual at each normalised time point before concatenation, retaining the existing encoders. |
| [`semantic.py`](slt/semantic.py) | Provides frozen text-semantic targets for auxiliary supervision of the translator. |
| [`pretrained.py`](slt/pretrained.py) | Loads BPEmb word2vec binary target embeddings with minimal dependencies. |
| [`pose_mae.py`](slt/pose_mae.py) | Implements multi-stream masked pose pretraining over body and hand streams with per-stream masks, scoring only observed masked keypoints. |
| [`llm_decoder.py`](slt/llm_decoder.py) | Feeds pose-encoder outputs through a bridge into a pretrained multilingual decoder with LoRA, bypassing that model's text encoder. |
| [`sign_vq.py`](slt/sign_vq.py) | Implements a vector-quantised tokeniser that encodes visual features, quantises them against a codebook and reconstructs the input. |
| [`llm_vq_decoder.py`](slt/llm_vq_decoder.py) | Treats vector-quantised sign codes as new tokens for a pretrained encoder-decoder language model with LoRA on the decoder. |
| [`mbart_decoder.py`](slt/mbart_decoder.py) | Connects the pose encoder through a learned adapter to a pretrained mBART decoder with its subword tokeniser and LoRA, bypassing the mBART encoder. |
| [`speech.py`](slt/speech.py) | Converts translated English into timed speech for the utterance-final demonstrator, which voices output only when the model is confident. |

## Trainers for alternatives that were not adopted

| File | Description |
| --- | --- |
| [`train.py`](train.py) | Trains the original gloss-free baseline translator from TSV manifests and evaluates it against the BLEU decision gate. |
| [`train_ddp.py`](train_ddp.py) | Wraps the single-GPU trainer in DistributedDataParallel with rank-zero logging and checkpointing; its header states it is untested on multiple GPUs. |
| [`pretrain_islr.py`](pretrain_islr.py) | Pretrains the shared pose encoder on MM-WLAuslan word-level sign recognition and saves an encoder checkpoint for translator initialisation. |
| [`pretrain_pose_mae.py`](pretrain_pose_mae.py) | Pretrains an Auslan-specific multi-stream masked pose encoder. |
| [`pretrain_rgb_vlp.py`](pretrain_rgb_vlp.py) | Provides the command-line entry for contrastive RGB-text pretraining implemented in `slt/rgb_vlp.py`. |
| [`train_llm.py`](train_llm.py) | Trains a pose encoder feeding a pretrained multilingual decoder adapted with LoRA for gloss-free translation. |
| [`train_vq.py`](train_vq.py) | Trains a vector-quantised tokeniser over pose or RGB features as the first stage of a two-stage route. |
| [`train_vq_llm.py`](train_vq_llm.py) | Trains a pretrained language-model decoder with LoRA to translate vector-quantised sign tokens into English, as the second stage. |
| [`train_mbart.py`](train_mbart.py) | Trains a pose encoder, adapter and pretrained mBART decoder with subword targets, warm-starting the pose encoder from an earlier checkpoint. |
| [`train_grpo.py`](train_grpo.py) | Fine-tunes a trained translator with group-relative policy optimisation, using sampled hypotheses, a BLEU plus ROUGE-L reward and a KL penalty. |
| [`train_presence.py`](train_presence.py) | Trains the RGB and DWPose translator with an order-invariant, temporally max-pooled multi-label head on pseudo-content tokens instead of CTC. |
| [`train_temporal_pyramid.py`](train_temporal_pyramid.py) | Runs a paired validation-only ablation of a feature-level temporal-motion adapter on DWPose plus frozen RGB, using translation cross-entropy only. |
| [`train_temporal_ctc.py`](train_temporal_ctc.py) | Trains temporal RGB fusion with the single pseudo-CTC head, optionally pooling decoder memory while CTC stays on uncompressed pose memory. |
| [`build_sentence_embeddings.py`](build_sentence_embeddings.py) | Precomputes frozen sentence-embedding targets from a manifest's text for the validation-only semantic-supervision experiment. |
| [`train_temporal_semantic_ctc.py`](train_temporal_semantic_ctc.py) | Adds a learned projection from fused encoder memory to a frozen reference-sentence embedding to the temporal RGB pseudo-CTC recipe, validation-only. |
| [`train_face_temporal_ctc.py`](train_face_temporal_ctc.py) | Appends a frozen DINO face-crop vector to each frame's RGB vector, otherwise keeping the temporal RGB pseudo-CTC recipe unchanged. |
| [`train_gated_temporal_ctc.py`](train_gated_temporal_ctc.py) | Replaces concatenation of pose and RGB memories with residual gated fusion, leaving the pose-memory CTC head unchanged, validation-only. |

## Tools (`tools/`)

### Data preparation and manifests

| File | Description |
| --- | --- |
| [`build_manifest.py`](tools/build_manifest.py) | Builds train, validation and test TSV manifests from Auslan-Daily translation spreadsheets, auto-detecting columns and resolving clip identifiers to pose files. |
| [`train_target_hashes.py`](tools/train_target_hashes.py) | Writes SHA-1 digests of the training targets, so that seen-target and unseen-target scores can be recomputed without the training subtitles. |
| [`restore_references.py`](tools/restore_references.py) | Puts the Auslan-Daily reference sentences back into the saved outputs from the dataset's spreadsheet, verifies each file against `results/SHA256SUMS`, and writes the text manifests; used by `make data`. |
| [`extract_clips.py`](tools/extract_clips.py) | Splits Auslan-Daily per-video pose files into one compressed file per clip holding keypoint and bounding-box arrays. It takes the entry annotated as the signer (`Annotation['Signer']`) and stops if a clip has none; an earlier version took the first-listed person, a non-signer in most Communication clips (see `REPRODUCING.md`, A32). |
| [`extract_from_zip.py`](tools/extract_from_zip.py) | Reads per-video pose files from an Auslan-Daily zip archive in memory and writes per-clip arrays without unpacking the archive. |
| [`build_combined_manifests.py`](tools/build_combined_manifests.py) | Builds News manifests and joint Communication-plus-News training manifests, keeping Communication validation and test for selection and evaluation. |
| [`build_vocab_file.py`](tools/build_vocab_file.py) | Builds one fixed vocabulary over several manifests with the trainer's vocabulary builder, so pretraining and fine-tuning share token indices. |
| [`halpe_to_cwb133.py`](tools/halpe_to_cwb133.py) | Re-indexes AlphaPose HALPE keypoints onto COCO-WholeBody slots so tools written for that layout run unchanged. |
| [`annotation_subsets.py`](tools/annotation_subsets.py) | Reports leak-free weak-label subsets by intersecting sign-spotting and fingerspelling annotations with the Communication training split. |
| [`real_subsets.py`](tools/real_subsets.py) | Draws deterministic subsets of the Communication training manifest, either within each video or by whole videos, for the learning-curve experiment. |
| [`audit_bbox_format.py`](tools/audit_bbox_format.py) | Infers whether manifest bounding boxes are stored as xywh or xyxy from keypoint containment, without modifying data. |
| [`audit_dwpose_inputs.py`](tools/audit_dwpose_inputs.py) | Audits per-region confidence statistics of extracted DWPose clips without altering manifests or pose files. |
| [`audit_pseudo_labels.py`](tools/audit_pseudo_labels.py) | Writes a deterministic JSON audit of pseudo-content label statistics for one manifest. |
| [`gen_paraphrases.py`](tools/gen_paraphrases.py) | Prompts an instruction-tuned language model for paraphrases of training targets only, written as a TSV for multi-reference augmentation in `train_ctc.py`. |
| [`extract_face_features.py`](tools/extract_face_features.py) | Extracts DINO features from face crops located by DWPose face landmarks and records crop-validity statistics. |
| [`extract_glofe_features.py`](tools/extract_glofe_features.py) | Extracts features from the public WLASL-pretrained GloFE pose backbone after mapping Auslan HALPE keypoints into its joint ordering. |
| [`extract_pose_mae_features.py`](tools/extract_pose_mae_features.py) | Exports temporal features from a pretrained Auslan pose masked-autoencoder checkpoint. |

### Decoding and evaluation

| File | Description |
| --- | --- |
| [`eval_beam.py`](tools/eval_beam.py) | Decodes a checkpoint greedily and with beam search, reporting BLEU under two tokenisation protocols and unique-output counts. |
| [`tune_beam.py`](tools/tune_beam.py) | Tunes beam width and length penalty on validation by decoding once per width and rescoring the N-best for each penalty. |
| [`rescore.py`](tools/rescore.py) | Re-scores a checkpoint's greedy outputs under several BLEU protocols and counts unique outputs, separating measurement from modelling differences. |
| [`tta_decode.py`](tools/tta_decode.py) | Decodes over temporally subsampled views of each clip, combining them through the ensemble beam search, with selection on validation. |
| [`arm_readouts.py`](tools/arm_readouts.py) | Computes protocol readouts for one decoded arm, including BLEU, WER, length ratio, distinct outputs and content-word precision and recall. |
| [`diagnose_run.py`](tools/diagnose_run.py) | Decodes the test manifest with a saved checkpoint and reports per-corpus BLEU, degeneracy statistics and example pairs to standard output. |
| [`plot_metrics.py`](tools/plot_metrics.py) | Renders training-curve figures and a LaTeX summary table from a run's per-epoch metrics file. |
| [`reproduce_headline.py`](tools/reproduce_headline.py) | Re-scores every committed decode dump with the project's metric code and exits non-zero if any score differs from the record. |
| [`run_claims.py`](tools/run_claims.py) | Runs the checks written in `REPRODUCING.md` and compares the numbers in each new output with the recorded output; used by `make reproduce`. |

### Ensembles

| File | Description |
| --- | --- |
| [`ensemble_ablation.py`](tools/ensemble_ablation.py) | Runs incremental and leave-one-out ensemble ablations on validation in one decode pass, reusing the ensemble beam search. |
| [`ensemble_mbr_wer.py`](tools/ensemble_mbr_wer.py) | Decodes an ensemble N-best once, then rescores it across length penalties and by minimum-Bayes-risk selection to target word error rate. |
| [`ensemble_mechanism.py`](tools/ensemble_mechanism.py) | Tests whether ensembling gains arise from novel token-level combinations or member selection, comparing per-clip sentence chrF of ensemble and members. |
| [`ensemble_subsets.py`](tools/ensemble_subsets.py) | Decodes every non-empty subset of the ensemble's members with the reported decoding, then reports members alone, leave-one-out losses, by-kind subsets and Shapley values with sentence-bootstrap intervals. |
| [`ensemble_selective.py`](tools/ensemble_selective.py) | Applies selective prediction to ensemble N-best dumps, fixing confidence thresholds on validation and carrying them unchanged to test. |
| [`ensemble_latency.py`](tools/ensemble_latency.py) | Times per-clip ensemble translation at batch size one for prefixes of the member list, scoring quality from the same decodes. |
| [`cross_system_confidence.py`](tools/cross_system_confidence.py) | Tests whether model cross-entropy can select between hypotheses of two independently trained systems, scoring each candidate under both models. |
| [`seedens_analysis.py`](tools/seedens_analysis.py) | Computes the pre-registered readout comparing seed ensembles with their best validation member, choosing each system's length penalty on validation. |

### Statistics and decision rules

| File | Description |
| --- | --- |
| [`paired_bootstrap.py`](tools/paired_bootstrap.py) | Computes paired bootstrap intervals and seen/unseen BLEU from decode dumps, first asserting that each dump re-scores to its recorded value. |
| [`adequacy_bootstrap.py`](tools/adequacy_bootstrap.py) | Computes a paired sentence bootstrap of the mean difference in language-model-judge adequacy grades between two systems. |
| [`judge_spotcheck.py`](tools/judge_spotcheck.py) | Creates a blind human grading sheet sampled across judge grades and scores the completed sheet's agreement with the automatic adequacy judge. |
| [`seed_decision.py`](tools/seed_decision.py) | Computes per-seed paired deltas, sign counts and two-sided seed-level t-intervals for one arm against its in-batch control. |
| [`holm_decision.py`](tools/holm_decision.py) | Implements the pre-registered protocol: futility stopping, one-sided paired t-tests, Holm-Bonferroni correction, adoption criteria and max-T simultaneous intervals. |
| [`selective_ci.py`](tools/selective_ci.py) | Adds paired bootstrap intervals, a matched-coverage random-abstention contrast and retained-set composition to ensemble selective prediction. |
| [`paired_probe_mcnemar.py`](tools/paired_probe_mcnemar.py) | Runs an exact paired McNemar test between two sign-identification probe runs for each feature variant. |

### Synthetic (stitched) corpus

| File | Description |
| --- | --- |
| [`build_posestitch_corpus.py`](tools/build_posestitch_corpus.py) | First version: concatenates MM-WLAuslan isolated-sign pose sequences in sentence order to synthesise continuous clips paired with real training sentences. |
| [`stitch_lexicon.py`](tools/stitch_lexicon.py) | Provides the shared MM-WLAuslan sign lexicon with phrase-aware, longest-first matching of multi-word and parenthesised glosses for both stitching stages. |
| [`gen_stitch_sentences.py`](tools/gen_stitch_sentences.py) | Prompts a language model for novel conversational sentences restricted to signable words already in the training vocabulary, reading only the training manifest. |
| [`constrained_gen.py`](tools/constrained_gen.py) | Constrains language-model generation with a trie of signable phrases so that out-of-vocabulary and unsignable content words cannot be emitted. |
| [`merge_sentence_pool.py`](tools/merge_sentence_pool.py) | Merges generated sentence shards, removing duplicates and any sentence matching a real reference under a normalised key, then shuffles. |
| [`build_posestitch_corpus_v2.py`](tools/build_posestitch_corpus_v2.py) | Second version: stitches signs for generated sentences with phrase-aware matching, sampled transition budgets, optional boundary trimming and a coverage threshold. |
| [`stitch_corpus_audit.py`](tools/stitch_corpus_audit.py) | Measures novelty and variety of synthetic sentences and how often the same isolated-sign recordings are reused as the corpus grows. |
| [`audit_posestitch_motion.py`](tools/audit_posestitch_motion.py) | Compares motion statistics and join discontinuities of synthetic clips with real training clips, using a shallow held-out classifier as a diagnostic. |
| [`w10_ctc_map.py`](tools/w10_ctc_map.py) | Writes CTC targets whose content words follow the shuffled sign order of a synthetic corpus, with self-checks against the trainer's own targets. |

### Diagnostic probes and analyses

| File | Description |
| --- | --- |
| [`content_failure_analysis.py`](tools/content_failure_analysis.py) | Computes content-word failure statistics from a hypothesis dump, including zero-correct and all-correct rates and ordered and unordered recall. |
| [`continuous_id_probe.py`](tools/continuous_id_probe.py) | Tests whether DWPose features identify signs within continuous signing, using sign-spotting spans, clip-disjoint folds, shuffled-label and span-length controls. |
| [`spotting_probe.py`](tools/spotting_probe.py) | Matches sliding-window encoder embeddings of continuous clips to per-gloss isolated-sign prototypes and reports spotting recall and precision. |
| [`spot_order_probe.py`](tools/spot_order_probe.py) | Checks whether pairs of spotted signs within a clip occur in the same order as their English words in the subtitle. |
| [`fingerspelling_detect_probe.py`](tools/fingerspelling_detect_probe.py) | Compares fingerspelling-detection difficulty on Communication and News clips, using length-matched negatives and clip-disjoint folds. |
| [`hand_visibility_cv.py`](tools/hand_visibility_cv.py) | Uses classical computer vision and keypoint measures to separate image conditions from articulatory reduction as causes of lost hand information, without training. |
| [`hand_crop_sharpness.py`](tools/hand_crop_sharpness.py) | Measures hand-crop sharpness and tests whether it falls with hand speed and predicts sign identification within continuous data. |
| [`mmpose_crossarch_probe.py`](tools/mmpose_crossarch_probe.py) | Compares an MMPose HRNet whole-body estimate with extracted DWPose keypoints on one raw frame, using the full frame as the person box. |
| [`nmm_question_probe.py`](tools/nmm_question_probe.py) | Exploratory probe, on training clips only, of whether DWPose face keypoints carry question marking, with question type read from the English reference. |
| [`pose_reliance_probe.py`](tools/pose_reliance_probe.py) | Compares teacher-forced cross-entropy under true and shuffled pose to measure how much joint and control models use their visual input. |
| [`per_clip_reliance.py`](tools/per_clip_reliance.py) | Measures each clip's reliance on its own pose and tests whether it predicts per-clip translation quality, with shuffled and positive controls. |
| [`self_reliance.py`](tools/self_reliance.py) | Computes a partial correlation for reliance and an inference-time self-reliance score that substitutes the model's own beam output for the reference. |
| [`selective_prediction.py`](tools/selective_prediction.py) | Computes risk-coverage curves of retained-set BLEU for confidence rankings against random abstention, an oracle and output length. |
| [`translation_predictors.py`](tools/translation_predictors.py) | Tests whether input-side clip properties predict translation quality, with a shuffled-outcome null band and a video-grouped cross-validated model. |
| [`sdr_probe.py`](tools/sdr_probe.py) | Measures representation density of encoder frame features as a gate on revisiting the SignCL contrastive objective, using forward passes only. |

### Speech and latency

| File | Description |
| --- | --- |
| [`speak_clip.py`](tools/speak_clip.py) | Translates one clip and speaks the English output, using model confidence to choose between voicing and asking the signer to repeat. |
| [`speech_pipeline_bench.py`](tools/speech_pipeline_bench.py) | Times translation and speech synthesis per clip and tabulates voiced coverage against BLEU of the voiced subset across confidence thresholds. |
| [`speech_roundtrip.py`](tools/speech_roundtrip.py) | Runs a pre-registered synthesis-then-recognition round trip, in staged commands, to test whether spoken output conveys the translator's text. |
| [`pose_throughput_bench.py`](tools/pose_throughput_bench.py) | Measures laptop throughput of the DWPose front end and single-clip translator decoding across execution providers, excluding video decoding. |
| [`nonsigner_gate.py`](tools/nonsigner_gate.py) | Decodes non-signer clips with the speech chain's translator to test the confidence gate on hard negatives under a pre-registration. |
| [`nonsigner_audit_sheets.py`](tools/nonsigner_audit_sheets.py) | Draws saved skeletons onto frames of non-signer crops and assembles labelled sheets for visual audit of the gate. |

### Live-demonstrator export

| File | Description |
| --- | --- |
| [`export_live_onnx.py`](tools/export_live_onnx.py) | Exports the live demonstrator's translator encoders and decoders to ONNX with a manifest, then checks graph outputs against PyTorch. |
| [`onnx_reference_decode.py`](tools/onnx_reference_decode.py) | Provides a NumPy beam-search reference decoder over ONNX Runtime sessions mirroring the browser engine, compared with recorded PyTorch dumps. |

### Checks for individual experiments

| File | Description |
| --- | --- |
| [`w4_prep_check.py`](tools/w4_prep_check.py) | Verifies that the stitched corpora for one comparison match their pre-registered construction before writing joint manifests, exiting non-zero on any violation. |
| [`w5_prep_check.py`](tools/w5_prep_check.py) | Concatenates synthetic build shards, checks paths, audits, reference overlap and sign durations, then writes nested prefix manifests. |
| [`shuffle_order_check.py`](tools/shuffle_order_check.py) | Checks that the second stitcher's sign-order shuffling and frame-matching options change only what the pre-registration specifies. |
| [`w11_readout.py`](tools/w11_readout.py) | Reads out validation-only arm selection, choosing the cheapest arm within a tie margin of the best mean validation BLEU, with paired differences. |
| [`learning_curve.py`](tools/learning_curve.py) | Computes the pre-registered real-data learning-curve readout, comparing training-subset sizes and video-held-out sampling across seeds. |
| [`learning_curve_mechanism.py`](tools/learning_curve_mechanism.py) | Analyses the learning-curve mechanism through per-word content recall, after checking that recomputed recall and BLEU match recorded readouts. |
| [`news_decode_analysis.py`](tools/news_decode_analysis.py) | Reads News test decoding at each model's validation-selected length penalty and compares it with the recorded default decode. |
| [`news_signable_recall.py`](tools/news_signable_recall.py) | Computes in-vocabulary content recall on the News test set, split by whether each content type is signable under the adopted lexicon. |
| [`w12a_news_decomp_readout.py`](tools/w12a_news_decomp_readout.py) | Reads out the split of the adopted News bundle into stitched clips and pose augmentation (arms nA, nS, nSA) with the rule fixed in advance. |
| [`w12a_power.py`](tools/w12a_power.py) | Simulates how often the News-decomposition rule returns each reading at three seeds, using invented deltas with the spread of the recorded News runs. |
| [`w12b_prep_check.py`](tools/w12b_prep_check.py) | Corpus gate for the wrong-sense experiment: the rebuilt adopted corpus must match the stored one array for array before the training array starts. |
| [`w12b_readout.py`](tools/w12b_readout.py) | Reads out the wrong-sense experiment (14 wrongly rendered signs removed, against the adopted corpus) at three seeds. |
| [`w12d_powermean.py`](tools/w12d_powermean.py) | Decodes validation with power means of the members' probabilities (p = -1, 0.5, 1, 2) against the geometric mean used by the reported ensemble; no test decode. |
| [`w12e_dryrun.py`](tools/w12e_dryrun.py) | Runs the whole local half of the non-signer ensemble pipeline on a stand-in whose expected reading is known. |
| [`w12e_nonsigner_s6.py`](tools/w12e_nonsigner_s6.py) | Applies the confidence gate of the reported six-member ensemble to crops of people who are not signing and reads out the voiced share against signers. |
| [`w12f_extract3d.py`](tools/w12f_extract3d.py) | Extracts whole-body pose with an estimated depth for a list of signer clips (RTMW3D-x through rtmlib). |
| [`w12f_locate.py`](tools/w12f_locate.py) | Finds, for each isolated-sign clip of Auslan-Daily, the conversational clip and frame range it was cut from. |
| [`w12f_overlap.py`](tools/w12f_overlap.py) | Checks from labels only whether signs in the overlap of two clips could give a reference clip that is the query's own signing. |
| [`w12f_pixels.py`](tools/w12f_pixels.py) | Checks that the isolated-sign clips are the same pixels as their located frame range. |
| [`w12f_probe.py`](tools/w12f_probe.py) | Runs the two pose measurements of sign identity (isolated clips as reference set; estimated depth) with nearest-centroid probes and clip-disjoint folds. |
| [`w12f_synth_test.py`](tools/w12f_synth_test.py) | End-to-end test of the pose-measurement probe on an invented data set. |
| [`w12g_readout.py`](tools/w12g_readout.py) | Reads out BLEU-4 on signers absent from training and on the same clips with the signer in training. |
| [`w12g_signer_split.py`](tools/w12g_signer_split.py) | Builds the signer-disjoint split of Communication and its two-fold exposure control. |
| [`w12h_readout.py`](tools/w12h_readout.py) | Reads out extra pose-text pre-training before the main training, with new seeds, against the in-job control. |
| [`w12i_heavy_did.py`](tools/w12i_heavy_did.py) | Computes the heavy-switch-bin difference-in-differences of tracker-v2 pose against stored pose, on five new seeds. |
| [`w12i_power.py`](tools/w12i_power.py) | Planning numbers for the heavy-switch rule: how often it fires for a given true effect and spread. |
| [`wG_signerguided.py`](tools/wG_signerguided.py) | Chooses per frame the DWPose detection that matches the dataset's hand-corrected AlphaPose Signer track (signer-guided DWPose). |
| [`wG_readout.py`](tools/wG_readout.py) | Reads out signer-guided DWPose against stored pose and tracker-v2 pose, reusing the heavy-switch statistics. |

## Launchers for early laptop runs

The shell scripts that were run on a laptop during the early experiments are in `../jobs/local/` and are listed in `../jobs/slurm/README.md`.
