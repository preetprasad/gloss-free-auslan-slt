# Job scripts: `jobs/slurm/` (Kaya cluster) and `jobs/local/` (laptop)

The scripts below are the SLURM job scripts as they were submitted to the UWA Kaya cluster. They are the record of how each experiment was run: the settings of every run are the flags in these files. Paths under /group/pmc099 and /scratch/pmc099 are cluster locations, and the scripts assume the cluster's directory layout, so they are not expected to run elsewhere unchanged. Files beginning with an underscore are shell libraries sourced by the job scripts. The code folder was called `phase2_baseline/` when these jobs ran, so their paths say `phase2_baseline`; it is `src/` in this repository. The job numbers that match each script to the dissertation are in the tables of the root README, and the file-name prefixes (`w4_`, `w12a_` and so on) are explained there.

## Data staging and pose extraction

| Script | What it runs |
|---|---|
| [`data_prep.sbatch`](data_prep.sbatch) | Fetches the Auslan-Daily Communication signer video archive from Google Drive with rclone and unzips the clips to scratch storage. |
| [`extract_dwpose.sbatch`](extract_dwpose.sbatch) | Extracts DWPose (RTMW-DW-x) keypoints from the Communication signer clips as a resumable GPU job array. |
| [`agreement.sbatch`](agreement.sbatch) | Runs a pose-estimator agreement check over a sample of RGB clips with `agreement_rtmlib.py` on a GPU. |
| [`audit_dwpose_inputs.sbatch`](audit_dwpose_inputs.sbatch) | Audits DWPose clip quality on CPU and writes evidence for confidence and keypoint-subset ablations. |
| [`classical_cv.sbatch`](classical_cv.sbatch) | Runs classical, non-neural computer-vision analysis of the raw Communication signer clips as a CPU job array. |
| [`alphapose_stage.sbatch`](alphapose_stage.sbatch) | Fetches the shipped AlphaPose pose files, extracts per-clip arrays and builds manifests matched to the DWPose clips, text and split. These clips were extracted by the first version of `extract_clips.py`, which took the first-listed person of each clip rather than the signer; `apsig_extract.sbatch` below repeats the extraction with the signer. |
| [`mmwl_probe.sbatch`](mmwl_probe.sbatch) | Fetches a small MM-WLAuslan validation archive and inspects its file layout, label encoding and shipped pose format. |
| [`mmwl_dwpose_valid.sbatch`](mmwl_dwpose_valid.sbatch) | Runs a pilot DWPose extraction of the MM-WLAuslan validation videos, keyed by sample identifier, as a GPU job array. |
| [`mmwl_dwpose_train.sbatch`](mmwl_dwpose_train.sbatch) | Fetches, unzips and DWPose-extracts the MM-WLAuslan training videos as a GPU job array. |
| [`mmwl_classical_cv.sbatch`](mmwl_classical_cv.sbatch) | Applies the classical computer-vision worker to the MM-WLAuslan validation clips as a CPU job array. |

## Baselines and single models

| Script | What it runs |
|---|---|
| [`train_ctc.sbatch`](train_ctc.sbatch) | Trains the translator with an auxiliary pseudo-gloss CTC head on the shipped AlphaPose pose to encourage temporal grounding. It reads the first-listed-person clips (see `apsig_extract.sbatch`). |
| [`train_dwpose.sbatch`](train_dwpose.sbatch) | Builds manifests pointing at the DWPose clips, retrains the translator on DWPose pose and evaluates it with beam search. |
| [`dwpose_ctc.sbatch`](dwpose_ctc.sbatch) | Trains the translator on DWPose pose jointly with pseudo-gloss CTC grounding using `train_ctc.py`. |
| [`train_dwpose_ctc.sbatch`](train_dwpose_ctc.sbatch) | Trains joint translation and rule-based pseudo-gloss CTC on DWPose input, with the CTC weight supplied as an environment variable. |
| [`dwpose_ctc_seeds.sbatch`](dwpose_ctc_seeds.sbatch) | Repeats the DWPose plus CTC training at several seeds as a job array to estimate variation across seeds. |
| [`tune_dwpose_beam.sbatch`](tune_dwpose_beam.sbatch) | Selects beam width and length penalty on validation, then evaluates the selected setting once on test. |
| [`ctc_weight_sweep.sbatch`](ctc_weight_sweep.sbatch) | Sweeps the CTC loss weight on the DWPose plus CTC recipe, with selection on validation. |
| [`estimator_ablation.sbatch`](estimator_ablation.sbatch) | Separates estimator, face-keypoint and normalisation effects by training DWPose with body and hands only under two normalisations. |
| [`alphapose_ablation.sbatch`](alphapose_ablation.sbatch) | Trains on AlphaPose pose under bounding-box and frame normalisation to complete the estimator-by-normalisation comparison. It reads the first-listed-person clips, so its AlphaPose cells are superseded by `w5_estrep_signer.sbatch`. |
| [`video_disjoint.sbatch`](video_disjoint.sbatch) | Retrains the DWPose plus CTC recipe on a split that holds out whole videos, to measure generalisation to unseen sessions. |
| [`comm_long.sbatch`](comm_long.sbatch) | Trains the plain recipe at several epoch budgets and seeds to measure the effect of schedule length, with validation-only selection. |

## Pose-text pretraining

| Script | What it runs |
|---|---|
| [`dwpose_vlp.sbatch`](dwpose_vlp.sbatch) | Pretrains the DWPose encoder with contrastive pose-text (VLP) learning, then fine-tunes the translator with CTC. |
| [`gpu_batch_10h.sbatch`](gpu_batch_10h.sbatch) | Trains further seeds of the VLP-to-CTC and RGB-fusion recipes, plus a longer VLP pretraining run, within one GPU allocation. |
| [`vlp_scale.sbatch`](vlp_scale.sbatch) | Trains VLP for longer pretraining budgets, each followed by the standard CTC fine-tune, to relate pretraining length to translation quality. |

## RGB features and fusion

| Script | What it runs |
|---|---|
| [`rgb_extract.sbatch`](rgb_extract.sbatch) | Extracts DINO image features from crops for every clip as a resumable CPU job array. |
| [`rgb_fusion_smoke.sbatch`](rgb_fusion_smoke.sbatch) | Runs the pose-plus-RGB fusion model end to end on synthetic data on CPU to validate the code path. |
| [`dwpose_rgb_fusion.sbatch`](dwpose_rgb_fusion.sbatch) | Trains a two-stream pose and RGB translator warm-started from the VLP checkpoint. |
| [`rgb_vlp_smoke.sbatch`](rgb_vlp_smoke.sbatch) | Tests RGB-text pretraining followed by warm-started RGB fusion on synthetic data on CPU. |
| [`rgb_vlp_fusion.sbatch`](rgb_vlp_fusion.sbatch) | Runs RGB-text contrastive pretraining, then RGB fusion warm-started from both the pose-VLP and RGB-VLP encoders. |
| [`presence_set_rgb.sbatch`](presence_set_rgb.sbatch) | Trains DWPose plus RGB with order-invariant pseudo-content supervision, using training and validation data only. |
| [`temporal_pyramid_ablation.sbatch`](temporal_pyramid_ablation.sbatch) | Runs a paired, validation-only ablation of a lightweight multi-scale temporal adapter on the RGB architecture without CTC. |
| [`temporal_pyramid_ctc.sbatch`](temporal_pyramid_ctc.sbatch) | Trains the multi-scale temporal adapter together with the pseudo-gloss CTC head, using validation only for selection. |
| [`sentence_targets.sbatch`](sentence_targets.sbatch) | Builds a frozen cache of sentence-embedding targets for the training subtitles with a sentence-transformer model on CPU. |
| [`temporal_semantic_ctc.sbatch`](temporal_semantic_ctc.sbatch) | Adds a sentence-semantic auxiliary loss to the temporal-adapter CTC configuration, using validation only. |
| [`face_feature_smoke.sbatch`](face_feature_smoke.sbatch) | Validates DWPose-localised face crops and face feature extraction on a small sample without reading a test manifest. |
| [`face_feature_extract.sbatch`](face_feature_extract.sbatch) | Extracts face-only DINO features for training and validation clips as a GPU job array, excluding test clips. |
| [`face_temporal_ctc_smoke.sbatch`](face_temporal_ctc_smoke.sbatch) | Runs an interface smoke test of the face-channel temporal CTC model on small training and validation subsets. |
| [`face_temporal_ctc.sbatch`](face_temporal_ctc.sbatch) | Appends frozen face-region features to the RGB features of the temporal CTC model, using training and validation only. |
| [`gated_temporal_ctc_smoke.sbatch`](gated_temporal_ctc_smoke.sbatch) | Runs an interface smoke test of gated pose-RGB fusion whose partial validation metric is not used for selection. |
| [`gated_temporal_ctc.sbatch`](gated_temporal_ctc.sbatch) | Trains gated pose-RGB fusion matched to the temporal-adapter CTC configuration, using training and validation only. |
| [`stochastic_filtered_ctc_p8.sbatch`](stochastic_filtered_ctc_p8.sbatch) | Runs a validation-only ablation with stochastically filtered English-order pseudo-content CTC targets. |
| [`temporal_lift_pool_p9.sbatch`](temporal_lift_pool_p9.sbatch) | Tests adaptive pooling of the decoder memory, with CTC kept on the unpooled pose memory, using validation only. |

## Ensembles and decoding

| Script | What it runs |
|---|---|
| [`ensemble.sbatch`](ensemble.sbatch) | Decodes training-free ensembles that average logits across same-vocabulary checkpoints, on CPU. |
| [`tier1.sbatch`](tier1.sbatch) | Weight-averages the DWPose plus CTC seed checkpoints and re-tunes ensemble beam and length penalty on validation, on a GPU node. |
| [`tier1_cpu.sbatch`](tier1_cpu.sbatch) | Runs the same seed weight averaging and ensemble decoding re-tuning as `tier1.sbatch` on a CPU node. |
| [`ensemble_sweep.sbatch`](ensemble_sweep.sbatch) | Decodes candidate member sets by logit averaging on validation for selection, then on test, on CPU. |
| [`swa.sbatch`](swa.sbatch) | Retrains two recipes while saving late-epoch checkpoints and averages them within each run (stochastic weight averaging). |
| [`swa_ens.sbatch`](swa_ens.sbatch) | Decodes ensembles that substitute the weight-averaged members, with selection on validation. |
| [`ensemble_max.sbatch`](ensemble_max.sbatch) | Searches weighted logit averaging, member sets, beam width and length penalty, with selection on validation. |
| [`ensemble_ablation.sbatch`](ensemble_ablation.sbatch) | Measures each ensemble member's contribution by incremental and leave-one-out decoding on validation. |
| [`score_best_ensemble_wer.sbatch`](score_best_ensemble_wer.sbatch) | Evaluates the validation-selected ensemble once on test, reporting BLEU, WER and substitution, deletion and insertion counts. |
| [`mbr_wer_val.sbatch`](mbr_wer_val.sbatch) | Sweeps length penalty and minimum-Bayes-risk WER selection on the established ensemble, using validation only. |
| [`mbr_wer_val_ext.sbatch`](mbr_wer_val_ext.sbatch) | Extends the validation sweep to lower length penalties and adds minimum-Bayes-risk selection by chrF. |
| [`mbr_wer_test.sbatch`](mbr_wer_test.sbatch) | Scores the validation-selected length penalty once on test, with the default penalty alongside as a within-run baseline. |
| [`mbr_wer_test2.sbatch`](mbr_wer_test2.sbatch) | Scores the length-penalty settings selected by the extended validation sweep once on test. |
| [`ensemble_dur.sbatch`](ensemble_dur.sbatch) | Tests adding duration-stitch members to the established ensemble, selecting on validation and decoding test once. |
| [`ensemble_dur_remap.sbatch`](ensemble_dur_remap.sbatch) | Re-runs ensemble candidates with vocabulary remapping between members whose vocabularies are ordered differently. |
| [`ensemble_stage2.sbatch`](ensemble_stage2.sbatch) | Re-sweeps leading candidates over a wider length-penalty grid on validation, then evaluates the validation argmax once on test. |
| [`ensemble_stage2_test.sbatch`](ensemble_stage2_test.sbatch) | Evaluates the validation winner and the in-batch control ensemble once on test, each at its own validation length penalty. |
| [`ensemble_extended.sbatch`](ensemble_extended.sbatch) | Tests whether adding members of a different recipe changes ensemble quality more than adding seeds of one recipe. |
| [`ensemble_search.sbatch`](ensemble_search.sbatch) | Searches ensembles combining duration-stitch and fusion members on validation, with RGB features supplied and a wider length-penalty grid. |
| [`ensemble_p3_val.sbatch`](ensemble_p3_val.sbatch) | Tests on validation whether the temporal-adapter CTC checkpoint complements the established ensemble. |
| [`ensemble_p3_p6_micro_val.sbatch`](ensemble_p3_p6_micro_val.sbatch) | Tests on validation adding temporal-adapter or gated-fusion checkpoints to the established ensemble at small weights. |
| [`t1_enssel_s6val.sbatch`](t1_enssel_s6val.sbatch) | Writes the established ensemble's validation n-best dump so that selective-prediction thresholds can be fitted on validation. |
| [`ens_subsets.sbatch`](ens_subsets.sbatch) | Decodes all 63 member subsets of the final ensemble on validation in four array tasks, then scores them in a dependent job; no training and no test decode. |
| [`seedens.sbatch`](seedens.sbatch) | Decodes single seeds and same-recipe seed ensembles from several training batches, choosing length penalty on validation, without training. |
| [`t2_vdens.sbatch`](t2_vdens.sbatch) | Tests ensembling on the video-disjoint split, selecting configuration and length penalty on its validation set before test decoding. |
| [`vd_fusion_train.sbatch`](vd_fusion_train.sbatch) | Trains RGB-fusion members on the video-disjoint training split, warm-started from video-disjoint pose checkpoints. |
| [`vd_xmodal_decode.sbatch`](vd_xmodal_decode.sbatch) | Decodes the video-disjoint fusion members alone and in an ensemble with pose members, on validation and then test. |

## Evaluation and analysis

| Script | What it runs |
|---|---|
| [`dump_hyps.sbatch`](dump_hyps.sbatch) | Writes test reference and hypothesis pairs for several checkpoints, for later semantic grading. |
| [`metric_suite.sbatch`](metric_suite.sbatch) | Computes metrics beyond BLEU and a seen versus unseen sentence breakdown on saved hypothesis dumps. |
| [`grade_semantic.sbatch`](grade_semantic.sbatch) | Grades the semantic adequacy of saved beam outputs on CPU, using a local instruction-tuned language model as judge. |
| [`semantic_regrade.sbatch`](semantic_regrade.sbatch) | Repeats semantic grading with the same judge model from a durable cache, after first checking that it reproduces a recorded grading. |
| [`bench_latency.sbatch`](bench_latency.sbatch) | Measures single-stream CPU decoding latency of a pose-only checkpoint with greedy and beam decoding. |
| [`bench_latency_fusion.sbatch`](bench_latency_fusion.sbatch) | Measures single-stream CPU decoding latency of the RGB-fusion checkpoint with greedy and beam decoding. |
| [`dwpose_fps.sbatch`](dwpose_fps.sbatch) | Benchmarks DWPose front-end throughput and latency on GPU and on CPU. |

## Stitched synthetic data

| Script | What it runs |
|---|---|
| [`posestitch_build.sbatch`](posestitch_build.sbatch) | Builds a stitched synthetic continuous-signing corpus from MM-WLAuslan validation clips matched to Auslan-Daily training sentences. |
| [`posestitch_train.sbatch`](posestitch_train.sbatch) | Pretrains on the stitched corpus, then fine-tunes on real Auslan-Daily, with selection on Communication validation. |
| [`posestitch_ft.sbatch`](posestitch_ft.sbatch) | Repeats the real-data fine-tuning stage from the completed stitched pretraining checkpoint. |
| [`posestitch_v2.sbatch`](posestitch_v2.sbatch) | Trains real-only, compute-matched and joint real-plus-synthetic arms on a rebuilt corpus of generated novel sentences. |
| [`posestitch_v2_seeds.sbatch`](posestitch_v2_seeds.sbatch) | Repeats the paired control and joint arms at fresh seeds on the same synthetic corpus. |
| [`posestitch_v2_scale.sbatch`](posestitch_v2_scale.sbatch) | Generates further synthetic sentences with a different seed and trains a joint arm on the enlarged corpus. |
| [`vlp_stitch_2x2.sbatch`](vlp_stitch_2x2.sbatch) | Runs a factorial design crossing VLP pretraining data and fine-tuning data, each either real-only or real plus synthetic. |
| [`stitch_diversity_k1k2.sbatch`](stitch_diversity_k1k2.sbatch) | Holds clip count fixed and compares many sentences rendered once against fewer sentences rendered twice. |
| [`stitch_temporal.sbatch`](stitch_temporal.sbatch) | Compares temporal treatments of stitched clips, including frame-rate correction and duration modelling, within one controlled job. |
| [`stitch_dur_seeds.sbatch`](stitch_dur_seeds.sbatch) | Repeats the duration-modelled stitch comparison against same-seed controls at further seeds. |
| [`dur_vlp_stack.sbatch`](dur_vlp_stack.sbatch) | Fine-tunes a VLP-pretrained encoder on real data with and without duration-modelled synthetic clips. |
| [`gen_constrained.sbatch`](gen_constrained.sbatch) | Generates synthetic sentences with logits masked to signable phrases, applying morphology and sense-disambiguation lexicon fixes, without training. |
| [`stitch_scale.sbatch`](stitch_scale.sbatch) | Trains joint arms at several synthetic-to-real ratios, with real-only and compute-matched controls. |
| [`stitch_scramble.sbatch`](stitch_scramble.sbatch) | Renders each sentence from randomly chosen signs, keeping targets, sign counts and durations, to separate lexical from structural effects. |
| [`stitch_motion.sbatch`](stitch_motion.sbatch) | Tests a motion-corrected synthetic corpus, without onset trimming and with motion-aware resampling, against same-seed controls. |
| [`dur_more_seeds.sbatch`](dur_more_seeds.sbatch) | Trains further duration-stitch seeds with the unchanged recipe to supply ensemble members. |
| [`dur_long.sbatch`](dur_long.sbatch) | Trains the duration-stitch recipe at a longer epoch budget against an in-batch control at the standard budget. |
| [`stitch_audit.sbatch`](stitch_audit.sbatch) | Rebuilds the synthetic corpus with a provenance sidecar, checks clip by clip that it matches the trained corpus, and computes statistics. |

## Pose augmentation and later controlled experiments

| Script | What it runs |
|---|---|
| [`pose_aug.sbatch`](pose_aug.sbatch) | Compares training-time pose augmentation (speed, rotation and scale perturbation) against paired controls on the adopted stitch corpus. |
| [`wave1_r2.sbatch`](wave1_r2.sbatch) | Compares post-normalisation hand masking and a longer epoch budget against the in-batch adopted recipe at several seeds. |
| [`wave2_prep.sbatch`](wave2_prep.sbatch) | Builds paired synthetic corpora from the same sentences, with and without the fingerspelled-letter lexicon fix, and checks row alignment. |
| [`wave2_r2.sbatch`](wave2_r2.sbatch) | Trains the letter-fix corpus against its paired control, and a follow-up stage of the longer-budget comparison at fresh seeds. |
| [`w4_prep.sbatch`](w4_prep.sbatch) | Builds the adopted, untrimmed-sign and repeated-realisation stitch corpora from one sentence pool and seed, then gates them. |
| [`w4_train.sbatch`](w4_train.sbatch) | Trains the adopted corpus against the untrimmed-sign and repeated-realisation corpora with deterministic kernels. |
| [`w4_trim_proxies.sbatch`](w4_trim_proxies.sbatch) | Computes motion proxies (join gap, velocity, pause density, separability) for trimmed and untrimmed corpora on CPU, without a model. |
| [`w4_news.sbatch`](w4_news.sbatch) | Trains News real-only arms against News plus synthetic clips with pose augmentation at several seeds. |
| [`w4_news_s2.sbatch`](w4_news_s2.sbatch) | Runs the follow-up stage of the News comparison at further seeds. |
| [`w5_estrep.sbatch`](w5_estrep.sbatch) | Replicates the DWPose versus AlphaPose comparison across seeds, with matched body and hand keypoints and frame normalisation. It reads the first-listed-person AlphaPose clips (the +4.97 difference of REPRODUCING.md, A32, which `w5_estrep_signer.sbatch` supersedes). |
| [`w5_null.sbatch`](w5_null.sbatch) | Repeats one configuration at one seed without determinism to estimate the run-to-run variation of a paired difference. |
| [`w5_gen.sbatch`](w5_gen.sbatch) | Generates constrained synthetic sentences using the full MM-WLAuslan dictionary, as a GPU job array. |
| [`w5_prep.sbatch`](w5_prep.sbatch) | Merges generated sentences, builds and gates a large sharded synthetic clip set, and writes nested prefix manifests and a fixed vocabulary. |
| [`w5_train.sbatch`](w5_train.sbatch) | Pretrains on nested synthetic prefixes at fixed compute, then fine-tunes with the adopted recipe, against scratch and compute controls. |
| [`w5_train_s2.sbatch`](w5_train_s2.sbatch) | Runs the follow-up stage at further seeds for the retained prefix arm, with fresh scratch and compute controls. |
| [`w5_summary.sbatch`](w5_summary.sbatch) | Writes per-seed summaries and seed-level readouts for the synthetic pretraining arms on CPU. |
| [`w6_corpus.sbatch`](w6_corpus.sbatch) | Trains jointly on the full-dictionary synthetic corpus at the adopted dose and a larger dose, against the adopted corpus. |
| [`w6_corpus_s2.sbatch`](w6_corpus_s2.sbatch) | Runs the follow-up stage of the full-dictionary corpus comparison at fresh seeds with a fresh control. |
| [`w7_prep.sbatch`](w7_prep.sbatch) | Builds and gates the adopted corpus and a variant that omits signs for function words, both from the same sentences. |
| [`w7_density.sbatch`](w7_density.sbatch) | Trains on the adopted corpus versus the reduced-sign-density corpus at several seeds. |
| [`w7s2_prep.sbatch`](w7s2_prep.sbatch) | Builds and gates a reduced-density corpus whose sign lengths are rescaled so total clip length matches the adopted corpus. |
| [`w7s2_train.sbatch`](w7s2_train.sbatch) | Runs the follow-up stage of the sign-density comparison at fresh seeds, adding the length-matched arm. |
| [`w8_realdata.sbatch`](w8_realdata.sbatch) | Measures how translation quality varies with the amount of real training data, using within-video and whole-video subsets at matched budget. |
| [`w9_prep.sbatch`](w9_prep.sbatch) | Builds pose arms and manifests for three signer-selection methods from one all-detections extraction. |
| [`w9_tracker.sbatch`](w9_tracker.sbatch) | Trains and evaluates each signer-selection method on its own pose, with English text and synthetic data held identical. |
| [`w10_prep.sbatch`](w10_prep.sbatch) | Builds the adopted corpus and a sign-order-shuffled variant, and checks that they differ only in sign order. |
| [`w10_order.sbatch`](w10_order.sbatch) | Trains arms with English-order signs, shuffled signs, shuffled signs with re-ordered CTC targets, and real data only. |
| [`w10_summary.sbatch`](w10_summary.sbatch) | Writes per-seed summaries and seed-level contrasts of each sign-order arm against the adopted arm on CPU. |
| [`w11_vlpepochs.sbatch`](w11_vlpepochs.sbatch) | Selects the number of pose-text pretraining epochs on validation, including a no-pretraining control, at several seeds. |
| [`w11_newsframes.sbatch`](w11_newsframes.sbatch) | Selects the input frame limit for the News subset on validation, comparing two limits at several seeds. |

## News split

| Script | What it runs |
|---|---|
| [`news_stage.sbatch`](news_stage.sbatch) | Fetches and unzips the News signer videos and translation spreadsheet from Google Drive after checking the DWPose runtime. |
| [`news_extract_dwpose.sbatch`](news_extract_dwpose.sbatch) | Extracts DWPose keypoints from the News clips with the Communication extraction pipeline, as a resumable GPU job array. |
| [`joint_train.sbatch`](joint_train.sbatch) | Runs VLP pretraining and CTC training on combined Communication and News data, with selection and evaluation on Communication. |
| [`news_baseline.sbatch`](news_baseline.sbatch) | Trains on News and evaluates on News, for comparison with published News results. |
| [`news_long.sbatch`](news_long.sbatch) | Retrains News arms at a longer schedule with two input frame limits, with selection on validation. |
| [`news_bpe.sbatch`](news_bpe.sbatch) | Compares a subword (BPE) target vocabulary against word-level targets on News. |
| [`news_oov.sbatch`](news_oov.sbatch) | Re-decodes existing News checkpoints to check whether the subword model emits out-of-vocabulary words, and whether they are correct. |
| [`news_xfer.sbatch`](news_xfer.sbatch) | Tests News as a pretraining stage before Communication fine-tuning, against a Communication-only control. |
| [`news_decode.sbatch`](news_decode.sbatch) | Decodes each News model alone with a length-penalty sweep, reading test at the model's validation-selected penalty. |
| [`news_ens.sbatch`](news_ens.sbatch) | Decodes the News seed ensemble and its members on validation, with a futility gate applied before any test decoding. |

## Person tracking and non-signer control

| Script | What it runs |
|---|---|
| [`extract_alldet.sbatch`](extract_alldet.sbatch) | Re-extracts all Communication clips keeping every detected person and the existing tracker outputs, as a resumable CPU job array. |
| [`extract_alldet_gt.sbatch`](extract_alldet_gt.sbatch) | Extracts the tracker ground-truth clips first, into the same all-detections output directory used for training data. |
| [`t3_ndet_export.sbatch`](t3_ndet_export.sbatch) | Exports per-clip detection counts and tracker choices from the tracked re-extraction to a CSV file. |
| [`stage_nonsigner.sbatch`](stage_nonsigner.sbatch) | Fetches and extracts the Communication non-signer crops to the group filesystem and verifies that every expected crop is present. |
| [`extract_nonsigner.sbatch`](extract_nonsigner.sbatch) | Runs person-tracked DWPose extraction of the development and test non-signer crops, with an extractor consistency check on signer clips. |

## Alternatives not adopted

| Script | What it runs |
|---|---|
| [`dwpose_d512.sbatch`](dwpose_d512.sbatch) | Trains a wider and deeper translator on DWPose pose with `train.py`. |
| [`train_dwpose_scaled.sbatch`](train_dwpose_scaled.sbatch) | Re-tests a larger translator on the DWPose manifests with the target protocol unchanged. |
| [`gaslt.sbatch`](gaslt.sbatch) | Trains a GASLT-style local temporal attention encoder on DWPose pose with the CTC objective. |
| [`temporal_pool_ctc.sbatch`](temporal_pool_ctc.sbatch) | Average-pools consecutive pose frames before the encoder at several pooling factors, with selection on validation. |
| [`graph_encoder.sbatch`](graph_encoder.sbatch) | Adds a learned-adjacency graph block over keypoints before the encoder, after a smoke test of the model build. |
| [`graph_encoder_fair.sbatch`](graph_encoder_fair.sbatch) | Repeats the graph front-end test with adjacency initialised from the anatomical skeleton, against an in-batch control. |
| [`hand_local_norm.sbatch`](hand_local_norm.sbatch) | Trains with each hand normalised around its own wrist at palm scale. |
| [`rdrop.sbatch`](rdrop.sbatch) | Adds R-Drop consistency regularisation (two dropout passes with symmetric KL divergence) to the DWPose plus CTC recipe. |
| [`regularisation_sweep.sbatch`](regularisation_sweep.sbatch) | Tunes dropout and related regularisation one setting at a time on validation, with a longer schedule for every arm. |
| [`ctc_freqweight.sbatch`](ctc_freqweight.sbatch) | Weights pseudo-gloss CTC targets by inverse corpus frequency and dumps hypotheses for analysis of unseen targets. |
| [`bpe_subword.sbatch`](bpe_subword.sbatch) | Trains SentencePiece subword models and retrains the DWPose plus CTC recipe with subword targets. |
| [`paraphrase_aug.sbatch`](paraphrase_aug.sbatch) | Paraphrases training subtitles with a local language model and trains with a probability of substituting a paraphrase. |
| [`pgg_slt.sbatch`](pgg_slt.sbatch) | Generates ordered pseudo-glosses with a local language model and uses them as CTC targets, warm-started from the VLP encoder. |
| [`grpo_smoke.sbatch`](grpo_smoke.sbatch) | Runs the GRPO reinforcement-learning loop on synthetic pose data on CPU to validate sampling, reward and update steps. |
| [`grpo.sbatch`](grpo.sbatch) | Fine-tunes the VLP translator with group-relative policy optimisation, using a BLEU and ROUGE-L reward with a KL penalty. |
| [`islr_pretrain.sbatch`](islr_pretrain.sbatch) | Pretrains a DWPose isolated-sign recognition encoder on MM-WLAuslan for loading into the translator. |
| [`islr_transfer.sbatch`](islr_transfer.sbatch) | Initialises the DWPose plus CTC translator from the isolated-sign recognition encoder. |
| [`islr_vlp_ctc.sbatch`](islr_vlp_ctc.sbatch) | Initialises VLP pretraining from the isolated-sign recognition encoder, then fine-tunes with CTC. |
| [`spotting_probe.sbatch`](spotting_probe.sbatch) | Probes whether the isolated-sign recognition encoder spots known signs in continuous Auslan-Daily clips with a sliding window. |
| [`llm_mt5.sbatch`](llm_mt5.sbatch) | Trains a pretrained multilingual text-to-text decoder with LoRA on the DWPose encoder, after a smoke test on a small subset. |
| [`vq_llm.sbatch`](vq_llm.sbatch) | Learns a vector-quantised pose tokeniser, then decodes English from the tokens with a pretrained text-to-text model and LoRA. |
| [`mbart_smoke.sbatch`](mbart_smoke.sbatch) | Runs a smoke test of the pose-encoder, adapter and mBART pipeline on a small subset for one epoch. |
| [`mbart_full.sbatch`](mbart_full.sbatch) | Trains the pose encoder, adapter and mBART decoder with LoRA on the official split, warm-started from the VLP encoder. |
| [`mbart_align.sbatch`](mbart_align.sbatch) | Aligns the pose encoder and adapter to a frozen mBART decoder, then fine-tunes with LoRA enabled. |
| [`gfslt_env_build.sbatch`](gfslt_env_build.sbatch) | Builds an isolated conda environment with the older package versions required by the GFSLT-VLP repository. |
| [`phoenix_stage.sbatch`](phoenix_stage.sbatch) | Fetches and extracts the PHOENIX dataset archive to scratch storage on CPU, for the GFSLT-VLP reproduction. |
| [`gfslt_vlp_phoenix.sbatch`](gfslt_vlp_phoenix.sbatch) | Runs multi-GPU GFSLT-VLP pretraining and translation training on PHOENIX for the reproduction attempt. |
| [`mmpose_probe.sbatch`](mmpose_probe.sbatch) | Checks whether MMPose and its whole-body estimators install cleanly in an isolated environment. |
| [`mmpose_hrnet_frame_probe.sbatch`](mmpose_hrnet_frame_probe.sbatch) | Compares HRNet whole-body and DWPose keypoints on one frame, bypassing the detector, to check layout and coordinates on CPU. |

## Utilities

| Script | What it runs |
|---|---|
| [`_w4lib.sh`](_w4lib.sh) | Shell library that enters a frozen code snapshot, checks the determinism fix, defines shared flags and provides a pose-file checksum gate. |
| [`_w5lib.sh`](_w5lib.sh) | Shell library equivalent to `_w4lib.sh`, pointed at a later code snapshot that supports a fixed vocabulary file. |
| [`_w11lib.sh`](_w11lib.sh) | Shell library equivalent to `_w4lib.sh`, pointed at a snapshot that also contains seeded pose-text pretraining. |
| [`env_build.sbatch`](env_build.sbatch) | Creates the Python virtual environment, installs PyTorch, rtmlib, OpenCV and evaluation packages, and pins a CUDA-compatible ONNX Runtime. |
| [`gpu_smoke.sbatch`](gpu_smoke.sbatch) | Reinstalls the GPU ONNX Runtime build and checks that the CUDA execution provider is available on a GPU node. |
| [`backup_scratch.sh`](backup_scratch.sh) | Copies derived DWPose extractions and manifests from purge-prone scratch storage to the durable group filesystem. |
| [`mirror_datasets_kaya.sh`](mirror_datasets_kaya.sh) | Mirrors selected datasets from Google Drive to Kaya storage by preset with rclone, from a login or data node. |

## Signer-keypoint correction and later follow-ups

The `w12*` scripts source `_w12lib.sh`, the shared library of that group of experiments, which is not part of this release. The scripts are listed so that each experiment's settings can be read; they cannot be re-launched from this repository alone.

| Script | What it runs |
|---|---|
| [`apsig_extract.sbatch`](apsig_extract.sbatch) | Re-extracts the released AlphaPose keypoints taking the annotated Signer entry of every clip, checks that rows and texts are unchanged, and writes new clips and manifests beside the old ones. |
| [`w5_estrep_signer.sbatch`](w5_estrep_signer.sbatch) | Repeats the DWPose versus AlphaPose comparison with the signer's AlphaPose keypoints; AlphaPose and DWPose are trained in the same job at three seeds. |
| [`w12a_news_decomp.sbatch`](w12a_news_decomp.sbatch) | Splits the adopted News bundle into stitched clips and pose augmentation (arms nA, nS, nSA) at three seeds in one array. |
| [`w12a_news_decomp_summary.sbatch`](w12a_news_decomp_summary.sbatch) | Reads out the News decomposition with the rule fixed in advance. |
| [`w12b_prep.sbatch`](w12b_prep.sbatch) | Builds the synthetic corpus without 14 wrongly rendered signs and gates it against the adopted corpus. |
| [`w12b_wrongsense.sbatch`](w12b_wrongsense.sbatch) | Trains the adopted recipe on the adopted corpus (A) and on the same sentences without the wrong-sense signs (W), three seeds each. |
| [`w12b_summary.sbatch`](w12b_summary.sbatch) | Reads out the wrong-sense experiment. |
| [`w12d_powermean.sbatch`](w12d_powermean.sbatch) | Decodes validation with power means of the ensemble members' probabilities, one array task per rule; CPU only, no test decode. |
| [`w12e_extract.sbatch`](w12e_extract.sbatch) | Extracts the DINOv2 crop features that the three RGB-fusion ensemble members need for crops of people who are not signing. |
| [`w12e_decode.sbatch`](w12e_decode.sbatch) | Decodes those crops with the reported ensemble after reproducing its recorded validation and test scores. |
| [`_w12flib.sh`](_w12flib.sh) | Shell library for the three-step pose-measurement chain below; it is sourced after the shared wave-12 library, which is not in this release. |
| [`w12f_stage.sbatch`](w12f_stage.sbatch) | Stages Auslan-Daily's isolated-sign clips and finds the conversational frames each was cut from. |
| [`w12f_extract.sbatch`](w12f_extract.sbatch) | Extracts DWPose and estimated-depth pose for the isolated-sign clips and for the clips of the depth measurement. |
| [`w12f_probe.sbatch`](w12f_probe.sbatch) | Runs the two pose measurements of sign identity on CPU; no model is trained. |
| [`w12g_prep.sbatch`](w12g_prep.sbatch) | Builds and checks the signer-disjoint split and its exposure control. |
| [`w12g_train.sbatch`](w12g_train.sbatch) | Trains on the signer-disjoint split and on the two exposure arms, three seeds each. |
| [`w12g_summary.sbatch`](w12g_summary.sbatch) | Reads out the signer-disjoint experiment. |
| [`w12h_pretrain.sbatch`](w12h_pretrain.sbatch) | Compares the adopted recipe from scratch with extra pre-training on the joint corpus and on synthetic clips only, with new seeds. |
| [`w12h_summary.sbatch`](w12h_summary.sbatch) | Reads out the extra pre-training experiment. |
| [`w12i_tracker_heavy.sbatch`](w12i_tracker_heavy.sbatch) | Trains on stored pose and on tracker-v2 pose with five new seeds, to retest the heavy-switch difference-in-differences. |
| [`w12i_tracker_heavy_summary.sbatch`](w12i_tracker_heavy_summary.sbatch) | Reads out the heavy-switch retest after reproducing the recorded wave-9 values. |

## Local launchers (`jobs/local/`)

These shell scripts were run from the code folder (then called `phase2_baseline/`) during early laptop experiments and were later moved here. They are kept as they ran and work out their paths from their own location, so a path may need adjusting to run one from here.

| File | Description |
| --- | --- |
| [`run_sourcefaithful_slt.sh`](../local/run_sourcefaithful_slt.sh) | Trains the closest reproducible Auslan adaptation of the pinned public SignJoey model, feeding flattened raw body and hand keypoints directly. |
| [`run_improved_slt.sh`](../local/run_improved_slt.sh) | Trains the SL-Transformer with xywh box handling, confidence-based coordinate suppression, a smoothed token objective and warm-up with cosine scheduling. |
| [`run_gaslt_attention_control.sh`](../local/run_gaslt_attention_control.sh) | Replaces only the SL-Transformer encoder's self-attention with the GASLT gloss-attention operator, holding other settings to `run_improved_slt.sh`. |
| [`run_gaslt_paper_recipe.sh`](../local/run_gaslt_paper_recipe.sh) | Reconstructs the public GASLT recipe for Communication, leaving the pose-to-visual-feature adapter as the remaining non-public variable. |
| [`run_signjoey_vlp_pretrain.sh`](../local/run_signjoey_vlp_pretrain.sh) | Contrastively pretrains the SignJoey pose encoder against Communication subtitles before translation. |
| [`run_islr_pretrain.sh`](../local/run_islr_pretrain.sh) | Pretrains the translator's pose encoder on MM-WLAuslan word-level recognition to produce a transferable encoder checkpoint. |
| [`run_islr_transfer_ab.sh`](../local/run_islr_transfer_ab.sh) | Runs a paired Communication test training identical translators from scratch and from the recognition-pretrained encoder. |
| [`extract_glofe_full.sh`](../local/extract_glofe_full.sh) | Extracts public GloFE features for the Communication train, validation and test splits on CPU, resuming from existing output. |
| [`run_glofe_slt.sh`](../local/run_glofe_slt.sh) | Trains the SL-Transformer on public GloFE features as a controlled input-pipeline experiment. |
| [`run_auslan_pose_mae.sh`](../local/run_auslan_pose_mae.sh) | Runs masked pose pretraining on official Auslan-Daily training rows from Communication and News only. |
| [`extract_pose_mae_communication.sh`](../local/extract_pose_mae_communication.sh) | Exports pose-MAE features for the official Communication train, validation and test splits from a pretrained checkpoint. |
| [`extract_pose_mae_full.sh`](../local/extract_pose_mae_full.sh) | Exports train, validation and test temporal features after Auslan pose-MAE pretraining. |
| [`run_pose_mae_slt.sh`](../local/run_pose_mae_slt.sh) | Trains the SL-Transformer recipe on Auslan pose-MAE temporal features as a controlled transfer experiment. |
