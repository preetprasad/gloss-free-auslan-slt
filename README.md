# Investigating Gloss-Free Auslan Sign Language to Speech Translation

This repository contains the code for the Honours dissertation *Investigating Gloss-Free Auslan Sign Language to Speech Translation* (Preet Prasad, The University of Western Australia, 2026). The dissertation addresses automatic translation from Auslan, the language of the Australian Deaf community, into English. The training data is the Auslan-Daily database, which has a small training set and no gloss sequences. The system has four stages:

1. The DWPose estimator extracts whole-body keypoints.
2. A gloss-free Transformer encoder–decoder, trained with an auxiliary pseudo-gloss CTC loss, translates the keypoints into English.
3. A decode-time ensemble averages the log-probabilities of six such translators.
4. A speech synthesiser voices a translation only when the translator's cross-entropy for its own output, which serves as a confidence score, is below a threshold fixed on validation data. Otherwise the system asks the signer to sign again.

On the 800-clip official Communication test set of Auslan-Daily, the final ensemble scores 11.80 BLEU-4. Published translators that read body-pose keypoints score 9.20 to 9.95, and the best of them, GASLT, scores 9.95. The best single-model configuration chosen on validation scores 10.97, which is the mean of three runs with different seeds. The dissertation also reports two results whose generating code is not in this release. Fine-tuning a released model that its authors pre-trained on 984 hours of American signing scores 18.21 BLEU-4 on the same test set (mean of five seeds) and 20.60 as an ensemble of five; this is a different kind of system, because its weights are licensed for non-commercial use only and it is not trained on Auslan-Daily alone. On the signer's AlphaPose keypoints, the authors' own SL-Transformer and GASLT code scored 8.77 and 9.16 (published 9.20 and 9.95; three seeds). The saved hypotheses behind both, and this repository's own scoring of them, are included and re-derived by checks A36 to A38.

Two qualifiers apply to these results:

- The reported system is not pose-only. It is an ensemble of three pose-only translators and three translators that also use RGB image features.
- Generalisation is lower than the headline score suggests. On the 570 test clips whose reference does not occur verbatim in the training set (unseen targets), the ensemble scores 6.60 BLEU-4. In addition, paired-bootstrap 95% intervals of about ±1.2 BLEU-4 include zero for most individual improvements.

## Repository layout

The code folder `src/` was called `phase2_baseline/` on the UWA Kaya cluster. The job scripts in `jobs/slurm/` record the cluster paths that actually ran, so they still say `phase2_baseline`; read that as `src/`. Each of `src/`, `jobs/slurm/` (which also covers `jobs/local/`), `pose/` and `results/` has its own README that lists every script.

| Path | Contents |
|---|---|
| `Makefile` | `make setup`, `make data`, `make headline`, `make reproduce`, `make test` and `make smoke`. |
| `ATTRIBUTION.md` | Sources, citations and licences of the two datasets, and how each is used. |
| `REPRODUCING.md` | Every check behind the reported numbers: the command, the recorded output and what the check needs. |
| `requirements.txt` | Pinned Python packages for training, evaluation, tests and figures. |
| `src/config.py` | Shared configuration. The trainers expose every field as a command-line flag. |
| `src/slt/` | Package containing data loading and normalisation, model definitions, the training and decoding engine, metrics, RGB fusion, contrastive pretraining and speech output. |
| `src/train_ctc.py` | Translator trainer with the auxiliary pseudo-gloss CTC loss. It also trains the RGB-fusion model when `--rgb_feat_dir` is given. |
| `src/pretrain_vlp.py` | Contrastive pose–text pretraining of the pose encoder. |
| `src/ensemble_decode.py` | Decode-time ensemble that averages per-step log-probabilities across pose-only and RGB-fusion members. |
| `src/tools/` | Manifest building, synthetic-corpus construction, evaluation, statistical decision tools, diagnostic probes, and speech and latency benchmarks. |
| `jobs/slurm/` | SLURM job scripts and shared shell libraries used on the Kaya cluster, kept with the cluster paths that ran. |
| `jobs/local/` | Shell scripts for runs outside the cluster (pose-MAE, GloFE, GASLT recipe, ISLR transfer and SignJoey pretraining runs). |
| `pose/` | DWPose and RGB feature extraction, person tracking, person-switch audits, person labels and signing-activity analysis. |
| `results/` | Saved outputs of the reported systems and experiments (decoded hypotheses with their reference sentences, per-seed summaries) and the read-out scripts that compute the reported numbers from them, in directories named after their content. The table below connects each directory to the dissertation and to the cluster job numbers. `results/checks/` holds one folder per verified claim (see Reproducing). |
| `figures/` | Scripts that generate the dissertation figures. |
| `demo/` | Live demonstrator: local translator service, static website and in-browser engine, with its own tests and validation harnesses. |
| `tests/` | Repository-level tests of metric conventions and safety guards. |

## Where each result lives

Folder names in `results/` describe their content. The last column gives the cluster job numbers of the runs that produced the saved outputs, so that a job script in `jobs/slurm/` and a log entry can be matched. Job scripts record the cluster paths that actually ran (the code folder was then called `phase2_baseline`; it is `src/` here).

| Folder | What it holds | Dissertation | Cluster job numbers |
|---|---|---|---|
| `results/per_clip_quality_analysis` | Per-clip scoring of two models against pose quality and reference; early search for causes of the score plateau. | Background to Results sec:res_errors (no table of its own) | none (local) |
| `results/ensemble_three_duration_members_decodes` | Decoded hypotheses of the three-member pose-only duration-stitch ensemble (11.54 on test). | Discussion and Results, unclaimed ensembles (sec:res_bootstrap); tab:app_additional | 1191189 (validation sweep); test decode run locally |
| `results/ensemble_three_duration_two_fusion_decodes` | Decodes of the retracted five-member ensemble (12.11 on test). | sec:res_bootstrap, tab:res_bootstrap (not claimed) | 1191189, 1196605; test decode run locally |
| `results/ensemble_one_to_one_duration_members_decodes` | Decodes of the ensemble whose duration members use the 1.1:1 mixing ratio (12.84 on test, not claimed). | sec:res_bootstrap, tab:res_bootstrap | test decode run locally |
| `results/ensemble_six_duration_two_fusion_decodes` | Decodes of the validation-best ensemble that scored lower on test (11.56). | sec:res_bootstrap, tab:res_bootstrap | 1197719; test decode run locally |
| `results/final_ensemble_member_test_decodes` | Test decodes of each of the six final-ensemble members alone. | tab:res_ens_ablation, fig:res_ens_members | 1101821, 1125067 (ensemble decoding) |
| `results/final_ensemble_test_decodes` | Test decodes of the six-member final ensemble at several length penalties (11.80 at 0.3); basis of the headline score. | tab:res_ladder, tab:res_bootstrap, tab:res_seen_unseen | 1101828, 1125067, 7921 |
| `results/final_ensemble_validation_decodes` | Validation decodes of the final ensemble (12.470); thresholds for selective prediction are fitted on these. | tab:res_selective_ens, tab:res_ens_ablation | 1101821, 1101909, 8001 |
| `results/dwpose_signer_tracking_check` | Measures which person the stored DWPose keypoints follow against the released signer track. | subsec:res_estimator_signer, subsec:res_poseswitch | none (local) |
| `results/dwpose_signer_guided_selection` | Measures DWPose chosen to match the annotated signer track against first-detection and tracked pose. | subsec:res_poseswitch | none (local) |
| `results/beam5_validation_selector_check` | Compares beam-search and greedy validation as selectors of the better training arm. | sec:method-protocol / sec:res_protocol (selection rule) | none (local; uses logs of 1196567, 1199393, 1215231) |
| `results/ensemble_member_subsets_ablation` | Decodes of all 63 member subsets of the final ensemble, with members alone, leave-one-out and Shapley values. | tab:res_ens_ablation | 120153, 120154 |
| `results/news_decoding_length_check` | Brevity penalty and length ratio of the News system before and after revised decoding. | Results News paragraph; tab:app_additional | 27382 |
| `results/news_stage2_decision` | Second-stage adoption rule for the News recipe (stitched data plus augmentation). | tab:res_negatives / News paragraph in sec:res_ceiling | 7950 (stage 1), 16398 (stage 2) |
| `results/pose_augmentation_decodes` | Beam-5 test decodes of pose-augmented and control models (three seeds) and the pooled bootstrap. | subsec:method-poseaug, tab:res_ladder | 1215231 |
| `results/semantic_adequacy_grading` | Test hypotheses of three models and a local language model's adequacy grades. | subsec:res_adequacy | 1075705, 1075949, 10288 |
| `results/stitched_corpus_audit` | Counts for the stitched corpus: sentences, signs, recordings and overlap. | sec:method-stitch; REPRODUCING.md A31 | 1216905 |
| `results/train_target_hashes.txt` | SHA-1 digests of the training targets, used in place of the subtitles. | sec:res_measurement (seen and unseen targets) | none |
| `results/stitch_trim_proxies` | Join-gap, velocity and pause statistics of stitched corpora with and without trimming. | tab:res_negatives (trimming row) | 25865 (and 7948 corpora) |
| `results/video_disjoint_cross_modal_ensemble` | Ensemble of pose and appearance models on the video-disjoint split. | tab:app_additional (video-disjoint rows) | 31204, 31205 |
| `results/person_tracking_stage2` | Five-seed test of tracked against first-detection keypoints. | subsec:res_poseswitch | 6591, 6592, 9444 |
| `results/video_disjoint_recipe_confirmation` | Plain baseline against the adopted recipe on the video-disjoint split, five seeds. | sec:res_measurement (video-disjoint split) | 6596, 7965 |
| `results/estimator_signer_keypoints_rerun` | Estimator comparison re-run on the signer's own AlphaPose keypoints (three seeds, with hypothesis dumps). | tab:res_estimator, fig:res_estimator, subsec:res_estimator_signer | 196406, 196407 |
| `results/run_to_run_noise` | Six replicate runs at one seed and the cross-batch noise of the adopted recipe. | sec:res_reproducibility (appendix app:moved), sec:method-protocol | 7952 |
| `results/validation_selected_pretraining_rerun` | Pre-training length re-run with selection on validation only (five pre-training lengths, three seeds). | fig:res_pretrain_rerun | 116619, 116620 |
| `results/hand_masking_and_longer_training` | Hand masking and 60-epoch training against the adopted recipe (three seeds). | tab:res_negatives | 1216919 |
| `results/letter_fix_and_longer_training_stage2` | Fingerspelled-letter fix and the second stage of the 60-epoch test. | tab:res_negatives | 5402, 5403, 7966 |
| `results/pretraining_selection_audit` | Audit of the extra-pre-training gain by selection artefact and seen and unseen targets. | tab:res_negatives (synthetic pre-training rows) | 12629, 16397 (inputs) |
| `results/wider_corpus_joint_training` | Wider synthetic corpus in joint training (two stages). | tab:res_negatives | 16435, 25863 |
| `results/sign_density_stage1` | Sign-density arms, first stage and futility rule. | tab:res_negatives | 31219, 31220 |
| `results/sign_density_stage2` | Sign-density arms, second stage with fresh seeds and a length-matched arm. | tab:res_negatives | 66643, 66644 |
| `results/real_data_learning_curve` | Real-only training at three data sizes and a whole-video arm. | subsec:res_learning_curve, tab:res_learning_curve, fig:res_learning_plot | 21734 |
| `results/tracker_v2_translation_test` | Translation test of the second-version tracker against first-detection pose. | subsec:res_poseswitch | 31736, 43506 |

### Verified claim checks (`results/checks/`)

Each folder holds the check script, the saved data it reads and the dissertation sentence it verifies. The check column gives its section in `REPRODUCING.md`, and `make reproduce` runs them all.

| Folder | Check | What it checks | Dissertation | Cluster job numbers |
|---|---|---|---|---|
| `results/checks/pretrained_ensemble` | A36 | Five fine-tuned pre-trained models and their ensemble (20.60 test); own-code three-model ensemble. | sec:res_transfer, tab:app_transfer, tab:app_transfer_diff | 193342, 195359-195362, 238134-238139 |
| `results/checks/clean_room_reimplementation` | A37 | Independent implementation reproducing the private port's logits and own-code fine-tuning. | sec:res_transfer, tab:app_transfer | 167035, 175088, 186422, 193342, 195640 |
| `results/checks/published_systems_own_code` | A38 | Published SL-Transformer and GASLT code run on the signer's AlphaPose and on DWPose. | sec:res_published_cmp, tab:res_published, tab:app_ownccode | 167758, 179411, 197330-197332 |
| `results/checks/rest_cut_stitching` | A39 | Cutting hand-rest frames from stitched signs (two depths, five seeds). | tab:app_additional; Results stitching discussion | 174914-174917 |
| `results/checks/keypoints_69_input` | A40 | 69-keypoint input, graph front ends and the tracked-keypoint check. | tab:app_additional (69-keypoint rows), tab:res_negatives | 168264-168267, 171802-171806, 172447, 172448, 179415, 179416 |
| `results/checks/three_training_ideas` | A41 | Sign targets, re-timing and preference stage, three then five seeds. | tab:app_additional | 144829-144832, 152915, 152916 |
| `results/checks/extra_pretraining_cycle` | A42 | Extra first training cycle with fresh seeds, and a synthetic-only cycle. | tab:app_additional | 138389, 138390, 140946, 140947 |
| `results/checks/news_stitching_vs_augmentation` | A43 | News bundle split into stitched data and pose augmentation. | tab:app_additional (News rows) | 138375-138378, 172449, 195461 |
| `results/checks/nonsigner_confidence_gate` | A45 | Confidence and grounding score on non-signing people, single model and ensemble. | sec:res_nonsigner, fig:res_nonsigner_plot | 138384, 138385, 171529, 171530 |
| `results/checks/ensemble_averaging_rule` | A44 | Power-mean averaging of ensemble members against the geometric mean. | tab:app_additional; sec:res_bootstrap | 120153, 138382, 138383 |
| `results/checks/speech_chain` | A46 | Latency by stage and voiced output by coverage for the speech chain. | sec:res_speech, tab:res_speech_latency, fig:res_latency_plot, tab:res_speech_coverage | none (local) |
| `results/checks/rouge_convention` | A47 | ROUGE-L tokenisation convention table. | tab:res_rouge_tokenisation | none (local) |

## File-name prefixes

Script names keep the prefix of the experiment that they belong to, because the job scripts and the pre-registration records use these names. `w` followed by a number is the numbered experiment round; `12a` to `12i` are the follow-up experiments of round 12.

| Prefix | Experiment | Dissertation | Cluster job numbers |
|---|---|---|---|
| `w4_` | Training-data distribution arms (trim, realisations) and News stitched data | tab:res_negatives; News paragraph | 7948-7952, 16398, 25865 |
| `w5_` | Large synthetic pre-training; estimator re-runs; run-to-run noise | tab:res_negatives; tab:res_estimator; sec:res_reproducibility | 12627-12630, 16397, 7951, 7952, 196406, 196407 |
| `w6_` | Wider synthetic corpus in joint training | tab:res_negatives | 16435, 25863 |
| `w7_ / w7s2_` | Sign density, stages 1 and 2 | tab:res_negatives | 20075, 31220, 66643, 66644 |
| `w8_` | Real-data learning curve | tab:res_learning_curve | 21734 |
| `w9_` | Second-version person tracker, translation test | subsec:res_poseswitch | 31736, 43506 |
| `w10_` | Sign order: shuffled signs with and without matching targets | tab:app_additional | 49134-49136 |
| `w11_` | Validation-selected pre-training length and News frame limit | fig:res_pretrain_rerun | 116619, 116620 |
| `w12a_` | News: stitched data against pose augmentation | tab:app_additional | 138375-138378, 172449, 195461 |
| `w12b_` | Wrong-sense signs in the stitched corpus | tab:app_additional | 138379-138381 |
| `w12d_` | Ensemble averaging rule (power mean) | tab:app_additional | 138382, 138383 |
| `w12e_` | Non-signer gate on the final ensemble | sec:res_nonsigner | 138384, 138385 |
| `w12f_` | Pose measurements: isolated signs and estimated depth | tab:disc_litgap; tab:app_additional | 140890-140892 |
| `w12g_` | Signer-disjoint split | tab:app_additional | 138386-138388 |
| `w12h_` | Extra pre-training cycle with fresh seeds | tab:app_additional | 138389, 138390, 140946, 140947 |
| `w12i_` | Tracker retest on heavy-switch clips | subsec:res_poseswitch; tab:app_additional | 138391, 138392 |
| `wave1_ / wave2_` | Hand masking, 60 epochs, fingerspelled-letter fix | tab:res_negatives | 1216919, 5402, 5403, 7966 |
| `wG_` | Signer-guided DWPose selection | subsec:res_poseswitch | none (local) |
| `apsig_` | Re-extraction of the released AlphaPose keypoints for the signer | subsec:res_estimator_signer | 196406 |
| `t1_ / t2_ / t3_` | Research-queue items: selective prediction on the final ensemble, video-disjoint ensembling, detection-count export | tab:res_selective_ens; tab:app_additional; subsec:res_poseswitch | 8001; 8117, 10674; 8050 |
| `tier1` | Weight-averaging seeds and beam and length-penalty retune | tab:res_negatives | 1081983, 1082033 |
| `posestitch_ / stitch_ / dur_ / temporal_` | Stitched synthetic data and its duration, ratio and scramble tests | tab:res_temporal, tab:res_ratio | 1170751, 1173725, 1175390, 1175508, 1193231, 1196567 |
| `news_` | News-subset training, decoding and pre-transfer | News paragraphs; tab:res_negatives | 1191190, 1196181, 1196431 |
| `islr_ / mmwl_` | Isolated-sign pre-training on MM-WLAuslan | tab:res_negatives | 1076388, 1076436 |
| `ensemble_ / mbr_ / ens_ / swa_` | Ensemble decoding, length penalty, minimum-Bayes-risk and subset ablation | tab:res_ladder, tab:res_ens_ablation | 1101821, 1125067, 120153 |
| `dwpose_ / alphapose_ / extract_ / estimator_ / mmpose_ / face_` | Pose extraction and estimator ablation | sec:res_input, tab:res_estimator | 1101249, 1101250 |
| `graph_ / ctc_ / gated_ / rgb_ / vlp_ / mbart_ / llm_ / vq_ / grpo_` | Model and objective variants (negative results) | tab:res_negatives | see `jobs/slurm/README.md` |

## What is and is not included

All code is included, together with the saved outputs needed to re-derive the reported numbers without a GPU (`results/`): decoded hypotheses and per-seed summary files. The subtitles of the datasets are not reproduced in the repository. In the saved outputs, each reference sentence is replaced by its SHA-1 digest. `make data` fetches the Auslan-Daily translation spreadsheet from the dataset's official Google Drive folder and restores the sentences from it. `results/train_target_hashes.txt` holds digests of the training targets, from which the seen-target and unseen-target scores are computed. `ATTRIBUTION.md` gives the sources, citations and licences of both datasets.

The following material is not included:

- **The datasets.** Auslan-Daily and MM-WLAuslan video, pose and subtitle files are not redistributed. The datasets are distributed by their authors under their own licence terms, and the recordings show identifiable signers.
- **Material derived from the datasets.** This covers extracted pose files, RGB features, clip-to-reference manifests, example clips and test fixtures. Derived files carry the same signer and licence considerations as the source data. `demo/tests/fixtures/` therefore contains only the scripts that generate the fixtures.
- **Code that stays private.** The code that fine-tuned a released pre-trained model, the authors' own code of the published systems that were re-run, and an independent re-implementation of the pre-trained model are not published here. Their saved hypotheses, logits and per-seed summaries are included under `results/checks/`, and the checks A36 to A38 re-score them with this repository's own code.
- **Model checkpoints.** They are not included because of their size. The weights used by the live demonstrator are published separately (see Licence).
- **The project's private laboratory notebook.** Comments in the code that cite dated entries such as "09-10m", or that cite `CLAUDE.md`, refer to this notebook. References to pre-registration documents, and to result folders that are absent from `results/`, also point to material that is not part of this repository.

Some paths in the code refer to machines rather than to this repository:

- Paths beginning `/group/pmc099` or `/scratch/pmc099` are locations on the UWA Kaya cluster.
- Paths beginning `/Users/` were on the author's laptop; the build replaces them with repository-relative paths, so a few scripts that read the dissertation source (the figure scripts) expect it in a folder named `SignLanguage_Preet` beside this repository.
- Where a job script calls an extraction worker under `results/dwpose_experiment/` on the cluster, the same script is in `pose/extraction/` in this repository.

## Requirements and installation

The code was run locally on CPython 3.14.3. The file `requirements.txt` pins the following packages:

| Package | Version |
|---|---|
| sacrebleu | 2.6.0 |
| jiwer | 4.0.0 |
| numpy | 2.5.3 |
| scipy | 1.17.1 |
| pytest | 9.0.3 |
| matplotlib | 3.10.8 |
| torch | 2.11.0 |

Install these pins with pip into a fresh environment.

The pose-extraction and demonstrator stack is listed separately in `demo/requirements.txt`. It comprises fastapi, uvicorn, onnxruntime, rtmlib, opencv-python, pillow, transformers, and the same metric and PyTorch packages. RGB feature extraction loads DINOv2-small through transformers. The demonstrator never downloads model assets, so these must be cached in advance.

On the cluster, each run used one NVIDIA V100 GPU of the University of Western Australia's Kaya cluster, with PyTorch 2.5.1 and CUDA 12. A 40-epoch Communication run took roughly 35–45 minutes. Throughput and latency were measured on the CPU of an Apple M3 Max laptop.

## Reproducing the reported numbers

The reported numbers fall into two groups, which `REPRODUCING.md` sets out check by check.

- **From the saved outputs and the dataset spreadsheet, with no GPU.** `make reproduce` runs all 47 checks of this kind (A1 to A47) and compares each output with the output recorded in `REPRODUCING.md`. Not every number of the dissertation has a check: `make reproduce` passes when every existing check reproduces, and does not fail because a claim has none. The checks cover the reported system's 11.80 BLEU-4 and 0.907 word error rate, its seen-target and unseen-target scores (35.00 and 6.60), the single models, the paired-bootstrap intervals, the multi-seed decisions, the selective-prediction results, the analysis of the ensemble and the dataset statistics. A full run on a laptop took about 37 minutes (30 October 2026: 47 reproduced, 0 failed).
- **By decoding or training.** Numbers that need pose files, checkpoints or training runs are listed at the end of `REPRODUCING.md` with the script or job script that produces each.

```bash
make setup                  # creates .venv and installs requirements.txt
make data REMOTE=gdrive     # fetches the Auslan-Daily spreadsheet with rclone and restores the reference sentences
make headline               # the reported system's scores and the decision rule (seconds)
make reproduce              # all 47 checks
make test                   # runs the tests
make smoke                  # trains and decodes on synthetic clips
```

`make data` needs [rclone](https://rclone.org) with a Google Drive remote. `rclone config` creates one (storage type `drive`, any Google account, read-only access is enough); its name is passed as `REMOTE`. The target copies `Auslan-Daily_Communication.xlsx` from the dataset's official folder, which is public, into `data/auslan-daily/translation/`. It then runs `make references`, which restores the reference sentences in `results/`, verifies each restored file against `results/SHA256SUMS`, and writes text manifests to `comm_run/manifests/`. Without rclone, the spreadsheet can be downloaded by hand from the official folder into the same location, followed by `make references`. Both locations are ignored by git, and the restored outputs should not be committed.

Re-scoring a saved decode shows that a reported number is what the saved output contains. It does not re-run inference.

## Data

Two datasets are required. Both are openly available and required no data-use agreement. Neither is redistributed here; `ATTRIBUTION.md` gives their sources, citations and licences. `make data` fetches the one file that the no-GPU checks need, as described above.

- **Auslan-Daily.**
  - Project page: <https://uq-cvlab.github.io/Auslan-Daily-Dataset/>
  - Release: <https://drive.google.com/drive/folders/17E5wgq1ig7-WynNskG-8tTMoOnGlVh4V>
  - Required files: `Auslan-Daily_Communication.xlsx` (clip identifiers, English subtitles and the official split) and `Communication/Signer Only Video Clip/Signer.zip` (signer video).
  - The keypoints shipped with the dataset are AlphaPose keypoints. The reported system does not use them; it re-extracts DWPose keypoints from the video, so reproduction requires the video. Each shipped file lists every person detected in a clip and marks one entry as the signer, so any extractor must take `Annotation['Signer']`. The original program of this project took the first-listed person, a non-signer in 88.4% of Communication clips; `tools/extract_clips.py` now takes the signer and stops if it is missing.
- **MM-WLAuslan.**
  - Release: <https://drive.google.com/drive/folders/1EQ1Nh3lidEcu1QLFw0IjRN7YqEq1N48q>
  - Required files: `<split>/Kinect_F/rgb.zip` and `Annotation/Labels+Split/{Train,Valid}.json`.
  - This dataset is used only as an isolated-sign lexicon (45,010 front-view clips of 3,215 glosses) for building synthetic sentences. No result is evaluated on it.

The main results use the official Communication split of 12,441 training, 800 validation and 800 test clips (14,041 in total). The manifests used for training hold 12,439 training clips, because two official training clips are absent from the built manifest.

The rows of `Sign_Spotting.xlsx` and `Fingerspelling_Detection.xlsx` that are labelled as training include Communication validation and test clips. Intersect these spreadsheets with the training manifest rather than filtering on their own split column.

Preparation proceeds in the following order:

1. **Fetch and unzip the signer video.** `jobs/slurm/data_prep.sbatch` copies `Signer.zip` with rclone and unzips it.
2. **Extract DWPose keypoints.** `pose/extraction/dwpose_extract_worker.py` is driven by `jobs/slurm/extract_dwpose.sbatch` (see Running the code, step b). Each output `.npz` file holds `KPS`, the COCO-WholeBody keypoints as (x, y, confidence), and `BBOX`, which is all zeros. Because `BBOX` is empty, every model is trained with `--coord_norm frame`. MM-WLAuslan clips are processed by the same worker, with the `CLIP_SUFFIX` environment variable set to the dataset's file-name suffix.
3. **Build the manifests.** `tools/build_manifest.py` writes plain tab-separated manifests of the form `<path to .npz><TAB><English reference>`. Use the official `Split` column and do not use `--val_frac`. Otherwise the scores are not comparable with the published baselines.

   ```bash
   cd src
   python tools/build_manifest.py \
     --xlsx Auslan-Daily_Communication.xlsx \
     --pose_root <DATA_ROOT>/dwpose_clips \
     --out_dir <DATA_ROOT>/manifests
   ```

4. **Extract RGB features.** These features are needed only for the RGB-fusion members of the ensemble. `pose/extraction/extract_rgb_feats.py` is driven by `jobs/slurm/rgb_extract.sbatch`. It crops the full signer, left hand and right hand using the DWPose keypoints, encodes each crop with a frozen DINOv2-small model, and writes one `rgb_<clip>.npz` file per clip with the key `FEATURES`.

   ```bash
   export RGB_DIR=<DATA_ROOT>/rgb_clips
   export DWPOSE_DIR=<DATA_ROOT>/dwpose_clips
   export OUT_DIR=<DATA_ROOT>/rgb_feats
   python pose/extraction/extract_rgb_feats.py --shard 0 --nshards 1 --frames 24
   ```

## Running the code

Unless stated otherwise, the commands below are run from `src/`. Placeholders in angle brackets stand for locations chosen by the user.

### (a) Smoke test

The smoke test needs no data and no GPU. It generates synthetic clips, passes them through the real normaliser, trains the baseline Transformer for a few epochs, decodes, and computes BLEU-4 and WER. A pass shows that the pipeline is wired correctly; it does not validate research performance.

```bash
cd src
python smoke_test.py
```

### (b) Pose extraction

The worker runs RTMLib's Wholebody model in `balanced` mode, which pairs a YOLOX-m detector with RTMW-DW-x-l, through ONNX Runtime. It writes one `.npz` file per clip and skips clips that have already been written. It keeps the first detection in each frame and does not track people across frames. Tracked variants are discussed in the results table below. This step requires the Auslan-Daily video, which is not distributed.

```bash
export RGB_DIR=<DATA_ROOT>/rgb_clips
export OUT_DIR=<DATA_ROOT>/dwpose_clips
python pose/extraction/dwpose_extract_worker.py --shard 0 --nshards 1 --device cpu --mode balanced
```

### (c) Training one translator with the adopted recipe

The adopted training configuration combines two changes:

- 13,760 duration-modelled stitched clips are added to the real clips, giving the 1.1:1 corpus `joint_13760.tsv`.
- Pose augmentation is applied during training.

The command below is the one assembled by the job-script library `jobs/slurm/_w4lib.sh`. The library then performs two further steps:

- It averages the last five checkpoints with `weight_average.py`.
- It scores both the best checkpoint and the averaged checkpoint with beam search using `tools/eval_beam.py`.

`train_ctc.py` refuses flags that it would otherwise silently ignore. It also rejects pose augmentation when `--rgb_feat_dir` is set, because RGB-fusion training is never augmented. The members of the final ensemble were trained without synthetic data or augmentation. This step requires the manifests and pose files.

```bash
python train_ctc.py --train_manifest <DATA_ROOT>/stitch/joint_13760.tsv \
  --val_manifest <DATA_ROOT>/manifests/val.tsv --test_manifest <DATA_ROOT>/manifests/test.tsv \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --batch_size 64 --ctc_weight 0.3 --lr 5e-4 --lr_schedule cosine --warmup_steps 800 \
  --bleu_tokenize 13a --detokenize True --device cuda --num_workers 8 \
  --deterministic True --epochs 40 --seed 42 \
  --pose_aug True --aug_speed 0.2 --aug_rotate_deg 10.0 --aug_scale 0.1 \
  --swa_last_k 5 --out_dir <RUN_DIR>
python weight_average.py --ckpts <comma-separated swa_ep*.pt files> --out <RUN_DIR>/soup_last5.pt
python tools/eval_beam.py --ckpt <RUN_DIR>/best.pt --manifest <DATA_ROOT>/manifests/test.tsv \
  --beam 5 --device cuda --output_json <RUN_DIR>/beam5_test_best.json
```

### (d) Building the stitched synthetic corpus

The corpus is built in three stages:

1. **Sentence generation.** `tools/gen_stitch_sentences.py` uses `tools/constrained_gen.py` to generate the sentence pool with an instruction-tuned Qwen2.5 model. Generation is constrained so that every content word has a matching sign.
2. **Stitching.** `tools/build_posestitch_corpus_v2.py` stitches MM-WLAuslan citation clips into DWPose-format sequences. It resamples each sign to a duration drawn from the distribution of the Auslan-Daily sign-spotting instances.
3. **Mixing ratios.** `jobs/slurm/stitch_scale.sbatch` builds the full corpus once and takes prefixes of it, so that the smaller mixing ratios are nested subsets of the larger ones.

This step requires the DWPose-extracted MM-WLAuslan clips.

```bash
python tools/build_posestitch_corpus_v2.py \
  --labels <MM_ROOT>/labels/Train.json <MM_ROOT>/labels/Valid.json \
  --npz_dirs <MM_ROOT>/dwpose_clips/train <MM_ROOT>/dwpose_clips/valid \
  --sentences <SENTENCE_POOL>/sentences_constrained.txt --min_cov 0.9 --min_signs 3 --realizations 1 --seed 42 \
  --morphology --no_paren_alias --sign_dur --out_dir <OUT>/corpus --max_clips 20608
head -n 13760 <OUT>/corpus/posestitch_v2_train.tsv > <OUT>/syn_13760.tsv
cat <DATA_ROOT>/manifests/train.tsv <OUT>/syn_13760.tsv > <OUT>/joint_13760.tsv
```

### (e) Ensemble decoding

The headline decode script `jobs/slurm/mbr_wer_test2.sbatch` decodes the six-member ensemble with beam width 10 and evaluates the length penalties 0.2, 0.3 and 0.6. The reported operating point uses length penalty 0.3, which was selected on the validation set. The checkpoints must be passed in the recorded order:

1. `comm_dwpose_ctc_vlplong80`
2. `comm_dwpose_rgb_fusion`
3. `comm_dwpose_vlp_ctc`
4. `comm_dwpose_ctc_s202`
5. `comm_dwpose_fus_s101`
6. `comm_dwpose_fus_s202`

The decode stops with an error if the RGB features are missing. `ensemble_decode.py` provides the same member handling as a standalone decoder. This step requires the checkpoints and the RGB features.

```bash
python tools/ensemble_mbr_wer.py \
  --ckpts <ckpt1>,<ckpt2>,<ckpt3>,<ckpt4>,<ckpt5>,<ckpt6> \
  --num_keypoints 133 --keypoint_subset all --coord_norm frame \
  --rgb_feat_dir <DATA_ROOT>/rgb_feats --rgb_feature_dim 1152 --device cpu --beam 10 \
  --eval_manifest <DATA_ROOT>/manifests/test.tsv --lps 0.2,0.3,0.6
```

### (f) Significance tests and decision tools

Three tools support the statistical comparisons:

- **`tools/paired_bootstrap.py`** computes paired-bootstrap intervals and seen-target and unseen-target BLEU-4 from saved decode dumps. It uses 2,000 resamples of the test sentences, and both systems are rescored on each resample. The `--expect` option first checks that a dump reproduces a recorded score.
- **`tools/seed_decision.py`** prints per-seed paired differences and seed-level t-intervals.
- **`tools/holm_decision.py`** implements the staged decision rule: the stage-1 futility check, a one-sided paired t-test, and Holm correction over a pre-registered family.

The saved outputs under `results/` are sufficient for the commands below, and `REPRODUCING.md` lists the full set.

```bash
python tools/paired_bootstrap.py --system S6=../results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl \
  --system F=../results/ensemble_three_duration_two_fusion_decodes/hyps_LP=0.7.jsonl --pairs F:S6 \
  --train_hashes ../results/train_target_hashes.txt --expect S6=11.802
python tools/seed_decision.py --control w3C --arm w3T ../results/person_tracking_stage2/summary_s42.json \
  ../results/person_tracking_stage2/summary_s101.json ../results/person_tracking_stage2/summary_s202.json
python tools/holm_decision.py --self_test
python tools/holm_decision.py --reproduce
```

### (g) Speech output and latency

The speech and latency tools cover the following measurements:

- **End-to-end speech chain.** `tools/speech_pipeline_bench.py` times pose-to-English translation and English-to-speech synthesis on test clips, using the macOS `say` command with an Australian English voice. It also reports quality against coverage when the confidence gate is applied.
- **Speech round trip.** `tools/speech_roundtrip.py` runs the pre-registered intelligibility test in stages: build, synth, transcribe, g5sheet and score.
- **Ensemble latency.** `tools/ensemble_latency.py` measures per-clip latency of the ensemble at batch size 1.
- **Pose throughput.** `tools/pose_throughput_bench.py` measures the throughput of the DWPose front end.

Speech synthesis requires macOS, and every measurement requires checkpoints and pose files.

```bash
python tools/speech_pipeline_bench.py --n 120
python tools/speech_roundtrip.py selftest
python tools/ensemble_latency.py --ckpts <six checkpoints, recorded order> --manifest <DATA_ROOT>/manifests/test.tsv \
  --rgb_feat_dir <DATA_ROOT>/rgb_feats --sizes 1,3,6 --beams 5,10 --lp 0.3 --expect <recorded BLEU-4>
```

### (h) The live demonstrator locally

The demonstrator runs in one of two modes:

- **Model mode.** `demo/run_demo.py` starts the local translator service. It loads one or both operating points: the validation-best single model and the six-model ensemble. Each operating point answers only when its confidence score is below a threshold fitted on validation data for the chosen coverage.
- **Showcase mode.** `--showcase` serves the pages without camera translation.

The service loads checkpoints from the paths listed in `demo/service.py` and checks each one against a SHA-256 hash. The weights are available from the Hugging Face repository named under Licence.

`demo/serve_static.py` serves the static site and the in-browser engine locally, with the same routes and headers as the deployed site. It expects the ONNX graphs written by `tools/export_live_onnx.py`.

```bash
python demo/run_demo.py --showcase
python demo/run_demo.py --models single,ensemble --coverage 0.5 --device cpu
python demo/serve_static.py
```

## Where each result in the dissertation comes from

All paths under `tools/`, `jobs/slurm/`, `slt/` and the trainers are relative to `src/`.

| Dissertation result | Code |
|---|---|
| Pose-estimator comparison on the signer's keypoints (DWPose minus AlphaPose +0.27, unresolved) and the first comparison on the first-listed person (5.58 to 10.03 BLEU-4, superseded); bounding-box format audit | `jobs/slurm/apsig_extract.sbatch`, `jobs/slurm/w5_estrep_signer.sbatch`, `results/estimator_signer_keypoints_rerun/`, `jobs/slurm/alphapose_ablation.sbatch`, `jobs/slurm/extract_dwpose.sbatch`, `pose/extraction/dwpose_extract_worker.py`, `tools/halpe_to_cwb133.py`, `tools/audit_bbox_format.py`, `pose/extraction/agreement_rtmlib.py` |
| Single models (baseline with auxiliary CTC loss; beam-search evaluation; checkpoint averaging) | `train_ctc.py`, `slt/model.py`, `slt/data.py`, `tools/eval_beam.py`, `weight_average.py`, `jobs/slurm/_w4lib.sh` |
| Pose–text pretraining | `pretrain_vlp.py`, `slt/vlp.py`, `jobs/slurm/dwpose_vlp.sbatch` |
| RGB-feature fusion | `pose/extraction/extract_rgb_feats.py`, `slt/rgb_fusion.py`, `jobs/slurm/rgb_extract.sbatch`, `jobs/slurm/dwpose_rgb_fusion.sbatch` |
| Stitched synthetic data, mixing ratio and compute-matched control | `tools/gen_stitch_sentences.py`, `tools/constrained_gen.py`, `tools/build_posestitch_corpus_v2.py`, `jobs/slurm/stitch_scale.sbatch` |
| Scrambled-sign control | `jobs/slurm/stitch_scramble.sbatch` (the `--scramble_glosses` option) |
| Pose augmentation | `config.py`, `slt/data.py`, `jobs/slurm/pose_aug.sbatch`, `results/pose_augmentation_decodes/pooled_bootstrap.py` |
| Ensemble decoding | `ensemble_decode.py`, `tools/ensemble_mbr_wer.py`, `jobs/slurm/mbr_wer_test2.sbatch` |
| Ensemble ablation over all member subsets | `tools/ensemble_subsets.py`, `jobs/slurm/ens_subsets.sbatch` |
| Paired bootstrap and multi-seed decision rule | `tools/paired_bootstrap.py`, `tools/seed_decision.py`, `tools/holm_decision.py`, `jobs/slurm/_w4lib.sh` (`w4summary`), `results/hand_masking_and_longer_training/stage1_decision.py` |
| Seen-target and unseen-target scores; metric conventions | `tools/paired_bootstrap.py`, `tools/arm_readouts.py`, `metric_suite.py`, `slt/metrics.py` |
| Real-data learning curve (unseen-target 3.15 to 5.00) | `tools/real_subsets.py`, `results/real_data_learning_curve/` |
| Sign-identification probe; content-word failure analysis | `tools/continuous_id_probe.py`, `tools/content_failure_analysis.py` |
| Selective prediction and confidence gate | `tools/selective_prediction.py`, `tools/ensemble_selective.py`, `demo/service.py` |
| Non-signer control | `tools/nonsigner_gate.py`, `pose/signing_activity/` |
| Speech round trip and latency | `tools/speech_roundtrip.py`, `tools/speech_pipeline_bench.py`, `slt/speech.py`, `tools/ensemble_latency.py`, `tools/pose_throughput_bench.py`, `bench_latency.py`, `jobs/slurm/bench_latency.sbatch` |
| Person tracking | `pose/extraction/dwpose_extract_worker_tracked.py`, `pose/extraction/test_tracking.py`, `pose/tracker_v2/`, `pose/person_switch_audit/`, `pose/person_labels/` |
| Alternatives tested and not part of the reported system | `train_mbart.py`, `train_llm.py`, `train_vq.py`, `train_vq_llm.py`, `train_grpo.py`, `train_temporal_ctc.py`, `train_temporal_pyramid.py`, `train_temporal_semantic_ctc.py`, `train_gated_temporal_ctc.py`, `train_face_temporal_ctc.py`, `train_presence.py`, `pretrain_pose_mae.py`, `pretrain_islr.py`, `pretrain_rgb_vlp.py`, and the scripts in `jobs/local/` |
| Figures | `figures/make_thesis_figures.py`, `figures/make_concept_figures.py`, `figures/make_failure_figure.py`, `figures/make_internals_figure.py`, `figures/make_uncertainty_figure.py`, `figures/extract_paper_figures.py`, `figures/figstyle.py` |

## The dissertation chapter by chapter

This section follows the dissertation's Results, Discussion and Appendix B. For each table, figure or claim it names the script in this repository that produces it and the check in `REPRODUCING.md` that re-derives it. "Dataset" means the Auslan-Daily files fetched by `make data` or the official release; "GPU" and "cluster" are as defined in `REPRODUCING.md`. Paths are relative to `src/` unless they begin with `results/`, `figures/` or `pose/`. The figure scripts expect the dissertation source in a folder named `SignLanguage_Preet` beside this repository, because several of them read the numbers they plot from its tables.

The dissertation also reports experiments of later waves. Where the code that produced them is public it is named below. For the transfer from a pre-trained model, the comparison with the published systems' own code and the independent re-implementation, the generating code is not in this release; the saved outputs and this repository's own scoring are, in `results/checks/`, and the rows below point to them. Three experiments of Appendix B (CTC-only pre-training, a masked pose autoencoder, a second tracker test) have neither code nor saved outputs in this release and are marked "not in this release".

### Results

| Table, figure or claim | Script | Check |
|---|---|---|
| Protocol, split sizes, verbatim overlap, metric conventions | `slt/metrics.py`, `metric_suite.py`, `tools/reproduce_headline.py` | A1, A14, A15, A16 |
| Pose estimators on one frame (`fig:res_pose_estimators`) | `figures/make_new_figures.py` | needs the dataset |
| Estimator comparison on the signer's keypoints (`tab:res_estimator`, `fig:res_estimator`) | `jobs/slurm/apsig_extract.sbatch`, `jobs/slurm/w5_estrep_signer.sbatch`, `figures/make_signerfix_figures.py`, `results/estimator_signer_keypoints_rerun/` | A32 (arithmetic), C2 (re-run on the cluster) |
| Which person the stored DWPose keypoints follow | `results/dwpose_signer_tracking_check/signer_check.py`, `results/dwpose_signer_guided_selection/wG_measure.py`, `tools/wG_signerguided.py` | needs the dataset and the all-detections pose |
| Hand identification probe (`fig:res_idprobe`, calibration of the estimator effect) | `tools/continuous_id_probe.py`, `tools/paired_probe_mcnemar.py`, `figures/make_thesis_figures.py` | B7, B8 |
| The ladder (`tab:res_ladder`, `fig:res_ladder`) | `tools/reproduce_headline.py`, `figures/make_signerfix_figures.py` | A1, A4 |
| Pre-training re-run (`fig:res_pretrain_rerun`) | `jobs/slurm/w11_vlpepochs.sbatch`, `tools/w11_readout.py`, `figures/make_extra_figures.py` | A35 |
| Duration-modelled stitching (`tab:res_temporal`, `fig:res_temporal`) | `tools/build_posestitch_corpus_v2.py`, `jobs/slurm/stitch_temporal.sbatch`, `jobs/slurm/stitch_dur_seeds.sbatch` | A31 (corpus audit), C4 |
| Mixing ratio (`tab:res_ratio`, `fig:res_ratio`) | `jobs/slurm/stitch_scale.sbatch`, `figures/make_thesis_figures.py` | C4 |
| Effect sizes with intervals (`fig:res_effects`) | `figures/make_signerfix_figures.py` | A5, A26, A32 |
| Learning curve (`tab:res_learning_curve`, `fig:res_learning_plot`) | `tools/real_subsets.py`, `tools/learning_curve.py`, `tools/learning_curve_mechanism.py`, `jobs/slurm/w8_realdata.sbatch` | cluster |
| Error analysis (`fig:failure_modes`, `fig:qualitative`, `fig:internals`) | `tools/content_failure_analysis.py`, `figures/make_failure_figure.py`, `figures/make_concept_figures.py`, `figures/make_internals_figure.py` | A13 |
| Published systems (`tab:res_published`) | literature values (Auslan-Daily benchmark paper); our earlier reimplementations are in `jobs/slurm/gaslt.sbatch` and `jobs/local/`; the authors' own code run with three seeds is scored in `results/checks/published_systems_own_code/` | A38; parameter counts of our models are A18 |
| Seen and unseen targets, per signer (`tab:res_seen_unseen`, `fig:res_seen`, `fig:res_signer_plot`) | `tools/paired_bootstrap.py`, `tools/arm_readouts.py`, `figures/make_thesis_figures.py`, `figures/make_extra_figures.py` | A3, A23, A24 |
| Bootstrap resolution (`tab:res_bootstrap`, `fig:res_bootstrap`) | `tools/paired_bootstrap.py`, `tools/holm_decision.py` | A5, A6, A7, A8, A9, A10 |
| Ensemble ablation over all member subsets (`tab:res_ens_ablation`); averaging rule of the members | `tools/ensemble_subsets.py`, `jobs/slurm/ens_subsets.sbatch`; `tools/w12d_powermean.py`, `results/checks/ensemble_averaging_rule/` | A34, A44 |
| Input-blind baselines and WER (`tab:res_null_models`, `fig:res_wer`) | `slt/metrics.py`, `figures/make_thesis_figures.py` | A19, A20, A22 |
| Identity switches and person tracking (`subsec:res_poseswitch`) | `pose/extraction/dwpose_extract_worker_tracked.py`, `pose/tracker_v2/`, `pose/person_switch_audit/` | A28 |
| Selective prediction (`tab:res_selective_ens`, `fig:res_selective`) | `tools/selective_prediction.py`, `tools/ensemble_selective.py`, `tools/selective_ci.py` | A30 |
| Ensemble cost and latency (`fig:res_ens_plot`, `tab:res_speech_latency`, `fig:res_latency_plot`) | `tools/ensemble_latency.py`, `tools/speech_pipeline_bench.py`, `figures/make_extra_figures.py`, `results/checks/speech_chain/` | A46, B5, B6 |
| Non-signer control (`fig:res_nonsigner_plot`) | `tools/nonsigner_gate.py`, `pose/signing_activity/`, `figures/make_extra_figures.py`, `results/checks/nonsigner_confidence_gate/` | A45, B10 |
| Transfer from a pre-trained model (`sec:res_transfer`) | saved decodes and scoring only: `results/checks/pretrained_ensemble/`, `results/checks/clean_room_reimplementation/`; the fine-tuning code is not in this release | A36, A37 |

### Discussion

| Table or claim | Script | Check |
|---|---|---|
| Summary of the research questions (`tab:disc_summary`) | derived from the Results rows above; no script | the checks named there |
| Comparison with the British Sign Language literature (`tab:disc_litgap`) | literature values; the Auslan-Daily rows come from `tools/continuous_id_probe.py` and the pose measurements `tools/w12f_probe.py` | B7 |
| Metric behaviour (brevity, constants, ROUGE-L convention) | `slt/metrics.py`, `metric_suite.py`, `tools/speech_roundtrip.py` | A19, A20, A21 |
| System properties (latency, confidence gate, speech) | `tools/speech_pipeline_bench.py`, `tools/ensemble_selective.py`, `demo/` | A30, B5, B6 |

### Appendix B

| Table, figure or claim | Script | Check |
|---|---|---|
| Annotation inventory (`tab:annotation-inventory`) | `tools/annotation_subsets.py` | needs the dataset |
| Negative results (`tab:res_negatives`) | one job script per row in `jobs/slurm/` (graph encoder, frequency-weighted CTC, BPE, R-Drop, paraphrase, regularisation sweep, temporal pooling, ISLR transfer, decoders, vector quantisation, GRPO, weight averaging, hand-local normalisation, test-time augmentation) | cluster; see `jobs/slurm/README.md` |
| ROUGE-L tokenisation (`tab:res_rouge_tokenisation`) | `metric_suite.py`, `slt/metrics.py`, `results/checks/rouge_convention/` | A21, A47 |
| Ensemble cost and speech coverage (`tab:res_ens_cost`, `tab:res_speech_coverage`) | `tools/ensemble_latency.py`, `tools/selective_prediction.py`, `results/checks/speech_chain/` | A30, A46, B5 |
| Speech round trip (`tab:res_speech_roundtrip`) | `tools/speech_roundtrip.py`, `tools/speak_clip.py` | needs the dataset and a local speech model |
| Seed spread (`fig:seed_spread`) | `figures/make_uncertainty_figure.py` | A25 |
| Further results (`tab:app_additional`), rows for the wave-12 follow-ups | News decomposition `tools/w12a_news_decomp_readout.py`; wrong-sense signs `tools/w12b_readout.py`; power-mean averaging `tools/w12d_powermean.py`; non-signer gate on the ensemble `tools/w12e_nonsigner_s6.py`; pose measurements `tools/w12f_probe.py`; signer-disjoint split `tools/w12g_signer_split.py`, `tools/w12g_readout.py`; extra pre-training `tools/w12h_readout.py`; tracker-v2 retest `tools/w12i_heavy_did.py`; the `w12*` job scripts in `jobs/slurm/`; saved outputs and checks in `results/checks/news_stitching_vs_augmentation/` (News decomposition), `results/checks/ensemble_averaging_rule/`, `results/checks/nonsigner_confidence_gate/` and `results/checks/extra_pretraining_cycle/` | A43, A44, A45, A42; the rest on the cluster |
| Further results, rows of the other later waves: a 69-keypoint input and its graph front ends, rest-cut stitching, three training ideas (sign targets, re-timing, preference stage) | saved outputs and checks only, in `results/checks/keypoints_69_input/`, `results/checks/rest_cut_stitching/` and `results/checks/three_training_ideas/`; the training code of these waves is not in this release | A40, A39, A41 |
| Further results, rows of the other later waves: CTC-only pre-training, a masked pose autoencoder, a second tracker test | not in this release | not in this release |
| Transfer from a pre-trained model; the published systems run with their own code (`app:transfer`, `tab:app_transfer`, `tab:app_ownccode`) | saved outputs and checks only: `results/checks/pretrained_ensemble/`, `results/checks/clean_room_reimplementation/`, `results/checks/published_systems_own_code/`; the code that produced them is not in this release | A36, A37, A38 |

## Tests

The directory `tests/` contains four test files:

- `test_guards.py` checks that the safety guards fire on invalid input and stay inactive on valid input. The guards cover strict boolean flags and the check that preprocessing settings match the checkpoints.
- `test_metrics.py` checks the metric conventions against the saved decode of the reported system. The tests that read saved decodes are skipped until `make data` has restored the reference sentences.
- `test_reproduction.py` checks that `tools/reproduce_headline.py` reproduces every recorded score and exits with an error when a recorded value is altered.
- `test_numpy_elision.py` checks that the installed NumPy does not have the temporary-elision fault of NumPy 1.x on Python 3.14, which can overwrite a named array inside a function.

The smoke test (`src/smoke_test.py`) runs without data.

The demonstrator has its own tests in `demo/tests/`. These are Node parity tests of the in-browser engine plus Python tests of the service. The Node tests are run from `demo/`. Most of them compare the engine with fixtures derived from dataset clips, which are not distributed; the scripts that generate the fixtures are in `demo/tests/fixtures/`.

```bash
pytest tests
cd demo && node --test tests/engine_*.test.mjs
```

Several tools have a built-in self-test:

- `tools/holm_decision.py --self_test`
- `tools/real_subsets.py --selftest` (needs the training manifest)
- The `selftest` stage of `tools/arm_readouts.py`, `tools/speech_roundtrip.py`, `tools/nonsigner_gate.py` and `pose/signing_activity/signing_activity.py`

## Use of AI tools

Generative AI (Claude, Anthropic) was used for brainstorming, refining structure and improving clarity of expression. It was also used for writing and checking the experiment and analysis code, running and checking analyses, searching for and summarising related work, and drafting and condensing the text. The final research direction, technical content, literature selection and wording were reviewed and verified by the student. Separately, a small language model (Qwen2.5-3B) is part of the method: it generates sentences for the synthetic training data and grades translation adequacy.

## Licence

The code is released under the MIT licence. The model weights for the live demonstrator are published separately at <https://huggingface.co/preetprasad/auslan-slt-live> under CC BY-NC-SA 4.0. The live demonstrator runs at <https://auslan-research.vercel.app>. The datasets are not covered by either licence and remain subject to their providers' terms.

## How to cite

```bibtex
@thesis{prasad2026glossfree,
  author      = {Preet Prasad},
  title       = {Investigating Gloss-Free Auslan Sign Language to Speech Translation},
  type        = {Bachelor of Advanced Computer Science (Honours) thesis},
  institution = {The University of Western Australia},
  year        = {2026},
  note        = {Supervisor: Dr Atif Mansoor},
  url         = {https://github.com/preetprasad/gloss-free-auslan-slt}
}
```

## Acknowledgements

The author thanks Shen et al., who created the Auslan-Daily and MM-WLAuslan datasets and made them openly available. The author also thanks the supervisor, Dr Atif Mansoor. Training and evaluation used the Kaya high-performance computing facility of The University of Western Australia.