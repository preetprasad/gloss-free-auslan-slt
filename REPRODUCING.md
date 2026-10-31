# Reproducing the results

This document lists the checks behind the empirical claims of the dissertation. For each check it gives the command and the output recorded when that command was run.

**Levels.** Each check belongs to one of the following levels.

- **A.** The check is re-derived on a laptop, without a GPU, from the saved outputs in `results/` and the Auslan-Daily translation spreadsheet. `make reproduce` runs all A checks. A32 and A36 to A83 live partly or wholly in `results/checks/`, one folder per check, each with its script, the saved data it reads and the dissertation sentence it verifies.
- **B and C.** These checks need the datasets and either a GPU (level B) or the cluster (level C). They are summarised in the two tables at the end of the document.

**How to run.** From the repository root, run:

```bash
make setup
make data REMOTE=<name of an rclone Google Drive remote>
make reproduce
```

- `make setup` creates `.venv` from `requirements.txt`, which holds the pinned packages.
- `make data` fetches `Auslan-Daily_Communication.xlsx` from the dataset's official Google Drive folder with rclone. It then puts the reference sentences back into the saved outputs, checks every restored file against `results/SHA256SUMS`, and writes the text manifests `comm_run/manifests/{train,val,test}.tsv`. The README describes the rclone remote. A spreadsheet downloaded by hand into `data/auslan-daily/translation/` serves equally; `make references` then does the rest.
- `make reproduce` runs `python src/tools/run_claims.py REPRODUCING.md`. That script executes every check and compares its output with the recorded output below. `make headline` runs A1, A10 and A11 only, in a few seconds.
- `make test` runs the tests.

Individual commands are run from `src/`.

The recorded outputs were produced on macOS with Python 3.14.3, sacreBLEU 2.6.0 and numpy 1.26.3, using the CPU only. `tools/holm_decision.py` emits a harmless `UserWarning` about the SciPy/NumPy version pair when it is run outside the `.venv` built by `make setup`. That warning does not affect the result and is omitted from the recorded outputs. Where a recorded output is abridged or wrapped for page width, the section says so. Checks A1 to A33 were run again on 9 October 2026 from a fresh clone, after `make setup` and `make data` (Python 3.14.3 and the package versions in `requirements.txt`), and every recorded number was reproduced; that run took 25 minutes on a laptop, most of it in the bootstrap checks A4 to A9 and A26. Checks A34 and A35 were added on 13 October 2026 and were run that day in a clean copy of the repository after `make data` (Python 3.12, numpy 1.26.3): every recorded number was reproduced, A34 in about 8 minutes, and the 55 tests passed. On 30 October 2026 all 47 checks (A1 to A47) were run in a clean copy of this build after `make data`: 47 reproduced, 0 failed, 0 skipped, in about 37 minutes. All 83 checks (A1 to A83) were run on 31 October 2026 from a fresh clone, after `make setup` with the dataset spreadsheet in place and `make references` (Python 3.14.3, numpy 2.5.3 and the versions in `requirements.txt`): 83 reproduced, 0 failed, about 42 minutes on a laptop, and the 55 tests and the smoke test passed. An earlier build of the same checks also reproduced all 83 on Python 3.12.1 with numpy 1.26.3.

**Coverage.** Not every number of the dissertation has a check here. `make reproduce` runs the checks that exist and passes when each of them reproduces its recorded output; it does not fail because some claim has no check. Examples of numbers that cannot be re-derived from saved outputs are in the level B and C tables, with the script that produces each; `CLAIM_INDEX.md` gives the status of every claim. The code that produced a few saved outputs (the fine-tuning of the released pre-trained model, the authors' published systems and the independent re-implementation) is not part of this release; their checks re-score the saved hypotheses and logits with this repository's own scoring code.

**Scope.** Re-scoring a saved decode shows that the reported number is what the saved output contains. It does not re-run inference.

**Saved outputs and the dataset.** `results/` holds the saved outputs: decoded hypotheses and per-seed summaries. The subtitles of Auslan-Daily are not reproduced in this repository. A check that needs a particular sentence names it by its SHA-1 digest. Each reference sentence in a saved output is replaced by its SHA-1 digest, and `make data` restores the sentences from the dataset's own spreadsheet. The dataset is the work of Shen et al. and is obtained from its authors' official folder; `ATTRIBUTION.md` gives the source, the citation and the licence.

**Seen and unseen subsets.** `results/train_target_hashes.txt` holds SHA-1 digests of the training targets, written by `tools/train_target_hashes.py`. The seen/unseen decompositions below are computed from these digests, which gives the same result as the training manifest.

## Summary of the A checks

| ID | What is checked | Runs with |
|---|---|---|
| A1 | Reported system: BLEU-4 **11.802**, WER **0.9069** | repository and `make data` |
| A2 | All six saved ensemble decodes re-score to their recorded values | repository and `make data` |
| A3 | seen 35.00 (n=230) / **unseen 6.60** (n=570) | repository and `make data` |
| A4 | Ladder: 9.522 / 10.272 / 10.335 / 11.802 | repository and `make data` |
| A5 | Only the ensembling step is resolvable: **+1.467 [+0.396, +2.563]** | repository and `make data` |
| A6 | Unseen-only intervals are wider; ordering preserved | repository and `make data` |
| A7 | **WITHDRAWN**: F = 12.114, +0.313 **[−0.887, +1.498]** | repository and `make data` |
| A8 | **NOT CLAIMED**: F′ = 12.836, +1.034 **[−0.102, +2.153]** | repository and `make data` |
| A9 | max-T over 4 looks: q = 2.406; F′ widens to [−0.363, +2.432]; bar ≳ +1.40 | repository and `make data` |
| A10 | Confirmatory family adopted **nothing** | repository and `make data` |
| A11 | The decision tool is self-testing | repository and `make data` |
| A12 | Wave-1 futility held against the project's interest; soup interval [+0.210, +0.612] | repository and `make data` |
| A13 | **66.92 %** of clips get zero content words; ordering excluded (18.796 vs 18.698) | repository and `make data` |
| A14 | Split sizes, vocabularies, and **28.75 %** verbatim test overlap | repository and `make data` |
| A15 | Detokenised test references average **5.144** words | repository and `make data` |
| A16 | detokenise→`13a` is an exact inverse: **0/800** round-trip mismatches | repository and `make data` |
| A17 | CTC targets: 12,439 clips, **2,988** symbols, **3.5992**/clip; stop lists 75 and 71 | repository and `make data` |
| A18 | Parameters **9.0481 M** / **10.9238 M** / **59.9155 M** | repository and `make data` |
| A19 | An input-blind constant scores **4.41** (1.09 unseen); 18 references are that sentence | repository and `make data` |
| A20 | WER is defeated by brevity: 3-word truncation **0.8999** beats the ensemble's 0.9069 | repository and `make data` |
| A21 | ROUGE-L moves ~19.5 points on tokenisation alone; β=1.2 trio 35.70/36.27/37.08 | repository and `make data` |
| A22 | The reported decode under-generates: length ratio **0.753**, 308 distinct, 44 exact | repository and `make data` |
| A23 | Signer 1 **14.39** / signer 2 **9.97** / all others pooled **3.12** | repository and `make data` |
| A24 | A signer-disjoint split can be built: 12,953 / 1,088, 19.3 % overlap | repository and `make data` |
| A25 | Measured noise: within-job sd **0.216** (±0.60); cross-batch sd **0.422** (±1.17) | repository and `make data` |
| A26 | Pose augmentation pooled **+0.695 [+0.048, +1.339]** | repository and `make data` |
| A27 | Video-disjoint, 5 seeds: **+0.939 [+0.155, +1.722]**, 5/5 | repository and `make data` |
| A28 | **NEGATIVE**: person tracking, 5-seed test **−0.057 [−0.658, +0.545]** | repository and `make data` |
| A29 | Ensembling works by token-level partial correctness (36 vs 13, novel clips) | repository and `make data` |
| A30 | Selective prediction: **19.07 @ 51.9 %** coverage, validation-fixed thresholds | repository and `make data` |
| A31 | Synthetic corpus: 0 train / 0 val / **1** test collision; 75.0 % novel trigrams | repository and `make data` |
| A32 | Estimator comparison on the signer's keypoints: DWPose minus AlphaPose **+0.272 [−1.027, +1.572]**, 2 of 3 seeds; arithmetic from the saved read-out | repository and `make data` |
| A33 | Output-side vocabulary work is excluded: **0.54 %** of test content tokens are unseen | repository and `make data` |
| A34 | Ensemble ablation over all 63 member subsets: Shapley values **1.87 to 2.39**; kinds differ by +0.56 [−0.57, +1.97] | repository and `make data` |
| A35 | Pretraining re-run selected on validation: 40 epochs minus none **−0.26 [−1.57, +1.06]** (validation) | repository and `make data` |
| A36 | Fine-tuned pre-trained model: five-model ensemble **20.60** (E\*) and 20.20 (E0) test BLEU-4; members 17.86; max-T intervals exclude zero | repository and `make data` |
| A37 | Independent implementation reproduces the pre-trained model's logits (about 2e-5) and own-code fine-tuning (18.27 test); stitched data plus augmentation stops at futility | repository and `make data` |
| A38 | Published systems' own code on the signer's AlphaPose: **8.77** and **9.16** (published 9.20 and 9.95); below this project's translator on DWPose | repository and `make data` |
| A39 | Rest-cut stitching: stitched clips drop the hands 4.68 times per clip against 0.25; test **+0.29** and **+0.49**, Holm rule rejects for both | repository and `make data` |
| A40 | 69-keypoint input, five seeds: validation **+0.94 [−0.28, +2.16]**, test +0.94 [−0.35, +2.23]; graph front ends and the tracked-keypoint check | repository and `make data` |
| A41 | Three training ideas (sign targets, re-timing, preference stage): none resolvable; the CTC head places 47.3 % of annotated spans correctly | repository and `make data` |
| A42 | Extra pre-training cycle with fresh seeds: test **−0.56 [−1.41, +0.28]**; not adopted | repository and `make data` |
| A43 | News: stitched data **+0.22**, augmentation **+0.31**, bundle **+0.53**; no part attributed | repository and `make data` |
| A44 | Averaging rule: no power mean is distinguishable from the geometric mean (−0.68, 0.00, +0.13, +0.23) | repository and `make data` |
| A45 | Non-signer control: single model voices **53.9 %** of non-signers and 53.5 % of signers; ensemble ratio **0.84 [0.75, 0.94]** | repository and `make data` |
| A46 | Speech chain: median delay **2.75 s**, pose estimation 71.7 %; voiced output 10.27 → **19.19** at half coverage | repository and `make data` |
| A47 | ROUGE-L tokenisation convention table | repository and `make data` |
| A48 | Citation versus conversational signs: clip lengths and durations | repository and `make data` |
| A49 | Dataset statistics | repository only |
| A50 | Real-clip lengths and frame rates behind the temporal-fidelity argument | repository and `make data` |
| A51 | Ensemble member-set ablation and member-set selection | repository only |
| A52 | Ensemble and translator decoding cost, pose front-end speed, GPU set-up | repository and `make data` |
| A53 | Why ensembling works: members, bootstrap, mechanism, MBR, seed ensembles | repository only |
| A54 | Early AlphaPose runs and the DWPose re-extraction | repository and `make data` |
| A55 | Estimator comparison on the signer's keypoints | repository and `make data` |
| A56 | The released Auslan-Daily pose files store HALPE-136 keypoints, shape (T, 136, 3) (26 body, 42 hand and 68 face keypoints), and list every detected person with one entry marked as the signer. | repository only |
| A57 | Content-word failure analysis | repository and `make data` |
| A58 | Figure settings and speed verdict | repository and `make data` |
| A59 | All 14,041 Communication clips were re-extracted with DWPose, giving a 399-dimensional frame vector (133 keypoints of x, y, confidence); the DWPose training list holds 12,439 of the 12,441 training clips (Methodology, Results). | repository and `make data` |
| A60 | Sign-identification probes and isolated-sign references | repository only |
| A61 | One five-word training sentence, output for every clip regardless of the video, scores 4.41 BLEU-4 on the 800-clip official test set and 1.09 on the 570 unseen-target clips; 18 test references are exactly that sentence. | repository and `make data` |
| A62 | Results ("Training-free measurements") states that neither eleven clip-level pose descriptors nor video identity predicts translation quality, and that each descriptor explains at most about 1% of variance. | repository and `make data` |
| A63 | Real-data learning curve (wave 8) | repository and `make data` |
| A64 | News: five-seed stitched data and pose augmentation, validation length penalty, signable recall, ensemble gate (wave 4) | repository only |
| A65 | Paired bootstrap, development ladder and the final ensemble against the published system | repository and `make data` |
| A66 | Pose augmentation replication, recipe and the 1.1:1 baseline | repository only |
| A67 | PoseStitch side experiments (sign order, wider vocabulary, function signs, letters, wrong-sense signs) | repository and `make data` |
| A68 | Transfer from a pre-trained sign-language model | repository and `make data` |
| A69 | The pose-text pre-training length and the News input-frame limit were re-chosen on validation alone (Method; Results, pre-training and News paragraphs; Discussion limitations). | repository and `make data` |
| A70 | CTC-only pseudo-gloss first stage against a compute-matched control (wave 23) | repository and `make data` |
| A71 | Published systems' own code and the project's translator with the published recipe | repository only |
| A72 | Training recipe, model size and decoding defaults | repository only |
| A73 | Run-to-run noise bands | repository and `make data` |
| A74 | Seen/unseen decomposition, split definition and per-signer gap | repository and `make data` |
| A75 | If the system answers only the clips it is most confident about (confidence = the model's own cross-entropy on its output, plus self-reliance), BLEU-4 of the answered clips rises. | repository only |
| A76 | Semantic adequacy grading | repository and `make data` |
| A77 | Signer-disjoint split | repository and `make data` |
| A78 | Speech round trip with Whisper small.en | repository and `make data` |
| A79 | Person tracking and the heavy-switch clips | repository and `make data` |
| A80 | The official split shares videos between train and test, so a video-disjoint split (11,341 / 1,444 / 1,254 clips from 73 / 9 / 8 videos) was built (Methodology, Results "Video-disjoint"). | repository and `make data` |
| A81 | Masked keypoint pre-training (wave 28): checks that ran, no scores quoted | repository and `make data` |
| A82 | Wave 4: stitched data without boundary trim, and two realisations of half the sentences | repository and `make data` |
| A83 | WER null floor, brevity trade and error decomposition | repository and `make data` |

## Level A checks

### A1 — Reported system score

**Claim.** The reported system scores BLEU-4 **11.802** and WER **0.9069** on the official 800-clip Auslan-Daily Communication test split, as stated in the dissertation.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 tools/reproduce_headline.py
```

```text
system                                                           BLEU-4      rec      WER     rec  n
--------------------------------------------------------------------------------------------------------
S6 ensemble (reported operating point, beam 10 / LP 0.3)         11.802   11.802   0.9069  0.9069  800
S6 ensemble, validation split                                    12.470   12.470   0.9164  0.9164  800
F' = 1.1:1 duration members + 2 RGB-fusion (NOT CLAIMED: CI vs   12.836   12.836   0.9213  0.9213  800
F = duration members + 2 RGB-fusion (RETRACTED 2026-09-10: not   12.114   12.114   0.9526  0.9526  800
G1 = 6 duration + 2 fusion (best on VALIDATION, worse on test    11.562   11.562   0.9706  0.9706  800
E = 3 duration members, pose-only                                11.537   11.537   0.9988  0.9988  800
--------------------------------------------------------------------------------------------------------
reported system decomposed by whether the reference occurs verbatim in training:
  seen   n= 230  BLEU-4  35.00   (recorded 35.00)
  unseen n= 570  BLEU-4   6.60   (recorded  6.60)  <- the honest number

OK: 6 system(s) reproduce their recorded BLEU-4 and WER exactly.
```

The command exits with code 0. The tool exits with a non-zero code on any mismatch, so it can serve as a continuous-integration gate.

**Scope.** This check re-scores a saved decode. It shows that the reported number is what the saved output contains; it does not re-run inference. Regenerating the decode from checkpoints is B4.

At full precision, computed directly with `slt/metrics`, the values are as follows:

- BLEU-4 **11.801735**
- WER **0.9069258809**
- n = 800
- signature `sacreBLEU|4-gram|tokenize=13a|detok=True|smooth=exp`

### A2 — All saved ensemble decodes re-score exactly

**Claim.** Every saved ensemble decode re-scores to its recorded BLEU-4 and WER. This includes the three decodes that scored above the reported system and were not claimed.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 tools/reproduce_headline.py
```

```text
system                                                           BLEU-4      rec      WER     rec  n
--------------------------------------------------------------------------------------------------------
S6 ensemble (reported operating point, beam 10 / LP 0.3)         11.802   11.802   0.9069  0.9069  800
S6 ensemble, validation split                                    12.470   12.470   0.9164  0.9164  800
F' = 1.1:1 duration members + 2 RGB-fusion (NOT CLAIMED: CI vs   12.836   12.836   0.9213  0.9213  800
F = duration members + 2 RGB-fusion (RETRACTED 2026-09-10: not   12.114   12.114   0.9526  0.9526  800
G1 = 6 duration + 2 fusion (best on VALIDATION, worse on test    11.562   11.562   0.9706  0.9706  800
E = 3 duration members, pose-only                                11.537   11.537   0.9988  0.9988  800
--------------------------------------------------------------------------------------------------------
reported system decomposed by whether the reference occurs verbatim in training:
  seen   n= 230  BLEU-4  35.00   (recorded 35.00)
  unseen n= 570  BLEU-4   6.60   (recorded  6.60)  <- the honest number

OK: 6 system(s) reproduce their recorded BLEU-4 and WER exactly.
```

The command and output are the same as for A1. Six systems give six exact matches, and the `OK:` line is printed.

### A3 — Seen and unseen decomposition

**Claim.** The reported system scores 35.00 on the 230 test clips whose reference occurs verbatim in training. It scores **6.60** on the 570 test clips whose reference does not. The figure of 6.60 is the one that measures generalisation.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 tools/reproduce_headline.py
```

```text
system                                                           BLEU-4      rec      WER     rec  n
--------------------------------------------------------------------------------------------------------
S6 ensemble (reported operating point, beam 10 / LP 0.3)         11.802   11.802   0.9069  0.9069  800
S6 ensemble, validation split                                    12.470   12.470   0.9164  0.9164  800
F' = 1.1:1 duration members + 2 RGB-fusion (NOT CLAIMED: CI vs   12.836   12.836   0.9213  0.9213  800
F = duration members + 2 RGB-fusion (RETRACTED 2026-09-10: not   12.114   12.114   0.9526  0.9526  800
G1 = 6 duration + 2 fusion (best on VALIDATION, worse on test    11.562   11.562   0.9706  0.9706  800
E = 3 duration members, pose-only                                11.537   11.537   0.9988  0.9988  800
--------------------------------------------------------------------------------------------------------
reported system decomposed by whether the reference occurs verbatim in training:
  seen   n= 230  BLEU-4  35.00   (recorded 35.00)
  unseen n= 570  BLEU-4   6.60   (recorded  6.60)  <- the honest number

OK: 6 system(s) reproduce their recorded BLEU-4 and WER exactly.
```

The two `seen`/`unseen` lines are computed from the digests in `results/train_target_hashes.txt`, which `tools/reproduce_headline.py` reads automatically.

### A4 — Development ladder

**Claim.** The development ladder runs as follows:

1. 9.522 (DWPose+CTC)
2. 10.272 (VLP-80)
3. 10.335 (RGB fusion)
4. 11.802 (S6 ensemble)

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 tools/paired_bootstrap.py \
  --system CTC=../results/semantic_adequacy_grading/hyps_dwpose_ctc.jsonl \
  --system VLP=../results/semantic_adequacy_grading/hyps_vlp.jsonl \
  --system RGB=../results/semantic_adequacy_grading/hyps_rgbfusion.jsonl \
  --system S6=../results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl \
  --pairs VLP:CTC,RGB:VLP,RGB:CTC,S6:RGB,S6:VLP,S6:CTC \
  --train_hashes ../results/train_target_hashes.txt \
  --expect S6=11.802
```

```text
test n=800  seen-in-train=230  unseen=570
CTC          BLEU-4   9.522  seen  29.72  unseen  4.88
VLP          BLEU-4  10.272  seen  29.91  unseen  5.31
RGB          BLEU-4  10.335  seen  30.22  unseen  5.81
S6           BLEU-4  11.802  seen  35.00  unseen  6.60
check: S6 reproduces the recorded 11.802
```

The output is abridged: the interval block printed by the same run is shown in A5. The runtime was 5 min 39 s, almost all of it spent inside sacreBLEU's 2,000 resamples.

`--expect` is a hard assertion. A wrong file or a changed metric aborts the run instead of printing a plausible number.

### A5 — Only the ensembling step is resolvable

**Claim.** Of the individual rungs of the ladder, **only ensembling is resolvable** at this sample size: +1.467 BLEU-4, 95 % CI [+0.396, +2.563]. Every single-model step has an interval containing zero. This includes the addition of appearance features (+0.063 [−1.054, +1.160]).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 tools/paired_bootstrap.py \
  --system CTC=../results/semantic_adequacy_grading/hyps_dwpose_ctc.jsonl \
  --system VLP=../results/semantic_adequacy_grading/hyps_vlp.jsonl \
  --system RGB=../results/semantic_adequacy_grading/hyps_rgbfusion.jsonl \
  --system S6=../results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl \
  --pairs VLP:CTC,RGB:VLP,RGB:CTC,S6:RGB,S6:VLP,S6:CTC \
  --train_hashes ../results/train_target_hashes.txt \
  --expect S6=11.802
```

```text
[all n=800] VLP vs CTC: observed +0.749  95% CI [-0.218, +1.708]  P(VLP>CTC) 93.2%  -> not resolvable
[all n=800] RGB vs VLP: observed +0.063  95% CI [-1.054, +1.160]  P(RGB>VLP) 54.3%  -> not resolvable
[all n=800] RGB vs CTC: observed +0.812  95% CI [-0.337, +1.970]  P(RGB>CTC) 91.5%  -> not resolvable
[all n=800] S6 vs RGB: observed +1.467  95% CI [+0.396, +2.563]  P(S6>RGB) 99.4%  -> significant
[all n=800] S6 vs VLP: observed +1.530  95% CI [+0.583, +2.523]  P(S6>VLP) 99.8%  -> significant
[all n=800] S6 vs CTC: observed +2.279  95% CI [+1.182, +3.382]  P(S6>CTC) 100.0%  -> significant
```

The command is the same as in A4. The output is abridged to the interval block; the header block of the same run is shown in A4. Every line matches the corresponding table in the dissertation exactly.

The method is as follows:

1. Resample the 800 test sentences with replacement.
2. Recompute **both** systems' corpus BLEU on the same resample.
3. Repeat for 2,000 draws using `default_rng(0)`.

The procedure is deterministic, so the numbers above are the numbers a re-run produces.

### A6 — Unseen-only intervals

**Claim.** On the 570 unseen-target clips the intervals are proportionally wider still, and the ladder's **ordering is preserved**. This is moderate evidence that the gains are real rather than memorisation; it is not a demonstration.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 tools/paired_bootstrap.py \
  --system CTC=../results/semantic_adequacy_grading/hyps_dwpose_ctc.jsonl \
  --system VLP=../results/semantic_adequacy_grading/hyps_vlp.jsonl \
  --system RGB=../results/semantic_adequacy_grading/hyps_rgbfusion.jsonl \
  --system S6=../results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl \
  --pairs VLP:CTC,RGB:VLP,S6:RGB,S6:CTC --subset unseen \
  --train_hashes ../results/train_target_hashes.txt --expect S6=11.802
```

```text
[unseen n=570] VLP vs CTC: observed +0.437  95% CI [-0.604, +1.489]  P(VLP>CTC) 78.2%  -> not resolvable
[unseen n=570] RGB vs VLP: observed +0.492  95% CI [-0.685, +1.604]  P(RGB>VLP) 80.2%  -> not resolvable
[unseen n=570] S6 vs RGB: observed +0.791  95% CI [-0.246, +1.762]  P(S6>RGB) 93.2%  -> not resolvable
[unseen n=570] S6 vs CTC: observed +1.720  95% CI [+0.433, +2.945]  P(S6>CTC) 99.6%  -> significant
```

The unseen subset is defined by the digests. The output is abridged: the command first reprints the four-system header block shown in A4.

### A7 — Withdrawn five-member configuration

**Claim — WITHDRAWN.** A five-member configuration scored **12.114**, above the reported 11.802, and was initially recorded as an improvement. Its paired bootstrap returned **+0.313 [−0.887, +1.498]** with P = 69.3 %, and the claim was retracted.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 tools/paired_bootstrap.py \
  --system S6=../results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl \
  --system F=../results/ensemble_three_duration_two_fusion_decodes/hyps_LP=0.7.jsonl \
  --system Fprime=../results/ensemble_one_to_one_duration_members_decodes/hyps_LP=0.6.jsonl \
  --pairs F:S6,Fprime:S6 --train_hashes ../results/train_target_hashes.txt \
  --expect S6=11.802 --expect F=12.114 --expect Fprime=12.836
```

```text
test n=800  seen-in-train=230  unseen=570
S6           BLEU-4  11.802  seen  35.00  unseen  6.60
F            BLEU-4  12.114  seen  33.40  unseen  6.92
Fprime       BLEU-4  12.836  seen  35.00  unseen  7.44
check: S6 reproduces the recorded 11.802
check: F reproduces the recorded 12.114
check: Fprime reproduces the recorded 12.836
[all n=800] F vs S6: observed +0.313  95% CI [-0.887, +1.498]  P(F>S6) 69.3%  -> not resolvable
[all n=800] Fprime vs S6: observed +1.034  95% CI [-0.102, +2.153]  P(Fprime>S6) 96.2%  -> not resolvable
```

This interval also serves as the known-good anchor for the harness. It reproduces bit-exactly the value recorded in `results/TEST_SPLIT_RESOLUTION_2026-09-10.md`. If it comes out as anything other than `[-0.887, +1.498]`, the harness has drifted, and no other result in this document should be trusted.

### A8 — Configuration reported but not claimed

**Claim — NOT CLAIMED.** A later configuration scores **12.836**, which is +1.034 over the reported system, with 95 % CI **[−0.102, +2.153]**. It is reported and explicitly not claimed.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 tools/paired_bootstrap.py \
  --system S6=../results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl \
  --system F=../results/ensemble_three_duration_two_fusion_decodes/hyps_LP=0.7.jsonl \
  --system Fprime=../results/ensemble_one_to_one_duration_members_decodes/hyps_LP=0.6.jsonl \
  --pairs F:S6,Fprime:S6 --train_hashes ../results/train_target_hashes.txt \
  --expect S6=11.802 --expect F=12.114 --expect Fprime=12.836
```

```text
test n=800  seen-in-train=230  unseen=570
S6           BLEU-4  11.802  seen  35.00  unseen  6.60
F            BLEU-4  12.114  seen  33.40  unseen  6.92
Fprime       BLEU-4  12.836  seen  35.00  unseen  7.44
check: S6 reproduces the recorded 11.802
check: F reproduces the recorded 12.114
check: Fprime reproduces the recorded 12.836
[all n=800] F vs S6: observed +0.313  95% CI [-0.887, +1.498]  P(F>S6) 69.3%  -> not resolvable
[all n=800] Fprime vs S6: observed +1.034  95% CI [-0.102, +2.153]  P(Fprime>S6) 96.2%  -> not resolvable
```

The command and output are the same as for A7. The output also shows that the entire margin sits on **unseen** targets (7.44 vs 6.60), with the seen subset unchanged at 35.00. This is the descriptive statement made in the dissertation.

The configuration is not claimed for three reasons:

- its interval contains zero;
- the max-T correction widens the interval further (A9);
- its member ratio was preferred on *test* ordering.

### A9 — Max-T correction over repeated looks

**Claim.** Five validation-selected ensembles have been scored on this split. A studentised max-T band over four of them gives **q = 2.406**, which widens F′'s interval to **[−0.363, +2.432]**. A further look would enter a family of five and would need an observed margin of **at least about +1.40**, which is larger than any margin actually observed.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 tools/holm_decision.py --maxt \
  --system S6=../results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl \
  --system F=../results/ensemble_three_duration_two_fusion_decodes/hyps_LP=0.7.jsonl \
  --system Fprime=../results/ensemble_one_to_one_duration_members_decodes/hyps_LP=0.6.jsonl \
  --system G1=../results/ensemble_six_duration_two_fusion_decodes/hyps_LP=0.7.jsonl \
  --system E=../results/ensemble_three_duration_members_decodes/hyps_LP=0.8.jsonl \
  --baseline S6 --expect S6=11.802
```

```text
check: S6 reproduces the recorded 11.802

max-T over 4 correlated looks at the same 800 test sentences (baseline S6, B=2000, alpha=0.05)
  look        observed      se   per-comparison 95%      max-T simultaneous 95%
  F             +0.313   0.609   [-0.876, +1.501]          [-1.152, +1.778]
  Fprime        +1.034   0.581   [-0.094, +2.162]          [-0.363, +2.432]
  G1            -0.240   0.531   [-1.292, +0.812]          [-1.516, +1.036]
  E             -0.265   0.609   [-1.448, +0.919]          [-1.730, +1.200]
  q(max-T) = 2.406 over these 4 looks.  A further look enters a family of 5, and q is non-decreasing in
  family size (self-test), so re-run this with the new look's own column included: at se ~ 0.582 the bar
  is AT LEAST ~+1.40, not exactly it.
  * = interval excludes 0.  Protocol rule 9: an official-test ensemble claim needs the max-T column.
```

One over-long line of this output is wrapped for page width. The check needs `scipy`, which is among the pinned packages installed by `make setup`.

### A10 — Confirmatory family adopted nothing

**Claim.** The pre-registered confirmatory family consisted of four interventions carried to five seeds under a Holm-corrected adoption rule. It adopted **none** of them.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 tools/holm_decision.py --reproduce
```

```text
Recorded confirmatory family, rebuilt from results/ artefacts (m = 4, alpha = 0.05)
  w2LF   val  -1.230 +0.611 +0.306  mean -0.104   [!] seed-202 val and all 3 test deltas are project-log-only (see docstring)
         test -0.243 +0.296 +0.026  mean +0.026
  w1H    val  +0.411 -0.660 -1.248  mean -0.499
         test -0.652 -0.069 -0.560  mean -0.427
  L60    val  +0.885 -0.347 -0.293 +0.397 +0.643  mean +0.257
         test +0.716 -0.144 +0.413 -0.665 -0.445  mean -0.025
  w3T    val  +0.609 +0.734 -0.012 +0.808 +0.148  mean +0.457
         test +0.703 -0.499 -0.269 -0.350 +0.131  mean -0.057

Stage 1 (futility, beam-5 VALIDATION, first 3 seeds)
  w2LF     mean val delta -0.1043  -> STOP (futile); enters Holm with p = 1
  w1H      mean val delta -0.4992  -> STOP (futile); enters Holm with p = 1
  L60      mean val delta +0.0817  -> CONTINUE; 5-seed test t=-0.096 one-sided p=0.536
  w3T      mean val delta +0.4433  -> CONTINUE; 5-seed test t=-0.262 one-sided p=0.597

Holm-Bonferroni step-down, m = 4, alpha = 0.05
  rank arm          raw p  Holm level   verdict
  1    L60          0.536      0.0125   retain
  2    w3T          0.597      0.0167   retain
  3    w2LF         1.000      0.0250   retain
  4    w1H          1.000      0.0500   retain

Adoption rule (Holm reject AND 5-seed val mean > 0 AND both fresh seeds positive on test)
  w2LF     holm_reject=N val_mean_positive=N fresh_seeds_positive=N  ->  NOT adopted
  w1H      holm_reject=N val_mean_positive=N fresh_seeds_positive=N  ->  NOT adopted
  L60      holm_reject=N val_mean_positive=Y fresh_seeds_positive=N  ->  NOT adopted
  w3T      holm_reject=N val_mean_positive=Y fresh_seeds_positive=N  ->  NOT adopted

check: reproduces the recorded {w2LF 1, w1H 1, L60 0.54, w3T 0.60}, nothing adopted
```

Note the tool's own `[!]` flag. For one arm, the seed-202 validation delta and all three test deltas of w2LF survive only in the project's working notes, which are not distributed. They are not in a saved summary JSON, and the tool reports this rather than hiding it. That arm enters the family with p = 1 in either case, so the family verdict does not depend on it. Its provenance is nevertheless weaker than that of the other three arms.

### A11 — Decision tool self-test

**Claim.** The decision rule is implemented in code rather than only described, and the implementation tests itself.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 tools/holm_decision.py --self_test
```

```text
self-test: all assertions passed (Holm levels + step-down, one-sided p, futility boundary, adoption conditions, max-T width)
```

### A12 — Wave-1 futility rule applied against the project's interest

**Claim.** The futility rule was applied even where it worked against the project's interest. In wave 1 the hand-masking arm was stopped on validation, with a mean of −0.499. At the same time, its last-five-checkpoint soup was positive on test in 3/3 seeds, with a 95 % interval of **[+0.210, +0.612]**. That was the only interval in the table to exclude zero.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
cd ../results/hand_masking_and_longer_training && python3 stage1_decision.py    # reads relative paths; run from its own directory
```

```text
beam-5 test, last-5 soup (secondary)
  w1A    s42=10.658  s101=11.155  s202=11.133   mean=10.982
  w1H    s42=11.010  s101=11.532  s202=11.636   mean=11.392
  w1L60  s42=11.642  s101=11.114  s202=11.892   mean=11.549
  w1H - w1A: +0.352 +0.377 +0.503  | mean +0.411 | 3/3 positive  95% t-interval [+0.210, +0.612]
  w1L60 - w1A: +0.985 -0.041 +0.759  | mean +0.568 | 2/3 positive  95% t-interval [-0.772, +1.907]

==============================================================================
STAGE-1 FUTILITY DECISION (validation only, best.pt):
  w1H    mean delta -0.4992  ->  STOP (futile)
  w1L60  mean delta +0.0817  ->  CONTINUE to Stage 2 (seeds 303/404)
```

The output is abridged to the two decisive blocks. The script also prints the validation and test results for `best.pt` and the validation soup. The soup readout was pre-registered as "not a claim in any case", and the arm was stopped.

### A13 — Content-word identification failure

**Claim.** The binding constraint is identification, not ordering:

- **66.92 %** of clips containing at least one reference content word receive **zero** correct content words;
- order-sensitive (LCS) and order-free (bag) content recall differ by only **0.098** points.

This excludes, by measurement, the whole family of reordering, monotonic-attention and alignment remedies.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 tools/content_failure_analysis.py \
  --hyps ../results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl --stop stitch_lexicon
```

```text
{"n": 786, "zero": 526, "zero_pct": 66.921, "all": 83, "all_pct": 10.56, "mean_content": 2.599, "recall_unordered": 18.796, "recall_ordered": 18.698}
```

`--stop` selects the stop list. `stitch_lexicon` is the 71-word list named in the dissertation; the CTC target uses a different list of 75 entries (see A17). The 14 clips whose reference has no content word are excluded, which gives n = 786 rather than 800.

### A14 — Split sizes, vocabularies and verbatim overlap

**Claim.** The splits and their overlap with training are as follows:

- Train 12,441 clips, validation 800 and test 800.
- The training set contains 3,065 word types.
- **28.75 %** of test clips have a reference occurring verbatim in training.
- **31.38 %** of validation clips have a reference occurring verbatim in training.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
S={}
for sp in ('train','val','test'):
    rows=[l.rstrip('\n').split('\t',1)[1].strip() for l in open('../comm_run/manifests/%s.tsv'%sp) if '\t' in l]
    S[sp]=rows; toks=[r.split() for r in rows]
    print('%-5s clips %6d  distinct %5d  word types %5d  mean len %.3f  median %d  total tok %d' % (
      sp, len(rows), len(set(rows)), len({w for t in toks for w in t}),
      sum(map(len,toks))/len(toks), sorted(map(len,toks))[len(toks)//2], sum(map(len,toks))))
tr=set(S['train'])
for sp in ('val','test'):
    seen=sum(r in tr for r in S[sp]); dist=len({r for r in S[sp] if r in tr})
    print('%s overlap clips %d/%d = %.2f%%   distinct %d/%d = %.2f%%' % (sp, seen, len(S[sp]),
          100*seen/len(S[sp]), dist, len(set(S[sp])), 100*dist/len(set(S[sp]))))
"
```

```text
train clips  12441  distinct  9535  word types  3065  mean len 8.087  median 7  total tok 100609
val   clips    800  distinct   732  word types   523  mean len 6.157  median 6  total tok 4926
test  clips    800  distinct   730  word types   470  mean len 6.144  median 6  total tok 4915
val overlap clips 251/800 = 31.38%   distinct 192/732 = 26.23%
test overlap clips 230/800 = 28.75%   distinct 166/730 = 22.74%
```

Every figure matches the dissertation. The split sizes also match Table 2 of the benchmark paper. The check needs manifests built from the dataset with `tools/build_manifest.py`.

### A15 — Length of detokenised test references

**Claim.** After detokenisation, which is the form in which scoring takes place, the test references average **5.144** words over **4,115** words in total. This short length is what makes a four-gram metric coarse on this corpus.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from slt.metrics import detokenize
rows=[l.rstrip('\n').split('\t',1)[1].strip() for l in open('../comm_run/manifests/test.tsv') if '\t' in l]
d=[detokenize(r) for r in rows]
print('test detok mean words %.3f  total %d' % (sum(len(x.split()) for x in d)/len(d), sum(len(x.split()) for x in d)))
"
```

```text
test detok mean words 5.144  total 4115
```

### A16 — Detokenisation round trip

**Claim.** Detokenising and then applying `13a` tokenisation is an exact inverse of the corpus's own pre-tokenisation on all 800 test sentences, with **0 round-trip mismatches**. This is what makes the BLEU scores here comparable to the SignJoey-convention figures used by the published baselines.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from slt.metrics import detokenize
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
t=Tokenizer13a()
refs=[l.rstrip('\n').split('\t',1)[1].strip() for l in open('../comm_run/manifests/test.tsv') if '\t' in l]
print('round-trip mismatches %d/%d' % (sum(t(detokenize(r)).split()!=r.split() for r in refs), len(refs)))
"
```

```text
round-trip mismatches 0/800
```

### A17 — CTC target statistics

**Claim.** The auxiliary CTC targets are **not glosses**. Over the 12,439 training clips actually used, they are English content-word sequences in English word order, yielding **2,988** distinct symbols and **3.5992** symbols per clip. The CTC stop list has 75 entries, whereas the corpus-building stop list used by the failure analysis has 71.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'tools')
from train_ctc import to_gloss, STOP
import stitch_lexicon as sl
DROP={'video_12_62','video_68_179'}          # in the official manifest, absent from the trained one
rows=[]
for l in open('../comm_run/manifests/train.tsv'):
    if '\t' not in l: continue
    p,t=l.rstrip('\n').split('\t',1)
    if p.split('/')[-1].replace('.npz','') in DROP: continue
    rows.append(t.strip())
g=[to_gloss(t) for t in rows]
print('clips',len(rows),'distinct symbols',len({w for x in g for w in x}),'per clip %.4f'%(sum(map(len,g))/len(g)))
print('train_ctc.STOP',len(STOP),'| stitch_lexicon.STOP',len(sl.STOP))
"
```

```text
clips 12439 distinct symbols 2988 per clip 3.5992
train_ctc.STOP 75 | stitch_lexicon.STOP 71
```

The two dropped clips matter. The official manifest has 12,441 rows, whereas the manifest actually trained on has 12,439. Running the same code over the full 12,441-row manifest gives 2,989 symbols and 3.5992 per clip, which is why the dissertation states the 12,439 denominator explicitly.

### A18 — Parameter counts

**Claim.** The parameter counts are as follows:

- a pose-only member has **9.0481 M** parameters;
- a fusion member has **10.9238 M** parameters;
- the six-member ensemble has **59.9155 M** parameters.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from config import Config
from slt.reference_model import build_model
from slt.rgb_fusion import RGBFusionSLT
cfg=Config(); V=3063
n1=sum(x.numel() for x in build_model(399,V,cfg).parameters())
n2=sum(x.numel() for x in RGBFusionSLT(399,1152,V,cfg).parameters())
print('pose %.4f M | fusion %.4f M | 3+3 total %.4f M' % (n1/1e6, n2/1e6, 3*(n1+n2)/1e6))
" 2>/dev/null
```

```text
pose 9.0481 M | fusion 10.9238 M | 3+3 total 59.9155 M
```

The check needs `torch`. `V = 3063` is the trained decoder vocabulary. This is a different quantity from the 3,065 training word types of A14, which counts surface types in the raw manifest rather than vocabulary entries.

The current `config.py` defaults already encode the adopted architecture, so no overrides are needed. `RGBFusionSLT` emits a harmless `enable_nested_tensor` warning, which is suppressed in the command above.

### A19 — Input-blind constant baseline

**Claim.** An input-blind constant string scores **4.41** BLEU-4 on the official split. The string consists of the same five words for every clip and ignores the video entirely. This score is roughly 44–48 % of the published pose baselines, and it **collapses to 1.09** on the unseen subset. The mechanism is bare memorisation: **18 test references are exactly that sentence**.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'tools')
from slt.metrics import corpus_bleu4, corpus_wer, detokenize
from restore_references import sentence
const=sentence('636ebd35c0929c80790e3b0c57919dfb05157d7f')   # the constant sentence, named by its SHA-1 digest
refs=[l.rstrip('\n').split('\t',1)[1].strip() for l in open('../comm_run/manifests/test.tsv') if '\t' in l]
R=[detokenize(r) for r in refs]; C=[detokenize(const)]*len(R)
print('constant BLEU-4 %.2f  WER %.4f' % (corpus_bleu4(C,R,'13a'), corpus_wer(C,R)))
print('test references equal to it:', sum(r==const for r in refs))
"
```

```text
constant BLEU-4 4.41  WER 1.1227
test references equal to it: 18
```

The seen/unseen decomposition of the constant (10.65 / **1.09**) is recorded in `results/NEWS_CONTEXT_AND_WER_FLOOR_2026-09-09.md`. The `--subset` flag of `tools/paired_bootstrap.py` recomputes it from the same definition, given a constant-string dump.

### A20 — WER is defeated by brevity

**Claim.** WER on this corpus has no regime that resists a degenerate strategy:

- Truncating the RGB-fusion model's output to three words scores **0.8999**, which beats the tuned six-member ensemble's 0.9069. The same truncation cuts that model's own BLEU-4 from 10.335 to 4.87.
- Truncating to two words scores 0.9077 at a BLEU-4 of **exactly zero**.
- An empty output scores exactly 1.000.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 -c "
import json, sys; sys.path.insert(0,'.'); sys.path.insert(0,'tools')
from restore_references import sentence
from slt.metrics import corpus_bleu4, corpus_wer, detokenize
rgb=sorted((json.loads(l) for l in open('../results/semantic_adequacy_grading/hyps_rgbfusion.jsonl')), key=lambda r: r['idx'])
H=[detokenize(r['hyp_beam']) for r in rgb]; R=[detokenize(r['ref']) for r in rgb]
print('empty            WER %.4f' % corpus_wer(['']*len(R), R))
print('best-WER constant   WER %.4f' % corpus_wer([detokenize(sentence('9b0921ab9c260fd270f2c384628d786bcad69796'))]*len(R), R))
print('RGB full         BLEU-4 %.3f  WER %.4f' % (corpus_bleu4(H,R,'13a'), corpus_wer(H,R)))
for k in (2,3):
    T=[' '.join(h.split()[:k]) for h in H]
    print('RGB trunc-%d      BLEU-4 %.2f  WER %.4f' % (k, corpus_bleu4(T,R,'13a'), corpus_wer(T,R)))
"
```

```text
empty            WER 1.0000
best-WER constant   WER 0.9526
RGB full         BLEU-4 10.335  WER 0.9497
RGB trunc-2      BLEU-4 0.00  WER 0.9077
RGB trunc-3      BLEU-4 4.87  WER 0.8999
```

The two constants used in the dissertation are different strings, each chosen for a different metric:

- the best-BLEU constant, a five-word training sentence (A19, WER 1.1227);
- the best-WER constant, a three-word training sentence (WER 0.9526, BLEU 1.45).

Quoting one string's number against the other string is an easy mistake to make.

### A21 — ROUGE-L and tokenisation convention

**Claim.** ROUGE-L moves by a large margin on tokenisation convention alone. Under the published comparators' own convention, the three single models score 35.70 / 36.27 / **37.08**. That convention is SignJoey's: a whitespace split and a recall-weighted F with β = 1.2. These scores are at **parity** with the comparators, not superior to them: SL-Transformer 35.65, GASLT 35.74 and SL-Luong 37.27.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
for f in dwpose_ctc vlp rgbfusion; do
  python3 metric_suite.py --in ../results/semantic_adequacy_grading/hyps_$f.jsonl \
    --train_hashes ../results/train_target_hashes.txt | grep -E "=====|BLEU-4="
done
```

```text
===== hyps_dwpose_ctc.jsonl (hyp_beam, n=800) =====
BLEU-4=9.52  BLEU-1=31.18  chrF=19.13  ROUGE-L=35.70   [BLEU/chrF detokenised; ROUGE-L whitespace-tokenised]
  seen-in-train targets:   n=230  BLEU-4=29.72
  UNSEEN targets (generalisation): n=570  BLEU-4=4.88
===== hyps_vlp.jsonl (hyp_beam, n=800) =====
BLEU-4=10.27  BLEU-1=32.41  chrF=19.65  ROUGE-L=36.27   [BLEU/chrF detokenised; ROUGE-L whitespace-tokenised]
  seen-in-train targets:   n=230  BLEU-4=29.91
  UNSEEN targets (generalisation): n=570  BLEU-4=5.31
===== hyps_rgbfusion.jsonl (hyp_beam, n=800) =====
BLEU-4=10.33  BLEU-1=32.22  chrF=19.29  ROUGE-L=37.08   [BLEU/chrF detokenised; ROUGE-L whitespace-tokenised]
  seen-in-train targets:   n=230  BLEU-4=30.22
  UNSEEN targets (generalisation): n=570  BLEU-4=5.81
```

The `seen`/`unseen` lines also match the `BLEU-4=` filter and are computed from the digests. The three ROUGE-L values constitute the claim. The seen/unseen lines reproduce the ladder's seen/unseen columns of A4 through a second, independent code path.

The size of the tokenisation swing under both conventions is computed as follows:

```bash
python3 -c "
import json, sys; sys.path.insert(0,'.')
import metric_suite as M
from slt.metrics import detokenize
rows=[json.loads(l) for l in open('../results/semantic_adequacy_grading/hyps_rgbfusion.jsonl')]
h=[r['hyp_beam'] for r in rows]; r_=[r['ref'] for r in rows]
for b in (1.0, 1.2):
    M.ROUGE_BETA=b
    print('beta=%.1f  whitespace %.2f  detokenised %.2f' % (b,
      100*sum(M.rouge_l(a,c) for a,c in zip(h,r_))/len(h),
      100*sum(M.rouge_l(detokenize(a),detokenize(c)) for a,c in zip(h,r_))/len(h)))
"
```

```text
beta=1.0  whitespace 37.67  detokenised 17.84
beta=1.2  whitespace 37.08  detokenised 17.61
```

**Discrepancy.** The corresponding passage of the dissertation is headed with a 16-point movement of ROUGE-L on tokenisation alone. It then gives the pair 17.84 / 37.67, which is a gap of **19.83** points, not 16. The comparator trio in the same passage is quoted at β = 1.2.

Both pairs of numbers reproduce exactly, but the figure of 16 points does not follow from either. The correct statement is **~19.8 points at β = 1.0 and ~19.5 at β = 1.2**.

### A22 — Under-generation of the reported decode

**Claim.** The reported decode under-generates systematically, which is the opposite of the failure mode that most decoding remedies target:

- it emits **308** distinct strings over 800 clips;
- it matches the reference exactly on **44** clips;
- it has a length ratio of **0.753**.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 -c "
import json, sys; sys.path.insert(0,'.')
from slt.metrics import detokenize
rows=sorted((json.loads(l) for l in open('../results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl')), key=lambda r: r['idx'])
H=[detokenize(r['hyp_beam']) for r in rows]; R=[detokenize(r['ref']) for r in rows]
print('distinct', len(set(H)), '| exact', sum(h==r for h,r in zip(H,R)),
      '| length ratio %.3f' % (sum(len(h.split()) for h in H)/sum(len(r.split()) for r in R)))
"
```

```text
distinct 308 | exact 44 | length ratio 0.753
```

### A23 — Per-signer breakdown

**Claim.** The corpus is close to a two-signer corpus. `Signer_ID` was joined to the reported decodes; the alignment was verified with **0 of 800** reference mismatches, and the joined dump re-scores to 11.802 exactly. The join gives the following scores:

- signer 1: **14.39** (n = 389);
- signer 2: **9.97** (n = 350);
- all other signers pooled: **3.12** (n = 61).

Reference lengths run against this result, and every one of these signers has their own clips in training.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
cd .. && python3 -c "
import json, sys, pandas as pd
sys.path.insert(0,'src')
from slt.metrics import corpus_bleu4, detokenize
man=[l.rstrip('\n').split('\t') for l in open('comm_run/manifests/test.tsv') if '\t' in l]
clips=[p[0].split('/')[-1].replace('.npz','') for p in man]
rows=sorted((json.loads(l) for l in open('results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl')), key=lambda r: r['idx'])
print('ref mismatches', sum(r['ref'].strip()!=p[1].strip() for r,p in zip(rows,man)), 'of', len(rows))
df=pd.read_excel('data/auslan-daily/translation/Auslan-Daily_Communication.xlsx')
sid=dict(zip(df['Video_Clip_Name'].astype(str), df['Signer_ID']))
ids=[sid[c] for c in clips]
H=[detokenize(r['hyp_beam']) for r in rows]; R=[detokenize(r['ref']) for r in rows]
print('overall %.3f' % corpus_bleu4(H,R,'13a'))
for lab, sel in (('signer 1', lambda s: s==1), ('signer 2', lambda s: s==2),
                 ('others  ', lambda s: s not in (1,2)), ('signer 10', lambda s: s==10)):
    k=[i for i,s in enumerate(ids) if sel(s)]
    print('%-9s n %3d  BLEU %5.2f  mean ref words %.2f' % (lab, len(k),
          corpus_bleu4([H[i] for i in k],[R[i] for i in k],'13a'),
          sum(len(R[i].split()) for i in k)/len(k)))
"
```

```text
ref mismatches 0 of 800
overall 11.802
signer 1  n 389  BLEU 14.39  mean ref words 5.28
signer 2  n 350  BLEU  9.97  mean ref words 4.95
others    n  61  BLEU  3.12  mean ref words 5.43
signer 10 n  42  BLEU  2.09  mean ref words 5.83
```

The check reads the test manifest and the dataset spreadsheet `Auslan-Daily_Communication.xlsx`, both of which `make data` provides. It also requires `pandas` and `openpyxl`, which `make setup` installs.

The figure of 2.09 for signer 10 rests on 42 clips and is described as not adjudicable in `results/SIGNER_BREAKDOWN_2026-09-22.md`. The pooled figure of 3.12 is the defensible form.

### A24 — A signer-disjoint split can be built

**Claim.** A signer-disjoint split **can** be built, contrary to a statement made earlier in the project. Holding out every signer except the two principals gives **12,953 train / 1,088 test** clips, with **19.3 %** verbatim target overlap. This is lower than the official split's 28.75 %.

Such a split would still be a poor generalisation test, and the breakdown in A23 already shows what it would measure. The earlier claim that it was impossible was nevertheless wrong.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
cd .. && python3 -c "
import pandas as pd
df = pd.read_excel('data/auslan-daily/translation/Auslan-Daily_Communication.xlsx')
top2 = df['Signer_ID'].value_counts().index[:2].tolist()
print('principals', top2, '= %.1f%% of clips' % (100*df['Signer_ID'].isin(top2).mean()))
tr = df[df['Signer_ID'].isin(top2)]; te = df[~df['Signer_ID'].isin(top2)]
ov = te['Subtitle'].astype(str).str.strip().isin(set(tr['Subtitle'].astype(str).str.strip()))
print('signer-disjoint train', len(tr), 'test', len(te), '| verbatim overlap %.1f%%' % (100*ov.mean()))
"
```

```text
principals [1, 2] = 92.3% of clips
signer-disjoint train 12953 test 1088 | verbatim overlap 19.3%
```

The check has the same dataset dependency as A23. `results/SIGNER_BREAKDOWN_2026-09-22.md` still carries the superseded sentence stating that no signer-disjoint split can be built. The dissertation carries the correction, and this command is the check.

### A25 — Measured run-to-run noise

**Claim.** Run-to-run noise was measured rather than assumed:

- Six runs of one identical configuration give a beam-5 test **sd of 0.216**, corresponding to a paired single-run band of **±0.60**.
- Five jobs running a verified-identical recipe at three seeds give a pooled within-seed **sd of 0.422**, corresponding to a band of **±1.17**.

`--seed 42` does not reproduce a run without the determinism flag.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
cd ../results/run_to_run_noise
python3 null_band.py
python3 adopted_crossbatch.py adopted_values.json
```

```text
beam5_test n=6 mean 9.745 sd 0.216 [95% 0.135, 0.529] | paired single-run band +-0.60 [0.37, 1.47] | range 9.493-10.119
beam5_val  n=6 mean 10.131 sd 0.587 [95% 0.367, 1.440] | paired single-run band +-1.63 [1.02, 3.99] | range 9.172-10.597
val_best   n=6 mean 9.393 sd 0.374 [95% 0.234, 0.918] | paired single-run band +-1.04 [0.65, 2.55] | range 8.907-9.998
```

```text
s42: pa_aug 11.443 w1A 10.637 w3C 10.091 r3_111 10.779 a3ctl 10.800 | mean 10.750 sd 0.483 range 1.353
s101: pa_aug 11.375 w1A 11.296 w3C 11.686 r3_111 12.118 a3ctl 11.223 | mean 11.539 sd 0.368 range 0.894
s202: pa_aug 11.193 w1A 11.345 w3C 11.483 r3_111 10.486 a3ctl 11.427 | mean 11.187 sd 0.407 range 0.997
POOLED within-seed cross-batch sd (beam-5 test) 0.422 [95% 0.302, 0.696], df 12 | paired single-run band +-1.17
```

Both outputs reproduce `null_band_output.txt` and `adopted_crossbatch_output.txt` byte for byte. The check needs `scipy`.

**Caution.** Always pass the argument to `adopted_crossbatch.py`. Run without an argument, the script enters its cluster-side extraction mode and reads run directories under `/group`, which do not exist off the cluster. It then **overwrites `adopted_values.json` with nulls**. Always run it as `python3 adopted_crossbatch.py adopted_values.json`.

### A26 — Pooled pose-augmentation effect

**Claim.** Pose augmentation is one of the four interventions that replicated. No individual seed is resolvable. Pooling the three seeds in one bootstrap gives the following:

- all targets: **+0.695 [+0.048, +1.339]**, P = 98.2 %;
- unseen targets: **+0.671 [−0.045, +1.453]**, which is not resolvable.

The pooled analysis was chosen *after* the run. The pre-registered criterion was 3/3 sign consistency, and that criterion was met.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
cd ../results/pose_augmentation_decodes && python3 pooled_bootstrap.py
```

```text
[pooled over seeds 42/101/202, all n=800] mean(aug - ctrl) observed +0.695  95% CI [+0.048, +1.339]  P(>0) 98.2%  -> significant
[pooled over seeds 42/101/202, unseen n=570] mean(aug - ctrl) observed +0.671  95% CI [-0.045, +1.453]  P(>0) 96.7%  -> not resolvable
```

The unseen subset is computed from the digests in `results/train_target_hashes.txt`. The runtime is about 7 min.

### A27 — Video-disjoint confirmation

**Claim.** The official split shares videos between train and test, so a **video-disjoint** split was held out. On that split, the full adopted recipe beats a plain baseline over five seeds: beam-5 test rises from 6.144 to 7.082, a gain of **+0.939 [+0.155, +1.722]**, positive in 5/5 seeds.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 tools/seed_decision.py --control vdA --arm vdB \
  ../results/video_disjoint_recipe_confirmation/summary_s42.json ../results/video_disjoint_recipe_confirmation/summary_s101.json \
  ../results/video_disjoint_recipe_confirmation/summary_s202.json ../results/video_disjoint_recipe_confirmation/summary_s303.json \
  ../results/video_disjoint_recipe_confirmation/summary_s404.json
```

```text
vdB - vdA beam5_val  n=5 per-seed +0.647 +0.336 +0.920 +0.563 +0.076 | positive 5/5 | mean +0.508 | 95% t [+0.112, +0.905]
vdB - vdA beam5_test n=5 per-seed +0.405 +0.956 +1.098 +1.898 +0.337 | positive 5/5 | mean +0.939 | 95% t [+0.155, +1.722]
  vdA      beam-5 val 5.498 5.631 5.244 5.366 5.623 | test 6.823 6.034 5.794 5.917 6.150
  vdB      beam-5 val 6.145 5.967 6.164 5.929 5.699 | test 7.228 6.990 6.892 7.815 6.486
```

**Scope.** This checks the decision arithmetic from the saved per-seed summaries. It does not re-run the ten trainings (C5).

It also does not cover the memorisation-controlled restatement, +0.934 [+0.408, +1.460]. That figure is computed on the 886 clips whose reference never occurs in the video-disjoint training text, and the per-clip dumps behind it are not included.

### A28 — Person tracking: negative result

**Claim — NEGATIVE, although the defect is real.** The pose extractor has no person tracking. In 23.8 % of training clips, at least one frame would have had a different person chosen by a tracker. Fixing this produced **no resolvable change**:

- five-seed test mean **−0.057 [−0.658, +0.545]**, positive in 2/5 seeds;
- validation mean +0.457 for comparison.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 tools/seed_decision.py --control w3C --arm w3T \
  ../results/person_tracking_stage2/summary_s42.json ../results/person_tracking_stage2/summary_s101.json \
  ../results/person_tracking_stage2/summary_s202.json ../results/person_tracking_stage2/summary_s303.json \
  ../results/person_tracking_stage2/summary_s404.json
```

```text
w3T - w3C beam5_val  n=5 per-seed +0.609 +0.734 -0.012 +0.808 +0.148 | positive 4/5 | mean +0.457 | 95% t [+0.002, +0.913]
w3T - w3C beam5_test n=5 per-seed +0.703 -0.499 -0.269 -0.350 +0.131 | positive 2/5 | mean -0.057 | 95% t [-0.658, +0.545]
  w3C      beam-5 val 10.731 11.083 11.517 10.429 11.166 | test 10.091 11.686 11.483 11.552 11.319
  w3T      beam-5 val 11.339 11.816 11.504 11.237 11.314 | test 10.793 11.186 11.214 11.202 11.450
```

This arm's validation interval only just excludes zero, while its test interval does not. That is the case the staged decision rule exists for, and the arm is retained by Holm in A10. The 23.8 % prevalence figure is a level C result (C5).

### A29 — Mechanism of the ensembling gain

**Claim.** The ensembling gain comes from **token-level partial correctness**, not from member selection:

- 20.2 % of the ensemble's outputs are produced by no member.
- On those clips the ensemble beats *every* member 36 times and falls below every member 13 times (sign test p = 0.0014).
- The true member oracle at pool 6 is 16.27, and the ensemble captures 18.3 % of the gap over its best member.

This claim is not stated in the dissertation. Its source is `results/ENSEMBLE_MECHANISM_2026-09-22.md`, which was pre-registered before any member was decoded alone.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 tools/ensemble_mechanism.py \
  --ensemble ../results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl \
  --members ../results/final_ensemble_member_test_decodes/*/hyps_LP=0.3.jsonl \
  --expect 11.802
```

```text
gate: ensemble BLEU-4 11.802 (recorded 11.802) | members 10.46 10.80 10.62 10.32 10.01 10.34
novel ensemble outputs: 162/800 = 20.2%
novel    clips 162 | ensemble beats EVERY member 36 (22.2%) | below EVERY member 13 (8.0%) | other 113
matched  clips 638 | ensemble beats EVERY member 0 (0.0%) | below EVERY member 0 (0.0%) | other 638
all      clips 800 | ensemble beats EVERY member 36 (4.5%) | below EVERY member 13 (1.6%) | other 751
DECISIVE (novel clips): beats-every 36 vs below-every 13 -> SUPPORTS token-level partial correctness
(not decisive) corpus BLEU on novel clips: ensemble 5.25 vs best member there 5.19 | matched clips: ensemble 14.10
member oracle by pool size (per-sentence best chrF; mean corpus BLEU over all subsets):
  pool 1:  10.42 (6 subsets)
  pool 2:  12.60 (15 subsets)
  pool 3:  13.94 (20 subsets)
  pool 4:  14.91 (15 subsets)
  pool 5:  15.66 (6 subsets)
  pool 6:  16.27 (1 subsets)
capture of the oracle gap: (ensemble 11.80 - best single 10.80) / (oracle 16.27 - best single) = 18.3%
EXPLORATORY decomposition from the best member (10.80): ensemble output on NOVEL clips only 11.34 | on MATCHED clips only 11.26 | both 11.80
EXPLORATORY matched clips: ensemble output shared by a median of 2.0 of 6 members; best member's output differs from the ensemble on 370 of them
member pairs with identical output: mean 20.8% (range 16.5-28.0)
```

The order of the member column follows shell glob order and may differ from the recorded run. The set of members and every statistic are identical. The runtime is about 1 min.

The between-subset BLEU contrast is printed but is explicitly **not** the decisive statistic, because membership in either subset tracks clip difficulty.

### A30 — Selective prediction with validation-fixed thresholds

**Claim.** Selective prediction transfers to the ensemble when thresholds are fitted on **validation** and applied unchanged to test. The result is **19.07 BLEU-4 at 51.9 % coverage**, against random abstention, which stays flat at about 11.78.

This is **not a BLEU improvement**. It is a different task, at a coverage that is not comparable, on a retained set enriched for short, formulaic, memorised utterances.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 tools/ensemble_selective.py \
  --val_dump ../results/final_ensemble_validation_decodes --test_dump ../results/final_ensemble_test_decodes --lp 0.3 \
  --train_hashes ../results/train_target_hashes.txt \
  --expect_test_bleu 11.802 --out /tmp/enssel_S6.json
```

```text
gate: 800/800 hyps match hyps_LP=0.3.jsonl; test BLEU 11.802 (recorded 11.802); WER 0.9069; seen val 251 test 230
all    val  n=800 | own-quantile  12.47  13.36  15.53  21.82  34.80 | longest  12.47  12.86  13.55  14.92  14.78 | random  12.47  12.48  12.44  12.58  13.04
all    test n=800 | own-quantile  11.80  12.86  14.68  19.56  34.06 | longest  11.80  12.30  12.82  13.76  15.95 | random  11.80  11.83  11.70  11.78  11.80
                     val-fixed thresholds ->  11.80 @ 100.0%   12.90 @ 89.4%   14.61 @ 75.6%   19.07 @ 51.9%   32.40 @ 26.4%
unseen val  n=549 | own-quantile   6.63   7.09   7.83   9.39  14.12 | longest   6.63   6.94   7.32   8.20   8.88 | random   6.63   6.61   6.61   6.66   6.53
unseen test n=570 | own-quantile   6.60   7.10   7.83   9.44  13.97 | longest   6.60   6.96   7.41   8.22   8.95 | random   6.60   6.60   6.49   6.63   6.70
                     val-fixed thresholds ->   6.60 @ 100.0%    7.18 @ 88.8%    7.71 @ 76.7%    8.95 @ 55.1%   13.36 @ 27.7%
```

The `longest` column is the control that matters. A confidence signal that secretly reflected output length would exploit the same brevity channel that makes WER uninformative here (A20). The `own-quantile` curve rises far above the `longest` curve, so that is not the case.

The seen counts asserted by the harness gate (251/230) are computed from the digests. The runtime is about 1 min. **The 25 %-coverage column must never be quoted on unseen targets**, because validation and test did not replicate there.

### A31 — Synthetic corpus audit

**Claim.** The adopted 13,760-sentence synthetic corpus is novel text, but structurally narrow:

- **0** sentences are identical to a real training sentence, **0** to a validation reference and **1** to a test reference.
- 42.3 % of its bigrams and **75.0 %** of its trigrams never occur in the real training text.
- Its 94,428 stitched signs draw on only **942** distinct signs and **8,581** distinct recordings. Each recording is re-used 11 times on average.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
cd .. && python3 -c "
import json; d=json.load(open('results/stitched_corpus_audit/stitch_corpus_audit.json'))
a=d['syn_13760_adopted']; v=d['visual_supply_adopted']
print({k:a[k] for k in ('sentences','vocab','2grams_unseen_in_real_train','3grams_unseen_in_real_train',
                        'exact_in_real_train','exact_in_val','exact_in_test')})
print({k:v[k] for k in ('stitched_sign_segments','distinct_glosses_used','distinct_recordings_used')},
      'mean segments/recording', v['segments_per_recording']['mean'])
print('real train vocab', d['real_train']['vocab'])
"
```

```text
{'sentences': 13760, 'vocab': 1289, '2grams_unseen_in_real_train': 0.4234, '3grams_unseen_in_real_train': 0.7496, 'exact_in_real_train': 0, 'exact_in_val': 0, 'exact_in_test': 1}
{'stitched_sign_segments': 94428, 'distinct_glosses_used': 942, 'distinct_recordings_used': 8581} mean segments/recording 11.0
real train vocab 3063
```

This check reads a saved audit artefact rather than recomputing it. The artefact, `results/stitched_corpus_audit/stitch_corpus_audit.json`, holds counts only. Regenerating it needs the synthetic corpus and MM-WLAuslan (C6).

The dissertation's figure of "1,271 distinct words at matched size" is a different statistic. It is computed on a 12,439-sentence sample of the synthetic corpus and recorded in `results/STITCH_CORPUS_AUDIT_2026-09-22.md`. The figure of 1,289 above refers to the full 13,760-sentence corpus.

### A32 — Pose-estimator comparison on the signer's keypoints (arithmetic from the saved read-out)

**Claim.** Whose keypoints are used mattered more than which estimator produced them. The first comparison of the two estimators gave DWPose a gain of **+4.970 BLEU-4 [+4.414, +5.527]**, 3/3 seeds. In that comparison the AlphaPose cells read the first-listed person of each clip, who is not the signer in 88.4 % of Communication clips: each released AlphaPose file lists every person detected and marks one entry as the signer, and the original extraction program took the first entry. Repeating the comparison with the signer's entry (job 196407, AlphaPose and DWPose in the same job, matched body-and-hand keypoints, frame normalisation, deterministic kernels, seeds 42/101/202) gives a difference of **+0.272 BLEU-4 on the official test set, 95 % t-interval [−1.027, +1.572], positive in 2 of 3 seeds**. The pre-registered rule (all three seeds positive) is not met, and the interval is wide enough that a difference of about one point can be neither confirmed nor excluded. The AlphaPose cells themselves rose from 4.97 to 10.08 on changing only the person. The earlier figure is therefore kept as measured and is superseded as an estimate of the estimator effect.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 -c "
import json, statistics
R = json.load(open('../results/estimator_signer_keypoints_rerun/readout.json'))
for split in ('val', 'test'):
    d = [b - a for a, b in zip(R[split]['eap'], R[split]['edw'])]          # Edw - Eap, beam 5, seeds 42/101/202
    m = statistics.mean(d); hw = 4.303 * statistics.stdev(d) / len(d) ** 0.5   # t(0.975, df=2)
    print(split, 'mean %+.3f  95%% t [%+.3f, %+.3f]  positive %d of %d' % (m, m - hw, m + hw, sum(x > 0 for x in d), len(d)))
"
```

```text
val mean +0.234  95% t [-0.743, +1.211]  positive 2 of 3
test mean +0.272  95% t [-1.027, +1.572]  positive 2 of 3
```

The per-seed beam-5 hypothesis dumps are under `results/estimator_signer_keypoints_rerun/runs/` (references replaced by digests), so the BLEU values in `readout.json` can be recomputed from them with `slt/metrics.py` after `make data`. The first comparison (job 7951) used the first-listed person; its dumps are not included. Re-running the experiment itself is C2, and the extraction that selects the signer is `jobs/slurm/apsig_extract.sbatch`. The probe that localises the effect to the hands is B8.

### A33 — Output-side vocabulary coverage

**Claim.** Output-side vocabulary work is excluded by measurement. Only **0.54 %** of test content tokens (11 of 2,043) never occur in the training targets. There is therefore no open-vocabulary problem on this split, which is why subword tokenisation was neutral here.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'tools')
import stitch_lexicon as sl
def content(t): return [w for w in t.split() if w.isalpha() and w not in sl.STOP]
tr=set()
for l in open('../comm_run/manifests/train.tsv'):
    if '\t' in l: tr.update(content(l.rstrip('\n').split('\t',1)[1].strip()))
tok=[w for l in open('../comm_run/manifests/test.tsv') if '\t' in l for w in content(l.rstrip('\n').split('\t',1)[1].strip())]
oov=[w for w in tok if w not in tr]
print('test content tokens', len(tok), 'never in train', len(oov), '= %.2f%%' % (100*len(oov)/len(tok)))
"
```

```text
test content tokens 2043 never in train 11 = 0.54%
```

The check needs the train and test manifests. It uses the same 71-word stop list as A13. The companion claim, that ordering is excluded, is A13.

### A34 — Ablation of the final ensemble over all subsets of its members

**Claim.** No member dominates the final ensemble. All 63 non-empty subsets of the six members were decoded on the 800 validation clips with the ensemble's decoder settings (beam width 10, length penalty 0.3; cluster jobs 120153 and 120154). The six-member subset reproduces the recorded validation score (**12.470**, WER 0.9164). Mean BLEU-4 rises with each added member, from 10.31 for one to 12.47 for six. Each member's Shapley value (its contribution averaged over every order of adding members, the empty set scored 0) lies between **1.87 and 2.39**. Leaving one member out costs 0.19 to 0.69, and the 95 % interval of that loss excludes zero for two members only, for one of them with a lower bound of +0.02. The three RGB-fusion members and the three pose-only members do not differ measurably (summed Shapley values 6.51 against 5.96; difference **+0.56 [−0.57, +1.97]**). One two-member subset scores above the six on validation (12.68); nothing is claimed from it. No test clip was decoded. This is the dissertation's ablation table.

**Runs with:** this repository and the dataset spreadsheet (`make data`). It takes about 8 minutes: the 500 bootstrap resamples re-score all 63 subsets.

```bash
python3 tools/ensemble_subsets.py report --out ../results/ensemble_member_subsets_ablation \
    --names VLP80,RGBfus,VLP40,VLP40s202,fus101,fus202 --types pose,fusion,pose,pose,fusion,fusion \
    --expect 12.470 --expect_wer 0.9164 --boot 500
```

```text
GATE PASSED: full set BLEU-4 12.470 WER 0.9164 | recorded 12.47 0.9164 | clips 800 | beam 10 LP 0.3
mean BLEU-4 by number of members (min, max):
  1: 10.305  (9.782, 10.686)  over 6 subsets
  2: 11.318  (10.590, 12.682)  over 15 subsets
  3: 11.610  (10.771, 12.222)  over 20 subsets
  4: 11.912  (11.551, 12.346)  over 15 subsets
  5: 12.077  (11.783, 12.279)  over 6 subsets
  6: 12.470  (12.470, 12.470)  over 1 subsets
per member: alone | leave-one-out loss | mean gain when added to a non-empty subset | Shapley (empty set = 0)
  VLP80          9.782 | +0.438 | +0.202 |  1.874
  RGBfus        10.629 | +0.333 | +0.339 |  2.091
  VLP40         10.686 | +0.191 | +0.521 |  2.195
  VLP40s202      9.998 | +0.348 | +0.218 |  1.887
  fus101        10.304 | +0.359 | +0.332 |  2.037
  fus202        10.431 | +0.687 | +0.753 |  2.385
by type:
  all fusion members (RGBfus+fus101+fus202): BLEU-4 11.418 WER 0.9244
  Shapley sum of fusion members: 6.513
  all pose members (VLP80+VLP40+VLP40s202): BLEU-4 11.218 WER 0.9377
  Shapley sum of pose members: 5.956
fusion minus pose, 95% interval: Shapley sums [-0.566, +1.965] | type-only ensembles [-0.986, +1.330]
95% intervals, 500 sentence resamples: leave-one-out loss | mean gain | Shapley
  VLP80        [+0.018, +0.912] | [-0.084, +0.484] | [+1.508, +2.291]
  RGBfus       [-0.153, +0.835] | [-0.010, +0.733] | [+1.640, +2.585]
  VLP40        [-0.041, +0.424] | [+0.258, +0.877] | [+1.818, +2.632]
  VLP40s202    [-0.104, +0.820] | [-0.044, +0.510] | [+1.486, +2.359]
  fus101       [-0.192, +0.984] | [-0.135, +0.897] | [+1.459, +2.667]
  fus202       [+0.122, +1.357] | [+0.374, +1.158] | [+1.868, +2.898]
```

The tool prints the gate line only, and exits with an error, unless the six-member subset reproduces `--expect`. Member names in the recorded order: `VLP80` is the model pre-trained for 80 epochs, `VLP40` and `VLP40s202` the two runs pre-trained for 40 epochs, `RGBfus`, `fus101` and `fus202` the three RGB-fusion runs. Intervals come from 500 resamples of the validation sentences with a fixed seed, so they reflect the sampling of sentences only. The decode files hold the hypotheses of every subset (`shard*of4.json`); the job logs are beside them. Regenerating the decodes needs the six checkpoints (`jobs/slurm/ens_subsets.sbatch`).

### A35 — Pretraining re-run with selection on validation only

**Claim.** Pose-text pretraining gives no resolvable gain when its length is chosen on validation. Five arms (no pretraining, and 40, 80, 128 and 256 pretraining epochs), three seeds each, were trained together (cluster job 116619). The rule fixed in advance (the cheapest pretraining length whose mean beam-5 validation BLEU-4 is within 0.10 of the best) selects 40 epochs, but no pretraining has the highest mean on both splits: 40 epochs minus no pretraining is **−0.26 [−1.57, +1.06]** on validation and **−0.89 [−2.14, +0.36]** on test.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 tools/w11_readout.py ../results/validation_selected_pretraining_rerun/runs_pre P0 0.10 P40 P80 P128 P256
```

```text
arm      n  greedy val  beam5 val  beam5 test   per-seed beam5 val / test
P0       3      9.625     10.450     10.122   s42 10.872/9.866  s101 10.498/10.492  s202 9.981/10.006
P40      3      9.325     10.192      9.232   s42 10.288/9.556  s101 9.955/9.317  s202 10.334/8.821
P80      3      9.271     10.001      9.810   s42 10.702/9.681  s101 9.995/10.208  s202 9.304/9.542
P128     3      9.136      9.827      9.697   s42 9.883/9.605  s101 9.430/9.544  s202 10.166/9.941
P256     3      9.149      9.907      9.398   s42 9.310/9.308  s101 10.048/9.429  s202 10.362/9.459
SELECTED ON VALIDATION: P40 (mean beam-5 validation 10.192); its mean beam-5 test 9.232
  P40 - P0 beam5_val: -0.258 [-1.573, +1.057]
  P40 - P0 beam5_test: -0.890 [-2.137, +0.358]
  P80 - P0 beam5_val: -0.450 [-1.089, +0.190]
  P80 - P0 beam5_test: -0.311 [-0.663, +0.040]
```

The read-out prints further rows (each longer pretraining against 40 epochs); the block above is the part compared. Scores are the stored beam-5 BLEU-4 of each run's validation-selected checkpoint. The News frame-limit arms of the same wave are not included, because their references are News subtitles, which `make data` does not fetch.

### A36 — Ensemble of the fine-tuned pre-trained sign-language model
**Claim.** Five models fine-tuned from the released How2Sign checkpoint on the Auslan-Daily Communication training clips score 17.86 on average on the official test set (best on validation 17.43). Their ensemble scores 20.60 (E*: beam 10, length penalty 0.8, chosen on validation) and 20.20 (E0: beam 5, length penalty 1.0) against the best published 9.95, which is +2.75 and +3.18 above the members under simultaneous (max-T, family of four) 95% intervals that exclude zero, with a predicted power of 0.99 and 0.72 (0.63 for E0), so the pre-registered claim rule is met. The three-model ensemble of models fine-tuned with our own code scores 19.77 (members 18.27, best 18.69); its second interval includes zero, so it is not claimed. On the 570 test clips whose reference is absent from training the five-model ensemble scores 16.39 against 6.60 for the 11.80 system (Results, Section on transfer, Discussion, Appendix table `tab:app_transfer` and `tab:app_transfer_diff`).

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/pretrained_ensemble/check.py
```
```text
=== Five fine-tuned models and their ensemble (n = 800 clips per split) ===
validation grid, five-model ensemble BLEU-4 (beam, length penalty):
  beam  5 lp 0.6: 19.73
  beam  5 lp 0.8: 19.70
  beam  5 lp 1.0: 19.54
  beam  5 lp 1.2: 17.82
  beam  5 lp 1.5: 16.15
  beam 10 lp 0.6: 19.69
  beam 10 lp 0.8: 19.88   <- selected on validation
  beam 10 lp 1.0: 19.25
  beam 10 lp 1.2: 17.32
  beam 10 lp 1.5: 13.02
E0 = beam 5, length penalty 1.0 (fixed in advance); E* = beam 10, length penalty 0.8 (validation argmax)
single models, beam 5 length penalty 1.0, BLEU-4 validation / test:
  seed  42: 17.65 / 17.23
  seed 101: 16.98 / 18.58
  seed 202: 17.25 / 18.10
  seed 303: 17.03 / 17.97
  seed 404: 17.84 / 17.43   <- best on validation
member mean: validation 17.35, test 17.86; best-on-validation member: test 17.43
five-model ensemble E0: validation 19.54, test 20.20
five-model ensemble E*: validation 19.88, test 20.60
test decodes of the five-model ensemble: two (E0: beam 5, length penalty 1.0; E*: beam 10, length penalty 0.8)
--- predicted power from the validation decodes (plug-in: test effect = validation effect) ---
  E0-meanS  validation effect +2.19, predicted power 0.99
  E0-bestS  validation effect +1.70, predicted power 0.63
  E*-meanS  validation effect +2.53, predicted power 0.99
  E*-bestS  validation effect +2.04, predicted power 0.72
--- test reading ---
max-T over the family of four (B = 2000, alpha 0.05), q = 2.379; observed on test:
  E0-meanS  +2.34  max-T 95% [+1.17, +3.52]  (paired-bootstrap 95% [+1.34, +3.30])
  E0-bestS  +2.78  max-T 95% [+1.08, +4.47]  (paired-bootstrap 95% [+1.36, +4.11])
  E*-meanS  +2.75  max-T 95% [+1.54, +3.95]  (paired-bootstrap 95% [+1.81, +3.80])
  E*-bestS  +3.18  max-T 95% [+1.43, +4.92]  (paired-bootstrap 95% [+1.72, +4.59])
  member mean 17.86; member best on validation 17.43
claim rule E0: both max-T intervals exclude 0 = True; predicted power 0.63 (>= 0.5: True) -> CLAIM
claim rule E*: both max-T intervals exclude 0 = True; predicted power 0.72 (>= 0.5: True) -> CLAIM
descriptive, outside the family: E* minus E0 +0.40 [-0.18, +1.13]
unseen-570 test BLEU-4 (references absent from training, n = 570):
  member mean unseen 13.49
  five-model E0 unseen 16.11, E* unseen 16.39
  E* length ratio 0.937, brevity penalty 0.935; verbatim-train outputs E* 0.585
  11.80-system (S6, beam 10, lp 0.3): test 11.80, unseen 6.60
  best published score on this subset (GASLT): 9.95 (literature value, not recomputed here)

=== Three models fine-tuned with our own code and their ensemble ===
three single models, test BLEU-4: seed 42 18.69, seed 101 17.44, seed 202 18.67; mean 18.27
best on validation: seed 42 (validation 18.14, test 18.69)
test decodes of the three-model ensemble: two (E0: beam 5, length penalty 1.0; E*: beam 10, length penalty 0.8)
three-model ensemble: E0 test 19.65, E* test 19.77
max-T over the family of four (B = 2000, alpha 0.05), q = 2.310; observed on test:
  E0-meanS  +1.39  max-T 95% [+0.29, +2.48]  (paired-bootstrap 95% [+0.51, +2.36])
  E0-bestS  +0.96  max-T 95% [-0.66, +2.57]  (paired-bootstrap 95% [-0.40, +2.40])
  E*-meanS  +1.50  max-T 95% [+0.38, +2.62]  (paired-bootstrap 95% [+0.59, +2.44])
  E*-bestS  +1.07  max-T 95% [-0.55, +2.69]  (paired-bootstrap 95% [-0.26, +2.51])
E*: both max-T intervals exclude 0 = False -> NO CLAIM: bounded estimate only
  three-model E0 unseen-570 test BLEU-4 14.81
  three-model E* unseen-570 test BLEU-4 14.71
  three-model member mean unseen 13.49
gates passed: 32 saved decodes re-scored equal to the BLEU-4 recorded by the original jobs and to slt.metrics
CHECK_DONE
```
**Scope.** Recomputed from the saved decodes: BLEU-4 of every member, ensemble and grid setting (sacreBLEU 13a on detokenised text, each re-scored value asserted equal to the BLEU-4 the original job recorded and to `slt.metrics`, 32 decodes), the validation selection of E*, the paired bootstrap (2,000 sentence resamples, seed 0, shared by all systems), the simultaneous max-T intervals, the predicted power from the validation decodes, the claim rule, and the unseen/seen, length and verbatim-train summaries; the 11.80 system is re-scored from the published dump `results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl`. References come from `comm_run/manifests` (written by `make data`). Not recomputed: training, fine-tuning and the beam-search decoding itself (the checkpoints are licensed CC BY-NC and are not part of the repository), the 9.95 published score (literature value), and the validation grid of the three-model ensemble (its selection is not stated as a number in the text). The five members were fine-tuned with a private port and the three members with our own code; the ensemble decoder is ours. Source jobs: five-model ensemble Kaya jobs 195359 (grid), 195360 (selection), 195361 (test decodes), 195362 (summary) and the clean-code single-model re-decodes of job 193342 (copied from `results/` of the working tree); three-model ensemble jobs 238134 to 238139. Saved files are in `results/checks/pretrained_ensemble/data/` (hypotheses only); see `SOURCES.md` there.

### A37 — Clean-room code and own-code fine-tuning of the pre-trained model
**Claim.** Appendix B and the transfer section of the Results state that an independent implementation, written from the paper and a behavioural description, reproduces the private port's output logits on the released checkpoint to within about 2e-5 on 20 validation clips; that re-decoding the five port-fine-tuned models with our own code gives a test mean of 17.86 (validation 17.35, unseen 13.49) against the port's 18.21 (17.37, 13.73), with most outputs identical; that fine-tuning with our own code gives 17.72 validation, 18.27 test and 13.49 unseen over three seeds, level with the port (+0.35 validation, +0.06 test); and that adding stitched data and pose augmentation (KSA) lowers validation BLEU-4 from 17.72 to 16.84 (mean difference -0.88, -2.99 to +1.23; test -0.64, -3.16 to +1.89; one of three seeds positive on validation) and is stopped by the futility rule (Table `tab:app_transfer`, `tab:app_transfer_diff`).

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/clean_room_reimplementation/check.py
```
```text
gates passed: every run re-scores to the BLEU-4 recorded by its own job (tolerance 1e-6)

[1] Independent implementation vs the private port, released checkpoint (read from the saved job summary gk_job186422.json)
  clips compared: 20; all in the validation manifest: True
  max |logit difference| = 1.9e-05  (rounds to 2e-05; tolerance 1e-03)
  exponent form: 1.9 x 10^-5; tolerance exponent -3
  max relative embedding difference = 5.3e-07; passed = True

[2] The five fine-tuned models decoded with our own code (beam 5, LP 1.0) vs the port's decodes
  seed | split | port BLEU-4 | our code | identical hypotheses
    42 | val   | 17.19 | 17.65 | 678/800 = 84.8%
   101 | val   | 17.68 | 16.98 | 654/800 = 81.8%
   202 | val   | 17.29 | 17.25 | 668/800 = 83.5%
   303 | val   | 17.24 | 17.03 | 680/800 = 85.0%
   404 | val   | 17.44 | 17.84 | 659/800 = 82.4%
    42 | test  | 17.80 | 17.23 | 645/800 = 80.6%
   101 | test  | 18.43 | 18.58 | 664/800 = 83.0%
   202 | test  | 18.16 | 18.10 | 665/800 = 83.1%
   303 | test  | 18.77 | 17.97 | 682/800 = 85.2%
   404 | test  | 17.87 | 17.43 | 658/800 = 82.2%
  val: port mean 17.37 / our code 17.35; unseen-570 11.95 / 11.92
  test: port mean 18.21 / our code 17.86; unseen-570 13.73 / 13.49
  identical hypotheses over the ten seed x split comparisons: 80.6% to 85.2%
  rounded as in the dissertation: 81 to 85% of outputs identical

[3] Fine-tuning with our own code, 3 seeds (42/101/202): K = plain, KSA = + stitched data and pose augmentation
  system | split | mean BLEU-4 | unseen-570 (test only)
  K   | val   | 17.72
  K   | test  | 18.27 | unseen 13.49
  KSA | val   | 16.84
  KSA | test  | 17.63 | unseen 13.21
  KSA minus K, per seed (seed-level 95% t-interval):
  val: -1.50, -1.22, +0.09; mean -0.88 (-2.99 to +1.23); positive in 1 of 3 seeds
  test: -1.20, +0.54, -1.24; mean -0.64 (-3.16 to +1.89); positive in 1 of 3 seeds
  futility rule (3-seed mean validation difference <= 0): STOP

[4] Our code minus the port, fine-tuned models (K 3 seeds minus port 5 seeds)
  val: 17.72 - 17.37 = +0.35
  test: 18.27 - 18.21 = +0.06
```
**Scope.** Re-scores 32 saved decodes (hypotheses against the official references, BLEU-4 with the project's sacreBLEU 13a convention; unseen = the 570 test references absent from training, from `results/train_target_hashes.txt`) and first requires every one to reproduce the BLEU-4 its own job recorded. The five port-fine-tuned models (training job 175088 and its first stage 167035; seeds 42, 101, 202, 303, 404), their clean re-decodes (job 193342) and the three-seed plain and KSA fine-tunes (job 195640) were trained and decoded on the cluster; the check does not train or decode. The seed-level t-intervals and the identical-hypothesis counts are recomputed from the saved decodes. The logit-parity figure (job 186422) is read from the saved job summary: its per-item dumps are derived from the checkpoint and were not kept. The 20 clip names it used are checked against the validation manifest. Not recomputed: the ensemble rows, the ten-point behavioural description, and the finding that the cause of the re-decode difference is unconfirmed. The identical-hypothesis range in the saved data is 80.6% to 85.2%, wider than the 82 to 85% in the text.

### A38 — Published systems run with their own code
**Claim.** The dissertation (Results section on the published systems' code, Table `tab:res_published` and the text below it; Appendix B, Table `tab:app_ownccode`, blocks 1 and 2) states that the authors' own SL-Transformer and GASLT code, run here with three seeds on the signer's AlphaPose keypoints (56 points, 200 epochs at a constant learning rate, batch size 64), scores 8.77 and 9.16 test BLEU-4 against the published 9.20 and 9.95, within the pre-registered tolerance of 1.0 (the intervals of GASLT and of the standardised SL-Transformer cell cross the tolerance edge, so they are only consistent with the published score). On 53 DWPose points the same code scores below this project's translator trained for 40 epochs in the same jobs (differences with 95% intervals, three and five seeds), and DWPose minus AlphaPose inside their code is not resolved.
**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/published_systems_own_code/check.py
```
```text
Gate passed: all 24 authors'-code runs (and the 10 further DWPose runs of the five-seed comparison) re-score to the
  recorded project and authors' BLEU-4 (1e-6); the 3 + 3 + 2 translator control runs re-score to their recorded values;
  references identical across the 24 runs (800 clips).
Training length of every authors'-code run (8 cells x 3 seeds): [(200, 'Training ended after 200 epochs.')]
Seeds per cell: 3 (42, 101, 202)

Block 1: the authors' own code on the signer's AlphaPose keypoints (test BLEU-4, authors' script, 3-seed mean, 95% t-interval)
  SL-Transformer  ap56_raw    8.77  [8.25, 9.29]  published 9.20; |diff| 0.43 <= 1.0: True; interval inside the band: True
  SL-Transformer  ap56_frame  8.87  [7.50, 10.23]  published 9.20; |diff| 0.33 <= 1.0: True; interval inside the band: False  (interval crosses the tolerance edge: consistent with, not shown)
  GASLT           ap56_raw    9.16  [7.36, 10.96]  published 9.95; |diff| 0.79 <= 1.0: True; interval inside the band: False  (interval crosses the tolerance edge: consistent with, not shown)
  GASLT           ap56_frame  9.67  [9.23, 10.11]  published 9.95; |diff| 0.28 <= 1.0: True; interval inside the band: True
  project scoring function, paper-literal cell (ap56_raw), SL-Transformer: 8.66
  project scoring function, paper-literal cell (ap56_raw), GASLT: 8.69

Block 2: authors' code on 53 DWPose points minus this project's translator (40 epochs, beam 5), test BLEU-4, project function
  SL-Transformer  3 seeds (8.44 against 10.30): -1.86  [-2.71, -1.02]
  GASLT           3 seeds (9.70 against 10.30): -0.60  [-1.44, +0.24]
  SL-Transformer  5 seeds (8.63 against 10.43): -1.80  [-2.27, -1.34]
  GASLT           5 seeds (9.58 against 10.43): -0.85  [-1.32, -0.39]
  Authors' code only: DWPose minus the signer's AlphaPose cut to the same 53 points (3 seeds, project function):
    SL-Transformer +0.07 [-0.86, +0.99];  GASLT +0.49 [-1.35, +2.34]
```
**Scope.** Recomputed from the saved per-clip test hypotheses of every run: the authors' scoring (sacreBLEU on the tokenised text, `tokenize='none'`) and this project's scoring (`slt/metrics.py`: detokenise, sacreBLEU 13a) are both re-applied and must equal the values the jobs recorded (the gate, 1e-6); the seed means, t-intervals and pooled-variance differences are computed here. Not recomputed: training and decoding (the runs were made with the authors' code, which is not part of this repository, on a cluster: jobs 197330 to 197332 for the three-seed cells and the in-job translator control; jobs 167758 and 179411 for the five-seed comparison, of which seeds 42, 101 and 202 come from the first and seeds 303 and 404 from the second); the published scores 9.20 and 9.95 are literature values entered as constants; the learning rate, batch size, number of keypoints, the 40-epoch length of the translator and the 56/53-point input layouts are job settings that the saved hypotheses do not carry (the check prints only the 200 epochs the scores files record). The earlier runs of this code on the first-listed AlphaPose person are not used.

### A39 — Rest-cut stitching (wave 16)
**Claim.** Stitched training clips drop both hands to rest between signs far more often than real clips do (4.68 times per clip against 0.25 with tracked keypoints and 0.67 as extracted). A pre-registered test cut those rest frames from the edges of each stitched sign at two depths (12.6% and 8.8% of frames removed, leaving 0.28 and 1.91 pauses per clip). Over five seeds the deeper cut raised test BLEU-4 by 0.29 (+0.13 to +0.45) and the shallower by 0.49 (-0.03 to +1.02), higher in all five seeds for both, with validation gains of +0.19 and +0.18 whose intervals include zero and length-adjusted test gains of +0.20 and +0.34 (Results, appendix table of additional results). Neither cut was added to the adopted configuration.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/rest_cut_stitching/check.py
```
```text
gate 1 OK: all 30 beam-5 decodes (3 arms x 5 seeds x val/test) re-score to the BLEU-4 recorded by the training jobs
gate 2 OK: control (A) test BLEU-4 per seed = [11.711, 10.745, 10.411, 11.305, 10.869] = wave-5 control

=== Five-seed paired differences against the control A (beam 5, best checkpoint) ===
R08 - A  test      mean +0.29 (+0.13 to +0.45), higher in 5 of 5 seeds, one-sided t 4.935, p 0.0039
R08 - A  validation mean +0.19 (-0.36 to +0.74), higher in 4 of 5 seeds; Stage-1 (seeds 42/101/202) mean +0.18
R08 - A  length-adjusted test gain +0.20 (per seed +0.40, -0.10, +0.10, +0.54, +0.07); fresh seeds 303/404 test +0.22, +0.27
R08 mean test/val 11.30/11.31  A 11.01/11.12
R10 - A  test      mean +0.49 (-0.03 to +1.02), higher in 5 of 5 seeds, one-sided t 2.610, p 0.0297
R10 - A  validation mean +0.18 (-0.54 to +0.90), higher in 3 of 5 seeds; Stage-1 (seeds 42/101/202) mean +0.11
R10 - A  length-adjusted test gain +0.34 (per seed +0.27, -1.00, +1.05, +0.82, +0.53); fresh seeds 303/404 test +0.69, +0.90
R10 mean test/val 11.50/11.30  A 11.01/11.12
Holm (m = 2): R08 p 0.0039 vs 0.025 -> reject; R10 p 0.0297 vs 0.05 -> reject

gate 3 OK: the audit's bootstrap interval of the stitched mean is reproduced: [4.596, 4.777]
=== Interior rest runs (both hands down between signs) per clip, 2,000 clips per group ===
stitched clips 4.68 (exact 4.6845); real clips, tracked keypoints 0.25 (exact 0.2450); real clips as extracted 0.67 (exact 0.6660)

=== Depth of the cut (per-segment table of the cut arms; 94,428 stitched signs) ===
R08: frames removed 12.6% as recorded by the build check (cut / (cut + trimmed length); matches its saved value exactly); cut / trimmed length alone: 16.1%; mean frames cut per sign: start 5.08, end 3.57
R08: interior rest runs left per clip (500 clips; read from the saved job summary PREP_OK_R08.json) 0.28
R10: frames removed 8.8% as recorded by the build check (cut / (cut + trimmed length); matches its saved value exactly); cut / trimmed length alone: 10.6%; mean frames cut per sign: start 3.84, end 1.98
R10: interior rest runs left per clip (500 clips; read from the saved job summary PREP_OK_R10.json) 1.91
gate 4 OK: the uncut arm on the same 500 clips gives 4.722 rest runs per clip (the saved build check: 4.722)
```
**Scope.** BLEU-4 is re-scored from the saved beam-5 decodes (best checkpoint, 800 validation and 800 test clips, 13a, detokenised) of the 15 training runs of Kaya job 174916 (arms A, R08, R10; seeds 42, 101, 202, 303, 404), against the restored official references, and must equal the scores recorded by the jobs; the paired differences, t intervals, one-sided t tests, Holm decision and brevity-penalty length adjustment are recomputed from them. The rest-run counts are recomputed from the per-clip table of the audit job 174914 (2,000 clips per group; the audit's bootstrap interval is reproduced). The cut depths are recomputed from the per-sign cut table taken from the build audit of job 174915 (94,428 signs). Not recomputed: training, and the number of rest runs left after the cut (0.28 and 1.91), which was measured on 500 cut clips whose pose arrays are not saved and is read from the saved job summaries `PREP_OK_R08.json` and `PREP_OK_R10.json` (the same 500 clips of the uncut arm are recomputed, 4.722). The "frames removed" percentages are the build check's statistic, cut frames divided by cut plus trimmed-exemplar frames; as a share of the trimmed exemplar alone the cut is 16.1% and 10.6%, printed beside them.

### A40 — 69-keypoint input, graph front ends and the tracked-keypoint check
**Claim.** The dissertation (Results, "Negative Results"; Appendix B, `tab:app_additional`, the 69-keypoint rows) states that a 69-keypoint input scaled per body part gave an unresolved small positive effect on the translation score. In the first submission it gave validation +0.35 (-0.85 to +1.56) and test +0.86 (-2.75 to +4.48) over three seeds; in a second submission with its own control, validation +0.64 (-0.10 to +1.37) and test +0.02 (-1.79 to +1.82). The same configuration scored up to 1.66 apart on test between the two submissions. Over five seeds the gain was +0.94 on validation (-0.28 to +2.16) and +0.94 on test (-0.35 to +2.23), with a one-sided test p of 0.057, and about 0.35 of the test gain came from longer output. A 133-keypoint input with median shoulder-width scaling lowered validation by 0.63, and two graph-convolution front ends on the 69 keypoints (15.7 M against 10.6 M parameters) scored 1.50 and 1.48 lower on validation than the plain linear input layer. The gain of the 69-keypoint input was not shown to differ between tracked and first-detection keypoints (difference of gains +0.09 on validation), and the check could not detect switches because the two keypoint sets changed the body scale by more than 10% in only 14 of 200 sampled clips.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/keypoints_69_input/check.py
```
```text
GATE PASSED: 183 of 183 values reproduced (BLEU-4 of every saved decode = the value its job recorded, to 1e-6; sweep at LP 0.6 = GPU decode; F69B seed 42 = F69 seed 42)
references: validation 800 clips, test 800; unseen-target clips 549 (validation), 570 (test)

=== Settings ===
input width per frame (x, y, confidence): 69-keypoint arms 207 = 3 x 69 keypoints; 133-keypoint arms 399 = 3 x 133
  every 69-keypoint run has width 207; every 133-keypoint run 399
first submission: job set 168265: arms ctrl, C133, F69, seeds 42, 101, 202 (3 seeds each)
second submission: job set 168266: arms ctrlB, F69B, P69TC, SG, seeds 42, 101, 202 (3 seeds each, with its own control ctrlB)
parameters: linear front end on 69 keypoints (F69) 10.60 M, control 10.65 M; graph front ends (P69TC, SG) 15.74 and 15.74 M -> 15.7 M against 10.6 M

=== First submission (array A): 69-keypoint input and 133-keypoint median scaling against the control ===
  ctrl  beam-5 val/test s42 11.307/11.711  s101 11.939/11.366  s202 11.199/10.492 | means 11.482/11.190
  C133  beam-5 val/test s42 10.516/10.924  s101 11.451/11.427  s202 10.584/11.698 | means 10.850/11.350
  F69   beam-5 val/test s42 11.856/11.109  s101 11.739/12.250  s202 11.909/12.800 | means 11.835/12.053
-- F69 - ctrl  (beam 5, best.pt, length penalty 0.6; job set 168265)
  validation                         per seed +0.549 -0.199 +0.711 | mean +0.35 95% t [-0.85, +1.56] | higher in 2 of 3
  test                               per seed -0.602 +0.884 +2.309 | mean +0.86 95% t [-2.75, +4.48] | higher in 2 of 3
  unseen-target validation           per seed +0.948 +0.695 +0.792 | mean +0.81 95% t [+0.49, +1.13] | higher in 3 of 3
  length matching (each arm at its own validation-best length penalty: F69 1.0, ctrl 1.0):
  validation, F69* - ctrl*           per seed +0.892 +0.218 +0.677 | mean +0.60 95% t [-0.26, +1.45] | higher in 3 of 3
-- C133 - ctrl  (beam 5, best.pt, length penalty 0.6; job set 168265)
  validation                         per seed -0.792 -0.488 -0.614 | mean -0.63 95% t [-1.01, -0.25] | higher in 0 of 3
  test                               per seed -0.787 +0.061 +1.207 | mean +0.16 95% t [-2.33, +2.65] | higher in 2 of 3

=== Second submission (array B): the same 69-keypoint configuration against its own control ===
  ctrlB beam-5 val/test s42 10.902/11.826  s101 11.939/11.366  s202 11.338/10.411 | means 11.393/11.201
  F69B  beam-5 val/test s42 11.856/11.109  s101 12.306/11.399  s202 11.927/11.144 | means 12.029/11.218
  P69TC beam-5 val/test s42 10.832/10.569  s101 10.709/9.941  s202 10.062/9.542 | means 10.534/10.018
  SG    beam-5 val/test s42 10.434/9.794  s101 10.789/10.085  s202 10.440/10.598 | means 10.554/10.159
-- F69B - ctrlB  (beam 5, best.pt, length penalty 0.6; job set 168266)
  validation                         per seed +0.954 +0.367 +0.589 | mean +0.64 95% t [-0.10, +1.37] | higher in 3 of 3
  test                               per seed -0.717 +0.033 +0.733 | mean +0.02 95% t [-1.79, +1.82] | higher in 2 of 3
  unseen-target validation           per seed +1.125 +1.063 +1.223 | mean +1.14 95% t [+0.94, +1.34] | higher in 3 of 3
  length matching (F69B at 1.0, ctrlB at 0.8):
  validation, F69B* - ctrlB*         per seed +0.925 +0.278 +1.017 | mean +0.74 95% t [-0.26, +1.74] | higher in 3 of 3
-- same configuration in the two submissions (F69B - F69, raw, not a contrast)
  test per seed +0.000 -0.851 -1.656 | largest absolute difference 1.66 | validation per seed +0.000 +0.566 +0.017

=== Graph-convolution front ends on the 69 keypoints (array B) ===
-- P69TC against the plain linear input layer on the same input (F69B)
  validation                         per seed -1.024 -1.596 -1.865 | mean -1.50 95% t [-2.56, -0.43] | higher in 0 of 3
  test                               per seed -0.540 -1.458 -1.602 | mean -1.20 95% t [-2.63, +0.23] | higher in 0 of 3
-- P69TC against the 133-keypoint control (ctrlB)
  validation                         per seed -0.070 -1.229 -1.276 | mean -0.86 95% t [-2.56, +0.84] | higher in 0 of 3
  test                               per seed -1.257 -1.425 -0.869 | mean -1.18 95% t [-1.89, -0.48] | higher in 0 of 3
-- SG against the plain linear input layer on the same input (F69B)
  validation                         per seed -1.422 -1.516 -1.487 | mean -1.48 95% t [-1.60, -1.36] | higher in 0 of 3
  test                               per seed -1.315 -1.314 -0.546 | mean -1.06 95% t [-2.16, +0.04] | higher in 0 of 3
-- SG against the 133-keypoint control (ctrlB)
  validation                         per seed -0.468 -1.149 -0.898 | mean -0.84 95% t [-1.69, +0.02] | higher in 0 of 3
  test                               per seed -2.032 -1.281 +0.187 | mean -1.04 95% t [-3.85, +1.76] | higher in 1 of 3
-- skeleton graph over identity graph (SG - P69TC)
  validation                         per seed -0.398 +0.080 +0.378 | mean +0.02 95% t [-0.95, +0.99] | higher in 2 of 3

=== Five seeds of the 69-keypoint input against the control ===
seeds 42, 101, 202, 303, 404; job sets 42: 168265, 101: 168265, 202: 168265, 303: 172447, 404: 172447 (the first three from the first submission, the fresh seeds 303 and 404 from the later one)
  validation                         per seed +0.549 -0.199 +0.711 +2.455 +1.168 | mean +0.94 95% t [-0.28, +2.15] | higher in 4 of 5
  test                               per seed -0.602 +0.884 +2.309 +1.194 +0.895 | mean +0.94 95% t [-0.35, +2.23] | higher in 4 of 5
  fresh seeds only (303, 404): validation +2.46 +1.17, test +1.19 +0.90: positive on both splits in 4 of 4
  unseen-target validation mean +1.18, test mean +1.29
  one-sided t test on test: t = 2.014, p = 0.057
  test gap at the control's brevity penalty +0.588 of the raw +0.936 -> longer output accounts for 0.35 of the 0.94 test gain

=== Tracked keypoints (second tracker) against first-detection keypoints, 69-keypoint input ===
  gain on first-detection (validation) per seed +1.354 +0.452 -0.251 | mean +0.52 95% t [-1.48, +2.52] | higher in 2 of 3
  gain on tracked (validation)       per seed +0.405 +0.161 +1.267 | mean +0.61 95% t [-0.83, +2.05] | higher in 3 of 3
  difference of the two gains        per seed -0.950 -0.291 +1.518 | mean +0.09 95% t [-3.08, +3.27] | higher in 1 of 3
  body scale differs by more than 10% between the two keypoint sets in 14 of 200 sampled clips (read from the saved job summary trackpose_pose_gate.json: the keypoint arrays are not shipped)
```
**Scope.** Recomputed: the beam-5 BLEU-4 of every saved decode (best checkpoint, length penalty 0.6, validation and official test) of the 25 training runs of the three job sets (and 12 further runs of the tracked-keypoint comparison, validation only), re-scored against the references restored by `make data` with the project's own scoring (`slt/metrics.py`, detokenised, 13a); the per-seed differences, their seed-level 95% t-intervals and sign counts; unseen-target BLEU (references absent from the training text); each arm at its own validation-best length penalty from the saved length-penalty sweeps (CPU re-decodes of the first and second submission's controls and 69-keypoint arms, 9 penalties each); the brevity share of the five-seed test gain; the one-sided test; and the difference of the two gains on tracked and first-detection keypoints from 12 saved validation decodes. The gate first reproduces the BLEU-4 each original job recorded (183 values to 1e-6). Original jobs: first submission 168265 (training array; sweeps 171802/171804), second submission 168266 (sweeps 171803/171805), the five-seed continuation 172447 (seeds 303 and 404; sweep 172448), the tracked-keypoint comparison 179416 (pose check 179415). Not recomputed (read from saved job files, as the output states): the parameter counts (the training script's smoke test, job 168264), the input widths (the first lines of each training log) and the 14 of 200 clip count (the pose check's saved summary, because the keypoint arrays are not shipped). No training or inference is repeated; the layout of the 69 keypoints is defined in training code that is not part of this repository.

### A41 — Three training ideas (sign targets, re-timing, preference stage)
**Claim.** Appendix B (Table of further results, row "Three training ideas") states that three ideas were each trained with three seeds in one job: CTC targets set to the signs shown (sign targets), every stitched sign re-timed by a random factor each epoch (re-timing), and a ten-epoch preference stage against a ten-epoch continuation. Against their in-job controls none is resolvable: sign targets validation +0.28 (-0.65 to +1.20), test -0.31 (-1.80 to +1.18), and with five seeds validation +0.19 and test +0.33, not adopted; re-timing validation +0.26, test +0.15, test content-word recall +1.27 points, and with five seeds validation +0.21 and test +0.24, not adopted (one fresh seed negative on test); preference stage validation -0.31, test -0.15, recall -0.34 points, not positive. The adopted CTC head places 47.3% of 603 annotated sign spans correctly against 26.2% for random placement, so the alignment gate (50% and twice random) failed.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/three_training_ideas/check.py
```
```text
GATE: all 42 decodes re-score to the BLEU-4 stored by their jobs; A seed 42 test = 11.715239 (recorded 11.715239)

Setup: 15 Stage-1 runs (5 arms x 3 seeds) all marked done by job ['144831']; E1/ST/P seeds [42, 101, 202] -> three seeds each, one job
Epochs: P (preference stage) 10/10/10 ; AC (continuation control) 10/10/10 -> ten-epoch stage and ten-epoch continuation

Stage 1, beam-5, seed-level 95% t-interval over seeds 42/101/202 (df 2), paired against the in-job control
Sign targets      validation +0.28 [-0.65, +1.20] (2/3 seeds higher) | test -0.31 [-1.80, +1.18] (1/3) | test content-word recall +0.82 [-1.56, +3.19] points | output length ratio val +0.013 (2/3 longer)
Re-timing         validation +0.26 [-1.81, +2.33] (1/3 seeds higher) | test +0.15 [-0.61, +0.92] (2/3) | test content-word recall +1.27 [-1.16, +3.70] points | output length ratio val +0.026 (3/3 longer)
Preference stage  validation -0.31 [-2.07, +1.44] (1/3 seeds higher) | test -0.15 [-1.36, +1.06] (1/3) | test content-word recall -0.34 [-2.47, +1.79] points | output length ratio val -0.000 (2/3 longer)
Sign targets: validation mean +0.28 > 0 -> passes the futility screen (stopped only if <= 0)
Re-timing: exploratory rule (validation, test and test recall means all > 0): POSITIVE  (val +0.26, test +0.15, recall +1.27)
Preference stage: exploratory rule (validation, test and test recall means all > 0): NOT POSITIVE  (val -0.31, test -0.15, recall -0.34)

Stage 2, five seeds 42/101/202/303/404 (df 4), paired against A
Sign targets  with five seeds, validation +0.19 [-0.18, +0.56], test +0.33 [-1.02, +1.68]; fresh seeds 303/404 test +0.60 +1.98 (2/2 positive)
Re-timing     with five seeds, validation +0.21 [-0.56, +0.98], test +0.24 [-0.48, +0.97]; fresh seeds 303/404 test -0.37 +1.12 (1/2 positive)
E1: one-sided t(4) on the five test differences = 0.681 (p = 0.267); Holm level 0.0100 needs t >= 3.747 -> not rejected, not adopted  [levels from the pre-registration: E1 0.01, ST 0.05/6]
ST: one-sided t(4) on the five test differences = 0.933 (p = 0.202); Holm level 0.0083 needs t >= 3.961 -> not rejected, not adopted  [levels from the pre-registration: E1 0.01, ST 0.05/6]

Alignment of the adopted CTC head (read from the saved job summary g1_alignment_summary.json; needs the checkpoints): 603 annotated sign spans: 47.3% hit against 26.2% for random placement
Gate rule: hit >= 0.50 and hit >= 2 x random and hit > shuffled, on both sets, means over checkpoints; hit >= 50%: False; hit >= 2 x random: False -> gate failed (saved verdict FAIL)
```
**Scope.** Recomputed: BLEU-4 (sacreBLEU 13a on detokenised text, `slt/metrics.py`) and content-word recall from the saved beam-5 decodes of the best checkpoint (validation and test, 5 arms x seeds 42/101/202 from job 144831, plus A/E1/ST x seeds 303/404 from the five-seed follow-up job 152915), each first checked against the BLEU the job itself stored; then the paired differences, seed-level t-intervals, seed counts and the five-seed one-sided t statistics. Not recomputed: training, the epoch counts (read from the saved per-run progress files), the 0.7 to 1.3 re-timing range (it lives in unpublished training code), and the alignment hit rates, which are read from the saved job summary (`g1_alignment_summary.json`) because recomputing them needs the model checkpoints and the sign-spotting annotation. The Holm levels are the ones fixed in the pre-registration, not recomputed from the other experiments of that family.

### A42 — Extra pre-training cycle with fresh seeds
**Claim.** Appendix B (the supplementary-results table) reports a pre-registered re-test of an extra first training cycle: 92 epochs on the adopted corpus before the usual 40-epoch training, on five new seeds. Against training without it, beam-5 test BLEU-4 was lower by 0.56 (interval -1.41 to +0.28), higher in one seed of five; validation was higher by 0.26 (-0.12 to +0.65); it was not adopted (one-sided p = 0.93). A first cycle of the same number of steps on the 13,760 synthetic clips alone scored -0.07 (-0.59 to +0.45) against no first cycle and +0.49 (-0.26 to +1.25) against the joint one; these are bounds only.
**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/extra_pretraining_cycle/check.py
```
```text
GATE: 30 decodes re-scored; max |recomputed - BLEU recorded by the jobs| = 0.00e+00
GATE: PJ seed 808 beam-5 test = 10.080669 (recorded in the read-out: 10.080669)
first cycle on the adopted corpus (PJ): [92] epochs; then main training: [40] epochs; seeds: 5 ([505, 606, 707, 808, 909])
first cycle on the synthetic clips alone (PS): [175] epochs
read from the job script header: PJ = pre-train 92 epochs on the adopted joint corpus (12,439 real + 13,760 synthetic clips = 26,199;
read from the job script header: PS = pre-train 175 epochs on the 13,760 synthetic clips of that corpus only (2,408,000 clip-passes,
read from the job script header: N_TRAIN=12439; N_SYN=13760; N_VAL=800; N_TEST=800

PJ - C, beam-5 test (primary): -0.56 (-1.41 to +0.28)  [-0.562 [-1.407, +0.284]]  higher in 1 of 5 seeds
   per seed: -1.230 / +0.466 / -0.456 / -1.136 / -0.452
PJ - C, beam-5 validation: +0.26 (-0.12 to +0.65)  [+0.265 [-0.122, +0.651]]  higher in 3 of 5 seeds
one-sided paired t on the five test differences: t = -1.844, p = 0.93 (0.931); Holm level 0.05 (m = 1): retain
both fresh seeds (808, 909) positive on test: False

PS - C, beam-5 test (synthetic first cycle vs none): -0.07 (-0.59 to +0.45)  [-0.069 [-0.589, +0.452]]  higher in 3 of 5 seeds
PS - PJ, beam-5 test (synthetic vs joint first cycle): +0.49 (-0.26 to +1.25)  [+0.493 [-0.264, +1.250]]  higher in 4 of 5 seeds

five-seed means, beam-5 val / test: C 11.385 / 11.336; PJ 11.650 / 10.775; PS 11.234 / 11.268
```
**Scope.** Re-scores the 30 saved beam-5 decodes of the best checkpoints (arms: no first cycle, joint first cycle, synthetic-only first cycle; seeds 505, 606, 707, 808, 909; validation and test; length penalty 0.6) through `slt/metrics.py` (detokenised, sacreBLEU 13a). The gate requires every recomputed score to equal the BLEU the original jobs recorded (maximum deviation 0) and the seed-808 joint-arm test score recorded in the read-out. Differences, 95% t-intervals (exact quantile, 4 degrees of freedom) and the one-sided paired t-test are then recomputed from the per-seed scores. The epoch counts come from the saved per-run progress files; the 13,760 synthetic clips are read from the header of the job script (not re-counted from clips, which are not shipped). Training is not re-run. Original jobs: Kaya 138389 (seeds 505-707, summary 138390) and 140946 (all seeds, summary 140947); the decodes were copied from `results/wave12h_140946/`.

### A43 — News: stitched data and pose augmentation separated
**Claim.** On the News subset, the adopted bundle (stitched clips plus pose augmentation) gains +0.53 BLEU-4 over the real-only control on the official News test (-0.40 to +1.45, seed-level 95% t-interval, three seeds). Separating the two parts gives stitched data alone +0.22 (-0.84 to +1.27) and augmentation added on top of it +0.31 (-0.50 to +1.12); augmentation alone (a later arm, three seeds, descriptive) gives +0.19 (-0.50 to +0.88), not lower in any seed, and its seeds 101 and 202 are compared with controls from an earlier job submission. No part is attributed (Appendix B, supplementary results table).

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/news_stitching_vs_augmentation/check.py
```
```text
gate: all 24 beam-5 BLEU-4 (13a, detokenised) recomputed from per-sentence n-gram counts equal the values recorded by the jobs
control nA seed 42 re-run in job 172449_0: 700/700 identical hypotheses on val, BLEU 2.00944 vs 2.00944
control nA seed 42 re-run in job 172449_0: 699/699 identical hypotheses on test, BLEU 1.87272 vs 1.87272
nRA jobs: s42 172449_9, s101 172449_10, s202 172449_11 | controls nA jobs: s42 138377_0, s101 138377_1, s202 138377_2
nRA seeds 101 and 202 are compared with controls from an earlier job submission (138377); seed 42's control was reproduced above
seeds per arm: {'nA': 3, 'nS': 3, 'nSA': 3, 'nRA': 3}
nA   beam-5 BLEU-4 mean  val 1.97  test 1.70
nS   beam-5 BLEU-4 mean  val 2.26  test 1.92
nSA  beam-5 BLEU-4 mean  val 2.15  test 2.23
nRA  beam-5 BLEU-4 mean  val 2.16  test 1.89
both together (B) = nSA - nA  test: per seed +0.23 +0.95 +0.40  positive 3/3  mean +0.53 (+0.525) [-0.40, +1.45]
both together (B) = nSA - nA  val: per seed -0.19 +0.35 +0.35  positive 2/3  mean +0.17 (+0.172) [-0.60, +0.95]
stitched alone (S) = nS - nA  test: per seed +0.25 +0.62 -0.23  positive 2/3  mean +0.21 (+0.215) [-0.84, +1.27]
stitched alone (S) = nS - nA  val: per seed +0.39 +0.47 -0.00  positive 2/3  mean +0.29 (+0.286) [-0.35, +0.92]
augmentation added (A) = nSA - nS  test: per seed -0.02 +0.33 +0.63  positive 2/3  mean +0.31 (+0.310) [-0.50, +1.12]
augmentation added (A) = nSA - nS  val: per seed -0.58 -0.12 +0.36  positive 1/3  mean -0.11 (-0.114) [-1.28, +1.05]
augmentation alone (R) = nRA - nA  test: per seed +0.00 +0.51 +0.07  positive 3/3  mean +0.19 (+0.193) [-0.50, +0.88]
augmentation alone (R) = nRA - nA  val: per seed +0.29 +0.11 +0.16  positive 3/3  mean +0.19 (+0.186) [-0.05, +0.42]
```
**Scope.** Recomputes the 24 beam-5 BLEU-4 scores (sacreBLEU 13a on detokenised text, best checkpoint, validation and test) from per-sentence n-gram match counts and lengths, requires each to equal the value recorded by its job, and then recomputes the seed-level contrasts and t-intervals (df = 2, t = 4.303, as `tools/seed_decision.py`). News reference sentences are not part of the spreadsheet that `make data` restores, so the saved files hold the per-sentence counts instead of the references; the hypotheses are in clear. Original jobs: 138377 (arms nA, nS, nSA, seeds 42/101/202; read out by 138378) and 172449 (nRA, plus a re-run of nA seed 42 whose 700/700 validation and 699/699 test hypotheses equal the original run). Not recomputed: training. Note: the stitched-alone test mean is +0.2150 (+0.21497 before rounding), which two-decimal rounding writes as +0.21; the dissertation prints +0.22 (rounded from the three-decimal value +0.215).

### A44 — Ensemble averaging rule (power mean)
**Claim.** The final ensemble combines its six members by the geometric mean of their probabilities. In a pre-registered comparison on validation (Results, ensemble-ablation paragraph; Appendix B, further-results table), no power mean of the members' probabilities with exponent -1, 0.5, 1 or 2 was distinguishable from the geometric mean: the BLEU-4 differences, each rule at its own length penalty, are -0.68, 0.00, +0.13 and +0.23 with 98.75% intervals that all include zero, and the nominal gain of exponents 1 and 2 is carried by output length (brevity penalty 0.80 for the geometric mean against 0.84 and 0.92) while n-gram precision falls.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/ensemble_averaging_rule/check.py
```
```text
GATE PASSED: control (geometric mean) at LP 0.3: BLEU-4 12.47 (12.470), WER 0.9164; hypotheses identical to job 120153: 800/800
GATE PASSED: recomputed point differences and 98.75% bootstrap intervals equal those saved by job 138383

Rules compared: geometric mean and power means of the members' probabilities with exponents -1, 0.5, 1, 2 (800 validation clips, six members, beam 10; no test clip)
Interval level: 98.75% per comparison (Bonferroni over 4 comparisons), 2000 paired resamples, length penalty re-chosen inside each resample

Each rule at its validation-chosen length penalty:
                          rule   LP  BLEU-4     BP n-gram prec.
   geometric mean (exponent 0)  0.3   12.47  0.797        15.65
       power mean, exponent -1  0.4   11.79  0.838        14.07
      power mean, exponent 0.5  0.4   12.47  0.812        15.35
        power mean, exponent 1  0.5   12.60  0.843        14.96
        power mean, exponent 2  0.7   12.70  0.919        13.82

BLEU-4 difference to the geometric mean, each rule at its own length penalty:
  exponent   -1: -0.68 (-1.79 to +0.10)   [-0.677; 98.75% interval [-1.790, +0.100]]   not distinguishable from the geometric mean
  exponent  0.5: 0.00 (-0.59 to +0.60)   [-0.001; 98.75% interval [-0.591, +0.602]]   not distinguishable from the geometric mean
  exponent    1: +0.13 (-0.53 to +0.74)   [+0.133; 98.75% interval [-0.530, +0.740]]   not distinguishable from the geometric mean
  exponent    2: +0.23 (-0.60 to +1.05)   [+0.230; 98.75% interval [-0.601, +1.050]]   not distinguishable from the geometric mean

Brevity penalty (dissertation rounding): geometric mean 0.80, exponent 1 0.84, exponent 2 0.92; n-gram precision falls from 15.65 to 14.96 (exponent 1) and 13.82 (exponent 2)

Rules measurably different from the geometric mean: 0 of 4
```
**Scope.** Re-chooses the candidate at every length penalty from the saved n-best lists (beam 10, six members, 800 validation clips; Kaya decode job 138382, one list per rule), re-scores them with `slt/metrics.py` (13a on detokenised text), and recomputes the paired bootstrap (2,000 resamples, seed 0, length penalty re-chosen in each resample, Bonferroni level 98.75%) with the original read-out's own functions (`tools/w12d_powermean.py`). Gates: the control reproduces the reported ensemble (12.470, WER 0.9164, 800/800 hypotheses equal to job 120153's six-member decode in `results/ensemble_member_subsets_ablation/`), and the recomputed differences and intervals equal those saved by the read-out job 138383. The decoding itself (running the six checkpoints with each averaging rule) is not repeated here. The n-best lists were copied from `results/wave12d_138382/out/` and split in halves to keep each file small; the saved job summary is used only for the second gate.

### A45 — Non-signer negative control: confidence and grounding score
**Claim.** The confidence threshold does not tell signing from not signing (Results, paragraph "Negative control with non-signing people", `fig:res_nonsigner_plot`; Discussion; Appendix B rows on the non-signer control). At the threshold fitted on the validation signers for 50% coverage the 40-epoch pre-trained model voices 53.9% of the 831 test crops of non-signing people and 53.5% of test signers (ratio 1.01, 95% interval 0.92 to 1.10, clustered by video; area under the ROC curve 0.51). The final six-member ensemble voices 43.8% against 51.9% (ratio 0.84, 0.75 to 0.94; area under the curve 0.56): a partial effect, with more than four non-signing people in ten still voiced. A grounding score (the log-probability margin of each output word given the true video over an empty or mismatched input) separates non-signers no better than confidence.
**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/nonsigner_confidence_gate/check.py
```
```text
GATE PASSED: confidence-gate ratio, interval and AUROC of both systems equal the values recorded by the original jobs
Crops: 847 test + 831 validation non-signer crops registered; 831 test and 805 validation crops after removing the 42 flagged as the signer's own person (flag list read from the saved job summary)
Test videos: signers 800 clips, crops 831; threshold fitted on the 800 validation signers

== Single model (40-epoch pre-trained), threshold fitted on validation for 50% coverage (realised 50.0% on validation)
  voiced: test signers 53.5% ; non-signers 53.9% ; ratio 1.01 (95% CI 0.92 to 1.10, clustered by video)
  AUROC 0.51 [0.48, 0.54] (0.50 = chance)
  non-signers who hardly move (wrist speed below the 10th percentile of validation signers; 390 crops) voiced: 50.3%
  a signer's pose frozen at one frame voiced: 46.5% of test signers

== Final six-member ensemble, production pose, threshold for 50% coverage
  voiced: test signers 51.9% ; non-signers 43.8% ; ratio 0.84 (95% CI 0.75 to 0.94)
  more than four non-signing people in ten still voiced: True
  non-signers voiced, whole percent: 44%; whole non-signers per ten voiced: 4 of 10
  AUROC 0.56 [0.53, 0.60]
  validation videos (805 crops): ratio 0.85 (0.77 to 0.93)
  threshold for 75% coverage, test: ratio 0.91 (0.85 to 0.98)
  person-tracked pose, test: ratio 0.84 (0.75 to 0.93)
  outputs of the 364 voiced non-signer crops that are verbatim training sentences: 99%

== Grounding score in place of confidence (wave 18), production pose, test, threshold for 50% coverage
  final ensemble: non-signers voiced / signers voiced, grounding 0.95 (0.86 to 1.04) against 0.84 for confidence; 831 test crops
    AUROC grounding 0.52 against 0.56 for confidence (difference -0.05, -0.08 to -0.01)
    true video beats a mismatched one (same output words) on 98.8% of test signer clips and 97.1% of non-signer crops
  40-epoch pre-trained model: non-signers voiced / signers voiced, grounding 1.01 (0.91 to 1.10) against 1.01 for confidence; 831 test crops
    AUROC grounding 0.49 against 0.51 for confidence (difference -0.02, -0.05 to +0.01)
    true video beats a mismatched one (same output words) on 98.6% of test signer clips and 98.9% of non-signer crops

GATE PASSED: grounding-score ratios and intervals equal the wave-18 record
```
**Scope.** Recomputed from saved per-clip outputs: voiced shares, ratios, video-cluster bootstrap intervals (2,000 resamples, seed 0, the resamples of `tools/nonsigner_gate.py`), areas under the ROC curve, paired differences, the verbatim-training-sentence share (against `results/train_target_hashes.txt`) and the true-versus-mismatched-video shares. The check first reproduces the recorded ratios, intervals and areas of both systems to 1e-9, and the recorded grounding-score ratios. The per-clip confidence values are the saved decodes of the 800 test and 800 validation signer clips and of the 1,678 non-signer crops (single model: local decode, `comm_dwpose_vlp_ctc`, beam 5, length penalty 0.6; six-member ensemble: Kaya jobs 138384/138385); the per-clip grounding scores come from Kaya array job 171529 (summary 171530). Not recomputed, read from saved job outputs: the list of 42 crops flagged as the signer's own person (needs the pose arrays), the per-clip grounding scores and the per-clip wrist speeds used for the "hardly move" stratum (computed once from pose arrays that are not shipped; `prep_features.py`). Training, pose estimation and decoding are not rerun.

### A46 — Speech chain: latency by stage and voiced output by coverage
**Claim.** The Results chapter (Section "Speech Output and Confidence-Based Abstention", Table `tab:res_speech_latency`, Figure `fig:res_latency_plot`, and the Discussion) states that over the 800 official Communication test clips, with the 40-epoch pre-trained pose-only model at beam width 5, the median delay from the end of signing to the start of speech is 2.75 s (95th percentile 5.27 s), of which pose estimation (estimated from clip length at 25.4 frames per second) is 71.7%, speech synthesis 24.1% and translation 3.3%. Appendix Table `tab:res_speech_coverage` and the Results text state the BLEU-4 of the voiced output as the confidence threshold tightens (10.27 at 100% coverage, 19.19 at 50%).

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/speech_chain/check.py
```
```text
references match comm_run/manifests/test.tsv: 800/800
GATE all 800 voiced: BLEU-4 10.2691 (recorded 10.27 for the 40-epoch pre-trained model)

Setting: 800 official test clips, beam 5, estimated pose time = frames / 25.4 fps
pose-only CPU rate 25.4 frames/s (read from the saved job summary pose_throughput_subset.json)

LATENCY per utterance (ms)
Stage                          Median     95th   Share
Pose estimation (estimated)    1968.5   4448.8   71.7%
Translation (beam width 5)       91.8    217.7    3.3%
Speech synthesis                660.8    697.1   24.1%
Translation + synthesis         752.6    901.5   27.4%
Whole chain                    2746.7   5271.0  100.0%

median delay 2.75 s, 95th percentile 5.27 s
translation with this benchmark's own beam search: median 92 ms

Detector on every frame: 5.2 frames/s (read from the saved job summary); median clip 50 frames -> pose estimation alone 9.6 s
Batched beam-5 search, single pose-only translator: mean 36 ms per clip (100 test clips; read from the saved job summary)

VOICED OUTPUT by coverage (threshold = quantile of the test-set mean per-token cross-entropy)
 Coverage  Voiced  Threshold   BLEU-4
    100%      800        ---    10.27
     90%      720      0.941    11.27
     75%      600      0.826    12.88
     50%      400      0.681    19.19
     25%      200      0.523    33.54
```
**Scope.** Recomputed from the per-clip records saved by `tools/speech_pipeline_bench.py` (800 test clips: hypothesis, reference, confidence, clip length, measured translation and synthesis time): stage medians and 95th percentiles, shares of the median chain, and corpus BLEU-4 of each voiced subset through `slt/metrics.py` (detokenised, sacreBLEU 13a), with the thresholds as quantiles of the saved confidence scores. The gate is that all 800 hypotheses score 10.27, the recorded test score of the 40-epoch pre-trained model; this also identifies the model, since the 80-epoch model scores 10.71. The per-clip translation and synthesis times are the original laptop (Apple M3 Max CPU) measurements; they are not re-timed here, and no inference is run. The pose stage was never run per clip: it is estimated, as in the dissertation, as frames / 25.4. Three numbers come from the saved summary of the pose-throughput benchmark (`tools/pose_throughput_bench.py`, no per-frame timings kept) and are printed as "read from the saved job summary": 25.4 frames/s (pose model alone, detector once per clip), 5.2 frames/s (detector on every frame), and 36 ms (mean batched beam-5 decode of a single pose-only translator over 100 test clips); the 9.6 s is the median clip length (50 frames, recomputed) divided by 5.2 frames/s. Files: `records.jsonl` (copied from `results/speech_pipeline_bench.json`), `pose_throughput_subset.json` (subset of `results/pose_throughput_m3max_2026-09-15.json`); see `results/checks/speech_chain/SOURCES.md`.

### A47 — ROUGE-L tokenisation convention

**Runs with:** this repository and the dataset spreadsheet

```bash
python3 ../results/checks/rouge_convention/check.py
```

Expected values are the frozen dissertation's table values. Whitespace is normalised; labels, values, ordering and repetitions must agree.

```text
Baseline model (DWPose + CTC) n=800 16.56 36.22 35.70 [ok | ok | ok]
40-epoch pre-trained model n=800 17.74 36.68 36.27 [ok | ok | ok]
RGB-fusion model n=800 17.84 37.67 37.08 [ok | ok | ok]
more than 19 points for the baseline and the RGB-fusion outputs: True
convention: ROUGE-L = mean over sentences of the F_beta of the longest common subsequence, x100
range for the three single models: 1.119 to 1.151 [ok]
```

This recomputes ROUGE and the ratio from saved outputs. It does not reproduce training or certify the originally executed beam-search setting. See claims.json for the partial gap.

### A48 — Citation versus conversational signs: clip lengths and durations
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


### A49 — Dataset statistics

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


### A50 — Real-clip lengths and frame rates behind the temporal-fidelity argument

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


### A51 — Ensemble member-set ablation and member-set selection
**Claim.** The dissertation (Results, Discussion, Appendix B, abstract) reports an ablation of the final six-member ensemble over all 63 member subsets on validation (kinds of member differ by +0.56, 95% interval -0.57 to +1.97), that validation and test BLEU-4 of four ensemble member sets are uncorrelated (r = -0.014), that three ensembles scored above 11.80 on test (12.11, 12.84 and 12.73), that the third, rebuilt from three pose-augmented models of the adopted configuration and two RGB-fusion models, was ahead in 91.5% of 1,000 validation halvings, three test-selected configurations (11.15, 11.51, 11.70), the video-disjoint ensemble (7.84 against 7.29 for the validation-selected single model), and that paired intervals include zero for most improvements, so the 12.11 gain was withdrawn.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/ensemble_member_subsets/check.py
```
```text
GATE six-member set, validation beam 10 / LP 0.3: BLEU-4 12.470 WER 0.9164 (recorded 12.470 / 0.9164, job 120153)

Decoded and scored 63 non-empty subsets of the 6 members (VLP80, RGBfus, VLP40, VLP40s202, fus101, fus202) on 800 validation clips  [c12-017]
Number of members 6, number of subsets 63  [c12-017]
Members added in the recorded order, validation BLEU-4: 9.78, 11.08, 11.47, 11.57, 11.78, 12.47  [c18-058]
Shapley sums: fusion members 6.51, pose members 5.96  [c12-021]
Fusion minus pose Shapley sum +0.56, 95% interval -0.57 to +1.97 (500 resamples)  [c12-021]
read from the saved job output ensemble_member_subsets_ablation/subsets_scores.json for comparison: type_diff [-0.5656478072422426, 1.9651782390706263]; recomputed matches: True
GATE S6 official test BLEU-4 11.802 (recorded 11.802)
GATE E_dur_x3 official test BLEU-4 11.537 (recorded 11.537)
GATE F official test BLEU-4 12.114 (recorded 12.114)
GATE G1 official test BLEU-4 11.562 (recorded 11.562)
GATE F-prime official test BLEU-4 12.836 (recorded 12.836)
GATE F-triple-prime official test BLEU-4 12.727 (recorded 12.727)

Member set | validation BLEU-4 (read from the saved job output for E, F, G1; S6 from the subset decode) | official test BLEU-4 (recomputed from the dump)
  S6                                  12.470   11.802
  three stitched-data models          12.472   11.537
  F                                   13.246   12.114
  six stitched-data models plus two   13.969   11.562
Four member sets scored on both splits  [c12-038]
Pearson r between validation and test BLEU-4 -0.014  [c12-038]
Larger variant (six stitched-data models plus two RGB-fusion): validation 13.97, test 11.56  [c12-037]
Pose-only ensemble of three stitched-data models: validation 12.472 against 12.470 for the final ensemble; test 11.54 against 11.80  [c18-097]
Ensembles that scored above 11.80 on test: F 12.11, F-prime 12.84, rebuilt 12.73; count above = 3  [c14-084]
Rebuilt ensemble (saved decode log): 5 members, 3 without RGB fusion, 2 RGB-fusion; test BLEU-4 12.73  [c12-040 c18-006]
Member kinds in the order of the decode (3 pose-augmented checkpoints, then 2 RGB-fusion): from the checkpoints' saved settings [False, False, False, True, True]; from the decode log [False, False, False, True, True]; equal: True  [c12-040 c18-006]
Pose augmentation switched on in the saved settings of 3 of the 3 pose-augmented checkpoints (seeds 42, 101, 202)  [c18-006]
Pose augmentation switched on in the 0 of 3 duration-stitch members of the earlier 1.1:1 ensemble (settings field present: 0)  [c18-006]
Settings of the three checkpoints: training manifest joint_13760.tsv (13,760 stitched clips added to the real training clips), 133 keypoints, frame normalisation, 40 epochs, learning rate 0.0005, cosine schedule, batch 64, pose augmentation on, no RGB input; all three agree: True  [c12-040]
Stitched to real training clips: 13,760 to 12,441 = 1.1:1  [c12-040]
The earlier 1.1:1 members of seeds 101 and 202 used the same training manifest and no pose augmentation: True  [c18-006]
(The decode log does not name its checkpoint files; which checkpoints formed the rebuilt ensemble is recorded in the pre-registration, not in a saved output, so the settings above are those of the checkpoints it names. The log's member kinds and the remapped vocabulary of the last two members agree.)
Video-disjoint ensemble (saved decode log): 5 members, 3 pose-only, 2 RGB-fusion  [c18-004]
Pose-only duration ensemble (saved job output ens-ext 1197079, X2_DUR3): 3 members, 0 with RGB fusion  [c18-096]

Exploratory configurations chosen with test scores in view, against the validation-selected configuration of the same job (read from the saved job outputs):
  beam/length-penalty re-tune of the two-member ensemble (tier1 1082033): validation-selected beam=5 lp=1.0 scored 10.71 on test; beam 10 / LP 1.0 scored 11.15 (+0.43)
  SWA ensemble candidates (swa-ens 1094775): validation-selected C:SV+SF+V0+C2+F1+F2 (C) scored 11.26 on test; candidate D scored 11.51 (+0.25)
  weighted ensembles (ens-max 1091645): validation-selected S6-weighted scored 11.41 on test; S9-weighted scored 11.70 (+0.28)
Test-selected configurations scoring above the validation-selected one of their job: 3 of 3; their BLEU-4 11.15, 11.51 and 11.70  [c08-009]

Rebuilt ensemble minus final ensemble, official test, 2,000 resamples: +0.93, 95% interval -0.28 to +2.11  [c18-007]
Test BLEU-4 12.73 against 11.80  [c18-007]
F minus final ensemble, official test: +0.31, 95% interval -0.89 to +1.50; includes zero: True  [c01-028]
F scored 12.11 on test against 11.80, having scored 13.25 against 12.47 on validation (validation read from the saved job output ens-search / the subset decode): an apparent gain of +0.31 whose interval includes zero, the condition under which the dissertation withdrew it (the withdrawal itself is the author's decision, not a computed number)  [c01-028]
GATE CTC official test BLEU-4 9.522 (recorded 9.522)
GATE VLP official test BLEU-4 10.272 (recorded 10.272)
GATE RGB official test BLEU-4 10.335 (recorded 10.335)
GATE Fp official test BLEU-4 12.836 (recorded 12.836)
  S6 minus RGB: +1.47, 95% interval +0.40 to +2.56, excludes zero: True
  S6 minus member ctc_vlplong80: +1.00, 95% interval -0.13 to +2.11, excludes zero: False
  RGB minus CTC: +0.81, 95% interval -0.34 to +1.97, excludes zero: False
  VLP minus CTC: +0.75, 95% interval -0.22 to +1.71, excludes zero: False
  F minus S6: +0.31, 95% interval -0.89 to +1.50, excludes zero: False
  Fp minus S6: +1.03, 95% interval -0.10 to +2.15, excludes zero: False
Of the 6 paired comparisons of Table res_bootstrap, 5 have an interval that includes zero and 1 excludes it; the intervals' half-widths are about 1.2 BLEU-4  [c01-028]
Rebuilt ensemble, unseen-target subset (570 clips): BLEU-4 7.89  [c18-008]

GATE rebuilt ensemble validation BLEU-4 at LP 0.8 13.472 (recorded 13.472); reported ensemble at LP 0.3 12.470 (recorded 12.470)
Rebuilt minus reported on the held-out half over 1,000 halvings: mean +0.71, 2.5th to 97.5th percentile -0.29 to +1.74  [c12-041]
Rebuilt ahead on the other half in 91.5% of 1,000 halvings (fixed threshold 95%)  [c12-041]
GATE vdB_s101 video-disjoint test BLEU-4 7.287 (recorded 7.287)
GATE VD5 video-disjoint test BLEU-4 7.841 (recorded 7.841)

Single models among VD5's members, validation (video-disjoint) / test BLEU-4, read from the saved job output g1_readout.txt: vdB_s42 6.60/7.41; vdB_s101 6.72/7.29; vdB_s202 6.37/7.15; fus_s303 6.00/6.14; fus_s404 6.03/6.82
Highest validation score among them: vdB_s101; the baseline named in the job output: vdB_s101; the baseline used here: vdB_s101; all equal: True  [c18-005]
(Highest test score among them: vdB_s42 7.41; the baseline is the validation-selected single model, not the test-best one.)

Video-disjoint ensemble VD5 7.84 against 7.29 for the validation-selected single model vdB_s101  [c18-005]
Difference +0.55; simultaneous 95% interval (max-T over 5 looks, 2,000 resamples) -0.23 to +1.34  [c18-005]
```
**Scope.** Subset decodes (shards of Kaya jobs 120153/120154), the test dumps of the final ensemble, F, F-prime, the pose-only duration ensemble, the six-plus-two variant, the rebuilt ensemble and the video-disjoint systems are re-scored with the repository scoring code; the bootstraps (500 resamples for the kind difference, 2,000 for the paired and video-disjoint intervals, 1,000 halvings; original seeds), Shapley values and the six Table res_bootstrap intervals are recomputed (sufficient-statistics BLEU, identical to corpus BLEU-4). Validation scores of three other member sets, the validation-selected configurations and the test-selected configurations are read from the saved job outputs (ens-ext 1197079, ens-search 1197719, tier1 1082033, swa-ens 1094775, ens-max 1091645), matched by section and configuration label rather than by line order. The single model used as the video-disjoint baseline is the one with the highest validation score among the ensemble's single members (read from the saved job output of 31205), not the highest on test. Member kinds are read from the saved decode logs; the pose augmentation, seeds, training manifest and learning settings of the rebuilt ensemble's checkpoints are read from the settings saved inside the checkpoints (`data/member_cfgs.json`, extracted once by `prepare_data.py`). The decode log does not name its checkpoint files, so which checkpoints formed the rebuilt ensemble is recorded only in the pre-registration, and the check cannot show that the decode used them; it shows that the checkpoints so named are pose-augmented, share one adopted configuration, and match the decode's member kinds. The withdrawal of the 12.11 gain is the author's decision; the check recomputes the interval that motivated it. No decoding or training is done; the 0.55:1 and 1.1:1 ratios of the older stitched-data ensembles are not recomputable from saved outputs.


### A52 — Ensemble and translator decoding cost, pose front-end speed, GPU set-up

**Claim.** The dissertation (Introduction, Method set-up, Results "latency", Discussion, appendix latency table) states that on a laptop CPU the pose front end runs at 5.2 frames/s with the detector on every frame and 25.4 with it run once per clip (55.0 with Core ML), against 25 in the source video; that a single pose-only translator decodes a clip in 36 ms at beam 5; that the six-member ensemble at beam 10 reaches 12.47 validation BLEU-4 at a median 269 ms per clip; the appendix table of 1, 3 and 6 members at beams 5 and 10; that at 5.2 frames/s pose estimation alone takes about 9.6 s for the median clip; and that each run used one NVIDIA V100 GPU.
**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/ensemble_cost_and_latency/check.py
```
```text
GATE six members, beam 10: validation BLEU-4 12.470 (recorded 12.470, the headline validation score)
read from the saved job output ensemble_latency_log.txt (M3 Max CPU, 12 threads, batch 1, 800 validation clips)
 members  beam  BLEU-4  median ms  p95 ms
       1     5    9.53         35      76  [c16-050]
       3     5   11.31        103     228  [c16-051]
       6     5   12.19        207     460  [c16-052]
       1    10    9.78         44     101  [c16-053]
       3    10   11.47        133     311  [c16-054]
       6    10   12.47        269     613  [c16-055]
members 1, 3 and 6 on 800 validation clips at batch size 1, 12 threads (log header: clips 800, torch threads 12)  [c16-049]
6 members (the six-member ensemble), beam 10: 12.47 validation BLEU-4 at a median 269 ms per clip  [c13-024 c13-026]
3 members (three), beam 10: 11.47 validation BLEU-4 at a median 133 ms per clip  [c18-056 c18-057]

read from the saved job output pose_throughput_subset.json (M3 Max CPU; frames per second recomputed as 1000 / mean ms per frame)
detector on every frame: 5.2 frames per second (839 frames)  [c02-027 c13-021 c13-025 c14-086]
detector once per clip, pose model alone: 25.4 frames per second  [c13-022 c14-086]
pose model alone on the Core ML provider: 55.0 frames per second  [c18-055]
single pose-only translator, beam width 5: mean 36 ms per clip (100 test clips)  [c02-028 c13-023]
source video: 25 frames per second (container rate of all 12 timed clips)  [c14-086]

median official test clip: 50 frames (800 clips, frames of each clip in the saved speech-chain benchmark)
pose estimation alone for the median clip at 5.2 frames/s: 9.6 s  [c14-089]

read from the job scripts: 64 scripts that call train_ctc.py; their `#SBATCH --gres` request:
    3 x gpu:1
   17 x gpu:v100-32gb:1
   41 x gpu:v100:1
    3 x none (CPU, summary or smoke script)
every one of the 61 scripts that request a GPU asks for exactly one GPU; 58 name the V100 type, 3 ask for any single GPU  [c07-069]
```
**Scope.** Parses saved timing outputs; nothing is timed again (timings depend on the machine). Gate: the six-member beam-10 validation BLEU-4 equals the recorded 12.470 (original tool run of 22 Sep 2026, local, no job id). Latency tables come from `tools/ensemble_latency.py` (M3 Max CPU, 12 threads, batch 1, 800 validation clips); front-end speeds from `tools/pose_throughput_bench.py` (839 frames from 12 clips; frames per second recomputed as 1000 / mean ms). The 9.6 s is the median test-clip length (from the saved speech-chain benchmark) divided by the 5.2 frames/s. The V100 statement is checked at the level of the job scripts: it counts the `#SBATCH --gres` request of the training scripts, so it shows what was requested, not what a job log reports; three scripts ask for "any single GPU" rather than naming the V100. Not covered: the 21.7 frames/s on one V100 (Kaya job 1081793, log only on Kaya) and the 35-45 minute training time (Kaya logs).


### A53 — Why ensembling works: members, bootstrap, mechanism, MBR, seed ensembles

**Claim.** The reported six-member ensemble (one model pre-trained 80 epochs, two pre-trained 40 epochs, three RGB-fusion; about 60 M parameters in total; BLEU-4 11.80) is 1.47 above its RGB-fusion member (Results, Table of comparisons) and 1.00 above its strongest member, which is not resolvable; five of six members are below it with intervals that exclude zero. On 20% of test clips no member produced the ensemble's sentence, and there it is above every member on 36 clips and below all on 13 (sign test p = 0.001). The CTC and pre-trained single models find the same clips hard (r = 0.78), the best training sentence per clip holds 81.9% of the content words, MBR decoding gives 10.69 / WER 0.956 against 11.80 / 0.907, and three-seed ensembles from seven job submissions add +0.35 (-0.05 to +0.75) over the validation-best seed (Appendix B).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/ensemble_mechanism/check.py
```

```text
GATE ensemble S6 official test BLEU-4 11.802 (recorded 11.802)
references match comm_run/manifests/test.tsv: 800/800

The ensemble job lists 6 members [c01-006 c03-006 c07-025]
  member comm_dwpose_ctc_vlplong80: pose-only, encoder pre-trained 80 epochs; fine-tuned 40 epochs, seed 101
  member comm_dwpose_rgb_fusion: RGB-fusion (fine-tuned from comm_dwpose_vlp_ctc/best.pt); fine-tuned 40 epochs, seed 42
  member comm_dwpose_vlp_ctc: pose-only, encoder pre-trained 40 epochs; fine-tuned 40 epochs, seed 42
  member comm_dwpose_ctc_s202: pose-only, encoder pre-trained 40 epochs; fine-tuned 40 epochs, seed 202
  member comm_dwpose_fus_s101: RGB-fusion (fine-tuned from runs/ctc_s101/best.pt); fine-tuned 40 epochs, seed 101
  member comm_dwpose_fus_s202: RGB-fusion (fine-tuned from runs/ctc_s202/best.pt); fine-tuned 40 epochs, seed 202
composition derived from the config stored in each checkpoint (members_config.json) and the pre-training epochs set in the job scripts gpu_batch_10h.sbatch and dwpose_vlp.sbatch: 1 model pre-trained 80 epochs, 2 models pre-trained 40 epochs (seeds [42, 202]), 3 RGB-fusion models (seeds [42, 101, 202])  [c07-026 c14-101]
members pre-trained for 40 and 80 epochs: ['pre-trained 40 epochs', 'pre-trained 80 epochs']  [c14-101]
GATE parameters per member (checkpoint tensors without the fixed position tables): pose-only 9.0481 M, RGB-fusion 10.9238 M (recorded in the 26 Sep audit: 9.0481 / 10.9238)
six members together: 59.92 M parameters (about 60 M)  [c08-004 c11-002]
ensemble decoded at beam 10, length penalty 0.3 (log-probabilities averaged by ensemble_mbr_wer.py)

MEMBERS decoded alone (official test, beam 10, LP 0.3)
  comm_dwpose_ctc_vlplong80    BLEU-4 10.80
  comm_dwpose_rgb_fusion       BLEU-4 10.01
  comm_dwpose_vlp_ctc          BLEU-4 10.34
  comm_dwpose_ctc_s202         BLEU-4 10.46
  comm_dwpose_fus_s101         BLEU-4 10.62
  comm_dwpose_fus_s202         BLEU-4 10.32
strongest member: comm_dwpose_ctc_vlplong80 10.80 (80-epoch pre-trained model)  [c12-005 c12-014]

PAIRED BOOTSTRAP, ensemble minus member (2,000 sentence resamples, seed 0)
  ensemble - comm_dwpose_ctc_vlplong80: +1.00 [-0.13, +2.11] ahead in 96.0% -> not resolvable
  ensemble - comm_dwpose_rgb_fusion: +1.79 [+0.64, +2.99] ahead in 99.7% -> interval excludes zero
  ensemble - comm_dwpose_vlp_ctc: +1.47 [+0.51, +2.51] ahead in 99.8% -> interval excludes zero
  ensemble - comm_dwpose_ctc_s202: +1.34 [+0.37, +2.41] ahead in 99.8% -> interval excludes zero
  ensemble - comm_dwpose_fus_s101: +1.18 [+0.16, +2.25] ahead in 99.2% -> interval excludes zero
  ensemble - comm_dwpose_fus_s202: +1.49 [+0.73, +2.32] ahead in 100.0% -> interval excludes zero
members scoring below the ensemble with an interval that excludes zero: 5 of 6  [c12-013]
Ensemble 11.80 vs strongest member (10.80): +1.00 [-0.13, +2.11], ahead in 96.0%  [c12-005 c12-014]
Ensemble (11.80) vs strongest member (10.80) 80-epoch, difference +1.00; interval includes zero: True  [c12-005 c12-014]

RGB-fusion model alone, beam 5: BLEU-4 10.33  (recorded 10.335 in the ladder job)
Final ensemble 11.80 vs its RGB-fusion member 10.33: +1.47 [+0.40, +2.56], ahead in 99.4%  [c12-002 c12-004]

Unseen references (n=570): ensemble minus strongest member +0.73 [-0.43, +1.95], ahead in 88.5%  [c18-010]
  unrounded: +0.7286 [-0.4322, +1.9545]  [c18-010]
  DISAGREES: the dissertation prints the upper bound as +1.96; the saved outputs give +1.9545, which rounds to +1.95  [c18-010]

MECHANISM: ensemble sentence equal to no member's on 162 of 800 clips = 20.2% (about 20%)  [c12-015]
on those clips the ensemble's sentence chrF is above every member's on 36 and below all on 13  [c12-016]
two-sided sign test over the 49 clips with a decisive sign: p = 0.001  [c12-016]

Oracle pool: 6 saved outputs per sentence (3 models x greedy and beam; six)  [c12-052]
GATE per-sentence oracle over the six outputs (lowest sentence WER, reference in hand): BLEU-4 14.33, WER 0.807 (recorded in the BLEU-WER figure data: 14.33, 0.807)
best of six saved outputs per sentence (oracle): BLEU-4 14.33, WER 0.807; the six outputs choose by sentence WER from 6 different sources  [c12-052]

Sentence chrF of the CTC and the pre-trained single model (beam 5) correlate r = 0.78  [c13-019]

MBR-WER decoding of the ensemble: BLEU-4 10.69, WER 0.956  [c18-009]
length-penalty decoding of the ensemble (beam 10, LP 0.3): BLEU-4 11.80, WER 0.907  [c18-009]

Training-sentence oracle: the best single training sentence per clip (chosen with the answer in hand) contains 81.9% of the reference content words, pooled over the clips that have one  [c10-037]

SEED ENSEMBLES (read from the saved job output seedens_rows.json, job 21736)
independent job submissions with adopted-recipe seeds: 7 (seven)  [c18-002]
seeds per ensemble: [3] (three)  [c18-003]
test BLEU-4, ensemble of three minus validation-best seed: mean +0.35 (95% t-interval -0.05 to +0.75); higher in 5 of 7 (five of seven)  [c18-001]
```

**Scope.** Recomputed from the saved test decodes (original jobs 1101828/1125067 for the ensemble; ladder jobs for the single models): all scores, bootstrap intervals (2,000 resamples, seed 0), the novel-output and sign-test counts, MBR scores, the training-sentence oracle, the chrF correlation and the per-sentence oracle over the six saved outputs (gated against the values of the BLEU-WER figure). Read from saved job inputs: the training configuration stored inside each of the six checkpoints (`members_config.json`, written once by `prepare_data.py`; the checkpoints themselves are not published) and the pre-training epochs set in the job scripts `gpu_batch_10h.sbatch` and `dwpose_vlp.sbatch`; the parameter count is the number of values in each checkpoint's tensors without the fixed position tables (9.0481 M pose-only, 10.9238 M RGB-fusion); the seed-ensemble rows (job 21736). Not done: the 11.50 beam-5 ensemble score (no saved decode) and the learned selector (script not saved). One disagreement is reported: for the unseen subset the saved outputs give an upper bound of +1.9545, i.e. +1.95, where the text has +1.96.


### A54 — Early AlphaPose runs and the DWPose re-extraction

**Claim.** The Introduction and Results (Section on the pose estimator, the table of the early attempts, Appendix B) state that a translator trained on the first-listed person's AlphaPose keypoints scored BLEU-4 2.87 at beam width 5 and 1.76 with greedy decoding on the 800-clip official test set, with only 53 distinct greedy sentences (output collapse); that DWPose re-extraction raised the mean hand-keypoint confidence to 0.82; that the same keypoints identified signs from the hands at 4.2% in the identification probe; the keypoint-set sizes (59, 68, 136, 133); that the SignJoey-compatible reconstruction scored 1.03 with 9 distinct sentences, the 25.95 M-parameter model 2.29, and the bounding-box-corrected generic run 2.13.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/estimator_first_listed_runs/check.py
```
```text
control_decodes.jsonl: references match comm_run/manifests/test.tsv: 800/800
control_decodes.jsonl: references match comm_run/manifests/test.tsv: 800/800
GATE recorded BLEU-4 of the first-listed AlphaPose control, greedy 1.76 and beam 5 2.87: recomputed 1.7604 and 2.8673

First-listed AlphaPose control, official test, 800 clips, one run
BLEU-4 2.87 at beam width 5 (length penalty 0.6), 1 run  [c02-006 c09-008]
BLEU-4 1.76 with greedy decoding  [c09-008]
53 distinct sentences for the 800 test clips with greedy decoding (output collapse)  [c02-007 c08-015 c09-008]
(beam width 5 gives 26 distinct sentences)
signjoey_decodes.jsonl: references match comm_run/manifests/test.tsv: 800/800

SignJoey-compatible reconstruction, greedy, official test: BLEU-4 1.03, 9 distinct sentences  [c17-010]

DWPose re-extraction: mean hand-keypoint confidence over 14041 clips 0.82 (0.8218)  [c08-020]

First-listed AlphaPose identification probe: 382 annotated spans usable (45 signs) in the saved probe log
hand keypoints, nearest-centroid top-1 on 333 scored spans (the probe scores a span only if its hand descriptor exists and its class is predictable in the held-out fold): 4.2%  [c08-049]
(333 of 382 usable spans are scored for the hand variant; the quoted 382 is the usable set, 4.2% is over the scored spans)

Keypoint sets (config.py)
COCO-WholeBody body+hands subset: 59 keypoints  [c06-030]
HALPE body+hands subset: 68 keypoints  [c06-030]
AlphaPose HALPE layout: 136 keypoints  [c06-032 c09-008]
DWPose COCO-WholeBody layout: 133 keypoints (default)  [c06-033]

1,024-to-512 capacity model: 25.95 M parameters (pos_enc buffer excluded)  [c17-009]
its test BLEU-4 2.29 on 800 clips (read from the saved job output capacity_accum2_metrics.json)  [c17-009]
bounding-box-corrected generic run, greedy test BLEU-4 2.13 (read from the saved job output bbox_xywh_test_rescore.txt)  [c17-011]
```
**Scope.** The control and the SignJoey-compatible reconstruction are decoded from their local checkpoints with `tools/eval_beam.py` (CPU, once, by `prepare_data.py`'s inputs; see SOURCES.md) and the saved hypotheses are scored through `slt/metrics.py`; the gate is the recorded 1.76 (greedy) and 2.87 (beam 5) of the control. The hand confidence, the probe top-1 and `config.py` are recomputed from per-clip and per-span files. The 4.2% is over the 333 spans the probe scores for the hand variant; the quoted 382 is the usable-span count printed in the same probe log. The 2.29 and 2.13 scores and the parameter count are read from saved job outputs and a checkpoint. Not checked: the beam-5 cells of the estimator 2x2 (job 1101250: 4.60, 5.58, 1.45, 10.03, the single collapsed output), the beam width and keypoint subset of those cells, and the DWPose re-extraction run (8.78, 7.92, 309), all on Kaya; the design labels of the 2x2 and the "as released, first-listed person" / "re-extracted" labels of its table are not checked. No training is done.


### A55 — Estimator comparison on the signer's keypoints
**Claim.** With the keypoints of the entry the distributed files mark as the signer, AlphaPose (136 body and hand keypoints) scored a test mean of 10.08 against 10.35 for DWPose, three seeds each in one job: a difference of +0.27 (95% t-interval -1.03 to +1.57, DWPose higher in two of three seeds), so no resolvable estimator effect (Results, estimator table and figure; Discussion). The hand sign-identification probe agrees in direction (18.1% against 13.1% top-1 on hands, McNemar p = 0.018). The signer entry is the right person in 95.1% of multi-person clips (Method, "Signer selection"; Appendix B).

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/estimator_signer_rerun/check.py
```
```text
references match comm_run/manifests/test.tsv: 800/800
GATE: all six recomputed beam-5 test BLEU-4 equal the values recorded in the job's dumps (to 1e-6)

Setting: Auslan-Daily Communication official test, 800 clips, beam width 5, three seeds, AlphaPose and DWPose trained in one job (196407)  [c08-029 c08-038 c14-018]
AlphaPose arm: HALPE-136 body+hands keypoints of the entry marked as the signer; DWPose arm: DWPose  [c08-034 c08-035]
Eap beam-5 test BLEU-4 per seed: 10.28 / 10.25 / 9.71
Edw beam-5 test BLEU-4 per seed: 10.44 / 10.06 / 10.55
AlphaPose, signer keypoints (3 seeds) test mean 10.08  [c08-034 c08-043 c08-044 c14-018]
DWPose (3 seeds, same job) test mean 10.35  [c08-035 c08-043 c08-044 c14-018]
DWPose minus AlphaPose per seed: +0.16 / -0.18 / +0.84
difference +0.27, 95% t-interval -1.03 to +1.57  [c08-043 c14-018]
DWPose higher in 2 of 3 seeds  [c08-043 c21-022]
interval contains zero: True; contains +1.0 (about one point): True; contains -1.0: True  [c14-021]
Figure bars: means of 3 seeds with 95% t-intervals (t = 4.303, 2 degrees of freedom); 2 estimators compared  [c08-038 c14-021]
two estimators, 3 seeds: neither a ranking (interval contains zero) nor exclusion of a one-point difference (interval spans -1 to +1)  [c14-021]

Unseen-reference test clips (570 of 800, reference absent from training, via results/train_target_hashes.txt)
unseen BLEU-4 DWPose minus AlphaPose per seed: +0.21 / +1.07 / +0.66
on test clips whose reference is absent from training: +0.65 (95% t-interval -0.43 to +1.72)  [c08-031]

Hand sign-identification probe (nearest centroid, 45 signs, annotated spans of the Communication training clips)
hand keypoints alone, top-1: DWPose 18.1%, signer's AlphaPose 13.1%  [c08-047]
hand keypoints alone, top-5: DWPose 34.7%, signer's AlphaPose 32.7% (read from the saved probe logs)  [c08-047]
body and hands, top-1: 18.7% and 11.9%  [c08-048]
exact McNemar test, hands: 335 shared spans, DWPose only right 35, AlphaPose only right 17, p = 0.018  [c08-048]

mean confidence of the hand keypoints of the signer entry in the distributed files: 0.12 (0.1156) over 14,041 clips (read from the saved probe output probe.json)  [c08-018]

Clips with several people: 12409 analysed + 2 without a DWPose file = 12,411 of 14,041 clips  [c19-057 c06-020]
the entry marked as the signer follows the tracked DWPose arm motion (r > 0.5) in 95.1% of those clips with a defined r (read from the saved per-clip check signer_check_results.json)  [c06-020]
```
**Scope.** Recomputed: BLEU-4 (13a, detokenised) of the six saved beam-5 test dumps of job 196407, the seed means, differences and t-intervals, the unseen-reference subset (reference not among training targets, by hash), the probe accuracies and the exact McNemar test from the per-span outputs. Read from saved outputs without recomputation: the probe's top-5 accuracy, the mean hand confidence and the signer-match share (computed by the original local runs from pose arrays that are not saved). The 2.2% first-listed match share has no saved output and is not checked. Original job: Kaya 196407 (seeds 42, 101, 202). No training or pose extraction is repeated.


### A56 — Extraction-error audit: signer entry, keypoint layouts, registration, bounding boxes
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


### A57 — Content-word failure analysis

**Claim.** In the Error Analysis (Results), Discussion and appendix, the dissertation states that 786 of the 800 official test clips have a reference content word (2.55 per clip, 2,043 tokens; content words are words outside a 71-word function list); that 48.8% of outputs are a verbatim training sentence sharing no content word and 5.2% are exact; that recall rises from 0% (seen twice or fewer) to 20.8% (seen over 50 times), those frequent words being 1,592 tokens (78%), 85% of them appearing somewhere in the outputs; that recall is 17.1% against 8.2% for a null model (2.1 times); and that the ensemble is exactly right on 0 of 564 unseen-target clips against 19.8% of seen ones.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/content_word_failure_analysis/check.py
```

```text
references match comm_run/manifests/test.tsv: 800/800
GATE single model beam-5 test BLEU-4 10.2716 (recorded 10.27 for the 40-epoch pre-trained model)
GATE S6 ensemble beam-10 LP 0.3 test BLEU-4 11.802 (recorded 11.802)

Content words are the words outside a list of 71 function words  [c06-009 c06-012 c10-028 c21-009]
Content-word recall and precision are scored on a scale of 0 to 100%; higher is better  [c06-009]
Setting: 40-epoch pre-trained pose-only model, beam width 5 (the 10.27 score above); ensemble panel at beam width 10  [c10-028 c10-030 c14-039 c21-009]

Clips with a reference content word: 786 of 800  [c10-023 c21-009]
Reference content tokens: 2,043  [c21-008 c10-032]
Mean reference content words per test clip: 2.55  [c10-024]
Recall of words seen twice or fewer: 0.0%  [c10-031]
Recall of words seen more than 50 times: 20.8%  [c10-031 c10-033 c14-039]
Tokens of words seen more than 50 times: 1,592 of 2,043 (78%)  [c10-032 c14-038]
Of these frequent words, 85% occur somewhere in the outputs (115 of 136 types)  [c10-033 c14-039]

Content-word recall of the model: 17.1%  [c10-036]
Null model (two most frequent training content words, every clip): 8.2%  [c10-036]
Ratio model / null: 2.1 times  [c10-036]

Failure modes over 800 clips: {'exact': 42, 'memorised': 390, 'partial': 160, 'novel': 161, 'reworded': 33, 'no content word': 14}
Clips classified: 800 (all test clips)  [c10-027]
Verbatim training sentence sharing no content word with the reference: 48.8%  [c10-027]
Exact matches: 5.2%  [c10-027]

S6 ensemble, beam width 10: exact match on 0 of the 564 unseen-target clips that contain content words  [c11-061 c21-009]
S6 ensemble exact match on seen-target clips: 19.8% (44 of 222)  [c11-061]
(single 40-epoch model, for comparison: unseen-target exact matches 1)
```

**Scope.** Recomputed from the saved test decodes (single 40-epoch pre-trained model, beam 5; S6 ensemble, beam 10 / LP 0.3) and the training references; two gates re-score BLEU-4 (10.27, 11.802). Not covered: the retrieval baseline numbers 2.72 and 1.47 (only a write-up exists). No training or inference.


### A58 — Figure settings and speed verdict
**Claim.** The effect-size figure (Results) uses the 800-clip official Communication test set with 95% intervals (seed-level t, paired bootstrap for the marked row); the qualitative figure shows outputs of the 40-epoch pre-trained pose-only model at beam width 5; the Discussion states that the pose-estimation stage, not the translator, limits speed.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/figure_settings/check.py
```
```text
GATE saved beam-5 decodes: BLEU-4 10.2716 (recorded 10.27 for the 40-epoch pre-trained model, 10.71 for the 80-epoch one)
official Communication test set: 800 clips (manifest rows 800, decoded 800) [c10-008]
95% two-sided t critical value, 3 seeds (2 degrees of freedom): 4.303; figure script uses 4.303 for the re-run row [c10-009]
figure axis label states 95% interval [c10-009]
qualitative figure: 5 of 5 example outputs equal the saved beam-5 decodes of the model scoring 10.27 (40-epoch pre-trained, pose-only) [c10-029]
decoding setting: beam width 5, 40-epoch pre-trained model (BLEU-4 10.27 = recorded 40-epoch score) [c10-029]
read from the saved job output: pose-only rate 25.4 frames/s
median pose estimation: 1969 ms (71.7% of the median chain) [c14-085]
median translation: 92 ms (3.3% of the median chain) [c14-085]
median synthesis: 661 ms (24.1% of the median chain) [c14-085]
pose estimation is the largest stage; translation is 4.7% of its median time [c14-085]
```
**Scope.** Recomputed: BLEU-4 of the saved beam-5 decodes (gate, 10.27), the match of the five example outputs to those decodes, stage shares from the saved per-clip timings. The interval constant and axis label are read from the figure script. The values of the intervals themselves belong to other checks. Timing is read from the saved benchmark output (M3 Max CPU, no Kaya job); nothing is trained or re-timed.


### A59 — DWPose front end: counts, hand confidence and hand speed

**Claim.** All 14,041 Communication clips were re-extracted with DWPose, giving a 399-dimensional frame vector (133 keypoints of x, y, confidence); the DWPose training list holds 12,439 of the 12,441 training clips (Methodology, Results). On four frames of one training clip the distributed hand keypoints have mean confidence 0.02 to 0.12 and the DWPose re-extraction 0.72 to 0.94 (`fig:res_pose_estimators`). Not recomputable here: the hand-speed ratios to citation form and the hand-confidence pair for correct versus wrong spans (see Scope).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/hand_speed_dwpose_frontend/check.py
```

```text
Communication clips with DWPose keypoints: 14,041  [c06-029 c08-019]
Training clips in the official list: 12,441; in the DWPose training list: 12,439  [c06-017 c08-002]
Official training clips absent from the DWPose list: 2 (video_12_62, video_68_179)
Keypoints per frame 133, values per keypoint 3 (x, y, confidence): frame vector of 399 dimensions  [c06-028]
Figure clip, frames [10, 28, 43, 59] (15/40/62/85% of the frames with DWPose hand confidence above 0.5)
Distributed hand keypoints, mean confidence per frame: 0.07, 0.05, 0.12, 0.02  range 0.02 to 0.12  [c08-023]
DWPose re-extraction, mean confidence per frame: 0.94, 0.72, 0.87, 0.84  range 0.72 to 0.94  [c08-023]

GATE hand speed per frame on first-detection keypoints, 591 clips: mean 0.391 (recorded 0.391)
Tracked keypoints, same 591 clips: mean 0.143; medians 0.097 (first-detection) and 0.086 (tracked)
First-detection speed is 2.7 times the tracked speed on the same clips (inflation of the first-detection figure)

Probe variant 'hands': 334 scored spans; mean hand confidence of correctly identified spans 0.817, of wrongly identified 0.814, correlation +0.019
The two means differ by 0.003 (hand confidence does not separate correct from wrong spans); the dissertation's particular pair of values is not reproduced by the saved probe output
```

**Scope.** Recomputed from saved per-clip files: row count of the DWPose confidence table, the training list against the official manifest, the keypoint tensor shape, per-frame hand confidence of both keypoint sets on the figure clip, and hand speed (`hand_stats`, unchanged) on 591 training clips held in first-detection and tracked form (gate: first-detection mean 0.391 as recorded 10 October 2026). Per-span hand confidence against probe correctness is recomputed and does not reproduce the dissertation's pair of values (c14-042, listed as disagreeing). Not recomputed: speed relative to MM-WLAuslan citation clips and News (9.86, 1.10, "about three times"): their inputs are no longer saved. No training or inference was run.


### A60 — Sign-identification probes and isolated-sign references
**Claim.** The Results, Discussion and Appendix state that on the 45 signs shared by MM-WLAuslan and Auslan-Daily (382 annotated conversational spans, 337 scored, chance 2.2%), DWPose hand keypoints identify the sign at 18.1% top-1 in conversational recordings (Figures on the identification probe, `tab:app_additional`). The Discussion and Appendix add that the 45 shared signs are an atypical subset, that using Auslan-Daily's own isolated clips (3,209 segments over 610 glosses; 379 usable for these signs, 7.5 same-sign references per query) as references gives 19.9% against 18.1% (p = 0.33), that the 630 MM-WLAuslan citation recordings (14 per sign) give 4.5%, and that estimated hand depth gives 5.9% alone (null 3.9%) and lowers x/y accuracy from 16.3% to 11.3% (p = 0.014).

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/sign_identification_probes/check.py
```
```text
GATE per-span file hands correct 61/337; recorded in the saved probe job output 61/337 (expected 61/337)

CONVERSATIONAL SPANS (the recorded probe, hands descriptor, clip-disjoint folds)
annotated spans 382 of 45 signs; chance 1/45 = 2.2%  [c08-046 c10-051]
signs 45  [c08-046 c10-051 c14-071 c14-070 c10-041 c14-047]
chance 2.2%  [c01-018 c02-021 c08-046 c10-040 c14-071]
spans 382  [c08-046 c10-051]
scored spans 337  [c14-070]
hands top-1 18.1% (61/337)  [c01-018 c02-021 c10-040 c14-072 c18-077]

AUSLAN-DAILY ISOLATED CLIPS: 3,209 segments over 610 glosses  [c04-039]
isolated clips of the 45 signs on a TRAIN clip with at least 3 frames: 379  [c14-070]
same-sign references per query (own clip excluded): mean 7.5, min 2, max 16  [c14-070]

Spotting glosses 120; also MM-WLAuslan glosses: 45 shared signs  [c10-041 c14-047]
MM-WLAuslan citation recordings of the 45 signs: 630  [c14-070]
recordings per sign: 14 (min) to 14 (max), i.e. 14 per sign  [c14-070]

ISOLATED CLIPS AS REFERENCES (read from the saved job output job_probe_a2_a5_depth.out)
isolated clips as references, top-1 19.9% (67/337) against conversational spans 18.1%  [c14-052 c14-072 c18-077]
citation-form recordings as references, top-1 4.5% (15/337)  [c14-052 c14-072 c18-078]
isolated clips (A2) minus spans: only first 16, only second 10; exact McNemar p recomputed 0.33 (log 0.33)  [c18-077]

ESTIMATED HAND DEPTH (read from the saved job output)
x/y/z top-1 11.3% against x/y 16.3% (same 3-D model)  [c18-079]
only first 13, only second 30; exact McNemar p recomputed 0.014 (log 0.014)  [c18-079]
depth alone 5.9% above a permutation null 95th percentile 3.9%  [c18-080]
```
**Scope.** Recomputed from per-item files: the 337 per-span hand-probe outcomes (61 correct; the gate matches the recorded run), the located table of the isolated clips (3,209; 379 references; 7.5 per query), and the MM-WLAuslan label counts (630 recordings, 14 per sign, 45 shared glosses). Read from the saved job output of Kaya probe job 140892 (wave 12f): the isolated-reference, citation-reference and depth accuracies; the McNemar p-values are recomputed from the printed discordant counts. Not recomputed: the 42.8% citation-form top-1 and the hand-confidence figures (only written up), and the SignCL density and spotting probes (outputs on Kaya, jobs 1174747 and 1171035). No training or pose extraction is done here.


### A61 — Input-blind constant baseline

**Claim.** One five-word training sentence, output for every clip regardless of the video, scores 4.41 BLEU-4 on the 800-clip official test set and 1.09 on the 570 unseen-target clips; 18 test references are exactly that sentence. The trained systems score 4.88 to 6.60 on the unseen clips (Abstract, Introduction, Results null-model paragraph, Discussion).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/input_blind_baselines/check.py
```

```text
references match comm_run/manifests/test.tsv: 800/800
GATE S6 11.802 (recorded 11.802), DWPose+CTC 9.52 (recorded 9.52)

Setting: 800 official test clips, one constant training sentence for every clip, BLEU-4 13a on detokenised text
constant sentence has 6 whitespace tokens incl. final full stop = 5 words
The null model outputs one sentence of 5 words for every clip  [c07-063 c11-079]
Constant, all 800 test clips: BLEU-4 4.41 (4.406)  [c01-025 c02-024 c11-079 c14-011 c14-077]
Test clips: 800 (800)  [c02-024 c11-079]
Constant, 570 unseen-target clips: BLEU-4 1.09 (1.093)  [c01-025 c02-023 c11-081 c14-011 c14-077]
Unseen-target clips: 570 (570)  [c02-023]
Constant, 230 seen-target clips: BLEU-4 10.65
Test references exactly equal to the constant: 18  [c11-079]
Constant: one sentence, the same for all clips (1 distinct output)  [c14-077 c02-023]

Trained systems, unseen-target BLEU-4 (beam 5; S6 = ensemble beam 10, LP 0.3):
DWPose+CTC single model 4.88  [c11-081]
S6 ensemble 6.60  [c11-081]
Constant 1.09 vs trained 4.88 to 6.60; ratio of the lowest trained score to the constant 4.5  [c11-081]
constant / published SL-Transformer (9.2 published) = 0.48  [c14-077]
constant / published GASLT (9.95 published) = 0.44  [c14-077]
```

**Scope.** Recomputed from the saved beam-5 test decodes (semantic-grading dump, job 1075705) and the S6 ensemble dump (job 1125067), the training-target digests and the test manifest; the constant sentence is named by its SHA-1 digest. The gate re-scores the recorded S6 (11.802) and DWPose+CTC (9.52) scores. The published baseline values (9.20, 9.95) are cited, not measured here. No training or inference.


### A62 — Input-side predictors of translation quality
**Claim.** Results ("Training-free measurements") states that neither eleven clip-level pose descriptors nor video identity predicts translation quality, and that each descriptor explains at most about 1% of variance.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/input_side_predictors/check.py
```
```text
references match comm_run/manifests/test.tsv
GATE: 800 clips, 13 descriptors x 4 outcomes, null bands and AUCs reproduce the recorded run (tolerance 1e-6)

clips 800; descriptors 13; constant on all clips: subsampled, invalid_frac
Non-degenerate clip-level pose descriptors: 11  [c10-047]
descriptor x outcome pairs exceeding the shuffled-outcome null band: 0 of 44  [c10-047]
null band (95th percentile of max|r| over shuffles): content_recall 0.099, any_content 0.101, chrf 0.103, exact 0.105
largest |r| = 0.103 (body_conf, exact)  [c10-048]
largest variance explained r^2 = 1.1% (about 1%)  [c10-048]
joint logistic model, video-grouped 5-fold AUC 0.518 vs shuffled-outcome control 0.519  [c10-047]
positive control (reference side, not available at inference): n_content_words vs any_content r = 0.150, exceeds band 0.101: True

video identity: 82 videos with >= 5 scored clips hold 760 of 786 scored clips; per-video success rates run 0.056 to 0.750
between-video sd of success rate 0.1275 vs shuffled null mean 0.1287 (95th percentile 0.1449), permutation p = 0.54 (1000 shuffles, seed 0)  [c10-047]
observed spread is within the shuffled null (p > 0.05): True  [c10-047]
```
**Scope.** Recomputed from per-clip data (`per_clip.jsonl`: 13 pose descriptors of the 800 official test clips, references, and beam-5 hypotheses of three single models): the correlations of each descriptor with four per-clip outcomes, the family-wise shuffled-outcome null band (500 permutations, seed 0, the tool's own random-number order), the video-grouped cross-validated AUC, the descriptor count, the exceedances, the largest r^2, and the video-identity permutation test. The gate requires the recomputed correlations, null bands and AUCs to equal the recorded 2026-09-09 run of `tools/translation_predictors.py` (local run, no cluster job). The video-identity test was not saved by that run, so it is re-implemented here (videos with at least 5 scored clips, 1000 shuffles); its p-value (0.54) differs slightly from the 0.526 in the write-up because the shuffle count and seed are this check's own, and the conclusion (p above 0.05) is the same. The pose descriptors were computed once by `prepare_data.py` from the pose arrays, which are not saved. Nothing is retrained.


### A63 — Real-data learning curve (wave 8)

**Claim.** Doubling the real training pairs from 6,241 to 12,439 raises BLEU-4 on the official Communication test set from 7.04 to 9.90 (+2.86) and on the 570 unseen-target clips from 3.15 to 5.00 (+1.84), positive in all five seeds, with seed-level 95% t-intervals; the table of nested subsets (Results, learning-curve table), the held-versus-kept episode contrast (appendix, -0.78 and -0.03) and the count-only prediction of content recall (appendix, 16.56 against 16.47) are recomputed too.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/learning_curve_with_real_data/check.py
```
```text
unseen flags match comm_run/manifests/train.tsv
GATE: all 40 decodes re-score to the BLEU-4 recorded by the original job (<1e-6); unseen clips = 570 of 800
Official test set: 800 clips  [c10-017]; unseen-target clips: 570  [c10-018]

Setting: five seeds, beam 5, best checkpoint; unseen = 570 of 800 test clips  [c10-010 c10-011 c10-021 c14-049]
Subset sizes (rows of the saved training manifests, seed 42): R025 3,141, R050 6,241, R100 12,439, V050 6,241  [c10-010 c10-016 c14-049]

Table (five-seed means): pairs / epochs / val / test / unseen / content recall
25% of each episode    3,141  320  6.80  6.03  2.27  8.8%  [c10-012]
50% of each episode    6,241  160  7.89  7.04  3.15  11.0%  [c10-013]
All training pairs     12,439  80  10.51  9.90  5.00  16.5%  [c10-014]
Whole episodes, 50%    6,241  160  8.08  7.63  3.53  11.6%  [c10-015]
  (epochs read from the saved progress.json of each run)

Unseen-target test BLEU-4, R050 3.15 -> R100 5.00 (mean of five seeds)  [c01-019 c03-015 c10-018 c14-049]
R100 - R050, official test (800 clips): per seed +2.897 +1.635 +2.849 +4.087 +2.821
  mean +2.86 (+2.858), 95% t-interval +1.78 to +3.93, positive in 5 of 5 seeds  [c10-017 c10-020 c14-003 c14-027 c10-019]
R100 - R050, unseen-target test (570 clips): per seed +2.071 +0.391 +1.823 +3.176 +1.752
  mean +1.84 (+1.842), 95% t-interval +0.61 to +3.08, positive in 5 of 5 seeds  [c01-020 c10-018 c10-020 c14-049 c10-019]
Test 6,241 -> 12,439 pairs: 7.04 -> 9.90  [c10-017]
Doubling from 6,241 to 12,439 pairs adds 2.86 on test and 1.84 on unseen targets  [c10-020 c14-003 c14-027 c14-049]

Episode familiarity: (V050 - R050) on clips of held-out episodes minus the same difference on kept episodes, at equal size (five seeds; difference between two differences; the one cut episode dropped)  [c17-111]
  all affected test clips: -0.78 (-2.06 to +0.49)  [c17-112]
  unseen-target clips: -0.03 (-0.98 to +0.93)  [c17-113]

Count-only curve (logistic on log(1+training count), fitted on R025 and R050): predicts R100 content recall 16.56% against 16.47% observed  [c18-091]
```
**Scope.** Recomputed from the saved beam-5 decodes (best checkpoint, test and validation, 20 runs each) of the original job 21734: BLEU-4 through `slt/metrics.py`, content recall through `tools/arm_readouts.py`, intervals from five seeds. Subset sizes, epochs and per-token training counts are derived once by `prepare_data.py` from the saved manifests and progress files; the unseen flags are re-checked against `comm_run/manifests/train.tsv`. Nothing is trained.


### A64 — News: five-seed stitched data and pose augmentation, validation length penalty, signable recall, ensemble gate (wave 4)
**Claim.** On the official News test the adopted recipe (10,614 stitched clips at 1.1:1 plus pose augmentation, 512 input frames) raises mean beam-5 BLEU-4 from 1.80 to 2.14 (+0.34; five-seed 95% t-interval +0.09 to +0.58, higher in all five seeds), 0.03 above the published SL-Transformer (2.11) and below GASLT (2.72) and SL-Luong (2.81) (Results, News paragraph; Discussion; Appendix B supplementary table). With beam 10 and a length penalty chosen on validation the mean is 2.33 (+0.19; -0.01 to +0.39) and the brevity penalty rises from 0.73 to 0.94. Recall of words the synthetic data can sign rises by +1.16 points, of words it cannot by +0.60. About a third of News clips (31.0%) exceed the 256-frame input cap, and the joint Communication-and-News run used 22,088 clips (12,439 + 9,649). A five-seed ensemble scores 2.948 on validation against 2.872 for the best single model (+0.076), below the pre-registered +0.10 gate.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/news_subset_results/check.py
```
```text
gate: 20 beam-5 BLEU-4 values (nA, nSA x 5 seeds x val/test) and all LP-sweep values recomputed from n-gram counts equal the values recorded by the jobs
seeds per arm: 5 (five); beam-5 BLEU-4 on the official News test, best checkpoint  [c11-017 c11-039 c18-072]
nA  (News real only) 5-seed mean test 1.80 (1.803)  [c11-038 c14-104]
nSA (stitched + pose augmentation) 5-seed mean test 2.14 (2.139)  [c11-017 c11-038 c14-104 c11-040]
nSA - nA test per seed +0.23 +0.60 +0.40 +0.38 +0.07  mean +0.34 (+0.336) from 1.80 to 2.14  [c11-038]
nSA - nA test 95% t-interval over the 5 seeds +0.09 to +0.58; higher in 5/5 seeds  [c11-039]
nSA mean 2.14 is 0.03 above published SL-Transformer 2.11  [c11-040 c14-104]
published News baselines (cited): SL-Transformer 2.11, GASLT 2.72, SL-Luong 2.81; nSA mean 2.14 is below GASLT 2.72 and below SL-Luong 2.81  [c14-104]
read from jobs/slurm/w4_news.sbatch (job 7950, seeds 42/101/202): --max_src_frames 512 in arm nA True and arm nSA True  [c11-035]
read from jobs/slurm/w4_news.sbatch (job 7950, seeds 42/101/202): the nSA arm trains on news_sa_train.tsv, required to have 20263 rows = 9649 real News rows + 10,614 stitched rows, with pose augmentation (nA: real only)  [c11-037]
read from jobs/slurm/w4_news_s2.sbatch (job 16398, seeds 303/404): --max_src_frames 512 in arm nA True and arm nSA True  [c11-035]
read from jobs/slurm/w4_news_s2.sbatch (job 16398, seeds 303/404): the nSA arm trains on news_sa_train.tsv, required to have 20263 rows = 9649 real News rows + 10,614 stitched rows, with pose augmentation (nA: real only)  [c11-037]
both scripts agree: 5-seed comparison used --max_src_frames 512 in both arms (two scripts, seeds 42/101/202 and 303/404)  [c11-035]
the recipe: 10,614 stitched clips added to 9649 real rows, ratio 1.1:1 (1.100), plus pose augmentation, in all five nSA runs  [c11-037]
nSA with beam 10 and the length penalty chosen on validation (five seeds): test mean 2.14 to 2.33 (2.327); difference +0.19 (+0.188), 95% t-interval -0.01 to +0.39; higher in 5/5  [c18-072 c18-073]
nSA brevity penalty (13a), beam 5 / LP 0.6 to validation-selected beam 10: 0.73 to 0.94  [c18-074]
read from the saved job output signable_recall.txt (job 16398 secondary): nSA - nA recall, words that the synthetic data can sign (signable): +1.16 percentage points [+0.64, +1.68], same sign as the mean in 5/5 seeds  [c18-075 c18-076]
read from the saved job output signable_recall.txt (job 16398 secondary): nSA - nA recall, words that it cannot (unsignable): +0.60 percentage points [-0.05, +1.25], same sign as the mean in 4/5 seeds  [c18-075 c18-076]
read from the saved job output ens5_val.log (job 21744): the ensemble decode loaded 5 members and checked 5 checkpoints; its five single-member validation decodes are the 5 nSA seeds (42, 101, 202, 303, 404)  [c18-098]
read from the saved job output ens_phase1.txt: pre-registered validation gate = best single +0.10  [c18-098]
ensemble validation 2.948 (LP 1.2) against 2.872 for the best single model: difference +0.076 (below the 0.10 gate)  [c18-099]
News clips: 11,069; input cap max_src_frames (config default) 256; clips longer than 256 frames 3,429 = 31.0% (about a third); median 191 frames  [c04-024]
joint run training clips: 12,439 Communication (lines of the saved real training list of job 21734) + 9,649 News (checked by the job scripts above) = 22,088  [c20-001]
```
**Scope.** Recomputes every beam-5 and length-penalty-sweep BLEU-4 (sacreBLEU 13a, detokenised) from per-sentence n-gram counts, requires each to equal the value recorded by its job, and derives means, differences, t-intervals (df = 4, t = 2.776), brevity penalties and the validation-selected length penalty. News reference sentences are not part of the spreadsheet that `make data` restores, so the saved files hold counts only (`make_stats.py`, run once). The signable-word recall differences and the ensemble gate threshold are read from saved job outputs; the underlying recall tool is not re-run. The five-member ensemble is read from the decode log `ens5_val.log` (it prints 5 members and 5 checkpoints). The share of News clips above 256 frames is computed from the per-clip frame lengths (End_Frame minus Begin_Frame) of the News alignment table, saved as numbers only by `make_extra.py`; it is the subtitle span, which matches the extracted clip length to within 2 frames on the Communication clips checked elsewhere. The joint run's own manifest is not saved: 22,088 is the sum of the saved real Communication training list of job 21734 (12,439 lines) and the 9,649 News rows the job scripts require. Original jobs: 7950 and 16398 (nA, nSA, five seeds), 27382 (length-penalty decoding), 21744 (validation ensemble, signable-recall secondary). The recipe facts (512 input frames in both arms, 20,263 training rows = 9,649 real + 10,614 stitched, pose augmentation only in nSA) are read from the two job scripts that trained the five seeds, `jobs/slurm/w4_news.sbatch` (seeds 42/101/202) and `w4_news_s2.sbatch` (seeds 303/404); these are script reads, not measurements, and the 1.1:1 ratio is computed from the two row counts the scripts require. The published values 2.11, 2.72 and 2.81 are cited literature. Not recomputed: training, the News-only run, News-to-Communication pretraining, joint training and the subword out-of-vocabulary check (outputs on the cluster).


### A65 — Paired bootstrap, development ladder and the final ensemble against the published system

**Claim.** On the 800-clip official test set the final six-member ensemble scores 11.80 BLEU-4, 1.47 above its RGB-fusion member (95% paired-bootstrap interval +0.40 to +2.56, 2,000 resamples), 1.00 above its strongest member (not resolvable) and 1.85 (18.6%) above the published GASLT score of 9.95; its own interval (9.87 to 13.89) includes 9.95. Only the ensemble-over-RGB-fusion step is resolvable (Results, Table res_bootstrap; ladder Table res_ladder; Discussion; appendix adequacy note).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/paired_bootstrap_ladder/check.py
```

```text
references match comm_run/manifests/test.tsv: 800/800
GATE final ensemble re-scores to 11.802 (recorded 11.802)
GATE baseline 9.522 (recorded 9.522), 40-epoch VLP 10.272 (recorded 10.272), RGB fusion 10.335 (recorded 10.335)
GATE stitched-data ensembles 12.114 (recorded 12.114), 11.562 (recorded 11.562), 12.836 (recorded 12.836)

DEVELOPMENT LADDER, official test, 800 clips
+ VLP pretraining, 40 epochs: greedy 8.75, beam 5 10.27  [c09-012]
+ RGB fusion: greedy 8.82, beam 5 10.33  [c09-013 c09-005 c14-031 c21-007]
RGB fusion adds +0.06 over the 40-epoch pre-trained model  [c09-013]
Final ensemble: beam 10, length penalty 0.3, BLEU-4 11.80 (11.802)  [c09-005 c14-005 c14-031 c21-007 c21-025 c11-028]
Final ensemble WER 0.907 on the 800-clip official test set  [c21-025]
Official test set size: 800 clips  [c21-025 c12-001 c19-017 c12-012 c02-025 c12-003 c14-031 c21-007]

Ensemble member list, read from jobs/slurm/mbr_wer_test2.sbatch (the job whose --ckpts decoded the final test ensemble at length penalty 0.3): 6 translators averaged  [c11-088]
the 6 listed checkpoints are exactly the 6 member dumps used below: True  [c11-088]
Members decoded alone with the ensemble decoder settings (6 dumps)  [c21-007]
  member comm_dwpose_ctc_s202: 10.46
  member comm_dwpose_ctc_vlplong80: 10.80
  member comm_dwpose_fus_s101: 10.62
  member comm_dwpose_fus_s202: 10.32
  member comm_dwpose_rgb_fusion: 10.01
  member comm_dwpose_vlp_ctc: 10.34
strongest member (comm_dwpose_ctc_vlplong80) 10.80; ensemble 11.80; ensemble minus strongest member +1.00  [c21-007]

GASLT published score 9.95 (published value, Auslan-Daily benchmark paper); ensemble minus 9.95 = +1.85  [c02-015 c11-004 c11-028 c13-053 c14-033 c21-007 c21-025]
relative to GASLT published: 18.6% (published value 9.95)  [c11-004]

Settings read from the source of src/tools/paired_bootstrap.py (the tool that produced Table res_bootstrap): default 2,000 resamples, seed 0, 95% interval, one shared draw matrix of sentence indices over the 800 sentences; this check uses the same convention  [c07-064 c07-066 c02-025 c12-001 c12-003 c19-017]

PAIRED BOOTSTRAP of BLEU-4 differences (BLEU-4, 4-gram): 2,000 resamples with replacement of the 800 test sentences, seed 0; interval = 2.5th and 97.5th percentiles of the difference = 95% interval  [c07-064 c07-066 c02-025 c12-001 c12-003 c12-012 c19-017]
GATE every interval below equals the recorded output of tools/paired_bootstrap.py (saved in recorded_bootstrap_tool_output.txt) to 0.0015
S6 minus RGB: +1.47 (observed +1.467), 95% interval [+0.40, +2.56], P(>0) 99.4%, resolvable, half-width 1.08  [c09-005 c14-005 c14-031 c21-007 c12-006 c07-058]
S6 minus M: +1.00 (observed +1.004), 95% interval [-0.13, +2.11], P(>0) 96.0%, not resolvable, half-width 1.12  [c21-007]
RGB minus CTC: +0.81 (observed +0.812), 95% interval [-0.34, +1.97], P(>0) 91.5%, not resolvable, half-width 1.15  [c12-006]
VLP minus CTC: +0.75 (observed +0.749), 95% interval [-0.22, +1.71], P(>0) 93.2%, not resolvable, half-width 0.96  [c12-007]
F minus S6: +0.31 (observed +0.313), 95% interval [-0.89, +1.50], P(>0) 69.3%, not resolvable, half-width 1.19  [c12-012]
Fp minus S6: +1.03 (observed +1.034), 95% interval [-0.10, +2.15], P(>0) 96.2%, not resolvable, half-width 1.13  [c12-012]
(M = comm_dwpose_ctc_vlplong80, same decoder settings, 10.80; table row 'its strongest member')  [c21-007]
RGB fusion (10.33) vs baseline (9.52): P(>0) 91.5%  [c12-006]
40-epoch pre-trained model (10.27) vs baseline (9.52): P(>0) 93.2%  [c12-007]
Figure res_bootstrap rows 5 and 7, which are not in Table res_bootstrap (stitched-data ensembles):  [c12-012]
G1 minus S6: -0.24 (observed -0.240), 95% interval [-1.34, +0.76], P(>0) 31.7%, not resolvable, half-width 1.05  [c12-012]
Fp minus F: +0.72 (observed +0.722), 95% interval [-0.30, +1.81], P(>0) 91.1%, not resolvable, half-width 1.06  [c12-012]
Stitched-data ensemble scores: 0.55:1 12.11, 1.1:1 12.84, six-model 11.56; final ensemble 11.80  [c12-012]
RGB-fusion checkpoint decoded alone with the ensemble decoder settings 10.01; its standalone beam-5 decode (the ladder row) 10.33
95% half-widths, the 6 rows of Table res_bootstrap: 1.08, 1.12, 1.15, 0.96, 1.19, 1.13; range 0.96 to 1.19, mean 1.11
95% half-widths, those 6 rows plus the two extra rows of Figure res_bootstrap (8 comparisons): range 0.96 to 1.19, mean 1.09; largest 1.19, median 1.10
(single-system interval, not a paired difference) half-width of the 800-sentence interval on the ensemble alone: 2.0
intervals excluding zero among the 8 compared pairs: ['S6-RGB']  [c07-058 c12-012]

Final ensemble's own 95% bootstrap interval over test sentences: 9.87 to 13.89  [c11-029 c21-023]
interval includes GASLT's published 9.95: True  [c21-023]

Judge adequacy (0-4), read from the saved per-clip grades of job 10288: final ensemble 1.311, RGB-fusion member 1.259  [c20-010]
difference +0.052, 95% interval -0.01 to +0.11 (paired bootstrap over 800 clips, 2,000 resamples, seed 0)  [c20-010]

src/tools/seed_decision.py: T975[df] (two-sided 95%) matches Student's t quantile for every df 1..6, df = n - 1 [GATE]
seed_decision.interval on 3 paired differences uses T975[2] = 4.303 (Student's t, n - 1 = 2 degrees of freedom): half-width 1.242 equals t(0.975, 2) x s / sqrt(3) = 1.242  [c07-057]
seed_decision.interval on 5 paired differences uses T975[4] = 2.776 (Student's t, n - 1 = 4 degrees of freedom): half-width 0.491 equals t(0.975, 4) x s / sqrt(5) = 0.491  [c07-057]
Stage 1 seeds, read from jobs/slurm/wave1_r2.sbatch line 33: SEEDS=(42 101 202); S=${SEEDS[$SLURM_ARRAY_TASK_ID]}  (three seeds: 42, 101, 202)  [c07-052 c07-054]
Stage 2 seeds, read from jobs/slurm/w4_train.sbatch line 19: S=(42 101 202 303 404); s=${S[$SLURM_ARRAY_TASK_ID]}  (303 and 404 follow 42, 101, 202)  [c07-055]
```

**Scope.** Recomputed from the saved per-sentence test decodes (baseline, 40-epoch pre-trained model, RGB-fusion model, six ensemble members, final ensemble, and the three stitched-data ensembles of rows 5 and 6 of Table res_bootstrap and rows 4 to 7 of Figure res_bootstrap) through `slt/metrics.py`; bootstrap with seed 0 and one shared draw matrix as in `tools/paired_bootstrap.py`, whose defaults (2,000 resamples, seed 0, 95%) are read from its source. The run first checks that the faster scorer equals the project scorer, re-scores the recorded BLEU-4 of every system, and requires every interval to equal the original tool run (a saved output, `recorded_bootstrap_tool_output.txt`). Adequacy grades are read from the saved per-clip output of the judge job 10288. The seed-level t quantiles are checked against the table `T975` in `tools/seed_decision.py`, which the staged decision rule uses; seed lists are read from the job scripts. The settings claims (2,000 resamples, seed 0, percentiles, 800 sentences) confirm that this convention produced these rows; they do not re-derive other intervals in the dissertation. Not done: training or decoding; the three-seed means, the 80-epoch single model's beam-5 decode and the first-listed-keypoint run are on Kaya or only written down (see claims.json). The dissertation's paired-bootstrap half-width of "about 1.2" is only partly reproduced: across the six rows of Table res_bootstrap the half-widths are 0.96 to 1.19 (mean 1.11; with the two extra rows of the figure, mean 1.09), so 1.2 is the upper end and "about 1.1" would be the typical value; the claim is listed in claims.json as disagreeing and its lines are deliberately untagged. The six-member count is read from the member list of the sbatch script that produced the test ensemble. The RGB-fusion checkpoint decoded alone with the ensemble decoder settings scores 10.01; the ladder row is its standalone beam-5 decode (10.33), while the strongest member (10.80) is compared under the ensemble decoder settings, as in the dissertation.


### A66 — Pose augmentation replication, recipe and the 1.1:1 baseline

**Claim.** Pose augmentation (temporal resampling by a factor from 0.8 to 1.2 with nearest-index frame selection, in-plane rotation within 10 degrees, x/y scaling 0.9 to 1.1; Methodology) was run on the 1.1:1 stitched corpus with three paired seeds and concurrent controls. Test BLEU-4 at beam width 5 rose from 10.34 to 11.44, 10.93 to 11.38 and 10.66 to 11.19 (means 10.64 to 11.34, +0.70); validation and greedy test also rose in all three seeds; the 95% t-interval over seeds, -0.18 to +1.57, includes zero; 11.34 is compared with 11.14 for 1.1:1 without augmentation (Results, ladder table, ratio and pose-augmentation paragraphs; Discussion). Two numbers do not reproduce: the second seed scores 11.37, not 11.38 (11.3749), and the pooled content-word precision rise (method chapter's 71-word content list) is about +2.38, +1.72, +1.73 points, not the quoted 1.6 to 2.3 (the 75-word training-script list is printed as a labelled second line).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/pose_augmentation_replication/check.py
```

```text
GATE recomputed aug s42 11.4435 vs recorded 11.4435 (read from the saved job output beam5_test.json)

Beam-5 official test BLEU-4, aug vs its same-seed control (recomputed from the saved hypotheses, 800 clips)
  seed 42: 10.34 -> 11.44  (+1.10)
  seed 101: 10.93 -> 11.37  (+0.45)
  seed 202: 10.66 -> 11.19  (+0.54)
Test BLEU-4 at beam width 5 rose in all 3 (three) seeds, aug above its same-seed control in each  [c09-074]
seed 42 from 10.34 to 11.44; seed 202 from 10.66 to 11.19; seed 101 from 10.93 to 11.37 (unrounded 11.3749; the dissertation says 11.38: DISAGREES, a rounding difference of 0.01)  [c09-075]
means 10.64 to 11.34, +0.70 (unrounded +0.695); control mean 10.64, augmented mean 11.34  [c09-018 c09-075 c09-076]
+ pose augmentation (3 seeds) 11.34, +0.70 over its concurrent controls (10.64), 3 of 3 seeds  [c09-018]
seed-level 95% t-interval of the paired differences: -0.18 to +1.57  (t=4.303, df=2; includes zero: True)  [c09-077 c14-026]
95% t-interval over the seeds -0.18 to +1.57  [c14-026]
validation BLEU-4 aug minus ctrl per seed (read from the saved job output progress.json): +0.188 +0.796 +0.190; rose in 3 of 3  [c09-076]
greedy test BLEU-4 aug minus ctrl per seed (read from the saved job output beam5_test.json): +0.349 +0.389 +0.423; rose in 3 of 3  [c09-076]
beam-5 test rose in 3 of 3 seeds: the pre-registered three-seed rule is met: True  [c09-074 c09-076 c09-077]
Pooled content-word precision of the beam-5 test outputs (stitch_lexicon.STOP, 71 words): seed 42 22.37 -> 24.75; seed 101 23.25 -> 24.97; seed 202 24.02 -> 25.75
Content-word precision rise per seed (points): +2.38 +1.72 +1.73; in every seed: True
range of the rise +1.72 to +2.38 points under the 71-word list; the dissertation says 1.6 to 2.3: DISAGREES (c09-078 not covered)
second line, labelled: under the 75-word train_ctc.STOP list the rise per seed is +2.26 +1.66 +1.55 points

Recipe (read from src/config.py defaults and the job script)
temporal resampling factor from [0.8, 1.2] (aug_speed 0.2; job 1215231 passed the same value)  [c06-034]
in-plane rotation from [-10, +10] degrees (aug_rotate_deg 10.0)  [c06-035]
x and y scaling from [0.9, 1.1] (aug_scale 0.1)  [c06-036]

Synthetic-to-real ratio 1.1:1 without augmentation, beam-5 test BLEU-4 (read from the saved job outputs stitch_ratio_job_logs.txt: stitch-scale 1196567 seed 42, stitch-scl2 1199393 seeds 101, 202)
  sc_r111_s101 (stitch-scl2-1199393.out): 11.03
  sc_r111_s202 (stitch-scl2-1199393.out): 10.87
  sc_r111_s42 (stitch-scale-1196567.out): 11.52
mean of the three logged aug-free 1.1:1 scores 11.14  [c09-052]
the higher means at 1.1:1 (11.14, and 11.34 with pose augmentation)  [c09-052]

Recipe checked against the augmentation code slt/data.py (augment_pose), run on synthetic keypoints
temporal resampling picks frame indices with linspace(0, T-1, new_T).round(), i.e. nearest-index frame selection (source line checked)  [c06-034]
temporal factor over 2000 draws: min 0.80, max 1.20, inside [0.8, 1.2]: True  [c06-034]
in-plane rotation over 2000 draws: min -10.0, max +10.0 degrees (limit +-10)  [c06-035]
x and y scale factors over 4000 draws: min 0.90, max 1.10 (limit [0.9, 1.1])  [c06-036]
```

**Scope.** BLEU-4 is recomputed from the saved beam-5 hypotheses of job 1215231 (800 official test clips); validation and greedy scores are read from the saved job outputs (`progress.json`, `beam5_test.json`); the interval is computed here from the three paired differences. The 1.1:1 baseline mean is parsed from the saved logs of jobs 1196567 and 1199393. The recipe ranges are measured by running the repository's `augment_pose` on synthetic keypoints. Content-word precision uses `STOP` from `src/tools/stitch_lexicon.py` (71 words); the 75-word `STOP` of `src/train_ctc.py` gives a labelled second line. Training is not re-run. Not covered: the dose sweep (job 6612; its script is not saved and its summaries are on the cluster) the 11.38 versus 11.37 difference and the 1.6 to 2.3 precision range.


### A67 — PoseStitch side experiments (sign order, wider vocabulary, function signs, letters, wrong-sense signs)
**Claim.** The dissertation reports five negative or null variations of the stitched-data recipe, each as a paired comparison with its in-batch control: a shuffled sign order costs about 0.7 BLEU-4 on validation unless the CTC targets follow the signs (Appendix B, wave 10); a wider-vocabulary corpus (B) and a doubled amount (C) give no gain (wave 6); not signing function words (32% of stitched signs dropped) gives no resolvable gain over five seeds (wave 7); the fingerspelled-letter signs (8.7% of stitched signs) can be removed with no measurable change (wave 2); and 3.45% of the signs are wrong-sense glosses (wave 12b corpus facts and its three-seed comparison).

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/stitching_side_experiments/check.py
```
```text
saved test references = comm_run/manifests/test.tsv (as a set of 800)
GATE 34 runs: BLEU-4 recomputed from the saved beam-5 hypotheses equals the job summary (expected 0 mismatches, found 0)

WAVE 10 shuffled sign order (3 seeds, beam-5 best.pt; difference to the English-order control A)
S - A validation per seed -0.456 -0.704 -0.917; mean -0.69  [c17-063]
S - A test per seed -1.550 +0.181 -0.662; mean -0.68  [c17-063]
SC - A validation per seed +0.791 +0.293 +0.266; mean +0.45  [c17-064]
SC - A test per seed -0.800 +0.357 +0.071; mean -0.12  [c17-064]

WAVE 6 wider-vocabulary stitched corpus (B: 1.1:1; C: twice the amount)
B - A validation -0.19 (2/5 positive)  [c16-032]
B - A test +0.09 [-0.45, +0.63] (95% t-interval over 5 seeds)  [c16-032]
C - A validation -0.80 (0/3 positive; stage 1 uses the first three seeds)  [c16-032]
seeds used for B vs A: 5; for C vs A: 3 (counted from the saved job summaries)

WAVE 7 function-word signs dropped (D vs A, 5 seeds; DL = length-matched arm)
D - A validation +0.18 (4/5 positive)  [c16-034]
D - A test -0.09 [-0.76, +0.59] (2/5 positive; fresh seeds 303 and 404: -0.444, -0.515)  [c16-034]
DL (length-matched) arm run at 2 seeds [303, 404]  [c16-034]
DL - A test per seed -0.072 -0.606 (descriptive)
content-word precision, D - A, test: mean -0.95 (0/5 positive) (read from the saved job output wave7_readout_output.txt; the tool that wrote it is src/tools/arm_readouts.py)  [c16-034]
function-word signs dropped: 30247 of 94428 spans = 32% (read from the saved job output wave7_prep_66643.out)  [c16-033]
seeds compared: 5  [c16-033]
constrained generation pool: 20608 sentences, 20608 unique (read from the saved corpus-audit output stitch_corpus_audit.json; the 6,880 unconstrained sentences are not saved locally)

ADOPTED STITCHED CORPUS (13,760 clips; segment list of the corpus audit)
GATE signs 94428 (recorded in the corpus-facts output: 94428); wrong-sense signs 3257 (recorded: 3257)
single-letter (fingerspelled alphabet) signs: 8187 of 94428 = 8.7% of stitched signs  [c17-079]
wrong-sense signs (the 11 sense-disambiguated glosses): 3257 of 94428 = 3.45% of the signs  [c17-048]

WAVE 12b wrong-sense signs removed (W - A, 3 seeds, beam-5 best.pt; per-run values read from the saved job output wave12b_summary_138381.out)
GATE per-seed W - A differences recomputed from the six run records equal the job's printed lines (3 seeds)
W - A validation per seed -0.657 +0.627 +0.374; mean +0.1146 = +0.115 to 3 decimals, which the dissertation rounds again to +0.12 (a single rounding gives +0.11); 95% t-interval [-1.57, +1.80] over 3 seeds  [c17-049 c17-048]
W - A test per seed -0.643 +0.525 +0.034; mean -0.03 [-1.48, +1.43] (95% t-interval over 3 seeds)  [c17-049 c17-048]
GATE mean and 95% t-interval equal the job's own seed_decision.py lines (validation and test)

WAVE 2 letter-fixed corpus (w2LF vs w2A, beam-5 best.pt, read from the saved job summaries)
w2LF - w2A validation per seed -1.230 +0.611 +0.306; mean -0.10  [c17-080]
w2LF - w2A test per seed -0.243 +0.296 +0.026; mean +0.03  [c17-080]
seeds: 3  [c17-079]
```
**Scope.** Recomputed from the saved beam-5 hypotheses of 34 runs (jobs 49135, 31220, 66644, 16435 and 25863) through `slt/metrics.py`, with the official references; every run first reproduces the BLEU-4 value in its job summary (the gate). Differences, signs, counts and the t-interval are computed here. The wave-2 differences are read from the saved job summaries (job 5403), the content-word precision and the 32% from the saved job outputs (66644, 66643), the 20,608 constrained sentences from the saved corpus audit, and the sign counts from the adopted corpus's segment list. The wave 12b differences (job 138380, summary 138381) are formed here from the six per-run records (beam-5 validation and test BLEU-4 of arms A and W) in the saved job output, gated against the job's own printed per-seed lines, mean and t-interval; the 3.45% comes from the adopted corpus's segment list. The 20,608 is printed untagged because its claim also states a half that is not saved locally (6,880 unconstrained sentences). Finding: the validation mean is +0.1147, which the job prints as +0.115 and the dissertation gives as +0.12; a single rounding gives +0.11 (immaterial; interval unchanged). Not done: training, the scrambled-sign control (job 1193231), the seed replication (job 1173725), wave-5 pre-training (job 16397) whose outputs are on Kaya.


### A68 — Transfer from a pre-trained sign-language model
**Claim.** The dissertation (abstract, introduction, Results "Transfer from a Pre-trained Sign-Language Model", Appendix `app_transfer`, conclusion) states that fine-tuning the released How2Sign checkpoint on Auslan-Daily Communication gives a five-seed mean test BLEU-4 of 18.21 (validation 17.37; 13.73 on the 570 unseen-target clips), against 10.37 for the smaller translator of this thesis trained from scratch in the same jobs (+7.84, 95% seed-level interval +7.04 to +8.64, five of five seeds; +5.79 after adjusting for output length), 6.60 with random pose-side weights (one run collapsed to 30 distinct outputs), and 20.60 for the five-model ensemble. The port reproduces the checkpoint's How2Sign output at 14.37 against 14.50.
**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/pretrained_transfer/check.py
```
```text
GATE: all 30 saved decodes re-score to the BLEU-4 and distinct-output count recorded by their jobs (tolerance 1e-6)

== Five-seed mean beam-5 BLEU-4 (seeds [42, 101, 202, 303, 404]); unseen = 570 test clips whose reference is absent from training ==
arm          validation     test  unseen (test)
ctrl_small   val 10.52 test 10.37 unseen 5.47 over 5 seeds  [c11-046 c21-002 c21-012 c19-020 c11-048]
u_scratch    val 7.06 test 6.60 unseen 3.21 over 5 seeds  [c11-049 c19-021]
u_asl        val 17.37 test 18.21 unseen 13.73 over 5 seeds  [c01-022 c03-008 c11-014 c11-045 c21-006 c21-012 c21-005 c13-055 c19-019 c11-048]
570 unseen-target test clips  [c11-048 c19-047]
per-seed test BLEU-4 of the fine-tuned arm u_asl: 17.80 18.43 18.16 18.77 17.87
BLEU-4: fine-tuned test mean 18.21 (five seeds) against published best 9.95 (published value, cited)  [c03-008 c11-045]
from scratch test mean 10.37; random pose-side weights 6.60  [c21-012 c11-046]

== Paired comparisons, seed-level 95% t-interval (df = 4) ==
95% t-interval over 5 seeds (df = 4)  [c03-010 c11-047]
test fine-tuned minus from scratch: +7.84 [+7.04, +8.64]  positive in 5 of 5 seeds  [c03-010 c11-047 c19-029 c19-030 c11-047]
test fine-tuned minus random pose-side weights: +11.61 [+10.09, +13.13]  positive in 5 of 5 seeds  [c19-031 c19-030 c11-047]
val  fine-tuned minus from scratch: +6.85 [+6.67, +7.02]  positive in 5 of 5 seeds  [c19-029 c19-030 c11-047]
val  fine-tuned minus random pose-side weights: +10.31 [+9.06, +11.56]  positive in 5 of 5 seeds  [c19-031 c19-030 c11-047]
test fine-tuned minus from scratch, adjusted for output length (brevity penalty): +5.79 [+4.82, +6.76] (unrounded 5.7917 [4.8187, 6.7648])  [c19-030]

== Distinct outputs (of 800 clips) ==
val  u_scratch  s42:204 s101:274 s202:242 s303:30 s404:159
val  u_asl      s42:568 s101:586 s202:587 s303:544 s404:585
test u_scratch  s42:193 s101:267 s202:216 s303:31 s404:170
test u_asl      s42:576 s101:594 s202:601 s303:543 s404:588
random-weight runs with fewer than 100 distinct validation outputs (the collapse guard of 100): 1 -> seed 303 has 30 distinct outputs  [c11-049 c19-032]

== Episode specificity of the per-clip gain (3-seed mean sentence-BLEU lift, 89 videos with at least 3 clips) ==
between-video permutation test, 2,000 permutations: p = 0.094  [c19-049]

GATE: ensemble re-scores to the BLEU-4 recorded by its job (20.6044)
five-model ensemble (beam 10, length penalty 0.8) test BLEU-4 20.60 (20.604)  [c13-055 c21-005 c21-006]

How2Sign port parity (read from the saved job output gate_how2sign_port.json, 2349 clips, beam 4): port 14.37 (14.372), released output 14.50  [c19-009]
GATE target 14.5 +- 0.5: passed
```
**Scope.** Recomputed from the saved beam-5 hypotheses of 3 arms x 5 seeds x 2 splits (Kaya jobs 167035/167036, seeds 42/101/202, and 175088/175089, seeds 303/404) and the saved ensemble decode (job 195361, beam 10, length penalty 0.8), scored against the official references; every decode must re-score to the BLEU-4 recorded by its job. The How2Sign parity numbers are read from the saved job output of the gate. Not recomputed: training, decoding, the YouTube-ASL identifier lookup (10,327 / 769) and the overlap with external label text (13 clips; 4.37 / 13.75), whose inputs were not saved. The upper bound of the length-adjusted interval comes out as +6.76 (+6.7648), the dissertation prints +6.77.


### A69 — Pre-training length and News frame limit chosen on validation alone

**Claim.** The pose-text pre-training length and the News input-frame limit were re-chosen on validation alone (Method; Results, pre-training and News paragraphs; Discussion limitations). Five pre-training settings (none, 40, 80, 128 and 256 epochs) were each trained with three seeds; the pre-registered rule selects 40 epochs, and no pre-trained length beats no pre-training (40 epochs minus none: validation −0.258 [−1.57, +1.06], test −0.890 [−2.14, +0.36]). On News, mean beam-5 validation BLEU-4 is 2.07 at 256 frames and 2.08 at 512 (+0.01, interval −0.37 to +0.39), so the cheaper 256 frames is selected.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/pretraining_length_rerun/check.py
```
```text
GATE P0 mean beam5_val 10.450 (recorded 10.450)
GATE P40 mean beam5_val 10.192 (recorded 10.192)
GATE P0 mean beam5_test 10.122 (recorded 10.122)
GATE P40 mean beam5_test 9.232 (recorded 9.232)
pre-training lengths in the job script w11_vlpepochs.sbatch: [40, 80, 128, 256] epochs (0 = no pre-training); seeds [42, 101, 202]  [c07-020 c07-021]
pre-training epochs completed per arm (saved progress files): P40: 40, P80: 80, P128: 128, P256: 256  [c07-020]
seeds per setting: 3 (P0: 3; P40: 3; P80: 3; P128: 3; P256: 3) = 3 seeds  [c07-021 c09-029 c09-034]

PRE-TRAINING RE-RUN (beam-5 BLEU-4, mean of three seeds)
  P0    validation 10.450  test 10.122
  P40   validation 10.192  test 9.232
  P80   validation 10.001  test 9.810
  P128  validation 9.827  test 9.697
  P256  validation 9.907  test 9.398
pre-registered rule (tie 0.10, as in the recorded read-out command): cheapest length within 0.10 of the best mean validation 10.192
SELECTED ON VALIDATION: P40 = 40 pre-training epochs
  P40 - P0 validation -0.258 [-1.57, +1.06]; test -0.890 [-2.14, +0.36]
  P80 - P0 validation -0.450 [-1.09, +0.19]; test -0.311 [-0.66, +0.04]
  P128 - P0 validation -0.624 [-2.37, +1.12]; test -0.425 [-1.58, +0.73]
  P256 - P0 validation -0.544 [-2.97, +1.88]; test -0.723 [-1.45, +0.01]
among the pre-trained lengths the highest mean is P40 on validation and P80 on test

NEWS FRAME LIMIT (150 epochs, News real clips only, three seeds)
  256 frames: mean beam-5 validation BLEU-4 2.07 (2.065); test 1.875  [c11-032]
  512 frames: mean beam-5 validation BLEU-4 2.08 (2.077); test 1.815  [c11-032]
  512 - 256 validation +0.01 (+0.012); 95% t-interval over the 3 seeds -0.37 to +0.39  [c11-033]
  rule (cheapest within 0.05 of the best mean validation): 256 frames selected  [c11-034 c06-026]
  job script w11_newsframes.sbatch (the wave-11 re-run, not the earlier runs): frame limits ['256', '512'], 150 epochs, seeds ['42', '101', '202']
  frame limit in the earlier News-only job scripts: w4_news.sbatch 512, news_bpe.sbatch 512, w12a_news_decomp.sbatch 512; selected on validation in wave 11: 256  [c06-026]
  news_long.sbatch (earlier single-run job, 150 epochs, outputs not saved locally) compares: 256-frame against 512-frame  [c18-069]
```

**Scope.** Recomputed from the stored beam-5 BLEU-4 of each of the 21 runs (cluster jobs 116619, pre-training arms, and 116620, News arms; read from the saved job outputs), with the selection rule and 95% t-intervals computed here. No training or decoding is repeated. The frame limits of the earlier News-only runs (c06-026) and of the earlier single-run comparison (c18-069) are read from the recorded job scripts (`w4_news.sbatch`, `news_bpe.sbatch`, `w12a_news_decomp.sbatch`, `news_long.sbatch`); the scores of that earlier comparison (job 1196181, c18-070), like the single-run 128- and 256-epoch scores, are on the cluster only and are not checked here.


### A70 — CTC-only pseudo-gloss first stage against a compute-matched control (wave 23)

**Claim.** Appendix B (supplementary results table, the row on CTC-only pseudo-gloss pre-training, wave 23) states that a first stage of 40 epochs with only the pseudo-gloss CTC loss, then the 40-epoch training from its encoder, was compared with 40 further epochs of the full loss with the same transfer, over three seeds: validation +0.39 (-0.81 to +1.60; higher in two of three seeds), test +0.38 (-0.34 to +1.10; all three); the 40 further epochs alone against no first stage: test +0.32 (-0.10 to +0.75); the CTC-only stage against no first stage: test +0.70 (+0.10 to +1.31); and that the control differed from an earlier job by 0.62 at one seed.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/pseudo_gloss_pretraining/check.py
```
```text
val references match comm_run/manifests/val.tsv: 800/800
test references match comm_run/manifests/test.tsv: 800/800
GATE: all 18 recomputed beam-5 BLEU-4 scores (validation and test) equal the job's recorded values to 1e-6
GATE: the 6 recomputed beam-5 test scores of the earlier jobs (wave 16 and wave 6 controls) equal those jobs' recorded values to 1e-6

Setting: beam 5, best.pt by validation, 800 clips per split, seeds 42/101/202; X = CTC-only first stage then 40-epoch training, Y = 40 further full-loss epochs then 40-epoch training, C = neither

Beam-5 BLEU-4 per seed (42 / 101 / 202) and mean
  C val: 11.307 / 11.939 / 11.338   mean 11.528
  C test: 11.711 / 11.366 / 10.411   mean 11.163
  X val: 11.845 / 12.162 / 12.453   mean 12.153
  X test: 12.185 / 12.045 / 11.372   mean 11.867
  Y val: 11.879 / 11.871 / 11.534   mean 11.761
  Y test: 12.136 / 11.491 / 10.827   mean 11.485

Paired differences, 95% t interval over three seeds (df = 2)
  X - Y val: mean +0.39 (-0.81 to +1.60); higher in 2 of 3 seeds; per seed -0.034 +0.291 +0.919  [c17-102]
  X - Y validation: higher in 2 of 3 seeds, interval -0.81 to +1.60  [c17-102]
  X - Y test: mean +0.38 (-0.34 to +1.10); higher in 3 of 3 seeds; per seed +0.049 +0.555 +0.545  [c17-103]
  X - Y test: higher in 3 of 3 seeds  [c17-103]
  Y - C test: mean +0.32 (-0.10 to +0.75); higher in 3 of 3 seeds; per seed +0.425 +0.124 +0.416  [c17-104]
  X - C test: mean +0.70 (+0.10 to +1.31); higher in 3 of 3 seeds; per seed +0.474 +0.679 +0.960  [c17-105]
  Y - C val: mean +0.23 (-0.56 to +1.03); higher in 2 of 3 seeds; per seed +0.571 -0.068 +0.196
  X - C val: mean +0.63 (-0.50 to +1.75); higher in 3 of 3 seeds; per seed +0.537 +0.223 +1.115

Control against earlier jobs of the same recipe: beam-5 test of control C (this job) minus the earlier control, per seed
  wave 16 control A: seed 42: this job 11.711, earlier 11.711, difference +0.000; seed 101: this job 11.366, earlier 10.745, difference +0.621; seed 202: this job 10.411, earlier 10.411, difference +0.000
  wave 6 control A: seed 42: this job 11.711, earlier 11.711, difference +0.000; seed 101: this job 11.366, earlier 11.366, difference +0.000; seed 202: this job 10.411, earlier 10.839, difference -0.428
  the control differed from the wave 16 control (same scores as the wave-5 control the read-out compares with) by 0.62 at seed 101, and by 0.00 and 0.00 at seeds 42 and 202  [c17-106]
  difference at seed 101 against wave 16 control A: +0.621; against wave 6 control A: +0.000 (this job's control reproduces wave 6's at this seed)
  read from the saved job output summary_179457.txt: differences from wave 5's control (seed: got - control) {42: 0.0, 101: 0.621, 202: 0.0}
GATE: the differences from wave 16 control A (recomputed here) equal the read-out job's recorded differences from wave 5's control at all three seeds to 1e-3, so the wave 16 control carries wave 5's scores  [c17-106]

Training length and seeds (read from the saved job output progress.json of each run)
  all 9 runs (C, X, Y x seeds 42/101/202) completed 40 training epochs of the full loss (progress.json)
  seeds per arm: 3  [c17-101]
```

**Scope.** Beam-5 BLEU-4 of best.pt is recomputed from the saved hypotheses and references of all 18 runs (arms C, X, Y; seeds 42/101/202; validation and official test) with the project's metric code, gated against the scores the job itself recorded (to 1e-6); differences, sign counts and 95% t intervals (df = 2) are computed here. The "earlier job" is not named in the dissertation. The read-out compares the control with wave 5's control, whose per-item output is on Kaya and not local; its saved difference line (read from the job output `summary_179457.txt`) is {42: 0.0, 101: 0.621, 202: 0.0}. The local control of wave 16 (same recipe, same scores) is recomputed from its saved hypotheses and gives the same three differences (seed 101: 11.366 against 10.745, +0.621), and wave 6's control, also recomputed, equals this job's control at seed 101. Training length (40 epochs) is read from progress.json. Not covered: that the "40 further epochs" of arm Y are a continuation of C's training (c17-100) and the 40-epoch CTC-only first stage itself (c17-099): both are stated only by the job script and the pre-registration, not by any saved output. Not done: training and decoding. Original jobs: training 179456, read-out 179457 (smoke 179455); earlier controls: wave 16 (174916), wave 6 (16435).


### A71 — Published systems' own code and the project's translator with the published recipe
**Claim.** The Results chapter (Section on the published systems, Table `tab:app_ownccode` in the appendix) states that the authors' SL-Transformer and GASLT code, run for 200 epochs on 53 DWPose points, scored 8.63 and 9.58 (five seeds) against 10.43 for the project's translator trained for 40 epochs in the same jobs (differences -1.80 and -0.85); that the project's translator given the authors' recipe scored 9.91 (O53), +1.48 above their SL-Transformer (8.44) and not resolvably above their GASLT (9.47); and that on its usual 133-point input at beam width 5 it scored 9.63 against 10.20 for its own recipe. The appendix also states that runs on the first-listed AlphaPose person scored 2 to 5 BLEU-4 and gives the scores of older single runs (2.08, 1.03, 7.11, 7.92). One appendix statement (60 epochs at learning rate 3e-4 for the larger model) is not reproduced; see Scope.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/published_systems_200_epochs/check.py
```
```text
GATE 1: 79 saved test decodes re-scored through slt.metrics; 79 equal the score recorded by the job that wrote them (1e-6)
GATE 2: the three-seed Rule B differences of the 12l2 summary (SL-T -1.91, -2.67 to -1.15; GASLT -0.86, -1.77 to +0.05) are reproduced: yes

== 1. The authors' code (200 epochs) on 53 DWPose points against this project's translator, five seeds (jobs 167758 + 179411), test BLEU-4, project scoring function
  SL-Transformer, dw53 cells, 5 seeds  mean 8.63   per seed [7.99, 8.74, 8.57, 9.01, 8.83]  [c11-022]
  GASLT, dw53 cells            mean 9.58   per seed [9.86, 9.69, 8.91, 9.57, 9.87]  [c11-022]
  translator of this thesis (Edw, 40 epochs, beam 5) mean 10.43   per seed [10.44, 10.05, 10.55, 10.6, 10.52]  [c11-023]
  Edw epochs completed / planned: [[40, 40], [40, 40], [40, 40], [40, 40], [40, 40]]  [c11-023]
  Rule B (authors' code minus translator, pooled t, 8 df):
  SL-Transformer - Edw                           -1.80 [-2.27, -1.34]  [c11-024]
  GASLT - Edw                                    -0.85 [-1.32, -0.39]  [c11-024]

== 2. This project's translator with the published recipe (200 epochs, constant 5e-5), job 239443, three seeds
  O53  (our layers, 53 DWPose points)  authors' decoding protocol: mean 9.91   beam 5: mean 9.80  [c11-025 c19-073 c19-078]
  Ocwb (our layers, usual input)       beam 5: mean 9.63  [c11-027 c19-076]
  Edw  (our recipe, 40 epochs)         beam 5: mean 10.20  [c11-027 c19-076]
  authors' SL-Transformer, retrained in this job: mean 8.44;  authors' GASLT: mean 9.47  [c11-025 c11-026 c19-073 c19-075]
  O53 (authors' protocol) minus the authors' code retrained in the same job:
  O53 - SL-Transformer                           +1.48 [+0.63, +2.33]  seeds positive 3 of 3  [c11-025 c19-073 c19-074]
  O53 - GASLT                                    +0.44 [-0.16, +1.05]  seeds positive 2 of 3  [c11-026 c19-075]
  Recipe effect on this project's translator, beam 5:
  Ocwb - Edw                                     -0.57 [-1.55, +0.42]  seeds positive 0 of 3  [c11-027 c19-076 c19-077]
  Input format at the authors' recipe, beam 5:
  Ocwb - O53 (O53 beam-5 mean 9.80)              -0.17 [-1.14, +0.80]  seeds positive 2 of 3  [c19-078]
  Ocwb settings: arrays of 133 keypoints, subset cwb_body_hands, confidence True, frame normalisation; 200 epochs, constant learning rate 5e-05, batch 64; decoded at beam width 5  [c19-062]

== 3. Earlier runs of the authors' code on the first-listed AlphaPose person (not the signer), test BLEU-4
  12l (stopped early)        gaslt__ap56_bbox       mean 2.18  (n=3)
  12l (stopped early)        gaslt__ap56_frame      mean 2.12  (n=3)
  12l (stopped early)        gaslt__ap56_raw        mean 2.46  (n=3)
  12l (stopped early)        slt__ap56_bbox         mean 4.37  (n=3)
  12l (stopped early)        slt__ap56_frame        mean 3.66  (n=3)
  12l (stopped early)        slt__ap56_raw          mean 3.33  (n=3)
  12l2 (200 epochs)          gaslt__ap53_frame      mean 4.49  (n=3)
  12l2 (200 epochs)          gaslt__ap56_frame      mean 4.33  (n=3)
  12l2 (200 epochs)          gaslt__ap56_raw        mean 4.82  (n=3)
  12l2 (200 epochs)          slt__ap53_frame        mean 4.24  (n=3)
  12l2 (200 epochs)          slt__ap56_frame        mean 4.63  (n=3)
  12l2 (200 epochs)          slt__ap56_raw          mean 4.27  (n=3)
  w22 (200 epochs, 2 seeds)  gaslt__ap53_frame      mean 4.08  (n=2)
  w22 (200 epochs, 2 seeds)  slt__ap53_frame        mean 3.79  (n=2)
  cell means range 2.12 to 4.82; single runs range 1.75 to 5.26 (40 runs, 14 cells)  [c19-056]

== 4. Older single runs. No per-item outputs (hypotheses) exist for them in this working tree: the folders of these runs hold a training log, a metrics.json or progress.json and a checkpoint only, so the scores are read, not recomputed
  exact SL-Transformer re-implementation (25,315,840 parameters), distributed keypoints, official test: beam 5 2.08, greedy 1.55  (read from the saved run summary exact_slt_run_summary.json; its training log ends at epoch 25 of 60 and exact_slt_progress.json says stopped_at_epoch_25_gate)  [c08-016]
  25.9 M SignJoey-compatible model (25,945,600 parameters), first-listed AlphaPose keypoints: greedy test 1.03  (read from the saved run summary signjoey_run_summary.json; its training log ends at epoch 10 of 60)  [c16-021]
  d=512, 6+6 layers on DWPose: greedy test 7.11  (d_model 512, 6+6 layers; read from the metrics.json that the job wrote, best epoch 36)  [c16-021]
  default size on DWPose: greedy test 7.92  (d_model 256, 4+4 layers; read from the metrics.json that the job wrote, best epoch 28)  [c16-021]
  recorded training settings of the runs in this row of the appendix table (c16-022 compares them):
    d512 DWPose run, test 7.11 (metrics.json of the job): 40 epochs, cosine learning rate 0.0005, batch 32 x 1 accumulation
    full DWPose run, test 7.92 (metrics.json of the job): 40 epochs, cosine learning rate 0.0005, batch 64 x 1 accumulation
    25.9 M model scored 1.03 (checkpoint settings): 60 epochs planned (ran 10), plateau learning rate 0.001, batch 16 x 2
    another 25.9 M-class model (512-d, 3+3 layers, 1024-d adapter, first-listed AlphaPose; not the 1.03 run): greedy test 2.29 (metrics.json of the job), 60 epochs, cosine learning rate 0.0003, batch 16 x 2
  c16-022 CHECK: the 7.11 run recorded 40 epochs at 0.0005 (same as the default-size run: 40 at 0.0005); '60 epochs at 3e-4' is recorded only for the 2.29 run above: DISAGREES with the dissertation sentence for the 7.11 run
```
**Scope.** Sections 1 to 3 re-score 79 saved test decodes (800 clips each) through `slt/metrics.py`; the gate requires each to equal the score its job recorded, and the three-seed differences printed by the 12l2 summary are reproduced. Means, differences and pooled-variance 95% t-intervals are computed here. Training is not repeated; original jobs 138567, 167758, 179411 (the authors' code and the project's translator, on the cluster) and 239443. Section 4: no hypothesis file exists for the four older runs (their folders hold a training log, a metrics.json or progress.json, and a checkpoint; `results/legacy_runs/` was searched too), so their scores are read from what the jobs saved, not recomputed. The 7.11 and 7.92 scores are in the metrics.json that the jobs wrote; the 2.08 and 1.03 scores exist only in run summaries written on 5 Aug 2026 (the training logs, copied, confirm where each run stopped). The appendix statement that the larger model used 60 epochs at learning rate 3e-4 against 40 at 5e-4 disagrees with the metrics.json of the 7.11 run (40 epochs at 5e-4): 60 epochs at 3e-4 is recorded for a different 25.9 M-class run that scored 2.29, and a submit script with those settings has no saved output. This is reported as a finding, not covered.


### A72 — Training recipe, model size and decoding defaults
**Claim.** The Methodology states the configuration of the translation model (Table "Training configuration of the translation model"): 133 COCO-WholeBody keypoints (399 features per frame), at most 256 source frames and 64 target tokens, model width 256 with 8 heads and a 1024-unit feed-forward layer, four encoder and four decoder layers, dropout and embedding dropout 0.1, label smoothing 0.1, auxiliary CTC weight 0.3, learning rate 5x10^-4 with weight decay 1x10^-4, gradient clip 1.0, cosine schedule with 800 warmup steps, batch 64 (32 for the RGB-fusion models) and 40 epochs; decoding defaults of beam width 5 and length penalty 0.6; the adopted training configuration (13,760 stitched clips plus pose augmentation); stage seeds 42, 101, 202 and 303, 404; and the Appendix B table convention (800 official test clips, beam width 5, 95% intervals).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/recipe_and_settings/check.py
```

```text
GATE saved beam-5 hypotheses of the pose-augmented seed-42 run: 800 clips, BLEU-4 11.443 (recorded 11.443, read from the saved job output)
references match comm_run/manifests/test.tsv

Members: the 6 checkpoints named in the ensemble job script (mbr_wer_test2.sbatch)
3 pose-only members, 3 RGB-fusion members

TABLE: training configuration (stored in every one of the six members)
keypoints / input feature dimension: 133 COCO-WholeBody / 399 (input projection weight 256 x 399; feature dim = 133 x 3)  [c07-001 c06-039]
coordinate normalisation: frame  []
max source frames / target tokens: 256 / 64  [c07-002 c06-025]
d_model / heads / FFN: 256 / 8 / 1024 (d_model = model width 256)  [c07-003 c06-039]
encoder / decoder layers: 4 / 4 (weights hold 4 encoder and 4 decoder layers)  [c07-004 c06-037]
dropout / embedding dropout: 0.1 / 0.1  [c07-005]
label smoothing: 0.1  [c07-006]
auxiliary CTC weight (default of train_ctc.py = value passed by the job script): 0.3  [c07-007 c07-014]
learning rate: 5x10^-4 (0.0005)  [c07-008]
weight decay / gradient clip: 1x10^-4 (0.0001) / 1.0  [c07-009]
schedule / warmup steps: cosine / 800  [c07-010]
batch size (pose-only members) / epochs: 64 / 40  [c07-011]
batch size of the RGB-fusion members: 32  [c06-040]

Config() defaults vs the stored configuration of the members: 17 of 17 fields agree

Source sequences longer than the 256-frame limit are subsampled uniformly (np.linspace in slt/data.py): a 300-frame clip keeps 256 frames with gaps of 1 or 2 frames  [c06-025]
Targets are limited to 64 tokens (stored max_tgt_tokens)  [c06-025]

DECODING defaults
beam width / length penalty (Config): 5 / 0.6  [c06-011 c07-012]
beam width / length penalty (tools/eval_beam.py): 5 / 0.6  [c06-011 c07-012]
beam width / length penalty (saved beam-5 test job output): 5 / 0.6 (read from the saved job output beam5_test.json)  [c06-011 c07-012]

Adopted training configuration (job log joblog_pose_aug.out, read from the saved job output): training set 26199 clips, of which 13,760 stitched; real clips 12,439  [c07-042]
pose augmentation switched on in that run (pose_aug True): speed +-0.2, rotation +-10.0 deg, scale +-0.1 (read from the saved job output)

Stage 1 job scripts train seeds 42, 101, 202; stage 2 job scripts train fresh seeds 303, 404  [c21-019]
seeds stored in the six members: 101, 42, 42, 202, 101, 202  [c21-019]

Table convention: the saved test run holds 800 official Communication test clips decoded at beam width 5  [c17-001 c17-002]
paired_bootstrap.py default interval: 95% (percentiles 2.5 and 97.5)  [c17-003]
```

**Scope.** Recomputed: the saved beam-5 test hypotheses of one adopted-configuration run (the gate) and every table value, read from the training configuration stored in the six ensemble members (subset in `cfg_members.json`), the current `Config` defaults, the training and decoding scripts and one saved job log. Not recomputed: training, the PyTorch/CUDA versions (no saved output prints them; listed as not covered). Original job: 1215231 (pose augmentation); members trained 11-23 Aug 2026.


### A73 — Run-to-run noise bands

**Claim.** The dissertation states (Introduction, Discussion, Appendix B "Reproducibility" and the seed-spread figure) that a difference of about one BLEU-4 point lies within the measurement variation: single-run differences have a 95% range of about +-0.6 within one job submission and +-1.2 across job submissions, from standard deviations of 0.22 (six runs, one job) and 0.42 (pooled, adopted configuration across five jobs). The seed-spread figure (`fig:seed_spread`) uses the 800-clip official test set at beam width 5 and plots three-seed means and ranges; its second and third rows are one configuration in two job submissions.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/run_to_run_noise_bands/check.py
```

```text
GATE six replicates sd 0.216, recorded 0.216 (read from the saved job output null_band_output.txt)
Baseline configuration, one seed, six runs in one job submission: n=6, test BLEU-4 sd 0.22 [95% 0.13, 0.53]
Within-job standard deviation of six identical-seed runs: 0.22
Paired single-run 95% band within one job submission: +-0.6 BLEU-4 (1.96 x sd x sqrt 2)  [c02-010 c14-029 c14-073]
seed 42: five job submissions 10.091-11.443, range 1.353
seed 101: five job submissions 11.223-12.118, range 0.894
seed 202: five job submissions 10.486-11.483, range 0.997
GATE pooled within-seed sd 0.422, recorded 0.422 (read from the saved job output adopted_crossbatch_output.txt)
Pooled within-seed standard deviation of the adopted configuration across 5 job submissions: 0.42 (df 12)  [c02-010 c14-073]
Paired single-run 95% band across job submissions: +-1.2 BLEU-4  [c02-010 c14-029 c14-073]
Differences below about 1 BLEU-4 point lie within the measurement variation (bands +-0.6 within a job, +-1.2 across jobs)  [c02-010 c14-029 c14-073]
same seed, same configuration, different job submission (beam-5 test BLEU-4, read from the saved job output for 1.1:1 runs):
  seed 42: 11.52 vs 10.34, gap 1.18
  seed 101: 11.03 vs 10.93, gap 0.10
  seed 202: 10.87 vs 10.66, gap 0.21
Largest gap between two submissions of one configuration at one seed: 1.18
Six same-configuration runs (three seeds x two submissions): mean 10.891, standard deviation 0.39
Figure setup: 800 official Communication test clips, beam width 5 (saved decode of job 1215231)  [c20-033]
second row (control of the later job, wave1 w1A): 3 seeds 10.64, 11.30, 11.35; three-seed mean 11.09, range 10.64-11.35  [c20-035]
third row (first job, pa_aug): 3 seeds 11.44, 11.37, 11.19; three-seed mean 11.34, range 11.19-11.44  [c20-035]
Second and third rows are one configuration in 2 job submissions of 3 seeds each; per-seed gaps 0.81, 0.08, 0.15  [c20-036]
deterministic-mode repeats of one configuration, two submissions (wave 6 and wave 16 controls; read from the saved job outputs):
  seed 42: 11.711 vs 11.711, gap 0.000
  seed 101: 11.366 vs 10.745, gap 0.621
  seed 202: 10.839 vs 10.411, gap 0.428
Of 3 pairs, 1 bit-identical score and 2 differ, by 0.62 and 0.43 BLEU-4
```

**Scope.** Recomputed from saved per-run beam-5 test BLEU-4 scores: the six same-seed replicates (job 7952), the adopted configuration across five job submissions (jobs 1215231, 1216919 and three wave-3 jobs), 1.1:1 runs against the control of job 1215231, and deterministic-mode repeats (waves 6 and 16). Scores are read from saved job outputs, not recomputed from hypotheses. The configuration identity of the repeated runs was checked on the cluster and is not rerun here. Not covered: the determinism A/A (job 6952), whose weight comparison is on the cluster. The figure rows for the real-only and August pre-trained runs are not saved locally.


### A74 — Seen/unseen decomposition, split definition and per-signer gap

**Claim.** The official Communication split has 12,441 training, 800 validation and 800 test clips (14,041 in all); all 90 test videos occur in training, 230 of the 800 test references (28.75%, over a quarter) occur verbatim in training, and the two most frequent signers supply 92% of the test clips. Three single models score about 30 BLEU-4 on the 230 seen-target clips and about 5 on the 570 unseen-target clips (5.2 to 6.1 times), and 72.6% of the pre-trained model's outputs are verbatim training sentences (Results, Discussion, Methodology, Appendix). The appendix per-signer row states the signer 1 minus signer 2 gap as +2.2 to +4.1 (p 0.02 to 0.29) and the unseen-target gap as +1.1 to -0.3; this check finds small differences (see Scope).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/seen_unseen_decomposition/check.py
```

```text
Split sizes: train 12,441, validation 800, test 800; total 14,041  [c06-016 c21-016 c09-001 c09-007 c20-011]
Test references occurring verbatim in training: 230 of 800 (28.75%); unseen 570  [c14-010 c14-074 c21-016 c01-026]
Share of test references seen in training 0.29: over a quarter (> 0.25)  [c01-026]
Test videos 90; of these in training 90 of 90  [c07-060]
Test clips by signer: signer 1 389, signer 2 350; two most frequent signers 739 of 800 = 92.4%  [c05-053]
GATE S6 ensemble (beam 10, LP 0.3) official test BLEU-4 11.802 (recorded 11.802)

Setting: all 800 official test clips, beam width 5, three single models
DWPose+CTC  all   9.52 (n=800)  seen  29.72 (n=230)  unseen   4.88 (n=570)  ratio 6.09  [c11-059 c11-066 c11-067 c14-075 c09-024]
VLP-40      all  10.27 (n=800)  seen  29.91 (n=230)  unseen   5.31 (n=570)  ratio 5.63  [c11-059 c11-066 c11-067 c14-075 c09-024]
RGB-fusion  all  10.33 (n=800)  seen  30.22 (n=230)  unseen   5.81 (n=570)  ratio 5.20  [c11-059 c11-066 c11-067 c14-075 c09-024]
GATE recorded beam-5 test scores 9.52 / 10.27 / 10.33 for DWPose+CTC / VLP-40 / RGB-fusion
Seen-target range 29.7 to 30.2; unseen-target range 4.88 to 5.81; ratio 5.2 to 6.1  [c11-059 c14-075 c11-066]
Seen about 30 (mean 29.9), unseen about 5 (mean 5.3); 230 seen / 570 unseen clips; 3 models  [c11-066 c14-075]
Outputs that are verbatim training sentences, DWPose+CTC: 80.8%  [c10-038]
Outputs that are verbatim training sentences, VLP-40: 72.6%  [c10-038]
Outputs that are verbatim training sentences, RGB-fusion: 70.8%  [c10-038]
Outputs that are verbatim training sentences, S6: 84.8%  [c10-038]

Signer 1 minus signer 2, official test, beam 5 (within-video permutation, 2000 draws, seed 0)
DWPose+CTC  signer1-signer2 +4.08  perm p 0.029  unseen-target only +0.61
VLP-40      signer1-signer2 +3.91  perm p 0.039  unseen-target only +0.42
RGB-fusion  signer1-signer2 +2.15  perm p 0.297  unseen-target only -0.28
S6 ensemble (not a single model) signer1-signer2 +4.42, unseen-target only +1.11
Single-model gap range +2.147 to +4.085 (to one decimal: +2.1 to +4.1); permutation p between 0.03 and 0.30  [c18-023]
  (recorded p 0.02 to 0.29 came from an earlier 300-draw run (seed not kept); p agrees only within permutation noise)  [c18-023]
Unseen-target gap, single models: +0.61 to -0.28; with the S6 ensemble the upper end is +1.11  [c18-024]
```

**Scope.** Recomputed from saved per-clip decodes (hypotheses and references), the training-target digests and the spreadsheet; gate is the reported ensemble's 11.802 and the three single models' recorded 9.52, 10.27 and 10.33. The permutation test shuffles Signer_ID within each video (2000 draws, seed 0); the original 300-draw run is not recoverable, so p differs slightly. The signer 1 minus signer 2 gap of the RGB-fusion model is 2.147 (dissertation +2.2) and the +1.1 upper end of the unseen-target gap belongs to the S6 ensemble, not a single model. No training or inference is done.


### A75 — Selective prediction: answering only the confident clips

**Claim.** If the system answers only the clips it is most confident about (confidence = the model's own cross-entropy on its output, plus self-reliance), BLEU-4 of the answered clips rises. For the single DWPose+CTC model on the 800 official test clips it goes from 9.52 at full coverage to 17.42 at 50% coverage (16.13 with the cross-entropy term alone), on validation from 9.64 to 18.42, and on the 570 unseen-target clips from 4.88 to 7.92. Keeping the longest outputs instead gives only 10.48 at 50% and 7.22 at 25%; at 25% coverage 98% of the answered outputs are verbatim training sentences. With the six-member ensemble and thresholds fixed on validation, 51.9% of test clips are answered at 19.07, +7.30 above random abstention at the same coverage (Results, selective prediction; Appendix tables of the selective analysis).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/selective_prediction/check.py
```

```text
GATE full-set beam-5 BLEU-4: test 9.52 (recorded 9.52), validation 9.64 (recorded 9.64)

Setting: single model, beam 5 hypotheses (field hyp_beam of the saved decodes), n test 800, n validation 800, unseen-target test clips 570 (validation 549)  [c16-044]
Coverage = clips answered / clips, from 0% to 100%; here 100% = all 800 answered  [c06-010]
At 50% coverage the system answers 400 of 800 test clips (its more confident half, coverage 0.5)  [c01-029 c02-029 c18-036]
Coverages evaluated: 90%, 75%, 50% and 25% (plus 100%)  [c07-073]

Retained-set BLEU-4 by coverage (ranking: self-CE + self-reliance, z-sum)
                           100%      90%      75%      50%      25%  [c16-046]
Test                       9.52    10.37    12.05    17.42    30.82  [c16-046]
Validation                 9.64    10.37    12.44    18.42    30.51  [c16-047]
Test, self-CE alone        9.52    10.28    11.97    16.13    29.24
Val, self-CE alone         9.64    10.45    12.15    17.06    32.32

Test, all 800 clips, 100% -> 50%: 9.52 -> 17.42  [c01-030 c02-030 c13-001 c13-003 c13-008]
Validation 100% -> 50%: 9.64 -> 18.42  [c13-005]
Self-CE term alone at 50% coverage: test 16.13, validation 17.06; two-term score 17.42 and 18.42  [c13-002 c18-035]
Beam width 5 (beam-5 decodes; the full-set score matches the recorded beam-5 value above)  [c13-004]

Unseen-target test clips (570): all 4.88; the more confident half of the 800 clips, unseen ones only (238): 7.92  [c13-007 c16-048 c02-030]
Unseen-target validation: all 4.27, confident half 7.24

Control, keep the longest outputs: 50% coverage 10.48  [c13-006]; 25% coverage 7.22  [c18-037]
Baseline model, 25% coverage: 98.0% of the answered outputs are verbatim training sentences (all 800: 80.8%)  [c13-018]
Signer repeats 50% of utterances at 50% coverage, i.e. one in two  [c14-092]

S6 ensemble, beam-10 n-best, length penalty 0.3; thresholds from validation, applied unchanged to test
GATE ensemble test BLEU-4 11.802 (recorded 11.802)
all    cov 0.90: test  12.90 @ 89.4% (715 clips); selected - random at matched coverage +1.10 [+0.39, +1.94]
all    cov 0.75: test  14.61 @ 75.6% (605 clips); selected - random at matched coverage +2.80 [+1.53, +4.29]
all    cov 0.50: test  19.07 @ 51.9% (415 clips); selected - random at matched coverage +7.30 [+4.55, +10.17]  [c13-012 c18-038]
unseen cov 0.90: test   7.18 @ 88.8% (506 clips); selected - random at matched coverage +0.59 [+0.05, +1.43]
unseen cov 0.75: test   7.71 @ 76.7% (437 clips); selected - random at matched coverage +1.12 [+0.14, +2.33]
unseen cov 0.50: test   8.95 @ 55.1% (314 clips); selected - random at matched coverage +2.36 [+0.47, +4.42]  [c18-039]
Ensemble answers 51.9% of the 800 official test clips with the validation threshold for 50% coverage  [c13-012]
```

**Scope.** Recomputed from per-clip saved outputs: the single model's beam-5 hypotheses, references, self-CE and self-reliance (800 test and 800 validation clips, saved by `tools/self_reliance.py`), re-scored through `slt/metrics.py`, and the ensemble's saved n-best lists (jobs 1101821 / 1125067), with the 2,000-resample bootstrap of `tools/selective_ci.py` (seed 0). Gates: the full-set scores 9.52 / 9.64 and 11.802 and the ensemble's three recorded coverage points must reproduce or the check exits non-zero. Not done: decoding or computing self-CE and self-reliance (model inference, run once locally). The 570/238-clip unseen figure ranks all 800 clips and then keeps the unseen ones among the answered half.


### A76 — Semantic adequacy grading
**Claim.** Appendix B (automatic adequacy rating, supplementary table row) states that a small instruction-tuned model (Qwen2.5-3B-Instruct) graded every output of four systems on the official test set for adequacy on a 0–4 scale, that mean adequacy rises from 1.155 for the baseline model to 1.311 for the final ensemble, and that the final ensemble was graded higher than the RGB-fusion model on 148 clips and lower on 134.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/semantic_adequacy_grades/check.py
```
```text
GATE dwpose_ctc: recomputed mean 1.155 vs recorded 1.155 (read from the saved job log dwpose_ctc.log)
GATE vlp: recomputed mean 1.232 vs recorded 1.232 (read from the saved job log vlp.log)
GATE rgbfusion: recomputed mean 1.259 vs recorded 1.259 (read from the saved job log rgbfusion.log)
GATE s6: recomputed mean 1.311 vs recorded 1.311 (read from the saved job log s6.log)

Judge model (read from the saved job logs): Qwen2.5-3B-Instruct  [c20-008]
Scale: grades take values 0 to 4, i.e. 0-4  [c20-008]
Systems graded: 4, each on 800 official test clips  [c20-008]

Mean adequacy
  dwpose_ctc  1.155
  vlp         1.232
  rgbfusion   1.259
  s6          1.311
Baseline model (DWPose+CTC) mean adequacy 1.155  [c20-009]
Final ensemble (S6) mean adequacy 1.311  [c20-009]

Final ensemble vs RGB-fusion model, per clip: graded higher on 148 clips and lower on 134 (equal on 518)  [c18-022]
```
**Scope.** Recomputed from the saved per-clip grades (800 per system) of the re-grading job 10288; the gate compares each recomputed mean with the mean printed in that job's saved log. The judge model name is read from the logs. The grading itself (running the language model) is not repeated.


### A77 — Signer-disjoint split

**Claim.** Results (signer overlap and signer-disjoint paragraph), Discussion, Method and Appendix B state that all nine official test signers also appear in training, and that baseline models trained on 12,024 clips of two signers (the presenter and the costumed character; three seeds, beam width 5) score 1.29 BLEU-4 on the 1,087 clips of the 47 other signers (0.87 on the 869 with unseen references), against 1.83 when other clips of those signers are in training (difference +0.55, bootstrap +0.23 to +0.91, t-interval -0.43 to +1.53; pre-registered thresholds 0.5 and 1.0); two signers supply 75.6% of the held-out clips, one alone 59%; a constant sentence chosen on validation scores 0.44.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/signer_disjoint_split/check.py
```

```text
official split: 9 signers in test, 9 of them also in training (train has 49 signers, 'all nine test signers appear in training': True)  [c11-084 c14-097]

held-out clips (signers other than 1 and 2, with pose): 1087; held-out signers 47; SD training clips of two signers (signers 1 and 2, outside the validation videos): 12024  [c18-028 c21-017]
signers 10 and 8 supply 637 + 185 = 822 clips: 75.6% of the 1087 clips with pose, 75.6% of the 1088 in the spreadsheet  [c18-027]
signer 10 alone: 637 of 1087 clips = 58.6%, rounded 59% of the held-out clips  [c18-034]
gate: 18 saved beam-5 decodes (3 arms x 3 seeds, test and validation) match the rebuilt manifests row for row and reproduce their recorded BLEU-4

baseline model, three seeds [42, 101, 202], beam width 5, trained on 12,024 clips of signers 1 and 2  [c11-085 c18-028]
signers absent from training (SD), held-out clips n = 1,087 of 47 other signers [c11-085 c18-029 c21-017]
  per seed 1.42 1.32 1.11 | mean 1.29 BLEU-4  [c11-085 c18-029]
  of which 869 clips have a reference that no training text of either arm contains: mean 0.87 BLEU-4  [c18-029]
other clips of the same signers in training (EXP), same 1,087 clips: per seed 1.87 1.54 2.09 | mean 1.83  [c11-085 c18-030]
difference EXP - SD: per seed +0.45 +0.21 +0.98 | mean +0.55  [c11-086 c18-031]
  bootstrap over clips (1,000 draws, test sampling only): [+0.23, +0.91]  [c18-031]
  t-interval over the three seeds: [-0.43, +1.53]  [c18-031]
pre-registered reading (tools/w12g_readout.py, reading()): R3: unresolved (the estimate and its intervals are reported, nothing is claimed)
  its thresholds, found by probing reading(): R1 needs a mean difference of at least 1.0 (all seeds positive, interval above 0); R2 needs a mean difference below 0.5 (interval inside -1.5 .. +1.5)  [c18-032]
input-blind constant output chosen on validation: 0.44 BLEU-4 on the held-out clips (validation 1.59)  [c18-033]
```

**Scope.** The split is rebuilt from the spreadsheet's Signer_ID and Split columns. The 18 saved beam-5 decodes (training array job 138387, summary job 138388) are a programmatic subset of the saved outputs (hypotheses, SHA-1 of the references, recorded BLEU-4; `prepare_data.py`, run once in the private tree); their reference digests must equal the references of the rebuilt manifests row for row, and each recorded BLEU-4 is reproduced through the project metrics (gate), before anything is printed. The bootstrap (1,000 draws, seed 0, clips resampled) is recomputed from per-clip sufficient statistics with the original draws; the reading thresholds are found by probing the read-out tool's `reading()` function. No training or decoding is done.


### A78 — Speech round trip with Whisper small.en
**Claim.** The appendix (supplementary speech round-trip table) and the discussion state that, when the voiced translator output is read back by a speech recogniser (Whisper small.en), the round-trip word error rate is 2.8% for the 40-epoch pre-trained model and 3.0% for the final ensemble (strict scoring), against 3.8% for the human reference sentences and 21.3% for references with scrambled word order; normalised scoring gives 1.1%, 1.6%, 2.2% and 19.4%. The paired difference to human text is -1.5 points (-2.8 to -0.1), and the round trip is said to add at most half a point in every variant (recomputed: true for the 40-epoch output, +0.8 for the ensemble in one variant).

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/speech_round_trip/check.py
```
```text
Transcriber: whisper.cpp 1.9.4, model ggml-small.en.bin (sha256 c6138d6d58ec), backend metal  [c07-072]
GATE human references strict RT-WER 0.037910 (recorded 0.037910)

Round-trip WER, 800 official test clips, 95% bootstrap interval (2000 resamples, seed 0)
Output of the 40-epoch pre-trained model (voiced)    strict 2.8% [1.7, 4.1]  normalised 1.1% [0.7, 1.5]  [c14-095 c20-012 c20-015 c20-016]
Final ensemble output                                strict 3.0% [1.8, 4.4]  normalised 1.6% [1.0, 2.6]  [c14-095 c20-012 c20-015 c20-017]
Human reference sentences                            strict 3.8% [2.8, 5.0]  normalised 2.2% [1.6, 3.1]  [c14-095 c20-012 c20-013 c20-015 c20-018]
References with scrambled word order                 strict 21.3% [19.6, 23.0]  normalised 19.4% [17.9, 21.0]  [c20-013 c20-015 c20-019]
scrambling the references raised strict RT-WER from 3.8% to 21.3%  [c20-013]
ensemble strict upper bound unrounded 4.446% (rounds to 4.4, not 4.5)  [c20-017]

Paired difference, voiced output minus human references (B standardised to the output's length mix), percentage points
strict     A   all 800          -1.5 (-2.8 to -0.1)  [c20-014 c18-054]
strict     A   757 scramblable  -0.9 (-2.1 to +0.5)  [c20-014]
strict     A6  all 800          -1.4 (-2.8 to +0.0)  [c20-014]
strict     A6  757 scramblable  -0.6 (-1.9 to +0.8)  [c20-014]
normalised A   all 800          -1.5 (-2.5 to -0.6)  [c20-014]
normalised A   757 scramblable  -1.1 (-2.0 to -0.3)  [c20-014]
normalised A6  all 800          -1.1 (-2.3 to +0.1)  [c20-014]
normalised A6  757 scramblable  -0.6 (-1.8 to +0.5)  [c20-014]
Voiced 40-epoch output, strict, all 800: -1.5 percentage points (-2.8 to -0.1)  [c18-054]
largest upper bound, 40-epoch output (A), all variants: +0.5 points, i.e. at most half a point  [c20-014]
largest upper bound, final ensemble (A6), all variants: +0.8 points (strict, 757 clips)  [c20-014]
DISAGREES for the ensemble: the appendix says 'every variant ... at most half a percentage point'; it holds for the 40-epoch output only
setting: 800 official test clips, 95% bootstrap intervals  [c20-015]
```
**Scope.** Recomputed from the saved synthesis-and-recognition outputs (the four spoken arms and their Whisper transcripts; transcripts of the spoken references are stored as edits against the reference text, rebuilt from `comm_run/manifests/test.tsv`), scored with the original functions of `tools/speech_roundtrip.py` (2,000 bootstrap resamples, seed 0). The gate reproduces the recorded strict round-trip WER of the human references. Not redone: speech synthesis and Whisper transcription (macOS voice and whisper.cpp on Metal, original job on the Mac), and the heard-BLEU analysis.


### A79 — Person tracking and the heavy-switch clips

**Claim.** The tracked re-extraction of all 14,041 Communication clips shows an identity switch in 23.8% of training clips (5.7% of frames), 77% of evaluation clips have none, and a blind two-rater audit finds the tracker on the signer in 87 of 96 decisive frames (91%). Training on tracked keypoints at five seeds changes test BLEU-4 by -0.06 (-0.66 to +0.55) and tracker v2 (wave 9) is stopped at stage 1; but on the 64 test clips with more than 30% switched frames, tracker v2 raises BLEU-4 by 3.52 more than on clips without switches (+1.44 to +5.61, five new seeds), with +0.80 (-0.54 to +2.13) on the whole test set (Results, person tracking; Discussion; appendix tables on tracked keypoints and tracker v2).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/tracker_v2_heavy_switch/check.py
```

```text
Tracked re-extraction covers 14,041 Communication clips (read from the saved job output tracking_prevalence_agg.json) [c12-062]
Training clips with at least one switched frame 23.8% (2955/12439); switched frames 5.7% (52996/935938) [c12-063]
Evaluation clips (validation+test) with no switched frame 77% (1227/1600) [c12-069]
  (the switched-clip numerators are recounted from the per-clip file and match the job totals)

Blind audit: 160 frames, each labelled by 2 raters [c12-064]
Tracker on the signer and the original extraction not: 87 of 96 decisive frames (91%) [c12-065]

Wave 3 (job 6592 + 9444): index-0 pose (w3C) vs person-tracked pose (w3T), seeds 42/101/202/303/404
GATE: all 20 beam-5 BLEU-4 values re-score to the recorded job values
Test (800 clips) tracked - original, per seed +0.70 -0.50 -0.27 -0.35 +0.13: mean -0.06 (-0.057) 95% t [-0.66, +0.54] (-0.658, +0.545), 2/5 seeds positive [c12-066]
  (the dissertation states the upper bound +0.55; the five-seed t-interval here is +0.545 rounded down to +0.54: a rounding difference, not forced)
Stage-2 seeds (303, 404): tracked scores lower in 1 of 2 (informational for c12-068)
Heavy bin = clips on which the tracker replaced more than 30% of the frames (frozen rule: share > 0.3): 64 test clips [c12-071]
Heavy-switch clips, validation, rounded to one decimal: +4.4 on 55 clips [c17-015]
Heavy-switch clips, test, rounded to one decimal: +0.6 on 64 clips [c17-016]
Heavy-switch clips, validation (n=55) tracked - original: mean +4.36 (+4.359) 95% t [+2.27, +6.45] (+2.273, +6.445), 5/5 seeds positive [c17-015]
Heavy-switch clips, test (n=64): mean +0.59 (+0.587) 95% t [-2.52, +3.69] (-2.521, +3.694), 3/5 seeds positive; negative in 2 seeds [c17-016]

Wave 9 (job 43506): tracker v2 - original, seeds 42/101/202
Validation mean -0.02 (-0.023): stage 1 stops (mean <= 0) [c17-017]
Test mean -0.04 (-0.038) 95% t [-1.79, +1.71] (-1.785, +1.710), 2/3 seeds positive [c17-018]

Wave 12i (jobs 138391/138392): tracker v2 vs original on the heavy-switch bin, five new seeds 303/404/505/606/707
Seeds: 5 new seeds [c12-070]
Heavy bin (test n=64) DiD tracker v2 - original, minus clips without switches: mean +3.52 (+3.523) 95% t [+1.44, +5.61] (+1.435, +5.611), 5/5 seeds positive [c12-071 c12-072 c17-019]
Heavy bin validation DiD: mean +4.74 (+4.738) [c12-072 c17-020]
Clip-sampling interval (test, 1000 resamples within bins, seed 0) [-0.13, +8.86] [c17-020]
Whole test set, tracker v2 - original: mean +0.80 (+0.798) 95% t [-0.54, +2.13] [c12-073 c17-021]
Original-trained model decoded on tracked keypoints, heavy-bin DiD vs decoding on its own pose: mean +4.49 (+4.490) [c17-022]

Hand speed on 591 training clips: first-detection mean 0.391, tracked mean 0.143 per frame
The citation-form comparison value is a recorded mean of 0.044 per frame (MM-WLAuslan clips are not held locally), recorded 9 Sep.
Ratio to the recorded citation value, per frame 3.2, per second 2.7  (informational, untagged)

Pose-jump census over locally stored clips with shoulders tracked on at least 90% of frames: 2,128 clips [c17-012]
Clips with at least one jump 25.0% (531); frame transitions with a jump 2.39% (3116/130353)
  (the dissertation states 25.2% and 2.31%; this definition of a jump, from pose/tracker_v2/research/alphapose_identity_check.py, gives the values above: not forced to agree)
```

**Scope.** Beam-5 hypotheses of every run are re-scored through `slt/metrics.py` and must equal the recorded BLEU-4 before any difference is printed; references come from the official manifests. Means, t-intervals and the 1,000-resample clip bootstrap (seed 0) are recomputed. Switch counts (job 6591) are read from the saved job output. Hand speed is recomputed on tracked keypoints only (the citation-form value is not held), the 552-event cause split has no saved output, and the pose-jump rate under this repository's jump definition differs slightly from the dissertation (25.0% and 2.39% against 25.2% and 2.31%). No training or decoding is repeated. Jobs: 6590/6591/6592/9444 (wave 3), 43506 (wave 9), 138391/138392 (wave 12i).


### A80 — Video-disjoint split and VD-CONFIRM
**Claim.** The official split shares videos between train and test, so a video-disjoint split (11,341 / 1,444 / 1,254 clips from 73 / 9 / 8 videos) was built (Methodology, Results "Video-disjoint"). On it the adopted recipe raises five-seed mean beam-5 test BLEU-4 from 6.14 to 7.08 (+0.94), 29.3% of test references occur verbatim in training (886 do not), and a six-model ensemble (7.12) does not exceed the best single model (7.29; -0.16, interval -0.86 to +0.57). The best single model is the adopted-recipe model with the highest video-disjoint validation score.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/video_disjoint_split/check.py
```
```text
video-disjoint split: train 11,341 clips from 73 videos, val 1,444 from 9, test 1,254 from 8; no video shared  [c07-061 c11-070]
GATE recorded job read-out (seed_decision_5seeds.txt) contains 'mean +0.939 | 95% t [+0.155, +1.722]': True

VD-CONFIRM: 5 seeds, read from the saved job outputs summary_s*.json (original job 6596 + 7965)  [c18-018 c07-062 c11-073]
beam-5 test BLEU-4, mean over 5 seeds: plain baseline 6.14 (6.144), adopted recipe 7.08 (7.082)  [c11-073]
difference +0.94 (+0.939); positive in 5 of 5 seeds  [c11-073]
seed-level 95% t-interval +0.16 to +1.72  [c11-073]
decode references equal the split's test manifest in order: 1254/1254

references occurring verbatim in the video-disjoint training text: 368 of 1254 test clips = 29.3%  [c11-075]
test clips whose reference never occurs in that training text: 886  [c11-076]
GATE recorded 7.123 (ensemble) / 7.287 (best single): 7.123 / 7.287
ensemble members (read from the saved read-out line v6_members_line.txt): 6 (six)  [c14-035 c11-078]
ensemble test BLEU-4 7.12 (7.123); best single model 7.29 (7.287)  [c14-035 c11-078]
difference -0.16 (-0.164); 95% paired-bootstrap interval -0.855 to +0.570, i.e. -0.86 to +0.57 (2,000 resamples, seed 0)  [c11-078]
GATE recorded vdB_s42 at LP 1.0: validation 6.600, test 7.407; recomputed 6.600 / 7.407
GATE recorded vdB_s101 at LP 0.9: validation 6.718, test 7.287; recomputed 6.718 / 7.287
GATE recorded vdB_s202 at LP 0.7: validation 6.373, test 7.149; recomputed 6.373 / 7.149
the best single model is the one with the highest video-disjoint validation BLEU-4 among the three adopted-recipe models: vdB_s101 (6.72 validation, 7.29 test)  [c14-035 c11-078]
scope of 'best single model': the highest-validation model among the three adopted-recipe (vdB) singles that make up the ensemble comparison; the plain-baseline (vdA) singles are not considered  [c14-035 c11-078]
(not the best on test: vdB_s42 scores 7.41 on test; the single model is selected on validation, so its test score is not a test-selected maximum)  [c14-035 c11-078]
```
**Scope.** Recomputed: split sizes and video counts from the clip identifiers, reference overlap against the training text, BLEU-4 and the 2,000-resample paired bootstrap (seed 0) from the saved test decodes of the ensemble and the best single model. Read from saved job outputs (not recomputed): the per-seed beam-5 summaries (jobs 6596 and 7965). The best single model is shown to be validation-selected: three adopted-recipe singles are re-scored on validation and test from their saved decodes, gated on the recorded read-out values (the test-best of the three differs). Not done: training; that the adopted-recipe arm adds 13,760 duration-modelled stitched clips (its own joint manifest is on the cluster, job 6596; not claimed by this check); the 'best single model' comparison covers the three adopted-recipe singles only (the plain-baseline singles are not considered); and the per-seed 886-clip subset scores, content precision/recall and WER, whose per-seed decodes are on the cluster. Gates: the recorded seed-level read-out and the recorded 7.123 / 7.287.


### A81 — Masked keypoint pre-training (wave 28): checks that ran, no scores quoted

**Claim.** The appendix table of supplementary results (and the departures table) state that nine masked-keypoint pre-training runs (three arms, three seeds) all met their pre-registered training check, that the decision criterion is validation, that the pre-training used 16,800 updates on 44,010 unlabelled dictionary clips (with the Auslan-Daily training clips added in one arm), and that one control seed failed the pre-registered reproduction check, so no comparison is quoted.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/masked_pose_pretraining/check.py
```
```text
read from the saved job output summary_239450.out (job 239450)
pre-training runs reported: 9; TRAINED under the pre-registered check: 9  [c17-110]
nine pre-training runs, all nine met the check: True  [c17-110]
final held-out loss / first evaluation: 0.179 to 0.449 (line 0.5)  [c17-110]
arm MD: pre-training seeds [42, 101, 202] = 3 seeds  [c17-109]
arm MDA: pre-training seeds [42, 101, 202] = 3 seeds  [c17-109]
arm MDR: pre-training seeds [42, 101, 202] = 3 seeds  [c17-109]
latest update at which a held-out loss was evaluated: 16800 (no evaluation beyond it)
(not checked: the 44,010 dictionary-clip count and the pre-training pool sizes are not printed by any saved job output; see claims.json)
arm S: finished recipe runs for seeds ['42', '101']  [c03-021]
arm MD: finished recipe runs for seeds ['42', '101', '202']  [c03-021]
arm MDA: finished recipe runs for seeds ['42', '101', '202']  [c03-021]
arm MDR: finished recipe runs for seeds ['42', '101', '202']  [c03-021]
control S finished for seeds [42, 101] of the three needed: 2 of 3  [c03-021]
reproduction check, control seed 202: epoch-1 ctc loss 24.340 here vs 24.341 recorded for wave 5 (third decimal differs)  [c03-021 c17-110]
number of control seeds failing the reproduction check: 1  [c03-021]
every contrast with the control is labelled NO VERDICT in the summary: 36 lines; contrasts without it carry 'HARNESS NOT REPRODUCED'  [c03-021]
decision rule of the summary reads the 3-seed mean beam-5 VALIDATION difference (2 contrasts: STOPPED, CONTINUES); the criterion is validation, not test  [c17-109]
```

**Scope.** Parsed from two saved job outputs (summary job 239450 and the training log of array 239449, task 2); no number is a score and none is quoted. Nothing is retrained. Checked: the nine pre-training runs and their check, the three seeds per pre-training arm, validation as the decision rule, the failed reproduction check of control seed 202. Not checked (no saved output prints them; listed in `claims.json`): the 44,010 dictionary-clip count, the 12,439 clips added to the pre-training pool (the 12,439 in the training log belongs to the fine-tuning manifest, not to the pre-training data), and the total of 16,800 updates (only the last evaluation point, update 16800, is seen).


### A82 — Wave 4: stitched data without boundary trim, and two realisations of half the sentences
**Claim.** Appendix B (negative results) states that stitched data built without the 15% boundary trim scored -0.44 (-0.90 to +0.01) on the beam-5 test set against the adopted corpus, lower in all three seeds, and that two stitched realisations of each of half as many sentences (five seeds) scored -0.29 (-1.15 to +0.57). The Methodology and Appendix B also state that the citation clips are trimmed by 15% at each end.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/stitch_trim_and_realisations/check.py
```
```text
seeds with summaries: [42, 101, 202, 303, 404] (read from the saved job output summary_*.json, job 7949)
GATE T-A beam-5 validation mean +0.256 (recorded +0.256 in the pre-registration outcome)

T (no trim) - A, beam-5 test, seeds [42, 101, 202]: per seed -0.54 -0.56 -0.23
T - A test mean -0.44 (95% t -0.90 to +0.01; unrounded -0.895 to +0.011) [c17-051]
T - A test lower in 3 of 3 seeds (all three seeds lower: True) [c17-051]
number of seeds 3 [c17-050]

Q (two realisations, half the sentences) - A, beam-5 test, seeds [42, 101, 202, 303, 404]: per seed +0.46 +0.10 -0.31 -0.32 -1.39
Q - A test mean -0.29 (95% t -1.15 to +0.57; unrounded -1.152 to +0.568) [c17-059]
number of seeds 5 [c17-060]

builder default --trim_frac 15% of each isolated clip's frames at each end (arm A uses the default) [c07-035 c15-016 c17-050]
arm T passes --trim_frac 0.0: True [c17-050]
arm Q passes --realizations 2 (A: 1) with the first 6,880 of A's sentences: True [c17-060]
```
**Scope.** Parses the five per-seed summary files of the original training job (7949, 3 seeds for T, 5 for Q), computes per-seed differences, means and two-sided 95% t intervals (the convention of `tools/seed_decision.py`), and reads the trim setting from the corpus builder and the job script. The gate is the validation mean recorded when the job was read out. No training is repeated. Not covered: the motion-aware resampling run (job 1197043) and the pose-reliance probe (job 1174944), whose outputs are on the cluster only.


### A83 — WER null floor, brevity trade and error decomposition

**Claim.** The Results chapter, Methodology and appendix state that WER is uninformative on this corpus: an empty output scores 1.000, a constant sentence 0.950 against the RGB-fusion model's 0.950 and the final ensemble's 0.907, and cutting the RGB-fusion output to three words lowers WER to 0.900 (0.8999) while BLEU-4 falls from 10.33 to 4.87. The appendix adds that substitutions are 56 to 58% of word edits (0.53 of reference length), that output is 0.79 to 0.86 of reference length, that mean output length correlates +0.990 with WER over six sets of outputs, and that a per-sentence oracle reaches 0.807. The Methodology defines BLEU-4 (sacreBLEU, 13a, detokenised) and ROUGE-L (beta 1.2) and the worked BLEU/WER example of clip 104.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/wer_bleu_null_models/check.py
```

```text
references match comm_run/manifests/test.tsv: 800/800
GATE RGB-fusion beam 5: BLEU-4 10.335 WER 0.9497 (recorded 10.335 / 0.9497)
GATE six-member ensemble: BLEU-4 11.802 WER 0.9069 (recorded 11.802 / 0.9069)

Test set: 800 official Communication clips (references only, no training data used)  [c12-050 c12-053]
BLEU-4 computed with sacreBLEU|4-gram|tokenize=13a|detok=True|smooth=exp  [c06-002 c06-007]
detokenised text, sacreBLEU 13a tokeniser, scale 0 to 100: scores below lie between 0 and 100 for 3 single models x 2 decodings, 4-gram  [c06-001 c06-007 c10-022]

NULL MODELS (no video used), WER and BLEU-4 on the official test set
3 null models scored for WER: 1 empty output and 2 constant sentences  [c07-067]
empty output                   WER 1.0000  BLEU-4 0.00
constant A (sha1 0edb4308)     WER 0.9988  BLEU-4 0.00
  constant A (sha1 0edb4308): WER 0.9988 BLEU-4 0.00  [c12-045]
constant B (sha1 d49b14b4)     WER 0.9504  BLEU-4 0.38
  constant B (sha1 d49b14b4): WER 0.9504 BLEU-4 0.38  [c12-046 c12-053 c12-054 c02-016]
empty output WER 1.000; lowest null-model WER (floor) 0.950; RGB-fusion 0.950; ensemble 0.907  [c14-079 c12-051]
constant B WER 0.9504, RGB-fusion WER 0.9497, difference -0.0007 -> 0.0007  [c12-053]
constant B BLEU-4 0.38, RGB-fusion BLEU-4 10.33  [c12-054]
constant WER 0.950, RGB-fusion WER 0.950, final ensemble 0.907  [c12-051 c02-016]

BREVITY: RGB-fusion output cut to its first three words
truncated to three words: WER 0.8999 (0.900) BLEU-4 4.87  [c07-068 c12-051]
RGB-fusion WER 0.9497 -> 0.8999, below the ensemble's 0.9069; BLEU-4 10.33 -> 4.87  [c12-055 c21-003]
one model's WER 0.9497 -> 0.8999, BLEU-4 10.33 -> 4.87, truncated to three words  [c21-003]

WORKED EXAMPLE: clip 104 of the 40-epoch pre-trained pose-only model (beam width 5)
model 40-epoch pre-trained, beam 5 (corpus BLEU-4 10.27, the recorded 10.27)  [c04-005]
output tokens 5, matching reference tokens 4 (full stop included)  [c04-004]
matching 4-token runs 0; sentence BLEU-4 0.00  [c04-004]
token WER 0.43 (tokenised, 7 reference tokens)  [c04-004]

ROUGE-L: whitespace tokens, punctuation kept, beta = 1.2 (read from metric_suite.py)  [c06-008]

ERROR DECOMPOSITION and length, six sets of outputs (3 single models x greedy / beam 5)
system                           WER  S/edits    S/N len ratio mean len
DWPose+CTC greedy             1.0678    57.7%  0.616     0.981     5.04
  DWPose+CTC greedy: insertions 889 exceed matched reference words 610, WER 1.0678 above 1.0  [c06-006]
DWPose+CTC beam 5             0.9810    56.7%  0.556     0.824     4.24
40-ep pre-trained greedy      1.0783    57.9%  0.625     1.016     5.22
  40-ep pre-trained greedy: insertions 965 exceed matched reference words 643, WER 1.0783 above 1.0  [c06-006]
40-ep pre-trained beam 5      0.9837    58.1%  0.571     0.861     4.43
RGB-fusion greedy             1.0296    58.2%  0.599     0.936     4.81
  RGB-fusion greedy: insertions 753 exceed matched reference words 631, WER 1.0296 above 1.0  [c06-006]
RGB-fusion beam 5             0.9497    56.0%  0.532     0.792     4.07
RGB-fusion beam 5: S 2188 D 1288 I 432 of N 4115 reference words; substitutions are 0.53 of reference length  [c18-013]
substitutions as % of word edits, six sets of outputs: 58, 57, 58, 58, 58, 56  (range 56-58 %)  [c18-013]
output / reference length ratio, three beam-5 systems: 0.82, 0.86, 0.79  (range 0.79-0.86)  [c18-015]
correlation of mean output length with WER across the six sets of outputs: +0.990  [c18-088]
per-sentence oracle WER over the six sets of outputs: 0.807  [c18-014]

sentence bootstrap (2000 resamples, seed 0) of the ensemble's BLEU-4 11.80: 95% interval [9.87, 13.89], half-width 2.0 points, wider than a difference of about one BLEU-4 point (one point lies within the variation)  [c14-009]
```

**Scope.** Recomputed from the saved per-clip test decodes (greedy and beam 5 of three single models; ensemble beam 10, length penalty 0.3) through `slt/metrics.py` and jiwer; gates reproduce the recorded 10.335 / 0.9497 and 11.802 / 0.9069. The two constants are named by digest. The identification simulation (recall 0.15 to 0.66; WER 0.8685, BLEU-4 12.45) has no saved output and is not covered. The ROUGE beta is read from `metric_suite.py`. No training or inference is done.


## Level B: needs the datasets and one GPU

These checks require Auslan-Daily and, where stated, MM-WLAuslan. Checkpoints are not distributed. The costs given below are taken from the recorded runs.

| ID | Claim | What is needed | Script or job script in this repository |
|---|---|---|---|
| B1 | A single pose-only model trained with the dissertation's recipe reaches the ladder's first rung. The expected beam-5 test figure lies in the 9.5–10.1 band; this is not an exact reproduction. Without the determinism flag, which was off for every reported run, `--seed 42` does not reproduce a run, and the measured cross-batch band is ±1.17 (A25). `--coord_norm frame` must be passed explicitly. | Auslan-Daily pose clips and manifests; one GPU; about 1 V100-day | `jobs/slurm/dwpose_ctc.sbatch` |
| B2 | A saved checkpoint decodes to its recorded beam-5 score. Report beam-5 validation and beam-5 test for the checkpoint selected on greedy validation. | A checkpoint and the test manifest; minutes on a GPU, about 10 min on a laptop CPU | `tools/eval_beam.py` |
| B3 | The adopted recipe adds duration-modelled synthetic data at 1.1:1 and pose augmentation: `--pose_aug True --aug_speed 0.2 --aug_rotate_deg 10.0 --aug_scale 0.1`. It uses a joint manifest of 12,439 real + 13,760 synthetic = 26,199 rows. Augmentation is applied in the training loader only. | Datasets and synthetic corpus; one GPU; about 1 V100-day per arm (the replication in A26 is six arms) | `jobs/slurm/pose_aug.sbatch` |
| B4 | The ensemble decode of A1 can be regenerated from the six checkpoints at beam 10 with length penalties 0.2, 0.3 and 0.6. `--rgb_feat_dir` is mandatory: without it, the three fusion members previously degraded silently to pose-only, at a cost of about 0.4 BLEU, and the tool now raises an error instead. | Six checkpoints; RGB features; test manifest; about 4 CPU-hours (269 ms/clip × 800 clips × 3 length penalties) or minutes on a GPU | `tools/ensemble_mbr_wer.py` |
| B5 | Six members at beam 10 cost **269.0 ms** median per clip (p95 612.7 ms), and three members recover most of the gain at half the cost. The tool gates on reproducing the recorded validation score (12.470) before timing. Timings are hardware-specific. | Six checkpoints; validation clips; about 1 CPU-hour on an Apple M3 Max (12 torch threads, batch 1, 800 validation clips) | `tools/ensemble_latency.py` |
| B6 | In the sign-to-speech chain the translator is not the bottleneck. Median per-clip latency is 1968.5 ms for pose estimation, 91.8 ms for translation and 660.8 ms for synthesis. The whole-chain median is **2746.7 ms** (p95 5271.0 ms), of which pose estimation is 72 %. The recorded per-clip timings are re-summarised from the saved records in A46. | A checkpoint, pose clips and the speech-synthesis stage; about 1 CPU-hour | `tools/speech_pipeline_bench.py`, `tools/pose_throughput_bench.py` |
| B7 | DWPose keypoints identify signs at **42.8 %** top-1 (40.0 % before the 2026-09-26 id-parsing fix) on a 45-way task with no learning. Continuous signing destroys 58 % of this (hands 42.8 % → 18.1 %), and the face contributes almost nothing (13.6 % → 3.3 %). Folds are clip-disjoint; chance is 2.22 %. | Extracted pose for both corpora and the sign-spotting annotation table; CPU-minutes | `tools/continuous_id_probe.py` |
| B8 | On the same spans, DWPose reaches **18.1 %** hand top-1 against **13.1 %** (top-5 34.7 % against 32.7 %) for AlphaPose taken from the annotated signer entry, paired McNemar counts 35 against 17, **p = 0.0175**. The earlier figure of 4.2 % was measured on the first-listed person's keypoints and is superseded. This is one calibration point, and its effect is smaller than first reported. | Both pose extractions (DWPose and the AlphaPose HALPE-136 pose shipped with the corpus); CPU-minutes | `tools/paired_probe_mcnemar.py`, `tools/halpe_to_cwb133.py` |
| B9 | Perception is bounded out on four measurements: hand size, motion blur, temporal resolution, and the absence of correlation between per-span hand conditions and identification. No descriptor exceeds its family-wise shuffled null band (largest \|r\| = 0.103 against a p95 null of 0.099–0.105). A joint logistic model reaches AUC 0.518 against a shuffled control of 0.519. | Pose clips and, for the sharpness check, the signer videos; CPU-hours | `tools/hand_visibility_cv.py`, `tools/hand_crop_sharpness.py`, `tools/translation_predictors.py` |
| B10 | The confidence gate does not detect whether anyone is signing. It voices 53.9 % of non-signers against 53.5 % of signers (ratio 1.01, 95 % CI [0.92, 1.10]) and separates them at AUROC 0.51 [0.48, 0.54], which is chance. The same numbers, the ensemble repeat and the grounding score are re-derived from saved decodes in A45. | Decodes of non-signer crops and a pose-extraction pass over them; GPU-minutes | `tools/nonsigner_gate.py` |

## Level C: needs the cluster

The job scripts in `jobs/slurm/` encode site-specific absolute paths and accounts, and their paths need changing for another site. The budgets below assume V100 (16 GB) nodes.

| ID | Claim | What is needed | Script or job script in this repository |
|---|---|---|---|
| C1 | The six reported members exist and are as the ladder states. The members were trained by scripts with no input assertions and no determinism option. `jobs/slurm/ensemble_sweep.sbatch`, which selected the member set, has no `set -e` and decodes candidate member sets on the test split in the same job as its validation sweep. The numbers re-verify under re-execution (A1–A3), but the process behind them is weaker than that behind the later waves. | Datasets and extracted features; about 6 V100-days plus extraction | `jobs/slurm/dwpose_ctc.sbatch`, `jobs/slurm/dwpose_vlp.sbatch`, `jobs/slurm/dwpose_rgb_fusion.sbatch`, `jobs/slurm/gpu_batch_10h.sbatch`, `jobs/slurm/mbr_wer_val_ext.sbatch`, `jobs/slurm/mbr_wer_test2.sbatch`, `jobs/slurm/ensemble_sweep.sbatch` |
| C2 | Estimator replication: `Edw` (DWPose) versus `Eap` (AlphaPose) at a matched keypoint subset (`cwb_body_hands`) and normalisation, three fresh seeds, `--deterministic True`. The AlphaPose clips must come from the annotated signer entry: `jobs/slurm/apsig_extract.sbatch` builds them and `jobs/slurm/w5_estrep_signer.sbatch` is the comparison that underlies A32. `jobs/slurm/w5_estrep.sbatch` is the first run, whose AlphaPose cells read the first-listed person. | Datasets and the AlphaPose HALPE-136 pose; about 4 V100-hours (job 196407: three tasks, each training both arms on one V100 in about 1 h 15 min) | `jobs/slurm/apsig_extract.sbatch`, `jobs/slurm/w5_estrep_signer.sbatch`, `jobs/slurm/w5_estrep.sbatch` |
| C3 | The measured noise bands: six runs of one identical configuration, seed 42, determinism off. The analysis is level A (A25). | Datasets; about 6 V100-days | `jobs/slurm/w5_null.sbatch` |
| C4 | Waves 1, 2, 4 and 5, which are the pre-registered arms behind A10 and A12. Each wave has a pre-registration written before any arm ran. | Datasets; 3–5 seeds × 2–6 arms per wave at about 1 V100-day per arm; wave 5 alone budgets about 100 V100-hours for Stage 1 | `jobs/slurm/wave1_r2.sbatch`, `jobs/slurm/wave2_prep.sbatch`, `jobs/slurm/wave2_r2.sbatch`, `jobs/slurm/w4_prep.sbatch`, `jobs/slurm/w4_train.sbatch`, `jobs/slurm/w4_news.sbatch`, `jobs/slurm/w5_gen.sbatch`, `jobs/slurm/w5_prep.sbatch`, `jobs/slurm/w5_train.sbatch`, `jobs/slurm/w5_summary.sbatch`, `jobs/slurm/_w4lib.sh`, `jobs/slurm/_w5lib.sh` |
| C5 | Wave 3: the video-disjoint confirmation and the tracking experiment. The wave-3 training scripts and their shared library are not included in this repository, so the runs cannot be re-launched from it. The per-seed summaries are included (`results/video_disjoint_recipe_confirmation/`, `results/person_tracking_stage2/`), so the decisions in A27 and A28 can be checked. The tracked re-extraction behind the 23.8 % prevalence figure (a 16-shard array over all 14,041 clips) is also not included. | Datasets; cluster | `jobs/slurm/video_disjoint.sbatch` (builds the held-out-video split), `jobs/slurm/t2_vdens.sbatch` |
| C6 | Pose and RGB feature extraction: signer video retrieval; DWPose to per-clip `.npz` (16-shard array over 14,041 clips); frozen DINOv2 to 1152-D per frame (32-shard array). This runs once. | Signer videos; hours of GPU array time | `jobs/slurm/data_prep.sbatch`, `jobs/slurm/extract_dwpose.sbatch`, `jobs/slurm/rgb_extract.sbatch` |
| C7 | A/A determinism: two runs of an identical configuration with `--deterministic True` give **0 of 132** differing tensors, and the flag is not slower (645 s vs 690 s at equal node co-tenancy). The essential setting is `warn_only=False`. With `warn_only=True` the first attempt gave 131 of 132 differing tensors, because memory-efficient attention backward only warned and kept split-K accumulation. | Datasets; about 2 × 10 minutes of GPU | wave-4 job scripts (`jobs/slurm/w4_*.sbatch`) |
| C8 | Synthetic-ratio optimum and its compute-matched control. Three-seed mean beam-5 test figures are: real-only 9.70, 0.55:1 10.80, **1.1:1 11.14** and 1.66:1 10.51. A compute-matched arm, run at seed 42 only, scores 8.54 against that seed's in-batch real-only control of 9.89, a paired **−1.35**. No per-seed summary JSON or decode dump for these arms is included, so neither the per-seed deltas nor the interval can be recomputed here, and the one-seed control is not independently checkable. | Datasets and the synthetic corpus build; about 5 V100-days plus the corpus build | `jobs/slurm/stitch_scale.sbatch`, `jobs/slurm/stitch_dur_seeds.sbatch` |