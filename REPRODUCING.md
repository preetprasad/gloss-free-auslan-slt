# Reproducing the results

This document lists the checks behind the empirical claims of the dissertation. For each check it gives the command and the output recorded when that command was run.

**Levels.** Each check belongs to one of the following levels.

- **A.** The check is re-derived on a laptop, without a GPU, from the saved outputs in `results/` and the Auslan-Daily translation spreadsheet. `make reproduce` runs all A checks. A32 and A36 to A47 live partly or wholly in `results/checks/`, one folder per check, each with its script, the saved data it reads and the dissertation sentence it verifies.
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

The recorded outputs were produced on macOS with Python 3.14.3, sacreBLEU 2.6.0 and numpy 1.26.3, using the CPU only. `tools/holm_decision.py` emits a harmless `UserWarning` about the SciPy/NumPy version pair when it is run outside the `.venv` built by `make setup`. That warning does not affect the result and is omitted from the recorded outputs. Where a recorded output is abridged or wrapped for page width, the section says so. Checks A1 to A33 were run again on 9 October 2026 from a fresh clone, after `make setup` and `make data` (Python 3.14.3 and the package versions in `requirements.txt`), and every recorded number was reproduced; that run took 25 minutes on a laptop, most of it in the bootstrap checks A4 to A9 and A26. Checks A34 and A35 were added on 13 October 2026 and were run that day in a clean copy of the repository after `make data` (Python 3.12, numpy 1.26.3): every recorded number was reproduced, A34 in about 8 minutes, and the 55 tests passed. On 30 October 2026 all 47 checks (A1 to A47) were run in a clean copy of this build after `make data`: 47 reproduced, 0 failed, 0 skipped, in about 37 minutes.

**Coverage.** Not every number of the dissertation has a check here. `make reproduce` runs the checks that exist and passes when each of them reproduces its recorded output; it does not fail because some claim has no check. The numbers that cannot be re-derived from saved outputs are listed in the level B and C tables with the script that produces each. The code that produced a few saved outputs (the fine-tuning of the released pre-trained model, the authors' published systems and the independent re-implementation) is not part of this release; their checks re-score the saved hypotheses and logits with this repository's own scoring code.

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
  w2LF   val  -1.230 +0.611 +0.306  mean -0.104   [!] seed-202 val and all 3 test deltas are CLAUDE.md-only (see docstring)
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
claim rule E0: both max-T intervals exclude 0 = True; predicted power 0.63 (>= 0.50: True) -> CLAIM
claim rule E*: both max-T intervals exclude 0 = True; predicted power 0.72 (>= 0.50: True) -> CLAIM
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
  (dissertation text: 'test mean of 17.86 against the port's 18.21, with 82 to 85% identical')

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

### A39 — Rest-cut stitching
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
Gate rule: hit >= 0.50 and hit >= 2 x random and hit > shuffled, on both sets, means over checkpoints; hit >= 0.50: False; hit >= 2 x random: False -> gate failed (saved verdict FAIL)
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