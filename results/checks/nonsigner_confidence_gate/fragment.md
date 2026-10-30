### AX — Non-signer negative control: confidence and grounding score
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
