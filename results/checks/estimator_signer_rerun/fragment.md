### AX — Estimator comparison on the signer's keypoints
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
