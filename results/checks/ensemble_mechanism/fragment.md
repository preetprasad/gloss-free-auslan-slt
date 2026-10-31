### AX — Why ensembling works: members, bootstrap, mechanism, MBR, seed ensembles

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
