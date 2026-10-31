### AX — Signer-disjoint split

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
