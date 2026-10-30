### AX — ROUGE-L tokenisation convention

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
