# Sources
Nothing is copied into this folder. The check reads three files that are already published (listed in build_report.json):
`results/semantic_adequacy_grading/hyps_dwpose_ctc.jsonl`, `hyps_vlp.jsonl`, `hyps_rgbfusion.jsonl` (800 official test clips each, saved greedy and beam-5 hypotheses with references; written by `results/semantic_adequacy_grading/dump_hyps.py`, jobs 1075705 and the ladder decodes).
Implementation: `src/metric_suite.py` (ROUGE-L, BLEU-1, `ROUGE_BETA`) and `src/slt/metrics.py` (detokenise).
Gate: the beam hypotheses must reproduce the recorded beam-5 test BLEU-4 9.52 / 10.27 / 10.33.
The ROUGE-L and BLEU-1 values of the three comparison systems in the ratio lines are published values (benchmark paper, Table 3), not this project's measurements; only their division is computed.
Beam 5 is established by the gate (beam hypotheses reproduce the recorded beam-5 scores; greedy hypotheses score differently).
