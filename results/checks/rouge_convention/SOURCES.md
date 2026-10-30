# ROUGE convention provenance

Inputs: `results/semantic_adequacy_grading/hyps_dwpose_ctc.jsonl`, `hyps_vlp.jsonl` and `hyps_rgbfusion.jsonl`.
Each contains saved hypotheses and references; the checker asserts the recorded BLEU-4 values before measuring ROUGE.
Implementation: `src/metric_suite.py`, including longest-common-subsequence F-beta, and `slt/metrics.py` detokenisation.
The expected table values below come from the frozen dissertation appendix and Results, as recorded by inventory c12/c16, rather than being generated as a new baseline from this run.

The check additionally computes ratios of transcribed published scores. This verifies their division only, not the external published results.
Beam width is inferred from `results/semantic_adequacy_grading/dump_hyps.py`; this does not establish the exact executed command. Claim c16-038 remains partly unsupported.
No inference is run. Subtitle-bearing data stays in the private workspace until reference stripping and distribution checks have been completed.
