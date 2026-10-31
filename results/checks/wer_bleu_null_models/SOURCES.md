# Sources (no file is copied: the check reads published saved outputs)
- results/semantic_adequacy_grading/hyps_rgbfusion.jsonl, hyps_vlp.jsonl, hyps_dwpose_ctc.jsonl: per-clip greedy and beam-5 test decodes (with references under "ref") of the RGB-fusion, 40-epoch pre-trained and DWPose+CTC single models; ladder re-decode job 6806 / semantic-grading dump job 1075705.
- results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl: six-member ensemble, beam 10, length penalty 0.3, test (ensemble decode, job 1125067).
- comm_run/manifests/test.tsv (from `make data`) is compared with the references when present.
- Two constant sentences are named by SHA-1 digest and fetched with tools/restore_references.py (`sentence`).
- Metrics: src/slt/metrics.py (BLEU-4 13a on detokenised text, jiwer WER); ROUGE beta read from src/metric_suite.py.
