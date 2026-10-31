# Sources (no file copied: the check reads published saved outputs)
- results/semantic_adequacy_grading/hyps_dwpose_ctc.jsonl: DWPose+CTC beam-5 test decodes with references under "ref" (job 1075705; ladder re-decode 6806).
- results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl: six-member ensemble, beam 10, LP 0.3 (job 1125067).
- results/train_target_hashes.txt: SHA-1 digests of training targets (seen/unseen).
- comm_run/manifests/test.tsv (from `make data`): compared with the references.
- The constant sentence is fetched by digest with tools/restore_references.py (`sentence`).
