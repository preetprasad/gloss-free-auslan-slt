# Sources

- `single_test.jsonl`: subset of results/semantic_adequacy_grading/hyps_dwpose_ctc.jsonl (beam-5 hypotheses and references of the 40-epoch DWPose+CTC model, 800 official test clips; ladder job 6806) joined by clip index with `self_ce` and `self_reliance` of results/self_reliance_ctc.json (src/tools/self_reliance.py, local MPS run of 9 Sep 2026). Built by `prepare_data.py`.
- `single_val.jsonl`: subset of results/hyps_val_dwpose_ctc.jsonl (same model, 800 validation clips, decoded locally) joined with results/self_reliance_val.json. Built by `prepare_data.py`.
- Ensemble: read from the published n-best dumps results/final_ensemble_validation_decodes/nbest.jsonl and results/final_ensemble_test_decodes/nbest.jsonl (six-member ensemble, beam 10, Kaya jobs 1101821 validation / 1125067 test). Seen/unseen membership from results/train_target_hashes.txt.
