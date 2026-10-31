# Sources

No data files are saved here. The check reads files that are already published or provided by `make data`:

- `results/semantic_adequacy_grading/hyps_{dwpose_ctc,vlp,rgbfusion}.jsonl`: beam-5 test decodes of the three single models (ladder jobs; hyps and references).
- `results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl`: reported ensemble, beam 10, LP 0.3 (gate 11.802).
- `results/train_target_hashes.txt`: SHA-1 digests of the training targets (from `tools/train_target_hashes.py`).
- `comm_run/manifests/test.tsv` and `data/auslan-daily/translation/Auslan-Daily_Communication.xlsx` (Split, Video_Name, Signer_ID), from `make data`.
