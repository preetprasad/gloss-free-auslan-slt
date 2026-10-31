# Sources
No data files are stored here; check.py reads files that are already in the repository:
- results/semantic_adequacy_grading/hyps_vlp.jsonl: beam-5 test decodes of the 40-epoch pre-trained pose-only model (ladder re-decode job 6806; 10.27).
- results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl: S6 ensemble test decodes, beam 10, LP 0.3 (11.802).
- comm_run/manifests/{train,test}.tsv: training and test references, written by `make data`.
- src/tools/stitch_lexicon.py: the 71-word function list (STOP).
