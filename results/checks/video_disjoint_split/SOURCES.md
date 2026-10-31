# Sources

All files in data/ come from `prepare_data.py` (run once by hand).

- summary_s42/101/202/303/404.json: byte copies of results/video_disjoint_recipe_confirmation/summary_s*.json (VD-CONFIRM, jobs 6596 and 7965).
- seed_decision_5seeds.txt: byte copy of results/video_disjoint_recipe_confirmation/seed_decision_5seeds.txt (recorded read-out; used as the gate).
- V6_test_LP1.0.jsonl: byte copy of results/t2_vdens_10674/dumps_V6_test/hyps_LP=1.0.jsonl (six-model ensemble, video-disjoint test, job 10674).
- vdB_s101_test_LP0.9.jsonl: byte copy of results/video_disjoint_cross_modal_ensemble/prior/seedens_vdB/s101_test/hyps_LP=0.9.jsonl (best single model, job 21736).
- vd_split_ids.csv: clip identifiers only (no text) of results/video_disjoint_cross_modal_ensemble/manifests/vd_{train,val,test}.tsv (built by jobs/slurm/video_disjoint.sbatch).
- v6_members_line.txt: the "V6" line of results/video_disjoint_cross_modal_ensemble/g1_readout.txt (names the six members).

References and training text come from comm_run/manifests (written by `make data`).
- vdB_s42/101/202_val_LP*.jsonl, vdB_s42/202_test_LP*.jsonl: byte copies of results/video_disjoint_cross_modal_ensemble/prior/seedens_vdB/s{42,101,202}_{val,test}/hyps_LP=<selected>.jsonl (job 21736, selected LP in the read-out).
- g1_singles_lines.txt: the three vdB single-model rows of results/video_disjoint_cross_modal_ensemble/g1_readout.txt (recorded LP, validation and test BLEU-4; used as gates).
