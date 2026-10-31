# Sources of the saved files

- `data/<arm>_s<seed>_<test|val>.json` (40 files): the keys `beam_hyps` and `refs`, plus `recorded_bleu4` (the `beam_result` metric `tokenize=13a;detok=True`), of `results/real_data_learning_curve/runs/<arm>_s<seed>/beam5_<test|val>_best.json`. Kaya job 21734 (`jobs/slurm/w8_realdata.sbatch`, 20 tasks: R025, R050, R100, V050 x seeds 42/101/202/303/404). References are under `refs` only.
- `derived.json`: written by `prepare_data.py` from `results/real_data_learning_curve/{manifests,train.tsv,test.tsv}` and each run's `progress.json`: subset sizes, epochs, per-token training counts of the test references (numbers only), unseen flags, held/kept episode flags for the V050 contrast.
- `prepare_data.py`, `check.py`, `claims.json`, `fragment.md`: written for this check.
