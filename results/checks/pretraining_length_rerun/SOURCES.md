# Sources
- `runs.json`: programmatic subset (by `prepare_data.py`, not run by make) of the saved job outputs
  `results/validation_selected_pretraining_rerun/runs_pre/{P0,P40,P80,P128,P256}_s{42,101,202}/{progress.json,beam5_val_best.json,beam5_test_best.json}`,
  `results/validation_selected_pretraining_rerun/runs_pre/vlp_e*_s*/progress.json` (cluster job 116619, pre-training length arms) and
  `results/validation_selected_pretraining_rerun/runs_news/{nF256,nF512}_s{42,101,202}/...` (cluster job 116620, News frame-limit arms).
  Only scores and settings are kept; no hypotheses or references.
- `check.py` also reads the recorded job scripts `jobs/slurm/w11_vlpepochs.sbatch`, `w11_newsframes.sbatch` (job 116620), and for the earlier News-only runs `w4_news.sbatch`, `news_bpe.sbatch`, `w12a_news_decomp.sbatch` and `news_long.sbatch` (job 1196181; settings only, its outputs are not local).
