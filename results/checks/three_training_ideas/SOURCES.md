# Sources of the saved files

All copies are made from the local result folder `results/wave12c_144831/` (itself copied from Kaya `/group/pmc099/pprasad/wave12c/` after jobs 144831 and 152915). No checkpoints, poses or videos.

- `data/<arm>_s<seed>_<split>.jsonl` (arms A, E1, ST, AC, P with seeds 42/101/202; arms A, E1, ST with seeds 303/404): the fields `beam_hyps` and `refs` of `runs/<arm>_s<seed>/beam5_<split>_best.json` (seeds 42/101/202, Kaya job 144831) and `runs_s2/<arm>_s<seed>/beam5_<split>_best.json` (seeds 303/404, Kaya job 152915), one line per clip in the official val/test order, written as {"id", "hyp", "ref"}.
- `data/runs_meta.json`: per run, the job id in its `DONE` marker, `epochs` and `status` from its `progress.json`, and the BLEU-4 the job stored in `beam5_<split>_best.json` (`beam_result.metrics["tokenize=13a;detok=True"]`).
- `data/g1_alignment_summary.json`: the fields verdict, mean_over_checkpoints, n_all, n_multi, min_spans and rule of `results/wave12c_144831/g1/G1.json` (the alignment probe on the three adopted-recipe checkpoints, written by the job's own tool; checkpoint paths dropped).
- `output.txt`: recorded output of check.py.
