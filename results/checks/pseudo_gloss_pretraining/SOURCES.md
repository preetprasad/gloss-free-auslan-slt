# Sources of the saved files

All from local copies of Kaya jobs; no retyping. Scripts: `prepare_data.py` (run once by hand).

- Wave 23, Kaya job set 179456 (training array 179456, read-out 179457), local `results/wave23_179456/`:
  - `runs/<arm>_s<seed>_progress.json` (arms C, X, Y; seeds 42, 101, 202): byte copies of `runs/<arm>_s<seed>/progress.json`.
  - `runs/<arm>_s<seed>_{val,test}.jsonl`: `beam_hyps` and `refs` (800 each) of `runs/<arm>_s<seed>/beam5_{val,test}_best.json`, re-keyed `hyp`/`ref`.
  - `summary_w23.json`: byte copy of `summary_w23.json` (read-out job 179457); used as a gate.
  - `summary_179457.txt`: byte copy of the read-out job's stdout; only its `GATE:` line (differences from wave 5's control) is parsed.
- Earlier controls with the same recipe, official test, beam 5:
  - `runs/earlier_wave16A_s<seed>_test.jsonl`: `beam_hyps`/`refs` of `results/wave16_174916/runs/A_s<seed>/beam5_test_best.json` (Kaya job 174916).
  - `runs/earlier_wave6A_s<seed>_test.jsonl`: same from `results/wider_corpus_joint_training/runs/A_s<seed>/beam5_test_best.json` (Kaya job 16435).
  - `earlier_recorded.json`: the recorded `beam_result.metrics["tokenize=13a;detok=True"]` of those six files (gate values), written by `prepare_data.py`.
- Wave 5's own per-item output (control C, array 12629) is on Kaya and is not used; its scores reach this check only through the read-out's saved difference line.
- `check.py`, `claims.json`, `fragment.md`, `prepare_data.py`: written for this check.
