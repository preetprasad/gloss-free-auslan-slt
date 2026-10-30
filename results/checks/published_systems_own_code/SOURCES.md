# Sources of the saved files

All files are copies of job outputs from the private tree (local `results/`, themselves copied from Kaya `/group/pmc099/pprasad/`).
Only test hypotheses and the recorded scores were kept (no checkpoints, no logs, no pose data). References sit under "ref"/"refs".

| folder here | copied from (local) | Kaya origin / job |
|---|---|---|
| `data/l3/<cell>_s<seed>/dump.jsonl` (24 runs) | `results/wave12l3_197331/runs/<cell>_s<seed>/w12l_test_dump.jsonl` | array job 197331 (tasks 0-23), prep 197330, summary 197332 |
| `data/l3/<cell>_s<seed>/scores.json` | same folder, `w12l_scores.json` | job 197331 |
| `data/l3/Edw_s<seed>.json` (3) | `results/wave12l3_197331/runs_own/Edw_s<seed>/beam5_test_best.json` (keys `beam_hyps`, `refs`, recorded 13a BLEU-4 only) | job 197331, tasks 24-26 |
| `data/l2/<system>__dw53_frame_s<seed>/` (6: seeds 42/101/202, `dump.jsonl`, `scores.json`) | `results/wave12l2_167758/runs/...` (`w12l_test_dump.jsonl`, `w12l_scores.json`) | job 167758 |
| `data/l2/Edw_s<seed>.json` (3) | `results/wave12l2_167758/runs_own/Edw_s<seed>/beam5_test_best.json` | job 167758 |
| `data/ext/<system>__dw53_frame_s<seed>/` (4: seeds 303/404) | `results/wave22_179411/runs/...` | job 179411 |
| `data/ext/Edw_s<seed>.json` (2) | `results/wave22_179411/runs_own/Edw_s<seed>/beam5_test_best.json` | job 179411 |

`output.txt` is the recorded output of `check.py`. The Edw files were rewritten compactly (three keys); the dumps and scores files are byte copies.
