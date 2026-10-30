# Sources of the saved files (all copied or derived by `prepare_data.py`, which runs only in the private tree)

All hypotheses are model outputs (the `beam_hyps` of the original jobs); no reference sentence is stored (references come from `comm_run/manifests` after `make data`).

| File(s) in `data/` | Origin (private tree, relative to the repository root) | Job |
|---|---|---|
| `hyps_{ctrl,C133,F69}_s{42,101,202}_168265.json` | `results/wave13_168265/runs/<arm>_s<seed>/beam5_{val,test}_best.json` (`beam_hyps`, recorded BLEU-4 from `beam_result`; job set from `DONE`) | training array 168265 (Kaya `/group/.../wave13/runs`) |
| `hyps_{ctrlB,F69B,P69TC,SG}_s{42,101,202}_168266.json` | `results/wave13_168266/runs/<arm>_s<seed>/beam5_{val,test}_best.json` | training array 168266 |
| `hyps_{ctrl,F69}_s{303,404}_172447.json` | `results/wave13x_172447/runs/<arm>_s<seed>/beam5_{val,test}_best.json` | five-seed continuation 172447 |
| `sweep_val_{ctrl,F69}_s{42,101,202}_168265.json` | `results/wave13_168265/sweep/<arm>_s<seed>/val_lp.json` (`hyps`, `bleu`; references dropped) | length-penalty sweeps 171802 / 171804 |
| `sweep_val_{ctrlB,F69B}_s{42,101,202}_168266.json` | `results/wave13_168266/sweep/<arm>_s<seed>/val_lp.json` | length-penalty sweeps 171803 / 171805 |
| `trackpose_{c0,v2}_{ctrl,F69}_s{42,101,202}.json` | `results/wave24_179416/dumps/runs_{c0,v2}/<arm>_s<seed>/beam5_val_best.json` (validation `beam_hyps` only) | tracked-keypoint array 179416 |
| `trackpose_pose_gate.json` | `results/wave24_179416/pose_gate.json` (three fields: `n`, `clips_scale_differs_gt10pct`, `body_scale_ratio_v2_over_c0_median`) | pose check 179415 |
| `params.json` | `results/wave13_168266/smoke.json` (`params_M` of each arm) | smoke test 168264 |
| `feat_dim.json` | `feat_dim=` line of `results/wave13_16826{5,6}/logs/w13-train-*.out` | training logs of 168265 / 168266 |

`output.txt` is the recorded output of `check.py` (twin interpreter). `fragment.md`, `claims.json`, `check.py`, `prepare_data.py` are written for this check.
