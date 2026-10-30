# Sources (all copied from the private tree, originally from Kaya)
- hyps_s{505,606,707,808,909}.json: `beam_hyps` of `results/wave12h_140946/{C,PJ,PS}_s<seed>/beam5_{val,test}_best.json` (Kaya `/group/.../wave12/runs_h/...`; jobs 138389 for seeds 505-707, 140946 for all). Model hypotheses only.
- refs_val.json, refs_test.json: the `refs` field of the same files (identical in all 30; equal to the pinned manifests). Restored by `make data`.
- recorded_s<seed>.json: three fields (`beam5_val`, `beam5_test`, `best_epoch`) of `results/wave12h_140946/summary_s<seed>.json` (written by the summary jobs 138390 / 140947); used only as the gate.
- epochs_s<seed>.json: `epochs` and `status` from `progress.json` of `results/wave12h_140946/{C,PJ,PS}_s<seed>/` and `pre_{PJ,PS}_s<seed>/`.
- recipe_excerpt.txt: lines 4-8 and 34-35 of `results/wave12h_140946/C_s505/w12h_pretrain.sbatch.ran` (the job script that ran, job 140946).
