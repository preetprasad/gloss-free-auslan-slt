# Sources (all copied from the local tree; originals produced by Kaya jobs 138382 (decode) and 138383 (read-out), wave 12d)

- `nbest_p{0,-1,0.5,1,2}_part{0,1}.json` : `results/wave12d_138382/out/nbest_p<p>.json` (Kaya `/group/.../wave12/w12d/out/`, job 138382), split into clips 0-399 and 400-799; only the `power` and `nbest` fields are kept.
- `refs.json` : the `refs` field of the same files (identical in all five; the 800 validation references, restored by `make data`).
- `meta.json` : `ckpt_md5`, `beam`, `power` fields of `nbest_p0.json` (used for the gate against job 120153).
- `job_summary_w12d_scores.json` : `results/wave12d_138382/out/w12d_scores.json` (written by read-out job 138383); used only as a cross-check of the recomputed bootstrap.
- Reused by path (already published): `results/ensemble_member_subsets_ablation/shard*of4.json` (job 120153, six-member decode), `src/tools/w12d_powermean.py`, `src/slt/metrics.py`.
