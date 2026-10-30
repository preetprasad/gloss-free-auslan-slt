# Sources of the saved files

- `records.jsonl`: 800 per-clip records (id, hyp, ref, ce, frames, translate_ms, synth_ms, audio_s), re-keyed from the `records` list of the local file `results/speech_pipeline_bench.json` (written 22 Sep 2026 by `src/tools/speech_pipeline_bench.py --n 800` on the M3 Max, checkpoint `results/ckpts/comm_dwpose_vlp_ctc.pt`, beam 5, LP 0.6, official test manifest). Reference sentences are under the key `ref` only. No Kaya job.
- `meta.json`: `n` and `beam` fields of the same file.
- `pose_throughput_subset.json`: the keys `machine`, `full_pipeline_cpu`, `pose_only_cpu`, `translator_cpu` of the local file `results/pose_throughput_m3max_2026-09-15.json` (`tools/pose_throughput_bench.py`, 15 Sep 2026, M3 Max CPU). The remaining keys (clip list with local paths, CoreML and MPS runs) were left out.
- `check.py`, `fragment.md`, `claims.json`: written for this check.
