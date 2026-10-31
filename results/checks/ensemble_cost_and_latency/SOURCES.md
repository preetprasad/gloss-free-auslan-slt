# Sources of the saved files

- `ensemble_latency_log.txt`: byte copy of the local file `results/ensemble_latency_2026-09-22.txt`, the printed output of `src/tools/ensemble_latency.py` (M3 Max CPU, 12 threads, batch 1, 800 validation clips, 22 Sep 2026). Local job, no Kaya job id.
- `pose_throughput_subset.json`: programmatic subset (by `prepare_data.py`) of the local file `results/pose_throughput_m3max_2026-09-15.json` written by `src/tools/pose_throughput_bench.py` (15 Sep 2026, M3 Max CPU). Kept: machine type, CPU count, onnxruntime version, the container frame rate of the 12 timed clips, and the timing summaries of the CPU full pipeline, CPU pose-only, Core ML pose-only and the translator. Left out: clip names, image sizes, MPS run, the Core ML failure text with local paths.
- `test_clip_frames.json`: `{clip: frames}` for the 800 official test clips, from the `records` list of the local file `results/speech_pipeline_bench.json` (`src/tools/speech_pipeline_bench.py`, 22 Sep 2026). Written by `prepare_data.py`; no subtitles kept.
- `prepare_data.py`: run once by hand to write the two files above; not run by `make`.
- `check.py`, `claims.json`, `fragment.md`: written for this check. `check.py` also reads the `#SBATCH --gres` lines of the job scripts in `jobs/slurm/`.
- Not included: the printed output of Kaya job 1081793 (DWPose front end on one V100). It exists only on Kaya; an earlier copy in this folder had no verifiable local origin and was removed.
