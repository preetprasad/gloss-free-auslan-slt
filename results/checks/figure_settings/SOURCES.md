# Sources of the saved files

- `qual_subset.jsonl`: fields `i`, `clip`, `hyp`, `ce` of each entry of `figures/qual_examples.json` (selection of the five qualitative-figure clips; `prepare_data.py`). Reference sentences left out.
- `stage_times.jsonl`: fields `id`, `frames`, `translate_ms`, `synth_ms` of the 800 per-clip records of `results/checks/speech_chain/records.jsonl` (itself from the local `results/speech_pipeline_bench.json`, 22 Sep 2026, M3 Max CPU). Hypotheses and references left out.
- `pose_rate.json`: key `pose_only_cpu` of `results/checks/speech_chain/pose_throughput_subset.json` (from `results/pose_throughput_m3max_2026-09-15.json`).
- Read in place (published files): `results/semantic_adequacy_grading/hyps_vlp.jsonl` (saved beam-5 decodes, 800 official test clips), `figures/make_extra_figures.py`.
- No Kaya job. `prepare_data.py` was run once by hand; `check.py` is the only script run by make.
