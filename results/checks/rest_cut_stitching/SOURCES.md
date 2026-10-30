# Sources of the saved files

All copied on 30 Oct 2026 from the private tree or read-only from Kaya; Kaya jobs: audit 174914, prep 174915, training 174916 (array 0-14), summary 174917.

- `hyps_{A,R08,R10}_{val,test}.json`: `beam_hyps` and the recorded 13a BLEU-4 (`beam_result.metrics`) of `results/wave16_174916/runs/<arm>_s<seed>/beam5_{val,test}_best.json` (copied from Kaya `/group/pmc099/pprasad/wave16/runs/`, job 174916). Only model hypotheses are kept; references are read from `comm_run/manifests` (identical to the saved decodes' own `refs`, checked on the original files).
- `audit_per_clip.csv`: columns group, index, interior_runs of Kaya `/group/pmc099/pprasad/wave16/audit/per_clip.csv` (job 174914; paths and other columns dropped).
- `audit_ids_runs.json`: keys `ci95`, `stitched_ids`, `stitched_runs`, `n` of Kaya `.../wave16/audit/AUDIT_OK.json` (job 174914).
- `cut_segments_R08.csv`, `cut_segments_R10.csv`: counts of (restcut_lead, restcut_trail, source_frames_after_trim) over all 94,428 segments of Kaya `.../wave16/corpora/audit_R08.jsonl` and `audit_R10.jsonl` (build audit written by prep job 174915; 36 MB each, aggregated here).
- `PREP_OK_R08.json`, `PREP_OK_R10.json`: Kaya `.../wave16/manifests/PREP_OK_*.json` (prep job 174915), also in `results/wave16_174916/`.
- `output.txt`: output of check.py.
