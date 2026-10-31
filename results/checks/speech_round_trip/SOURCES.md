# Sources
All files were written once by `prepare_data.py` (private tree only, not run by make) from `results/speech_roundtrip/` (the original round-trip job, Mac, Whisper.cpp 1.9.4 on Metal).
- `items.jsonl` : per clip, the model output (A, 40-epoch pre-trained model; A6, final ensemble) and its Whisper transcript, from `results/speech_roundtrip/arms.json` and `<arm>/NNN.wav.txt`. No reference text.
- `transcript_edits.json` : Whisper transcripts of the spoken human references (B) and scrambled references (C) as character edits against the spoken text; the spoken text is rebuilt in `check.py` from `comm_run/manifests/test.tsv` (and the original scramble function).
- `recorded_results.json` : the original job's `roundtrip_results.json` (subs_top20 removed); used only as a gate.
- `transcribe_metal.json`, `g2_pass.json` : the original job's transcription and synthesis gate files (model hash, flags, voice).
Related code (published): `src/tools/speech_roundtrip.py`.
