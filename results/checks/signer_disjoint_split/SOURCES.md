# Sources

`prepare_data.py` (run once in the private tree; make does not run it) wrote `data/{SD,EXPa,EXPb}_s{42,101,202}/beam5_{val,test}.json` (18 files)
from the saved beam-5 decodes `results/wave12g_138387/runs/<arm>_s<seed>/beam5_<split>_best.json` (local copies of the outputs of the
signer-disjoint experiment, wave 12g: training array job 138387, split preparation job 138386, summary job 138388).

These files are a programmatic SUBSET of the saved decodes, not byte copies. Kept: `beam_hyps` (identical to the source), `recorded_bleu4`
(= `beam_result.metrics["tokenize=13a;detok=True"]`, identical), `manifest_name` (the basename of the source's `manifest` field) and
`refs_sha1` (SHA-1 of each entry of the source's `refs`). Dropped: `checkpoint`, `manifest` (cluster paths), `split`, `beam`, `length_penalty`,
`greedy`, `greedy_hyps`, the other `beam_result` entries and `refs` itself, so no dataset sentence is stored; `check.py` takes the references
from the manifests rebuilt by `make data` and requires their digests to equal `refs_sha1`.

The split itself is rebuilt from the dataset spreadsheet (`make data`) with `src/tools/w12g_signer_split.py`; the decodes are
tied to the rebuilt manifests row for row and their recorded BLEU-4 reproduced (the gate) before any number is printed.
