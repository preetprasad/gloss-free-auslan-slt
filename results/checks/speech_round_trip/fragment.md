### AX — Speech round trip with Whisper small.en
**Claim.** The appendix (supplementary speech round-trip table) and the discussion state that, when the voiced translator output is read back by a speech recogniser (Whisper small.en), the round-trip word error rate is 2.8% for the 40-epoch pre-trained model and 3.0% for the final ensemble (strict scoring), against 3.8% for the human reference sentences and 21.3% for references with scrambled word order; normalised scoring gives 1.1%, 1.6%, 2.2% and 19.4%. The paired difference to human text is -1.5 points (-2.8 to -0.1), and the round trip is said to add at most half a point in every variant (recomputed: true for the 40-epoch output, +0.8 for the ensemble in one variant).

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/speech_round_trip/check.py
```
```text
Transcriber: whisper.cpp 1.9.4, model ggml-small.en.bin (sha256 c6138d6d58ec), backend metal  [c07-072]
GATE human references strict RT-WER 0.037910 (recorded 0.037910)

Round-trip WER, 800 official test clips, 95% bootstrap interval (2000 resamples, seed 0)
Output of the 40-epoch pre-trained model (voiced)    strict 2.8% [1.7, 4.1]  normalised 1.1% [0.7, 1.5]  [c14-095 c20-012 c20-015 c20-016]
Final ensemble output                                strict 3.0% [1.8, 4.4]  normalised 1.6% [1.0, 2.6]  [c14-095 c20-012 c20-015 c20-017]
Human reference sentences                            strict 3.8% [2.8, 5.0]  normalised 2.2% [1.6, 3.1]  [c14-095 c20-012 c20-013 c20-015 c20-018]
References with scrambled word order                 strict 21.3% [19.6, 23.0]  normalised 19.4% [17.9, 21.0]  [c20-013 c20-015 c20-019]
scrambling the references raised strict RT-WER from 3.8% to 21.3%  [c20-013]
ensemble strict upper bound unrounded 4.446% (rounds to 4.4, not 4.5)  [c20-017]

Paired difference, voiced output minus human references (B standardised to the output's length mix), percentage points
strict     A   all 800          -1.5 (-2.8 to -0.1)  [c20-014 c18-054]
strict     A   757 scramblable  -0.9 (-2.1 to +0.5)  [c20-014]
strict     A6  all 800          -1.4 (-2.8 to +0.0)  [c20-014]
strict     A6  757 scramblable  -0.6 (-1.9 to +0.8)  [c20-014]
normalised A   all 800          -1.5 (-2.5 to -0.6)  [c20-014]
normalised A   757 scramblable  -1.1 (-2.0 to -0.3)  [c20-014]
normalised A6  all 800          -1.1 (-2.3 to +0.1)  [c20-014]
normalised A6  757 scramblable  -0.6 (-1.8 to +0.5)  [c20-014]
Voiced 40-epoch output, strict, all 800: -1.5 percentage points (-2.8 to -0.1)  [c18-054]
largest upper bound, 40-epoch output (A), all variants: +0.5 points, i.e. at most half a point  [c20-014]
largest upper bound, final ensemble (A6), all variants: +0.8 points (strict, 757 clips)  [c20-014]
DISAGREES for the ensemble: the appendix says 'every variant ... at most half a percentage point'; it holds for the 40-epoch output only
setting: 800 official test clips, 95% bootstrap intervals  [c20-015]
```
**Scope.** Recomputed from the saved synthesis-and-recognition outputs (the four spoken arms and their Whisper transcripts; transcripts of the spoken references are stored as edits against the reference text, rebuilt from `comm_run/manifests/test.tsv`), scored with the original functions of `tools/speech_roundtrip.py` (2,000 bootstrap resamples, seed 0). The gate reproduces the recorded strict round-trip WER of the human references. Not redone: speech synthesis and Whisper transcription (macOS voice and whisper.cpp on Metal, original job on the Mac), and the heard-BLEU analysis.
