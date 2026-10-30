### AX — Speech chain: latency by stage and voiced output by coverage
**Claim.** The Results chapter (Section "Speech Output and Confidence-Based Abstention", Table `tab:res_speech_latency`, Figure `fig:res_latency_plot`, and the Discussion) states that over the 800 official Communication test clips, with the 40-epoch pre-trained pose-only model at beam width 5, the median delay from the end of signing to the start of speech is 2.75 s (95th percentile 5.27 s), of which pose estimation (estimated from clip length at 25.4 frames per second) is 71.7%, speech synthesis 24.1% and translation 3.3%. Appendix Table `tab:res_speech_coverage` and the Results text state the BLEU-4 of the voiced output as the confidence threshold tightens (10.27 at 100% coverage, 19.19 at 50%).

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/speech_chain/check.py
```
```text
references match comm_run/manifests/test.tsv: 800/800
GATE all 800 voiced: BLEU-4 10.2691 (recorded 10.27 for the 40-epoch pre-trained model)

Setting: 800 official test clips, beam 5, estimated pose time = frames / 25.4 fps
pose-only CPU rate 25.4 frames/s (read from the saved job summary pose_throughput_subset.json)

LATENCY per utterance (ms)
Stage                          Median     95th   Share
Pose estimation (estimated)    1968.5   4448.8   71.7%
Translation (beam width 5)       91.8    217.7    3.3%
Speech synthesis                660.8    697.1   24.1%
Translation + synthesis         752.6    901.5   27.4%
Whole chain                    2746.7   5271.0  100.0%

median delay 2.75 s, 95th percentile 5.27 s
translation with this benchmark's own beam search: median 92 ms

Detector on every frame: 5.2 frames/s (read from the saved job summary); median clip 50 frames -> pose estimation alone 9.6 s
Batched beam-5 search, single pose-only translator: mean 36 ms per clip (100 test clips; read from the saved job summary)

VOICED OUTPUT by coverage (threshold = quantile of the test-set mean per-token cross-entropy)
 Coverage  Voiced  Threshold   BLEU-4
    100%      800        ---    10.27
     90%      720      0.941    11.27
     75%      600      0.826    12.88
     50%      400      0.681    19.19
     25%      200      0.523    33.54
```
**Scope.** Recomputed from the per-clip records saved by `tools/speech_pipeline_bench.py` (800 test clips: hypothesis, reference, confidence, clip length, measured translation and synthesis time): stage medians and 95th percentiles, shares of the median chain, and corpus BLEU-4 of each voiced subset through `slt/metrics.py` (detokenised, sacreBLEU 13a), with the thresholds as quantiles of the saved confidence scores. The gate is that all 800 hypotheses score 10.27, the recorded test score of the 40-epoch pre-trained model; this also identifies the model, since the 80-epoch model scores 10.71. The per-clip translation and synthesis times are the original laptop (Apple M3 Max CPU) measurements; they are not re-timed here, and no inference is run. The pose stage was never run per clip: it is estimated, as in the dissertation, as frames / 25.4. Three numbers come from the saved summary of the pose-throughput benchmark (`tools/pose_throughput_bench.py`, no per-frame timings kept) and are printed as "read from the saved job summary": 25.4 frames/s (pose model alone, detector once per clip), 5.2 frames/s (detector on every frame), and 36 ms (mean batched beam-5 decode of a single pose-only translator over 100 test clips); the 9.6 s is the median clip length (50 frames, recomputed) divided by 5.2 frames/s. Files: `records.jsonl` (copied from `results/speech_pipeline_bench.json`), `pose_throughput_subset.json` (subset of `results/pose_throughput_m3max_2026-09-15.json`); see `results/checks/speech_chain/SOURCES.md`.
