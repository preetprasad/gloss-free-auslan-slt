### AX — Ensemble and translator decoding cost, pose front-end speed, GPU set-up

**Claim.** The dissertation (Introduction, Method set-up, Results "latency", Discussion, appendix latency table) states that on a laptop CPU the pose front end runs at 5.2 frames/s with the detector on every frame and 25.4 with it run once per clip (55.0 with Core ML), against 25 in the source video; that a single pose-only translator decodes a clip in 36 ms at beam 5; that the six-member ensemble at beam 10 reaches 12.47 validation BLEU-4 at a median 269 ms per clip; the appendix table of 1, 3 and 6 members at beams 5 and 10; that at 5.2 frames/s pose estimation alone takes about 9.6 s for the median clip; and that each run used one NVIDIA V100 GPU.
**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/ensemble_cost_and_latency/check.py
```
```text
GATE six members, beam 10: validation BLEU-4 12.470 (recorded 12.470, the headline validation score)
read from the saved job output ensemble_latency_log.txt (M3 Max CPU, 12 threads, batch 1, 800 validation clips)
 members  beam  BLEU-4  median ms  p95 ms
       1     5    9.53         35      76  [c16-050]
       3     5   11.31        103     228  [c16-051]
       6     5   12.19        207     460  [c16-052]
       1    10    9.78         44     101  [c16-053]
       3    10   11.47        133     311  [c16-054]
       6    10   12.47        269     613  [c16-055]
members 1, 3 and 6 on 800 validation clips at batch size 1, 12 threads (log header: clips 800, torch threads 12)  [c16-049]
6 members (the six-member ensemble), beam 10: 12.47 validation BLEU-4 at a median 269 ms per clip  [c13-024 c13-026]
3 members (three), beam 10: 11.47 validation BLEU-4 at a median 133 ms per clip  [c18-056 c18-057]

read from the saved job output pose_throughput_subset.json (M3 Max CPU; frames per second recomputed as 1000 / mean ms per frame)
detector on every frame: 5.2 frames per second (839 frames)  [c02-027 c13-021 c13-025 c14-086]
detector once per clip, pose model alone: 25.4 frames per second  [c13-022 c14-086]
pose model alone on the Core ML provider: 55.0 frames per second  [c18-055]
single pose-only translator, beam width 5: mean 36 ms per clip (100 test clips)  [c02-028 c13-023]
source video: 25 frames per second (container rate of all 12 timed clips)  [c14-086]

median official test clip: 50 frames (800 clips, frames of each clip in the saved speech-chain benchmark)
pose estimation alone for the median clip at 5.2 frames/s: 9.6 s  [c14-089]

read from the job scripts: 64 scripts that call train_ctc.py; their `#SBATCH --gres` request:
    3 x gpu:1
   17 x gpu:v100-32gb:1
   41 x gpu:v100:1
    3 x none (CPU, summary or smoke script)
every one of the 61 scripts that request a GPU asks for exactly one GPU; 58 name the V100 type, 3 ask for any single GPU  [c07-069]
```
**Scope.** Parses saved timing outputs; nothing is timed again (timings depend on the machine). Gate: the six-member beam-10 validation BLEU-4 equals the recorded 12.470 (original tool run of 22 Sep 2026, local, no job id). Latency tables come from `tools/ensemble_latency.py` (M3 Max CPU, 12 threads, batch 1, 800 validation clips); front-end speeds from `tools/pose_throughput_bench.py` (839 frames from 12 clips; frames per second recomputed as 1000 / mean ms). The 9.6 s is the median test-clip length (from the saved speech-chain benchmark) divided by the 5.2 frames/s. The V100 statement is checked at the level of the job scripts: it counts the `#SBATCH --gres` request of the training scripts, so it shows what was requested, not what a job log reports; three scripts ask for "any single GPU" rather than naming the V100. Not covered: the 21.7 frames/s on one V100 (Kaya job 1081793, log only on Kaya) and the 35-45 minute training time (Kaya logs).
