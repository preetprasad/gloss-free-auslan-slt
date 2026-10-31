### AX — Training recipe, model size and decoding defaults
**Claim.** The Methodology states the configuration of the translation model (Table "Training configuration of the translation model"): 133 COCO-WholeBody keypoints (399 features per frame), at most 256 source frames and 64 target tokens, model width 256 with 8 heads and a 1024-unit feed-forward layer, four encoder and four decoder layers, dropout and embedding dropout 0.1, label smoothing 0.1, auxiliary CTC weight 0.3, learning rate 5x10^-4 with weight decay 1x10^-4, gradient clip 1.0, cosine schedule with 800 warmup steps, batch 64 (32 for the RGB-fusion models) and 40 epochs; decoding defaults of beam width 5 and length penalty 0.6; the adopted training configuration (13,760 stitched clips plus pose augmentation); stage seeds 42, 101, 202 and 303, 404; and the Appendix B table convention (800 official test clips, beam width 5, 95% intervals).

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/recipe_and_settings/check.py
```

```text
GATE saved beam-5 hypotheses of the pose-augmented seed-42 run: 800 clips, BLEU-4 11.443 (recorded 11.443, read from the saved job output)
references match comm_run/manifests/test.tsv

Members: the 6 checkpoints named in the ensemble job script (mbr_wer_test2.sbatch)
3 pose-only members, 3 RGB-fusion members

TABLE: training configuration (stored in every one of the six members)
keypoints / input feature dimension: 133 COCO-WholeBody / 399 (input projection weight 256 x 399; feature dim = 133 x 3)  [c07-001 c06-039]
coordinate normalisation: frame  []
max source frames / target tokens: 256 / 64  [c07-002 c06-025]
d_model / heads / FFN: 256 / 8 / 1024 (d_model = model width 256)  [c07-003 c06-039]
encoder / decoder layers: 4 / 4 (weights hold 4 encoder and 4 decoder layers)  [c07-004 c06-037]
dropout / embedding dropout: 0.1 / 0.1  [c07-005]
label smoothing: 0.1  [c07-006]
auxiliary CTC weight (default of train_ctc.py = value passed by the job script): 0.3  [c07-007 c07-014]
learning rate: 5x10^-4 (0.0005)  [c07-008]
weight decay / gradient clip: 1x10^-4 (0.0001) / 1.0  [c07-009]
schedule / warmup steps: cosine / 800  [c07-010]
batch size (pose-only members) / epochs: 64 / 40  [c07-011]
batch size of the RGB-fusion members: 32  [c06-040]

Config() defaults vs the stored configuration of the members: 17 of 17 fields agree

Source sequences longer than the 256-frame limit are subsampled uniformly (np.linspace in slt/data.py): a 300-frame clip keeps 256 frames with gaps of 1 or 2 frames  [c06-025]
Targets are limited to 64 tokens (stored max_tgt_tokens)  [c06-025]

DECODING defaults
beam width / length penalty (Config): 5 / 0.6  [c06-011 c07-012]
beam width / length penalty (tools/eval_beam.py): 5 / 0.6  [c06-011 c07-012]
beam width / length penalty (saved beam-5 test job output): 5 / 0.6 (read from the saved job output beam5_test.json)  [c06-011 c07-012]

Adopted training configuration (job log joblog_pose_aug.out, read from the saved job output): training set 26199 clips, of which 13,760 stitched; real clips 12,439  [c07-042]
pose augmentation switched on in that run (pose_aug True): speed +-0.2, rotation +-10.0 deg, scale +-0.1 (read from the saved job output)

Stage 1 job scripts train seeds 42, 101, 202; stage 2 job scripts train fresh seeds 303, 404  [c21-019]
seeds stored in the six members: 101, 42, 42, 202, 101, 202  [c21-019]

Table convention: the saved test run holds 800 official Communication test clips decoded at beam width 5  [c17-001 c17-002]
paired_bootstrap.py default interval: 95% (percentiles 2.5 and 97.5)  [c17-003]
```

**Scope.** Recomputed: the saved beam-5 test hypotheses of one adopted-configuration run (the gate) and every table value, read from the training configuration stored in the six ensemble members (subset in `cfg_members.json`), the current `Config` defaults, the training and decoding scripts and one saved job log. Not recomputed: training, the PyTorch/CUDA versions (no saved output prints them; listed as not covered). Original job: 1215231 (pose augmentation); members trained 11-23 Aug 2026.
