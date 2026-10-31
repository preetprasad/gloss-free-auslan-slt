### AX — News: five-seed stitched data and pose augmentation, validation length penalty, signable recall, ensemble gate (wave 4)
**Claim.** On the official News test the adopted recipe (10,614 stitched clips at 1.1:1 plus pose augmentation, 512 input frames) raises mean beam-5 BLEU-4 from 1.80 to 2.14 (+0.34; five-seed 95% t-interval +0.09 to +0.58, higher in all five seeds), 0.03 above the published SL-Transformer (2.11) and below GASLT (2.72) and SL-Luong (2.81) (Results, News paragraph; Discussion; Appendix B supplementary table). With beam 10 and a length penalty chosen on validation the mean is 2.33 (+0.19; -0.01 to +0.39) and the brevity penalty rises from 0.73 to 0.94. Recall of words the synthetic data can sign rises by +1.16 points, of words it cannot by +0.60. About a third of News clips (31.0%) exceed the 256-frame input cap, and the joint Communication-and-News run used 22,088 clips (12,439 + 9,649). A five-seed ensemble scores 2.948 on validation against 2.872 for the best single model (+0.076), below the pre-registered +0.10 gate.

**Runs with:** this repository and the dataset spreadsheet (`make data`)
```bash
python3 ../results/checks/news_subset_results/check.py
```
```text
gate: 20 beam-5 BLEU-4 values (nA, nSA x 5 seeds x val/test) and all LP-sweep values recomputed from n-gram counts equal the values recorded by the jobs
seeds per arm: 5 (five); beam-5 BLEU-4 on the official News test, best checkpoint  [c11-017 c11-039 c18-072]
nA  (News real only) 5-seed mean test 1.80 (1.803)  [c11-038 c14-104]
nSA (stitched + pose augmentation) 5-seed mean test 2.14 (2.139)  [c11-017 c11-038 c14-104 c11-040]
nSA - nA test per seed +0.23 +0.60 +0.40 +0.38 +0.07  mean +0.34 (+0.336) from 1.80 to 2.14  [c11-038]
nSA - nA test 95% t-interval over the 5 seeds +0.09 to +0.58; higher in 5/5 seeds  [c11-039]
nSA mean 2.14 is 0.03 above published SL-Transformer 2.11  [c11-040 c14-104]
published News baselines (cited): SL-Transformer 2.11, GASLT 2.72, SL-Luong 2.81; nSA mean 2.14 is below GASLT 2.72 and below SL-Luong 2.81  [c14-104]
read from jobs/slurm/w4_news.sbatch (job 7950, seeds 42/101/202): --max_src_frames 512 in arm nA True and arm nSA True  [c11-035]
read from jobs/slurm/w4_news.sbatch (job 7950, seeds 42/101/202): the nSA arm trains on news_sa_train.tsv, required to have 20263 rows = 9649 real News rows + 10,614 stitched rows, with pose augmentation (nA: real only)  [c11-037]
read from jobs/slurm/w4_news_s2.sbatch (job 16398, seeds 303/404): --max_src_frames 512 in arm nA True and arm nSA True  [c11-035]
read from jobs/slurm/w4_news_s2.sbatch (job 16398, seeds 303/404): the nSA arm trains on news_sa_train.tsv, required to have 20263 rows = 9649 real News rows + 10,614 stitched rows, with pose augmentation (nA: real only)  [c11-037]
both scripts agree: 5-seed comparison used --max_src_frames 512 in both arms (two scripts, seeds 42/101/202 and 303/404)  [c11-035]
the recipe: 10,614 stitched clips added to 9649 real rows, ratio 1.1:1 (1.100), plus pose augmentation, in all five nSA runs  [c11-037]
nSA with beam 10 and the length penalty chosen on validation (five seeds): test mean 2.14 to 2.33 (2.327); difference +0.19 (+0.188), 95% t-interval -0.01 to +0.39; higher in 5/5  [c18-072 c18-073]
nSA brevity penalty (13a), beam 5 / LP 0.6 to validation-selected beam 10: 0.73 to 0.94  [c18-074]
read from the saved job output signable_recall.txt (job 16398 secondary): nSA - nA recall, words that the synthetic data can sign (signable): +1.16 percentage points [+0.64, +1.68], same sign as the mean in 5/5 seeds  [c18-075 c18-076]
read from the saved job output signable_recall.txt (job 16398 secondary): nSA - nA recall, words that it cannot (unsignable): +0.60 percentage points [-0.05, +1.25], same sign as the mean in 4/5 seeds  [c18-075 c18-076]
read from the saved job output ens5_val.log (job 21744): the ensemble decode loaded 5 members and checked 5 checkpoints; its five single-member validation decodes are the 5 nSA seeds (42, 101, 202, 303, 404)  [c18-098]
read from the saved job output ens_phase1.txt: pre-registered validation gate = best single +0.10  [c18-098]
ensemble validation 2.948 (LP 1.2) against 2.872 for the best single model: difference +0.076 (below the 0.10 gate)  [c18-099]
News clips: 11,069; input cap max_src_frames (config default) 256; clips longer than 256 frames 3,429 = 31.0% (about a third); median 191 frames  [c04-024]
joint run training clips: 12,439 Communication (lines of the saved real training list of job 21734) + 9,649 News (checked by the job scripts above) = 22,088  [c20-001]
```
**Scope.** Recomputes every beam-5 and length-penalty-sweep BLEU-4 (sacreBLEU 13a, detokenised) from per-sentence n-gram counts, requires each to equal the value recorded by its job, and derives means, differences, t-intervals (df = 4, t = 2.776), brevity penalties and the validation-selected length penalty. News reference sentences are not part of the spreadsheet that `make data` restores, so the saved files hold counts only (`make_stats.py`, run once). The signable-word recall differences and the ensemble gate threshold are read from saved job outputs; the underlying recall tool is not re-run. The five-member ensemble is read from the decode log `ens5_val.log` (it prints 5 members and 5 checkpoints). The share of News clips above 256 frames is computed from the per-clip frame lengths (End_Frame minus Begin_Frame) of the News alignment table, saved as numbers only by `make_extra.py`; it is the subtitle span, which matches the extracted clip length to within 2 frames on the Communication clips checked elsewhere. The joint run's own manifest is not saved: 22,088 is the sum of the saved real Communication training list of job 21734 (12,439 lines) and the 9,649 News rows the job scripts require. Original jobs: 7950 and 16398 (nA, nSA, five seeds), 27382 (length-penalty decoding), 21744 (validation ensemble, signable-recall secondary). The recipe facts (512 input frames in both arms, 20,263 training rows = 9,649 real + 10,614 stitched, pose augmentation only in nSA) are read from the two job scripts that trained the five seeds, `jobs/slurm/w4_news.sbatch` (seeds 42/101/202) and `w4_news_s2.sbatch` (seeds 303/404); these are script reads, not measurements, and the 1.1:1 ratio is computed from the two row counts the scripts require. The published values 2.11, 2.72 and 2.81 are cited literature. Not recomputed: training, the News-only run, News-to-Communication pretraining, joint training and the subword out-of-vocabulary check (outputs on the cluster).
