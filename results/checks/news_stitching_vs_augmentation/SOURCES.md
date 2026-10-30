# Sources
- make_stats.py: written here; run once in the private tree to make data/*.json from the saved decodes (needs the References, so it is not run by make).
- data/{nA,nS,nSA,nRA}_s{42,101,202}_{val,test}.json: from results/wave12a_172449/runs/<arm>_s<seed>/beam5_<split>_best.json (local copies of Kaya /group/pmc099/pprasad/wave12/a_news/runs_138377 (nA, nS, nSA; jobs 138377_0..8) and runs_172449 (nRA; jobs 172449_9..11)). Fields kept: beam hypotheses, per-sentence BLEU sufficient statistics computed with make_stats.py, the job's recorded beam-5 BLEU-4 and job id (slurm_job.txt). Read-out of the original job: 138378 (results/wave12_build_2026-10-13/readout_logs/w12a-news-sum-138378.out); nRA summary 195461.
- data/probe_nA_s42_{val,test}.json: Kaya /group/pmc099/pprasad/wave12/a_news/runs_172449/nA_s42/beam5_{val,test}_best.json (job 172449_0, re-run of nA seed 42); hypotheses and recorded BLEU only.
- output.txt: exact output of the command in fragment.md.
