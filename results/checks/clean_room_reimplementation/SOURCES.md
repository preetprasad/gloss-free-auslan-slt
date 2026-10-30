# Sources of the saved files in data/

All hypotheses are model outputs; references appear only under the key "refs" in refs_val.json / refs_test.json (json.dump defaults).

| file | what | copied from |
|---|---|---|
| refs_val.json, refs_test.json | the 800 validation / 800 official-test reference sentences, in clip order | the "refs" field of results/wave27_195640/runs/K_s42/beam5_{val,test}_best.json (identical in every run file; the build asserts it) |
| ownft_K_{val,test}.json, ownft_KSA_{val,test}.json | beam-5 (LP 1.0) hypotheses of the plain (K) and stitched+augmented (KSA) fine-tunes, seeds 42/101/202, with each job's recorded BLEU-4 | local results/wave27_195640/runs/{K,KSA}_s{42,101,202}/beam5_{val,test}_best.json (cluster job 195640, copied from the cluster earlier; field beam_hyps, beam_result.metrics) |
| port_u_asl_{val,test}.json | beam-5 hypotheses of the five port-fine-tuned models, seeds 42/101/202 and 303/404, with recorded BLEU-4 | local results/wave12nb_167035/runs/u_asl_s{42,101,202}/ (job 167035) and results/wave12nb_175088/runs_s2/u_asl_s{303,404}/ (job 175088), beam5_{val,test}_best.json |
| clean_redecode_{val,test}.json | the same five models decoded with our own code, with the job's recorded BLEU-4 and its gate fields | cluster /group/pmc099/pprasad/wave25b/redecode/{42,101,202,303,404}/beam5_{val,test}_clean.json and gate.json (job 193342, read-only scp) |
| gk_job186422.json | summary of the logit-parity job (tolerances, deviations, pass flag) | cluster /group/pmc099/pprasad/wave25b/gk/gk.json (job 186422), unchanged |
| gk_clips_names.json | file names (no text) of the 20 validation clips used by the parity job | file-name part of the "clip" paths in cluster /group/pmc099/pprasad/wave25b/gk/clips.json (job 186422) |

Not copied: the parity job's dump of the original model's tensors (derived from the checkpoint, 200 MB), the clip text of clips.json (reference sentences), checkpoints.
