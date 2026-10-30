# Sources of the saved files in data/

All files hold `{"hyps": [...], "recorded_bleu4": x, "beam": b, "length_penalty": lp}`: the model hypotheses of one decode and the BLEU-4 the original job recorded. No references are stored (they come from `comm_run/manifests`).

| files | copied from | job |
|---|---|---|
| `single5_s{42,101,202,303,404}_{val,test}.json` | local `results/wave26_195361/inputs/singles/s*_{val,test}.json` (clean-code re-decodes of the five fine-tuned models, beam 5, length penalty 1.0; field `beam_hyps`, recorded value `beam_result.metrics`) | Kaya 193342 (re-decode), copied for the five-model summary 195362 |
| `ens5_E0_{val,test}.json`, `ens5_Estar_{val,test}.json` | local `results/wave26_195361/inputs/E0_{val,test}.json`, `Estar_{val,test}.json` (test also identical to `results/wave26_195361/E0.json`, `E_star.json`) | Kaya 195359 / 195361 |
| `ens5_grid_b{5,10}_lp{0.6,0.8,1.0,1.2,1.5}.json` | Kaya `/group/pmc099/pprasad/wave26/runs/grid/` (read-only copy) | Kaya 195359 |
| `single3_s{42,101,202}_{val,test}.json` | Kaya `/group/pmc099/pprasad/wave26b/runs/singles/{seed}/beam5_{val,test}_clean.json` | Kaya 238135 |
| `ens3_{E0,Estar}_test.json` | Kaya `/group/pmc099/pprasad/wave26b/runs/test/{E0,E_star}.json` | Kaya 238138 |

The 11.80 comparison system is read from the published `results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl` (not copied).
Cross-check against the original read-outs (not copied): `results/wave26_195361/w26-summary-195362.out`, `results/wave26b_238138/w26b-summary-238139.out`.
