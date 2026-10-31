# Sources
`prepare_data.py` (run once by hand) cuts these files; references are not copied.
- `data/{ctrl_small,u_scratch,u_asl}_s{42,101,202}_{val,test}.json`: hypotheses and recorded BLEU-4 / distinct count from `results/wave12nb_167035/runs/<arm>_s<seed>/beam5_{val,test}_best.json` (Kaya job 167035).
- `data/{...}_s{303,404}_{val,test}.json`: the same from `results/wave12nb_175088/runs_s2/<arm>_s<seed>/beam5_{val,test}_best.json` (Kaya job 175088).
- `data/ensemble_Estar_test.json`: hypotheses and recorded BLEU-4 of `results/wave26_195361/E_star.json` (job 195361).
- `data/gate_how2sign_port.json`: subset of `results/wave12nb_167035/gate/h2s_test_port.json` (gate of job 167035; numbers only).
