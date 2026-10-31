### AX — Content-word failure analysis

**Claim.** In the Error Analysis (Results), Discussion and appendix, the dissertation states that 786 of the 800 official test clips have a reference content word (2.55 per clip, 2,043 tokens; content words are words outside a 71-word function list); that 48.8% of outputs are a verbatim training sentence sharing no content word and 5.2% are exact; that recall rises from 0% (seen twice or fewer) to 20.8% (seen over 50 times), those frequent words being 1,592 tokens (78%), 85% of them appearing somewhere in the outputs; that recall is 17.1% against 8.2% for a null model (2.1 times); and that the ensemble is exactly right on 0 of 564 unseen-target clips against 19.8% of seen ones.

**Runs with:** this repository and the dataset spreadsheet (`make data`)

```bash
python3 ../results/checks/content_word_failure_analysis/check.py
```

```text
references match comm_run/manifests/test.tsv: 800/800
GATE single model beam-5 test BLEU-4 10.2716 (recorded 10.27 for the 40-epoch pre-trained model)
GATE S6 ensemble beam-10 LP 0.3 test BLEU-4 11.802 (recorded 11.802)

Content words are the words outside a list of 71 function words  [c06-009 c06-012 c10-028 c21-009]
Content-word recall and precision are scored on a scale of 0 to 100%; higher is better  [c06-009]
Setting: 40-epoch pre-trained pose-only model, beam width 5 (the 10.27 score above); ensemble panel at beam width 10  [c10-028 c10-030 c14-039 c21-009]

Clips with a reference content word: 786 of 800  [c10-023 c21-009]
Reference content tokens: 2,043  [c21-008 c10-032]
Mean reference content words per test clip: 2.55  [c10-024]
Recall of words seen twice or fewer: 0.0%  [c10-031]
Recall of words seen more than 50 times: 20.8%  [c10-031 c10-033 c14-039]
Tokens of words seen more than 50 times: 1,592 of 2,043 (78%)  [c10-032 c14-038]
Of these frequent words, 85% occur somewhere in the outputs (115 of 136 types)  [c10-033 c14-039]

Content-word recall of the model: 17.1%  [c10-036]
Null model (two most frequent training content words, every clip): 8.2%  [c10-036]
Ratio model / null: 2.1 times  [c10-036]

Failure modes over 800 clips: {'exact': 42, 'memorised': 390, 'partial': 160, 'novel': 161, 'reworded': 33, 'no content word': 14}
Clips classified: 800 (all test clips)  [c10-027]
Verbatim training sentence sharing no content word with the reference: 48.8%  [c10-027]
Exact matches: 5.2%  [c10-027]

S6 ensemble, beam width 10: exact match on 0 of the 564 unseen-target clips that contain content words  [c11-061 c21-009]
S6 ensemble exact match on seen-target clips: 19.8% (44 of 222)  [c11-061]
(single 40-epoch model, for comparison: unseen-target exact matches 1)
```

**Scope.** Recomputed from the saved test decodes (single 40-epoch pre-trained model, beam 5; S6 ensemble, beam 10 / LP 0.3) and the training references; two gates re-score BLEU-4 (10.27, 11.802). Not covered: the retrieval baseline numbers 2.72 and 1.47 (only a write-up exists). No training or inference.
