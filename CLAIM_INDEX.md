# Claim index

One row per measured claim of the dissertation (literature figures and document metadata are left out). The *Check* column gives the section of `REPRODUCING.md` that verifies the claim from saved outputs (`make reproduce`), or the reason there is none. A claim marked *needs the cluster* depends on a job that has to be re-run; *no saved output* means the output that produced the number was not kept; *disagrees* means the saved data give a different number from the dissertation's (the reason is given at the end); *covered by A1-A35 (mapping, not reviewed)* means an older check of the same number exists and the mapping of claim to check has not been read claim by claim. Quotes are the dissertation's wording.

Counts: checked 644, mapped 143, unbuilt 94, needs the cluster 87, no saved output 60, disagrees 26, retired 8, not a measurement 7; 1069 claims.

| Claim | Check | File | Quote | Numbers |
|---|---|---|---|---|
| c01-003 | retired (text no longer in the dissertation) | abstract.tex | raised BLEU-4 by 4.97 over the AlphaPose keypoints distributed with the dataset as first read (three seeds) | 4.97, 3 |
| c01-004 | retired (text no longer in the dissertation) | abstract.tex | the first-listed person, which our first extraction took, is a non-signer in 88\% of clips | 88 |
| c01-005 | covered by A32 (A1-A35 mapping, not reviewed) | abstract.tex | DWPose scored $+0.27$ BLEU-4 above AlphaPose on test (mean of three seeds; 95\% interval $-1.03$ to $+1.57$) | 0.27, 95, -1.03, 1.57, three |
| c01-006 | A53 | abstract.tex | The final ensemble averages the log-probabilities of six translators | 6 |
| c01-007 | covered by A18 (A1-A35 mapping, not reviewed) | abstract.tex | three pose-only (9.0 million parameters each) | 3, 9.0 |
| c01-008 | covered by A18 (A1-A35 mapping, not reviewed) | abstract.tex | three adding image features (10.9 million each) | 3, 10.9 |
| c01-009 | covered by A1 (A1-A35 mapping, not reviewed) | abstract.tex | It scores 11.80 BLEU-4 on that test set | 11.80 |
| c01-011 | A38 | abstract.tex | run here with three seeds, scored 8.77 and 9.16 (published 9.20 and 9.95) | 3, 8.77, 9.16, 9.20, 9.95 |
| c01-012 | A38 | abstract.tex | within the pre-registered tolerance of 1.0 | 1.0 |
| c01-013 | needs the cluster | abstract.tex | scored 10.97 (mean of three runs with different random seeds) | 10.97, three |
| c01-014 | needs the cluster | abstract.tex | raised BLEU-4 over a same-seed control in all three seeds (mean gain +0.56) | 3, 0.56 |
| c01-015 | needs the cluster | abstract.tex | raised the mean gain to 1.48 (controls 9.49, stitched 10.97; three seeds) | 1.48, 9.49, 10.97, 3 |
| c01-016 | needs the cluster | abstract.tex | A control with random signs (two seeds, smaller synthetic set) | 2 |
| c01-017 | no saved output | abstract.tex | identified 45 signs from hand keypoints with 42.8\% accuracy in dictionary recordings | 45, 42.8 |
| c01-018 | A60 | abstract.tex | but 18.1\% in conversational recordings (chance 2.2\%) | 18.1, 2.2 |
| c01-019 | A63 | abstract.tex | Doubling the real conversational training pairs raises BLEU-4 on test clips whose reference is absent from training (unseen targets) from 3.15 to 5.00 | 3.15, 5.00 |
| c01-020 | A63 | abstract.tex | (+1.84 unrounded; in all five seeds) | 1.84, 5 |
| c01-022 | A68 | abstract.tex | scored 18.21 BLEU-4 on the same test set after fine-tuning (mean of five seeds) | 18.21, five |
| c01-023 | A36 | abstract.tex | an ensemble of five such models scored 20.60 | 20.60, five |
| c01-024 | covered by A1 (A1-A35 mapping, not reviewed) | abstract.tex | which is trained on \textsc{Auslan-Daily} alone | 11.80 |
| c01-025 | A61 | abstract.tex | scored 4.41 BLEU-4 on the official test set but 1.09 on unseen targets | 4.41, 1.09 |
| c01-026 | A74 | abstract.tex | over a quarter of the test references appear verbatim in training |  |
| c01-027 | covered by A5 (A1-A35 mapping, not reviewed) | abstract.tex | Paired-bootstrap 95\% intervals over resampled test sentences, about $\pm 1.2$ BLEU-4 | 95, 1.2 |
| c01-028 | A51 | abstract.tex | include zero for most improvements, so one apparent gain was withdrawn | 1 |
| c01-029 | A75 | abstract.tex | answered only its more confident half of the test clips | 0.5 |
| c01-030 | A75 | abstract.tex | its BLEU-4 rose from 9.52 to 17.42 in an exploratory analysis | 9.52, 17.42 |
| c02-002 | A56 | intro.tex | The 133 keypoints per frame (body, hands, face) | 133 |
| c02-005 | covered by A14 (A1-A35 mapping, not reviewed) | intro.tex | the 800-clip official test set of the conversational Communication subset | 800 |
| c02-006 | A54 | intro.tex | scored 2.87 on that test set (one run) | 2.87, 1 |
| c02-007 | A54 | intro.tex | produced only 53 distinct sentences for its 800 clips (output collapse) | 53, 800 |
| c02-009 | covered by A14 (A1-A35 mapping, not reviewed) | intro.tex | the 800-clip test set | 800 |
| c02-010 | A73 | intro.tex | a difference of about one \gls{bleu}-4 point between two systems lies within measurement variation | 1 |
| c02-011 | retired (text no longer in the dissertation) | intro.tex | raised \gls{bleu}-4 by 4.97 over the distributed AlphaPose keypoints as first read | 4.97 |
| c02-012 | covered by A32 (A1-A35 mapping, not reviewed) | intro.tex | scored only $+0.27$ higher on the official test set (mean of three seeds; | 0.27, three |
| c02-013 | covered by A32 (A1-A35 mapping, not reviewed) | intro.tex | 95\% $t$-interval $-1.03$ to $+1.57$), so the estimator had no resolvable effect | 95, -1.03, 1.57 |
| c02-014 | covered by A1 (A1-A35 mapping, not reviewed) | intro.tex | The final ensemble of six translators scored 11.80 \gls{bleu}-4 | 6, 11.80 |
| c02-015 | A65 | intro.tex | 1.85 above GASLT's published score | 1.85 |
| c02-016 | A83 | intro.tex | with a \gls{wer} of 0.907 (constant sentence: 0.950) | 0.907, 0.950 |
| c02-017 | needs the cluster | intro.tex | raised test \gls{bleu}-4 over a same-seed control in each of three seeds, from a mean of 9.49 to 10.97 ($+1.48$) | 3, 9.49, 10.97, +1.48 |
| c02-018 | needs the cluster | intro.tex | fell below that of real data alone (9.04 against 9.38; two seeds, smaller synthetic set) | 9.04, 9.38, 2 |
| c02-019 | covered by A13 (A1-A35 mapping, not reviewed) | intro.tex | correct content word (a word outside a fixed function-word list) for 66.9\% of the 786 test clips whose reference contains one | 66.9, 786, one |
| c02-020 | no saved output | intro.tex | identified 45 signs from hand keypoints with 42.8\% top-1 accuracy in citation form | 45, 42.8 |
| c02-021 | A60 | intro.tex | but 18.1\% in conversational signing (chance 2.2\%) | 18.1, 2.2 |
| c02-022 | covered by A14 (A1-A35 mapping, not reviewed) | intro.tex | The references of 230 of the 800 test clips occur verbatim in the training set | 230, 800 |
| c02-023 | A61 | intro.tex | on the other 570 clips (unseen targets), one constant training sentence scored 1.09 | 570, 1, 1.09 |
| c02-024 | A61 | intro.tex | against 4.41 on all 800 | 4.41, 800 |
| c02-025 | A65 | intro.tex | Paired-bootstrap 95\% intervals, which rescore both systems on the same 2,000 resamples of the 800 test sentences | 95, 2,000, 800 |
| c02-026 | covered by A5 (A1-A35 mapping, not reviewed) | intro.tex | about $\pm 1.2$ \gls{bleu}-4 | 1.2 |
| c02-027 | A52 | intro.tex | it ran at 5.2 frames per second on a laptop processor | 5.2 |
| c02-028 | A52 | intro.tex | a single pose-only translator decoded a clip in 36\,ms | 36 |
| c02-029 | A75 | intro.tex | answered only its more confident half of the test clips |  |
| c02-030 | A75 | intro.tex | \gls{bleu}-4 rose from 9.52 to 17.42, and from 4.88 to 7.92 on unseen targets | 9.52, 17.42, 4.88, 7.92 |
| c03-001 | covered by A1 (A1-A35 mapping, not reviewed) | conclusion.tex | scored 11.80 \gls{bleu}-4 on the 800-clip official Communication test set | 11.80, 4, 800 |
| c03-003 | retired (text no longer in the dissertation) | conclusion.tex | moving from the first-listed AlphaPose keypoints, mostly a non-signer's, to DWPose raised test \gls{bleu}-4 by 4.97 | 4, 4.97 |
| c03-004 | covered by A32 (A1-A35 mapping, not reviewed) | conclusion.tex | DWPose added only $+0.27$ over AlphaPose (95\% interval $-1.03$ to $+1.57$, three seeds) | 0.27, 95, -1.03, 1.57, three |
| c03-005 | no saved output | conclusion.tex | duration-corrected stitched synthetic sentences, replicated over three seeds | 3 |
| c03-006 | A53 | conclusion.tex | from a six-translator ensemble | 6 |
| c03-007 | covered by A13 (A1-A35 mapping, not reviewed) | conclusion.tex | the final ensemble produced no correct content word for 66.9\% of the 786 test clips whose reference contains one | 66.9, 786, 1 |
| c03-008 | A68 | conclusion.tex | 18.21 test \gls{bleu}-4 (five seeds; best published 9.95) | 18.21, 4, 5, 9.95 |
| c03-010 | A68 | conclusion.tex | 7.84 above the from-scratch translator in the same job (95\% interval $+7.04$ to $+8.64$) | 7.84, 95, +7.04, +8.64 |
| c03-011 | covered by A14 (A1-A35 mapping, not reviewed) | conclusion.tex | Of the 800 test references, 230 occur verbatim in the training set | 800, 230 |
| c03-012 | covered by A3 (A1-A35 mapping, not reviewed) | conclusion.tex | on the other 570 clips, the final ensemble scored 6.60 \gls{bleu}-4 | 570, 6.60, 4 |
| c03-013 | covered by A30 (A1-A35 mapping, not reviewed) | conclusion.tex | the final ensemble scored 19.07 \gls{bleu}-4 on the 52\% of test clips answered, against 11.80 on all clips | 19.07, 4, 52, 11.80 |
| c03-014 | A45 | conclusion.tex | the final ensemble's threshold still voiced 44\% of non-signing people | 44 |
| c03-015 | A63 | conclusion.tex | doubling the real training pairs raised unseen-target \gls{bleu}-4 from 3.15 to 5.00 (mean of five seeds) | 3.15, 5.00, 4, 5 |
| c03-016 | A49 | conclusion.tex | the Communication subset contains all 90 videos of its source programme | 90 |
| c03-017 | A47 | conclusion.tex | the tokenisation convention alone moved the ROUGE-L of one set of outputs by more than 19 points | 19 |
| c03-018 | covered by A19 (A1-A35 mapping, not reviewed) | conclusion.tex | Verbatim overlap lets a constant sentence score 4.41 \gls{bleu}-4 | 4.41, 4 |
| c03-021 | A81 | appendix_departures.tex | A failed reproduction check on one control seed leaves it without a verdict | 1 |
| c04-004 | A83 | review.tex | No matching 4-token run makes sentence BLEU-4 zero, though four of five output tokens (full stop included) match; WER is 0.43. | 4, 4, 5, 0.43 |
| c04-005 | A83 | review.tex | 40-epoch pre-trained pose-only model (Chapter~\ref{ch:results}), beam width 5. | 40, 5 |
| c04-007 | covered by A4 (A1-A35 mapping, not reviewed) | review.tex | Corpus BLEU-4 (800 clips): 10.27. | 4, 800, 10.27 |
| c04-010 | A49 | review.tex | & Auslan & 12,441 / 800 / 800 & 49/12/9 & 3,064$^\dagger$ | 12,441, 800, 800, 49, 12, 9, 3,064 |
| c04-011 | disagrees | review.tex | & Auslan & 9,665 / 700 / 700 & 18/17/17 & 12,346$^\dagger$ | 9,665, 700, 700, 18, 17, 17, 12,346 |
| c04-021 | covered by A14 (A1-A35 mapping, not reviewed) | review.tex | The \textit{Communication} subset contains 12,441 training, 800 development and 800 test clips | 12,441, 800, 800 |
| c04-022 | A49 | review.tex | contains 12,441 training, 800 development and 800 test clips, signed by 49, 12 and 9 signers | 12,441, 800, 800, 49, 12, 9 |
| c04-023 | disagrees | review.tex | contains 9,665, 700 and 700 clips, signed by 18, 17 and 17 signers | 9,665, 700, 700, 18, 17, 17 |
| c04-024 | A64 | review.tex | about a third of News clips exceed the 256-frame input limit (dashed line) | 256 |
| c04-025 | disagrees | review.tex | 25,106 clips, all splits pooled | 25,106 |
| c04-037 | disagrees | review.tex | All 25,106 clips (22,106 in the training splits) are annotated with English sentences. | 25,106, 22,106 |
| c04-039 | A60 | review.tex | the released files contain 3,209 segments over 610 glosses | 3,209, 610 |
| c05-005 | A56 | review.tex | The HALPE-136 format distributed with \textsc{Auslan-Daily} has 26 body, 42 hand and 68 face keypoints. | 136, 26, 42, 68 |
| c05-007 | A56 | review.tex | The HALPE-136 format covers 26 body (including the feet), 42 hand and 68 face keypoints; a clip of $T$ frames yields a tensor of shape $(T, 136, 3)$ | 136, 26, 42, 68, 136, 3 |
| c05-028 | disagrees | review.tex | it can use all 22,106 \textsc{Auslan-Daily} training clips | 22,106 |
| c05-034 | not a measurement | review.tex | SignCL added 0.38 on PHOENIX14T (21.97 to 22.35) | 0.38, 4, 21.97, 22.35 |
| c05-049 | covered by A14 (A1-A35 mapping, not reviewed) | review.tex | its Communication subset supplies 12,441 training sentences | 12,441 |
| c05-051 | covered by A14 (A1-A35 mapping, not reviewed) | review.tex | 230 of the 800 test clips (28.75\%) have a reference sentence that occurs verbatim in training | 230, 800, 28.75 |
| c05-053 | A74 | review.tex | The two most frequent signers account for 92\% of the test clips. | 92 |
| c06-001 | A83 | method.tex | BLEU-4~\cite{papineni2002bleu} (Equation~\eqref{eq:bleu4}; 0 to 100, higher is better) | 4, 0, 100 |
| c06-002 | A83 | method.tex | All BLEU scores use sacreBLEU~\cite{post2018sacrebleu} with the \texttt{13a} tokeniser on detokenised text. | 13a |
| c06-003 | A47 | method.tex | averaged over sentences and reported $\times 100$ | 100 |
| c06-004 | covered by A21 (A1-A35 mapping, not reviewed) | method.tex | whitespace tokens, kept punctuation and $\beta = 1.2$ | 1.2 |
| c06-005 | covered by A1 (A1-A35 mapping, not reviewed) | method.tex | Translation WER is reported as a fraction (for example 0.907; lower is better) | 0.907 |
| c06-006 | A83 | method.tex | It exceeds 1.0 whenever insertions outnumber correctly matched reference words. | 1.0 |
| c06-007 | A83 | method.tex | 0 to 100; higher is better & sacreBLEU, \texttt{13a} tokeniser, detokenised text | 0, 100, 3 |
| c06-008 | A83 | method.tex | Whitespace tokens, punctuation kept, $\beta = 1.2$ | 1.2 |
| c06-009 | A57 | method.tex | 0 to 100\%; higher is better & Content words are words outside a list of 71 function words | 0, 100, 71 |
| c06-010 | A75 | method.tex | Share of clips that the system answers when it may decline & 0 to 100\% | 0, 100 |
| c06-011 | A72 | method.tex | The defaults are beam width 5 and length penalty 0.6. | 5, 0.6 |
| c06-012 | A57 | method.tex | A \textit{content word} is a word outside a fixed list of 71 function words. | 71 |
| c06-013 | covered by A3 (A1-A35 mapping, not reviewed) | method.tex | the 230 of the 800 official Communication test clips whose reference occurs verbatim in the training set. The other 570 form the \textit{unseen-target subset}. | 230, 800, 570 |
| c06-014 | A49 | method.tex | 1,069 sign-spotting instances (single occurrences of 120 glosses with start and end frames) | 1,069, 120 |
| c06-015 | A49 | method.tex | 3,209 isolated gloss segments, 2,004 fingerspelling spans (words spelled letter by letter; 1,343 on Communication clips) | 3,209, 2,004, 1,343 |
| c06-016 | A74 | method.tex | 12,441 training, 800 validation and 800 test clips (14,041 in total) | 12,441, 800, 800, 14,041 |
| c06-017 | A59 | method.tex | The DWPose training list contains 12,439 of the 12,441 training clips. | 12,439, 12,441 |
| c06-018 | A56 | method.tex | AlphaPose HALPE-136 layout | 136 |
| c06-019 | A56 | method.tex | In 12,411 of the 14,041 clips (88.4\%) the file lists several people and the program kept a non-signer | 12,411, 14,041, 88.4 |
| c06-020 | no saved output | method.tex | its array matched the arm motion of the tracked DWPose keypoints from the signer video in 2.2\% of those clips, the entry marked as the signer in 95.1\% | 2.2, 95.1 |
| c06-021 | disagrees | method.tex | On 200 audited Communication clips | 200 |
| c06-022 | A56 | method.tex | for both box readings on 200 Communication clips | 200 |
| c06-023 | disagrees | method.tex | \texttt{xyxy} (assumed) & 14.4\% & 87 & $-5{,}036$ / $+12{,}912$ | 14.4, 87, -5,036, 12,912 |
| c06-024 | disagrees | method.tex | \texttt{xywh} (correct) & \textbf{88.9\%} & \textbf{0.60} & $-2.1$ / $+2.6$ | 8.9, 60, -2.1, 2.6 |
| c06-025 | A72 | method.tex | Source sequences are limited to 256 frames by uniform subsampling and targets to 64 tokens | 256, 64 |
| c06-026 | A69 | method.tex | experiments trained on News alone used a 512-frame limit, although validation selected 256 frames | 512, 256 |
| c06-027 | A49 | method.tex | 45,010 front-view clips of all 3,215 glosses, roughly 14 \emph{exemplars} (recorded instances) per gloss | 45,010, 3,215, 14 |
| c06-028 | A59 | method.tex | each an $(x, y, \text{confidence})$ triple, giving a 399-dimensional frame vector | 399 |
| c06-029 | A59 | method.tex | All 14,041 Communication clips were re-extracted with DWPose | 14,041 |
| c06-030 | A54 | method.tex | 59 COCO-WholeBody keypoints for DWPose, the 68-point HALPE subset for AlphaPose | 59, 68 |
| c06-031 | not a measurement | method.tex | matched $2 \times 2$ design | 2, 2 |
| c06-032 | A54 | method.tex | AlphaPose HALPE-136 (as released, first-listed person) | 136 |
| c06-033 | A54 | method.tex | DWPose COCO-WholeBody-133 (re-extracted) & cell C & cell D | 133 |
| c06-034 | A66 | method.tex | temporal resampling by a factor from $[0.8, 1.2]$ with nearest-index frame selection | 0.8, 1.2 |
| c06-035 | A66 | method.tex | in-plane rotation by one angle from $[-10^\circ, +10^\circ]$ | -10, +10 |
| c06-036 | A66 | method.tex | independent $x$ and $y$ scaling by factors from $[0.9, 1.1]$ | 0.9, 1.1 |
| c06-037 | A72 | method.tex | four encoder and four decoder layers | four, four |
| c06-038 | covered by A18 (A1-A35 mapping, not reviewed) | method.tex | approximately 9.0 and 10.9 million parameters respectively | 9.0, 10.9 |
| c06-039 | A72 | method.tex | DWPose (pose estimator) gives 133 keypoints (x, y, confidence) per frame | 133 |
| c06-040 | A72 | method.tex | RGB-fusion models use batch size 32 | 32 |
| c07-001 | A72 | method.tex | 133 COCO-WholeBody / 399 | 133, 399 |
| c07-002 | A72 | method.tex | 256 / 64 | 256, 64 |
| c07-003 | A72 | method.tex | 256 / 8 / 1024 | 256, 8, 1024 |
| c07-004 | A72 | method.tex | 4 / 4 | 4, 4 |
| c07-005 | A72 | method.tex | 0.1 / 0.1 | 0.1, 0.1 |
| c07-006 | A72 | method.tex | label smoothing & 0.1 | 0.1 |
| c07-007 | A72 | method.tex | auxiliary \gls{ctc} weight & 0.3 | 0.3 |
| c07-008 | A72 | method.tex | $5\times10^{-4}$ | 5, 10, -4 |
| c07-009 | A72 | method.tex | $1\times10^{-4}$ / 1.0 | 1, 10, -4, 1.0 |
| c07-010 | A72 | method.tex | cosine / 800 | 800 |
| c07-011 | A72 | method.tex | 64 / 40 | 64, 40 |
| c07-012 | A72 | method.tex | 5 / 0.6 | 5, 0.6 |
| c07-013 | covered by A14 (A1-A35 mapping, not reviewed) | method.tex | produced few distinct sentences for the 800 test clips | 800 |
| c07-014 | A72 | method.tex | \lambda_{\text{CTC}} = 0.3 | 0.3 |
| c07-015 | covered by A17 (A1-A35 mapping, not reviewed) | method.tex | 2,988 distinct words; mean 3.60 per clip | 2,988, 3.60 |
| c07-016 | check not yet built | method.tex | compared at 0.1, 0.3, 0.5 and 0.7 (one run each) | 0.1, 0.3, 0.5, 0.7, one |
| c07-017 | check not yet built | method.tex | BLEU-4 on the 800-clip official Communication test set was 8.83, 8.82, 8.54 and 8.36 respectively | 800, 8.83, 8.82, 8.54, 8.36 |
| c07-018 | check not yet built | method.tex | 0.1 and 0.3 differ by 0.01 | 0.1, 0.3, 0.01 |
| c07-019 | check not yet built | method.tex | the default of 0.3 was kept | 0.3 |
| c07-020 | A69 | method.tex | Pretraining lasted 40, 80, 128 or 256 epochs | 40, 80, 128, 256 |
| c07-021 | A69 | method.tex | re-run with three seeds | three |
| c07-022 | check not yet built | method.tex | three crops (full signer, left hand, right hand) on each of up to 24 frames per clip, as one 384-dimensional vector per crop | three, 24, 384 |
| c07-023 | check not yet built | method.tex | A 2-layer Transformer | 2 |
| c07-024 | check not yet built | method.tex | mean weights from the last five epochs of one run | five |
| c07-025 | A53 | method.tex | The \emph{final ensemble} has six members | six |
| c07-026 | A53 | method.tex | the 80-epoch pre-trained model, two seeds of the 40-epoch pre-trained model and three seeds of the RGB-fusion model | 80, two, 40, three |
| c07-027 | covered by A1 (A1-A35 mapping, not reviewed) | method.tex | decoded with beam width 10 and length penalty 0.3 | 10, 0.3 |
| c07-028 | not a measurement | method.tex | instruction-tuned 3-billion-parameter Qwen2.5 model | 3 |
| c07-029 | needs the cluster | method.tex | yielded 20,608 unique sentences in which every content word has a matching sign, against 6,880 sentences from unconstrained generation | 20,608, 6,880 |
| c07-030 | no saved output | method.tex | The stitched clip at 25\,fps | 25 |
| c07-031 | disagrees | method.tex | One citation clip (mean 79.4 frames) is about as long as a real sentence clip (73.1 frames) | 79.4, 73.1 |
| c07-032 | disagrees | method.tex | Over the 45 glosses present in both datasets, signs in continuous signing were 72\% shorter in seconds than their citation forms | 45, 72 |
| c07-033 | no saved output | method.tex | resamples the 30\,fps source clips to 25\,fps | 30, 25 |
| c07-034 | A48 | method.tex | distribution of the 1,069 sign-spotting instances, 61 of them on validation and test clips (mean 22.49 and median 20 frames at 25\,fps) | 1,069, 61, 22.49, 20, 25 |
| c07-035 | A82 | method.tex | trim 15\% of each isolated clip's frames at each end | 15 |
| c07-036 | no saved output | method.tex | the 6,880 synthetic clips and seed 42 | 6,880, 42 |
| c07-037 | no saved output | method.tex | replicated with three paired seeds | three |
| c07-038 | disagrees | method.tex | lasts 6.4 times as long in citation form (median 3.08\,s, $n=12$) as in conversation (median 0.48\,s, $n=6$) | 6.4, 3.08, 12, 0.48, 6 |
| c07-039 | check not yet built | method.tex | A single corpus of 20,608 duration-modelled clips was built. Its first 6,880, first 13,760 and all 20,608 clips | 20,608, 6,880, 13,760, 20,608 |
| c07-040 | covered by A17 (A1-A35 mapping, not reviewed) | method.tex | nested synthetic-to-real ratios of 0.55:1, 1.1:1 and 1.66:1 against the 12,439 real clips | 0.55:1, 1.1:1, 1.66:1, 12,439 |
| c07-041 | needs the cluster | method.tex | verified on 200 of 200 sampled clips, with 99.9\% of glosses changed | 200, 200, 99.9 |
| c07-042 | A72 | method.tex | The \emph{adopted training configuration} adds 13,760 duration-modelled stitched clips and pose augmentation | 13,760 |
| c07-043 | check not yet built | method.tex | BLEU-4 on the 800-clip official Communication test set (beam width 5) was 9.63 as a three-seed mean and 9.52 for the one run whose outputs were saved | 800, 5, 9.63, three, 9.52, one |
| c07-044 | check not yet built | method.tex | Baseline model & pose & none & none & no & 40 & 5 / 0.6 | 40, 5, 0.6 |
| c07-045 | check not yet built | method.tex | 40-epoch pre-trained model & pose & 40 & none & no & 40 & 5 / 0.6 | 40, 40, 40, 5, 0.6 |
| c07-046 | check not yet built | method.tex | 80-epoch pre-trained model & pose & 80 & none & no & 40 & 5 / 0.6 | 80, 40, 5, 0.6 |
| c07-047 | check not yet built | method.tex | RGB-fusion model & \raggedright pose + RGB & 40 & none & no & 40 & 5 / 0.6 | 40, 40, 5, 0.6 |
| c07-048 | no saved output | method.tex | Baseline model with stitched data & pose & none & \raggedright 6,880 (0.55:1) & no & 40 & 5 / 0.6 | 6,880, 0.55:1, 40, 5, 0.6 |
| c07-049 | check not yet built | method.tex | Adopted training configuration & pose & none & \raggedright 13,760 (1.1:1) & yes & 40 & 5 / 0.6 | 13,760, 1.1:1, 40, 5, 0.6 |
| c07-050 | covered by A1 (A1-A35 mapping, not reviewed) | method.tex | Final ensemble (three pose-only and three RGB-fusion members) & \raggedright pose; pose + RGB & varies & none & no & 40 & 10 / 0.3 | 40, 10, 0.3, three, three |
| c07-051 | check not yet built | method.tex | the 1.1:1 ratio was preferred on test BLEU-4 because validation BLEU-4 did not separate the ratios | 1.1:1 |
| c07-052 | A65 | method.tex | (seeds 42, 101 and 202) | 42, 101, 202 |
| c07-053 | no saved output | method.tex | Three comparisons (scrambled signs, training length, News pretraining) used only seeds 42 and 101 | Three, 42, 101 |
| c07-054 | A65 | method.tex | the change and its control are trained at three seeds | three |
| c07-055 | A65 | method.tex | trained at seeds 303 and 404 | 303, 404 |
| c07-056 | covered by A10 (A1-A35 mapping, not reviewed) | method.tex | the five test differences rejects at the 0.05 level after Holm correction | five, 0.05 |
| c07-057 | A65 | method.tex | Student's $t$ with $n-1$ degrees of freedom | 1 |
| c07-058 | A65 | method.tex | 95\% interval excludes zero | 95, zero |
| c07-059 | covered by A14 (A1-A35 mapping, not reviewed) | method.tex | 28.75\% of test clips (230 of 800) and 22.74\% of distinct test sentences (166 of 730) occur verbatim in training | 28.75, 230, 800, 22.74, 166, 730 |
| c07-060 | A74 | method.tex | all 90 test videos occur in training | 90 |
| c07-061 | A80 | method.tex | (11,341, 1,444 and 1,254 clips from 73, 9 and 8 videos) | 11,341, 1,444, 1,254, 73, 9, 8 |
| c07-062 | no saved output | method.tex | used to retrain the baseline model once and for a pre-registered five-seed comparison | once, five |
| c07-063 | A61 | method.tex | outputs one five-word training sentence for every clip | one, five |
| c07-064 | A65 | method.tex | 2,000 resamples with replacement | 2,000 |
| c07-065 | covered by A5 (A1-A35 mapping, not reviewed) | method.tex | over the 800 test sentences | 800 |
| c07-066 | A65 | method.tex | The 2.5th and 97.5th percentiles of the difference form a 95\% interval | 2.5, 97.5, 95 |
| c07-067 | A83 | method.tex | Three null models are scored for WER on the official test set: an empty output and two constant sentences | Three, two |
| c07-068 | A83 | method.tex | truncated to three words | three |
| c07-069 | A52 | method.tex | Each run used one NVIDIA V100 GPU | one |
| c07-070 | no saved output | method.tex | PyTorch 2.5.1 and CUDA 12 | 2.5.1, 12 |
| c07-071 | needs the cluster | method.tex | a 40-epoch Communication run took roughly 35--45 minutes | 40, 35, 45 |
| c07-072 | A78 | method.tex | Whisper small.en | small.en |
| c07-073 | A75 | method.tex | coverage of 90\%, 75\%, 50\% and 25\% | 90, 75, 50, 25 |
| c07-074 | covered by A30 (A1-A35 mapping, not reviewed) | method.tex | the 800 Communication test clips | 800 |
| c08-001 | A49 | result.tex | 12,441 training clips (12,439 in the list used for DWPose input) and 800 test clips | 12,441, 12,439, 800 |
| c08-002 | A59 | result.tex | 12,439 in the list used for DWPose input | 12,439 |
| c08-003 | covered by A18 (A1-A35 mapping, not reviewed) | result.tex | The translator (Chapter~\ref{ch:method}) has 9.0\,M parameters (10.9\,M with \gls{rgb} fusion). | 9.0, 10.9 |
| c08-004 | A53 | result.tex | averages the log-probabilities of six such translators (about 60\,M parameters in total) | six, 60 |
| c08-007 | covered by A4 (A1-A35 mapping, not reviewed) | result.tex | gives the same \gls{bleu}-4 as sacreBLEU~\cite{post2018sacrebleu} on three saved sets of single-model test outputs (9.52, 10.27 and 10.33) | 3, 9.52, 10.27, 10.33 |
| c08-009 | A51 | result.tex | Three exploratory configurations (\gls{bleu}-4 11.15, 11.51 and 11.70) scored higher on test than the validation-selected configuration | Three, 11.15, 11.51, 11.70 |
| c08-010 | disagrees | result.tex | A 95\% paired-bootstrap interval over the 800 test sentences has a half-width of about 1.2 \gls{bleu}-4 | 95, 800, 1.2 |
| c08-011 | covered by A25 (A1-A35 mapping, not reviewed) | result.tex | a 95\% range of about $\pm0.6$ for the difference between two same-seed runs of the baseline configuration within one job submission and $\pm1.2$ for the adopted configuration at one seed across job submissions | 95, 0.6, 1.2 |
| c08-012 | A56 | result.tex | AlphaPose HALPE-136~\cite{fang2022alphapose} | 136 |
| c08-013 | check not yet built | result.tex | scored \gls{bleu}-4 2.87 on the 800-clip official Communication test set at beam width 5 and 1.76 with greedy decoding (one run) | 2.87, 800, 5, 1.76, one |
| c08-015 | A54 | result.tex | Greedy decoding produced only 53 distinct sentences for the 800 test clips | 53, 800 |
| c08-016 | A71 | result.tex | an exact re-implementation of SL-Transformer (2.08 at beam width 5) | 2.08, 5 |
| c08-017 | check not yet built | result.tex | the highest, 3.87, used vision--language pretraining and beam width 10 | 3.87, 10 |
| c08-018 | A55 | result.tex | The mean confidence of the hand keypoints of the signer entry in the distributed files was 0.12 | 0.12 |
| c08-019 | A59 | result.tex | Re-extracting all 14,041 Communication clips with DWPose | 14,041 |
| c08-020 | A54 | result.tex | raised it to 0.82 | 0.82 |
| c08-021 | needs the cluster | result.tex | scored 8.78 at beam width 5 and 7.92 with greedy decoding, with 309 distinct sentences | 8.78, 5, 7.92, 309 |
| c08-022 | no saved output | result.tex | it scored 9.63 at beam width 5 (mean of three seeds), within the published range | 9.63, 5, three |
| c08-023 | A59 | result.tex | hand keypoints distributed with the dataset (AlphaPose) have a mean confidence of 0.02 to 0.12 and the DWPose re-extraction 0.72 to 0.94 | 0.02, 0.12, 0.72, 0.94 |
| c08-025 | not a measurement | result.tex | dashed lines and open joints: confidence below 0.3 | 0.3 |
| c08-026 | A56 | result.tex | aligned to the crop through the shared body keypoints (median residual 5.3 pixels) | 5.3 |
| c08-027 | not a measurement | result.tex | The matched $2\times2$ comparison | 2, 2 |
| c08-028 | needs the cluster | result.tex | BLEU-4 on the official Communication test set (800 clips), beam width 5, body and hand keypoints only | 800, 5 |
| c08-029 | A55 | result.tex | Lower block: AlphaPose keypoints of the signer, mean of three seeds in one job | three |
| c08-030 | covered by A32 (A1-A35 mapping, not reviewed) | result.tex | Signer-keypoint difference, DWPose minus AlphaPose, on validation: $+0.23$ ($-0.74$ to $+1.21$) | +0.23, -0.74, +1.21 |
| c08-031 | A55 | result.tex | on test clips whose reference is absent from training: $+0.65$ | +0.65 |
| c08-032 | needs the cluster | result.tex | AlphaPose HALPE-136, first-listed person & 4.60 & 5.58 | 136, 4.60, 5.58 |
| c08-033 | needs the cluster | result.tex | DWPose RTMW-DW-x (re-extracted) & 1.45\textsuperscript{$\dagger$} & 10.03 | 1.45, 10.03 |
| c08-034 | A55 | result.tex | AlphaPose HALPE-136, signer keypoints (3 seeds) & --- & 10.08 | 136, 3, 10.08 |
| c08-035 | A55 | result.tex | DWPose RTMW-DW-x (3 seeds, same job) & --- & 10.35 | 3, 10.35 |
| c08-036 | needs the cluster | result.tex | 1 distinct output for the whole test set | 1 |
| c08-037 | covered by A32 (A1-A35 mapping, not reviewed) | result.tex | DWPose scored $+0.27$ \gls{bleu}-4 relative to AlphaPose under frame normalisation; the 95\% interval ($-1.03$ to $+1.57$) includes zero | +0.27, 95, -1.03, +1.57 |
| c08-038 | A55 | result.tex | \textsc{Auslan-Daily} Communication test set (800 clips), beam width 5. Bars: means of three seeds with 95\% $t$-intervals; dots: the three seeds | 800, 5, three, 95, three |
| c08-040 | needs the cluster | result.tex | AlphaPose HALPE-136, first-listed person & 4.60 & 5.58 | 136, 4.60, 5.58 |
| c08-041 | retired (text no longer in the dissertation) | result.tex | DWPose scored \gls{bleu}-4 10.03 and AlphaPose 5.58 on the official test set at beam width 5 ($+4.45$, one run each) | 10.03, 5.58, 5, +4.45, one |
| c08-042 | retired (text no longer in the dissertation) | result.tex | three new seed pairs gave $+4.97$ (DWPose 10.13, 9.96 and 9.72; AlphaPose 5.27, 4.73 and 4.90) | three, +4.97, 10.13, 9.96, 9.72, 5.27, 4.73, 4.90 |
| c08-043 | A55 | result.tex | AlphaPose scored a test mean of 10.08 and DWPose 10.35, a difference of $+0.27$ (95\% $t$-interval $-1.03$ to $+1.57$; DWPose higher in two of three seeds | 10.08, 10.35, +0.27, 95, -1.03, +1.57, two, three |
| c08-044 | A55 | result.tex | AlphaPose scored a test mean of 10.08 and DWPose 10.35 | 10.08, 10.35 |
| c08-045 | needs the cluster | result.tex | the body-and-hands translator scored 10.03 (one run), against 9.63 for all 133 keypoints including the face (mean of three seeds) | 10.03, one, 9.63, 133, three |
| c08-046 | A60 | result.tex | On the same 382 annotated spans of 45 signs (chance 2.2\%) | 382, 45, 2.2 |
| c08-047 | A55 | result.tex | top-1 accuracy from hand keypoints alone was 18.1\% with DWPose and 13.1\% with the signer's AlphaPose keypoints (top-5: 34.7\% and 32.7\% | 18.1, 13.1, 34.7, 32.7 |
| c08-048 | A55 | result.tex | body and hands, top-1: 18.7\% and 11.9\%; exact McNemar test, hands, $p = 0.018$ | 18.7, 11.9, 0.018 |
| c08-049 | A54 | result.tex | first-listed AlphaPose had scored 4.2\% | 4.2 |
| c09-001 | A74 | result.tex | Communication test set (800 clips) | 800 |
| c09-002 | covered by A4 (A1-A35 mapping, not reviewed) | result.tex | \gls{rgb} fusion, for example, moved it from 10.27 to 10.33 | 10.27, 10.33 |
| c09-003 | needs the cluster | result.tex | beam width 5 of 10.97 against 9.49 for the concurrent | 5, 10.97, 9.49 |
| c09-004 | needs the cluster | result.tex | real-clips-only controls, $+1.48$) | +1.48 |
| c09-005 | A65 | result.tex | the final ensemble (11.80 against 10.33 for its \gls{rgb}-fusion member, $+1.47$; 95\% | 11.80, 10.33, +1.47, 95 |
| c09-006 | covered by A5 (A1-A35 mapping, not reviewed) | result.tex | paired-bootstrap interval over the 800 test sentences, $+0.40$ to $+2.56$). | 800, +0.40, +2.56 |
| c09-007 | A74 | result.tex | Official Communication test set (800 clips); one run unless a seed count is given. | 800 |
| c09-008 | A54 | result.tex | First-listed AlphaPose HALPE-136 control      & 1.76  & 2.87   & output collapse: 53 distinct outputs (greedy) | 136, 1.76, 2.87, 53 |
| c09-009 | check not yet built | result.tex | VLP pretraining, tuned decoding  & 2.44  & 3.87   & beam width 10, length penalty 0.9 | 2.44, 3.87, 10, 0.9 |
| c09-010 | needs the cluster | result.tex | DWPose re-extraction             & 7.92  & 8.78   & 309 distinct outputs | 7.92, 8.78, 309 |
| c09-011 | needs the cluster | result.tex | \;\;+ pseudo-gloss \gls{ctc} (3 seeds) & 8.92 & 9.63 & within the published range (9.20--9.95) | 3, 8.92, 9.63, 9.20, 9.95 |
| c09-012 | A65 | result.tex | \;\;+ \gls{vlp} pretraining, 40 epochs & 8.75 & 10.27 | 40, 8.75, 10.27 |
| c09-013 | A65 | result.tex | \;\;+ \gls{rgb} fusion                 & 8.82 & 10.33 & \gls{rgb} fusion adds $+0.06$ | 8.82, 10.33, +0.06 |
| c09-014 | needs the cluster | result.tex | (pose-only; replaces 40 epochs)\textsuperscript{*} & --- & 10.71 & highest of the four pretraining lengths | 40, 10.71, four |
| c09-015 | check not yet built | result.tex | + checkpoint averaging (last 5 epochs)\textsuperscript{*}  & 9.13 & 10.89 | 5, 9.13, 10.89 |
| c09-016 | needs the cluster | result.tex | Duration-modelled synthetic data (0.55:1) & ---   & \textbf{10.97} & 3-seed mean (10.74, 11.08, 11.08) | 0.55:1, 10.97, 3, 10.74, 11.08, 11.08 |
| c09-017 | check not yet built | result.tex | at 1.1:1 synthetic:real (3-seed mean) & --- & 11.14 & highest on test in 3 of 3 seeds | 1.1:1, 3, 11.14, 3, 3 |
| c09-018 | A66 | result.tex | + pose augmentation (3 seeds) & --- & 11.34 & $+0.70$ over its concurrent controls (10.64), 3 of 3 seeds | 3, 11.34, +0.70, 10.64, 3, 3 |
| c09-019 | no saved output | result.tex | 6-member ensemble, beam width 5  & ---   & 11.50 | 6, 5, 11.50 |
| c09-020 | covered by A1 (A1-A35 mapping, not reviewed) | result.tex | Final ensemble: 6 members, beam width 10, length penalty 0.3 & --- & \textbf{11.80} & \gls{wer} 0.907 | 6, 10, 0.3, 11.80, 0.907 |
| c09-022 | no saved output | result.tex | Test \gls{bleu}-4 rose from 2.87 with the keypoints as first extracted (Section~\ref{sec:method-data}) to 11.80 with the final ensemble | 2.87, 11.80 |
| c09-023 | needs the cluster | result.tex | (2.87 to 8.78 with the same architecture) | 2.87, 8.78 |
| c09-024 | A74 | result.tex | 800-clip official Communication test set, beam width 5 unless stated. | 800, 5 |
| c09-026 | needs the cluster | result.tex | Whisker: range of three seeds. | three |
| c09-027 | needs the cluster | result.tex | of 10.27, 10.71, 9.69 and 9.66 after 40, 80, 128 and 256 pretraining epochs | 5, 10.27, 10.71, 9.69, 9.66, 40, 80, 128, 256 |
| c09-028 | no saved output | result.tex | and 80 epochs were chosen with these test scores in view. | 80 |
| c09-029 | A69 | result.tex | A three-seed re-run read on validation alone | three |
| c09-030 | covered by A35 (A1-A35 mapping, not reviewed) | result.tex | selected 40 epochs among the pretraining lengths, but 40 epochs of pretraining scored a mean beam-5 validation \gls{bleu}-4 of 10.19 against 10.45 without pretraining | 40, 40, 5, 4, 10.19, 10.45 |
| c09-031 | covered by A35 (A1-A35 mapping, not reviewed) | result.tex | ($-0.26$; 95\% $t$-interval over the three seeds, $-1.57$ to $+1.06$), and 9.23 against 10.12 on test. | -0.26, 95, -1.57, +1.06, 9.23, 10.12 |
| c09-032 | covered by A35 (A1-A35 mapping, not reviewed) | result.tex | at 40 and 80 epochs they lay above every seed of the re-run (8.82 to 9.56, and 9.54 to 10.21), about one point above its means | 40, 80, 8.82, 9.56, 9.54, 10.21 |
| c09-033 | not a measurement | result.tex | (overlap of sequences of one to four words with the reference; 0 to 100, higher is better) against pretraining epochs | one, four, 0, 100 |
| c09-034 | A69 | result.tex | three seeds per setting (dots) and their mean (line). | three |
| c09-035 | needs the cluster | result.tex | beat the concurrent real-clips-only control in all three paired seeds, on validation and test (test at beam width 5: $+0.71$, $+0.75$ and $+0.23$; mean | three, 5, +0.71, +0.75, +0.23 |
| c09-036 | needs the cluster | result.tex | $+0.56$). The stitched signs were too long: an isolated citation clip averages 79.4 | +0.56, 79.4 |
| c09-037 | A48 | result.tex | frames and a whole real \textsc{Auslan-Daily} sentence averages 73.1. | 73.1 |
| c09-038 | no saved output | result.tex | Two temporal corrections were compared at one seed | Two, one |
| c09-039 | no saved output | result.tex | the synthetic rows add the same 6,880 stitched sentences. | 6,880 |
| c09-040 | no saved output | result.tex | ``Validation'' is greedy BLEU-4; ``Beam 5'' and WER are on the official test set. | 4, 5 |
| c09-041 | needs the cluster | result.tex | Real clips only (control)  & ---   & 9.908  & 9.49  & 1.073 | 9.908, 9.49, 1.073 |
| c09-042 | needs the cluster | result.tex | + synthetic, no resampling   & 246.1 & 9.987  & 10.80 & 1.064 | 246.1, 9.987, 10.80, 1.064 |
| c09-043 | needs the cluster | result.tex | (30 to 25\,fps) & 205.9 & 10.363 & 10.80 & 1.069 | 30, 25, 205.9, 10.363, 10.80, 1.069 |
| c09-044 | needs the cluster | result.tex | \textbf{108.4} & \textbf{10.625} & \textbf{11.08} & \textbf{1.055} | 108.4, 10.625, 11.08, 1.055 |
| c09-045 | A48 | result.tex | \textit{Real \textsc{Auslan-Daily} clips} & \textit{73.1} | 73.1 |
| c09-046 | needs the cluster | result.tex | as the mean synthetic clip length falls from 246 to 108 frames (real clips: 73.1), validation \gls{bleu}-4 rises from 9.99 to 10.62 and test \gls{bleu}-4 from 10.80 to 11.08. | 246, 108, 73.1, 4, 9.99, 10.62, 4, 10.80, 11.08 |
| c09-047 | no saved output | result.tex | One run per point (seed 42); synthetic-data rows of Table~\ref{tab:res_temporal}. | 42 |
| c09-048 | needs the cluster | result.tex | per-sign duration modelling over three paired seeds, test \gls{bleu}-4 at beam width 5 rose | three, 4, 5 |
| c09-049 | needs the cluster | result.tex | from 9.49 to 11.08, from 9.03 to 10.74 and from 9.94 to 11.08 over the concurrent | 9.49, 11.08, 9.03, 10.74, 9.94, 11.08 |
| c09-050 | needs the cluster | result.tex | real-clips-only controls (means 9.49 to 10.97, $+1.48$), and validation \gls{bleu}-4 rose in | 9.49, 10.97, +1.48, 4 |
| c09-051 | needs the cluster | result.tex | all three pairs. This 10.97 is the highest replicated mean | three, 10.97 |
| c09-052 | A66 | result.tex | the higher means at 1.1:1 (11.14, and 11.34 with pose augmentation) | 1.1:1, 11.14, 11.34 |
| c09-053 | needs the cluster | result.tex | signs in each synthetic clip is changed. Over two paired seeds, mean validation \gls{bleu}-4 | two, 4 |
| c09-054 | needs the cluster | result.tex | was 9.38 for the real-clips-only control, 9.70 with correct signs and 9.04 with scrambled | 9.38, 9.70, 9.04 |
| c09-055 | needs the cluster | result.tex | signs (below the control in both seeds). Mean test \gls{bleu}-4 at beam width 5 was 0.69 | 4, 5, 0.69 |
| c09-056 | needs the cluster | result.tex | appears to come from showing the correct sign for each word, not from regularisation (two seeds). | two |
| c09-057 | A39 | result.tex | At two cut depths, five-seed test \gls{bleu}-4 rose by 0.29 ($+0.13$ to $+0.45$) and 0.49 ($-0.03$ to $+1.02$) | two, five, 4, 0.29, +0.13, +0.45, 0.49, -0.03, +1.02 |
| c09-058 | check not yet built | result.tex | Three nested corpora of duration-modelled stitched clips, at synthetic-to-real ratios of | Three |
| c09-059 | check not yet built | result.tex | 0.55:1, 1.1:1 and 1.66:1, were each compared with a concurrent real-clips-only control over | 0.55:1, 1.1:1, 1.66:1 |
| c09-060 | check not yet built | result.tex | three seeds (Table~\ref{tab:res_ratio}, Figure~\ref{fig:res_ratio}). Mean test \gls{bleu}-4 at beam width 5 was 9.70 for | three, 4, 5, 9.70 |
| c09-061 | check not yet built | result.tex | the controls, 10.80 at 0.55:1, 11.14 at 1.1:1 and 10.51 at 1.66:1; 1.1:1 scored highest, and | 10.80, 0.55:1, 11.14, 1.1:1, 10.51, 1.66:1, 1.1:1 |
| c09-062 | check not yet built | result.tex | 1.66:1 below it, in all three seeds. At seed 42 the compute-matched control scored 1.35 below | 1.66:1, three, 42, 1.35 |
| c09-063 | check not yet built | result.tex | the real-clips-only control (9.89), so additional optimisation steps do not explain the gain. | 9.89 |
| c09-064 | check not yet built | result.tex | that the gain does not saturate at 0.55:1, not that 1.1:1 is optimal. | 0.55:1, 1.1:1 |
| c09-065 | check not yet built | result.tex | Official test set (beam width 5), with validation in parentheses; each row is one seed with its own concurrent control. | 5 |
| c09-066 | check not yet built | result.tex | 42   & 9.89 (---)   & 10.90 (10.27) & \textbf{11.52} (10.27) & 10.67 (10.11) | 42, 9.89, 10.90, 10.27, 11.52, 10.27, 10.67, 10.11 |
| c09-067 | check not yet built | result.tex | 101  & 10.10 (---)  & 10.80 (9.64)  & \textbf{11.03} (10.91) & 10.46 (10.35) | 101, 10.10, 10.80, 9.64, 11.03, 10.91, 10.46, 10.35 |
| c09-068 | check not yet built | result.tex | 202  & 9.11 (---)   & 10.70 (11.09) & \textbf{10.87} (9.97)  & 10.41 (10.40) | 202, 9.11, 10.70, 11.09, 10.87, 9.97, 10.41, 10.40 |
| c09-069 | check not yet built | result.tex | Mean & 9.70         & 10.80         & \textbf{11.14}         & 10.51 | 9.70, 10.80, 11.14, 10.51 |
| c09-070 | check not yet built | result.tex | every seed scores highest at a synthetic-to-real ratio of 1.1:1 (mean 11.14), but validation does not separate the ratios (means 10.33, 10.38 and 10.28). | 1.1:1, 11.14, 10.33, 10.38, 10.28 |
| c09-071 | check not yet built | result.tex | Grey lines: one seed each (circle 42, square 101, triangle 202); purple: three-seed mean. | 42, 101, 202, three |
| c09-072 | check not yet built | result.tex | (a) Official Communication test set, beam width 5. (b) Validation set, greedy decoding | 5 |
| c09-073 | check not yet built | result.tex | (real clips repeated to the same number of training steps), seed 42. | 42 |
| c09-074 | A66 | result.tex | corpus with paired seeds and concurrent controls. Test \gls{bleu}-4 at beam width 5 rose in | 4, 5 |
| c09-075 | disagrees | result.tex | all three seeds: from 10.34 to 11.44, from 10.93 to 11.38 and from 10.66 to 11.19 (means 10.64 | three, 10.34, 11.44, 10.93, 11.38, 10.66, 11.19, 10.64 |
| c09-076 | A66 | result.tex | to 11.34, $+0.70$). Validation \gls{bleu}-4 and greedy test \gls{bleu}-4 also rose in all | 11.34, +0.70, 4, 4 |
| c09-077 | A66 | result.tex | three seeds, which met the pre-registered three-seed rule. The 95\% $t$-interval over the three seeds, $-0.18$ to $+1.57$, includes zero | three, three, 95, three, -0.18, +1.57 |
| c09-078 | disagrees | result.tex | Content-word precision rose by 1.6 to 2.3 points in every seed | 1.6, 2.3 |
| c10-001 | check not yet built | result.tex | a learned skeleton-graph encoder scored 8.78 against 9.40 | 8.78, 9.40 |
| c10-002 | A40 | result.tex | a 69-keypoint input scaled per body part scored $+0.94$ on validation over five seeds ($-0.28$ to $+2.15$) | 69, 0.94, -0.28, 2.15, five |
| c10-003 | check not yet built | result.tex | changed test \gls{bleu}-4 by between $-0.68$ and $+0.05$ | -0.68, +0.05 |
| c10-004 | check not yet built | result.tex | 7.66 for mT5 and 8.97 for mBART, against 9.63 in an earlier job submission | 7.66, 8.97, 9.63 |
| c10-005 | check not yet built | result.tex | An encoder pre-trained for isolated sign recognition scored 9.98 | 9.98 |
| c10-006 | check not yet built | result.tex | against 9.63 at beam width 5 but lower with greedy decoding (8.46 against 8.92) | 9.63, 5, 8.46, 8.92 |
| c10-007 | needs the cluster | result.tex | after five-seed comparisons (mean test differences $+0.04$ and $+0.09$, intervals including zero) | five, +0.04, +0.09 |
| c10-008 | A58 | result.tex | Test set: official \textsc{Auslan-Daily} Communication, 800 clips, except News and video-disjoint rows. | 800 |
| c10-009 | A58 | result.tex | with 95\% intervals (seed-level $t$; paired bootstrap for $\dagger$) | 95 |
| c10-010 | A63 | result.tex | nested subsets of 3,141, 6,241 and 12,439 pairs, each with five seeds in one job submission | 3,141, 6,241, 12,439, five |
| c10-011 | A63 | result.tex | Five-seed means at beam width 5 for translators trained without synthetic data; ``Unseen'' is the 570 unseen-target test clips | 5, 570 |
| c10-012 | A63 | result.tex | 25\% of each episode   & 3,141  & 320 & 6.80  & 6.03 & 2.27 & 8.8\% | 25, 3,141, 320, 6.80, 6.03, 2.27, 8.8 |
| c10-013 | A63 | result.tex | 50\% of each episode   & 6,241  & 160 & 7.89  & 7.04 & 3.15 & 11.0\% | 50, 6,241, 160, 7.89, 7.04, 3.15, 11.0 |
| c10-014 | A63 | result.tex | All training pairs     & 12,439 & 80  & 10.51 & 9.90 & 5.00 & 16.5\% | 12,439, 80, 10.51, 9.90, 5.00, 16.5 |
| c10-015 | A63 | result.tex | Whole episodes, 50\%   & 6,241  & 160 & 8.08  & 7.63 & 3.53 & 11.6\% | 50, 6,241, 160, 8.08, 7.63, 3.53, 11.6 |
| c10-016 | A63 | result.tex | one doubling of the training set, from 6,241 to 12,439 | 6,241, 12,439 |
| c10-017 | A63 | result.tex | on the official test set (800 clips, beam width 5) from 7.04 to 9.90 ($+2.86$; 95\% $t$-interval over five seeds, $+1.78$ to $+3.93$) | 800, 5, 7.04, 9.90, +2.86, 95, five, +1.78, +3.93 |
| c10-018 | A63 | result.tex | On the 570 unseen-target test clips it raised mean \gls{bleu}-4 from 3.15 to 5.00 ($+1.84$ before rounding; 95\% $t$-interval over five seeds, $+0.61$ to $+3.08$) | 570, 3.15, 5.00, +1.84, 95, five, +0.61, +3.08 |
| c10-019 | A63 | result.tex | Both differences were positive in all five seeds. | five |
| c10-020 | A63 | result.tex | one doubling, from 6,241 to 12,439 pairs, adds 2.86 on the test set (95\% interval over five seeds $+1.78$ to $+3.93$) and 1.84 on unseen targets ($+0.61$ to $+3.08$) | 6,241, 12,439, 2.86, 95, five, +1.78, +3.93, 1.84, +0.61, +3.08 |
| c10-021 | A63 | result.tex | on the official Communication test set (800 clips) and on its 570 unseen-target clips | 800, 570 |
| c10-022 | A83 | result.tex | (overlap of sequences of one to four words with the reference; 0 to 100, higher is better) | one, four, 0, 100 |
| c10-023 | A57 | result.tex | 786 official Communication test clips (of 800) whose reference contains a content word | 786, 800 |
| c10-024 | A57 | result.tex | References contain 2.55 content words on average. | 2.55 |
| c10-025 | covered by A13 (A1-A35 mapping, not reviewed) | result.tex | The final ensemble produced no correct content word on 66.9\% of these clips and every reference content word on 10.6\%. | 66.9, 10.6 |
| c10-026 | covered by A13 (A1-A35 mapping, not reviewed) | result.tex | content-word recall of the final ensemble is 18.7\% when the words must appear in reference order and 18.8\% when order is ignored | 18.7, 18.8 |
| c10-027 | A57 | result.tex | In nearly half of the 800 test clips (48.8\%) the output is a verbatim training sentence that shares no content word with the reference; 5.2\% are exact matches. | 800, 48.8, 5.2 |
| c10-028 | A57 | result.tex | Content words: all words outside a list of 71 function words. Saved outputs of the 40-epoch pre-trained pose-only model, beam width 5 | 71, 40, 5 |
| c10-029 | A58 | result.tex | translated by the 40-epoch pre-trained pose-only model (beam width 5) | 40, 5 |
| c10-030 | A57 | result.tex | Content-word recall of the 40-epoch pre-trained model (single model, beam width 5) | 40, 5 |
| c10-031 | A57 | result.tex | from 0\% for words seen twice or fewer to 20.8\% above 50 occurrences | 0, twice, 20.8, 50 |
| c10-032 | A57 | result.tex | words seen more than 50 times make up 1,592 of the 2,043 reference content tokens (78\%) | 50, 1,592, 2,043, 78 |
| c10-033 | A57 | result.tex | Of these frequent words, 85\% occur somewhere in the model's outputs, but only 20.8\% are produced for the clip that contains them. | 85, 20.8 |
| c10-034 | covered by A17 (A1-A35 mapping, not reviewed) | result.tex | among 2{,}988 labels at its best frame | 2,988 |
| c10-035 | check not yet built | result.tex | One official \textsc{Auslan-Daily} Communication test clip, 40-epoch pre-trained model. | 40 |
| c10-036 | A57 | result.tex | its content-word recall of 17.1\% is 2.1 times the 8.2\% of a null model that outputs the two most frequent training content words | 17.1, 2.1, 8.2, two |
| c10-037 | A53 | result.tex | An oracle that picks the single best \emph{training} sentence for each clip reaches 81.9\%. | 81.9 |
| c10-038 | A74 | result.tex | Of the model's outputs, 72.6\% are verbatim training sentences. | 72.6 |
| c10-039 | no saved output | result.tex | the sign-identification test identifies 45 glosses with 42.8\% top-1 accuracy in citation form | 45, 42.8 |
| c10-040 | A60 | result.tex | and 18.1\% in conversational signing (chance 2.2\%; | 18.1, 2.2 |
| c10-041 | A60 | result.tex | the 45 shared glosses are an atypical subset | 45 |
| c10-042 | no saved output | result.tex | confidence is the same on correctly and incorrectly identified spans (0.814 against 0.816) | 0.814, 0.816 |
| c10-043 | no saved output | result.tex | Communication hands move at 3.2 times citation speed per frame, and at 2.7 times per second after allowing for the frame rates | 3.2, 2.7 |
| c10-044 | no saved output | result.tex | (MM-WLAuslan 30 frames per second, \textsc{Auslan-Daily} 25) | 30, 25 |
| c10-045 | no saved output | result.tex | about three times faster than in citation form, not about ten. The 9.86 from first-detection keypoints was inflated about threefold | three, ten, 9.86, threefold |
| c10-046 | no saved output | result.tex | its hands move at 1.10 times citation speed (first-detection keypoints, not recomputed) | 1.10 |
| c10-047 | A62 | result.tex | Neither eleven clip-level pose descriptors nor video identity predicts translation quality | eleven |
| c10-048 | A62 | result.tex | each descriptor explains at most about 1\% of variance | 1 |
| c10-049 | A49 | result.tex | extraction is exact to two frames | two |
| c10-050 | A49 | result.tex | about 1\% of clips are affected by overlapping subtitles | 1 |
| c10-051 | A60 | result.tex | over 45 glosses (written sign labels): grey, isolated citation signs (MM-WLAuslan); hatched, 382 conversational | 45, 382 |
| c11-001 | covered by A1 (A1-A35 mapping, not reviewed) | result.tex | final ensemble (six translators, about 60\,M parameters) scored 11.80 \gls{bleu}-4 | 6, 60, 11.80 |
| c11-002 | A53 | result.tex | about 60\,M parameters | 60 |
| c11-004 | A65 | result.tex | (+1.85, or +18.6\% relative; | 1.85, 18.6 |
| c11-008 | needs the cluster | result.tex | duration-modelled synthetic data at 0.55:1 (3-seed mean; best single-model configuration supported by validation) & 10.97 | 0.55:1, 3, 10.97 |
| c11-009 | check not yet built | result.tex | duration-modelled synthetic data at 1.1:1 (3-seed mean; ratio preferred on test, different job submission) & 11.14 | 1.1:1, 3, 11.14 |
| c11-010 | covered by A21 (A1-A35 mapping, not reviewed) | result.tex | \gls{rgb}-fusion model (one run) & 10.33 & 37.08 | 10.33, 37.08, one |
| c11-011 | covered by A1 (A1-A35 mapping, not reviewed) | result.tex | final ensemble (6 members; trained from scratch) & \textbf{11.80} | 6, 11.80 |
| c11-012 | A38 | result.tex | SL-Transformer, authors' code run for this thesis (signer's AlphaPose keypoints; 3-seed mean) & 8.77 | 3, 8.77 |
| c11-013 | A38 | result.tex | GASLT, authors' code run for this thesis (signer's AlphaPose keypoints; 3-seed mean) & 9.16 | 3, 9.16 |
| c11-014 | A68 | result.tex | the pre-trained model model fine-tuned (external pre-training; 5-seed mean) & 18.21 | 5, 18.21 |
| c11-015 | A36 | result.tex | ensemble of five fine-tuned the pre-trained model models (external pre-training) & 20.60 | 20.60, five |
| c11-016 | needs the cluster | result.tex | News training data only (one run) & --- & --- & 1.99 | 1.99 |
| c11-017 | A64 | result.tex | News with stitched synthetic data and pose augmentation (5-seed mean) & --- & --- & 2.14 | 5, 2.14 |
| c11-018 | no saved output | result.tex | 56 points; 200 epochs at a constant learning rate of $5\times10^{-5}$, batch size 64; three seeds | 56, 200, 5, -5, 64, 3 |
| c11-019 | A38 | result.tex | authors' own code scored 8.77 for SL-Transformer (95\% interval 8.25 to 9.29; published 9.20) | 95, 8.77, 8.25, 9.29, 9.20 |
| c11-020 | A38 | result.tex | 9.16 for GASLT (7.36 to 10.96; published 9.95) | 9.16, 7.36, 10.96, 9.95 |
| c11-021 | A38 | result.tex | within the pre-registered tolerance of 1.0 | 1.0 |
| c11-022 | A71 | result.tex | On 53 DWPose points the same code scored 8.63 and 9.58 (five seeds) | 53, 8.63, 9.58, 5 |
| c11-023 | A71 | result.tex | 10.43 for the translator of this thesis trained in the same jobs for 40 epochs | 10.43, 40 |
| c11-024 | A71 | result.tex | ($-1.80$, $-2.27$ to $-1.34$; $-0.85$, $-1.32$ to $-0.39$) | -1.80, -2.27, -1.34, -0.85, -1.32, -0.39 |
| c11-025 | A71 | result.tex | the translator scored 9.91 (three seeds), $+1.48$ ($+0.63$ to $+2.33$) above their SL-Transformer retrained in the same job (8.44) | 9.91, 3, +1.48, +0.63, +2.33, 8.44 |
| c11-026 | A71 | result.tex | not resolvably above their GASLT (9.47; $+0.44$, $-0.16$ to $+1.05$) | 9.47, +0.44, -0.16, +1.05 |
| c11-027 | A71 | result.tex | on its usual input it scored 9.63 at beam width 5, against 10.20 for its own recipe ($-0.57$, $-1.55$ to $+0.42$) | 9.63, 5, 10.20, -0.57, -1.55, +0.42 |
| c11-028 | A65 | result.tex | margin of 11.80 over 9.95 | 11.80, 9.95 |
| c11-029 | A65 | result.tex | the ensemble's own 95\% bootstrap interval over test sentences is 9.87 to 13.89 | 95, 9.87, 13.89 |
| c11-030 | needs the cluster | result.tex | 512-frame input limit scored 1.99 \gls{bleu}-4 at beam width 5 | 512, 1.99, 5 |
| c11-032 | A69 | result.tex | mean beam-5 News validation \gls{bleu}-4 was 2.07 at 256 frames and 2.08 at 512 | 5, 2.07, 256, 2.08, 512 |
| c11-033 | A69 | result.tex | ($+0.01$ for 512; 95\% $t$-interval over the three seeds, $-0.37$ to $+0.39$) | +0.01, 512, 95, 3, -0.37, +0.39 |
| c11-034 | A69 | result.tex | the cheaper 256 is selected | 256 |
| c11-035 | A64 | result.tex | five-seed comparison below used 512 frames in both arms | 5, 512 |
| c11-036 | needs the cluster | result.tex | 95\% of the missed News content words are in the training vocabulary | 95 |
| c11-037 | A64 | result.tex | adding 10,614 conversational stitched clips (1.1:1) and pose augmentation | 10,614, 1.1, 1 |
| c11-038 | A64 | result.tex | raised mean beam-5 \gls{bleu}-4 on the News test set from 1.80 to 2.14 | 5, 1.80, 2.14 |
| c11-039 | A64 | result.tex | 95\% $t$-interval over the five seeds, $+0.09$ to $+0.58$; higher in all five seeds | 95, 5, +0.09, +0.58, 5 |
| c11-040 | A64 | result.tex | The mean of 2.14 is 0.03 above SL-Transformer (2.11) | 2.14, 0.03, 2.11 |
| c11-044 | covered by A17 (A1-A35 mapping, not reviewed) | result.tex | Fine-tuned on the 12,439 Communication training clips | 12,439 |
| c11-045 | A68 | result.tex | it scored a five-seed mean test \gls{bleu}-4 of 18.21 (best published 9.95) | 18.21, 9.95, five |
| c11-046 | A68 | result.tex | against 10.37 for the smaller translator of this thesis trained from scratch on real clips only in the same job | 10.37 |
| c11-047 | A68 | result.tex | ($+7.84$; 95\% $t$-interval over seeds $+7.04$ to $+8.64$; all five seeds positive) | +7.84, 95, +7.04, +8.64, 5 |
| c11-048 | A68 | result.tex | the scores were 13.73 and 5.47 | 13.73, 5.47 |
| c11-049 | A68 | result.tex | one of its five runs produced only 30 distinct outputs | 30, one, five |
| c11-050 | A36 | result.tex | scored 20.60 (best published 9.95) | 20.60, 9.95 |
| c11-051 | A36 | result.tex | $+2.75$ over its members decoded with the same code (mean 17.86; 95\% simultaneous interval $+1.54$ to $+3.95$) | 2.75, 17.86, 95, 1.54, 3.95 |
| c11-052 | A36 | result.tex | $+3.18$ over the member best on validation ($+1.43$ to $+4.92$) | +3.18, +1.43, +4.92 |
| c11-053 | A37 | result.tex | reproduces its output logits to within $2\times10^{-5}$ | 2, -5 |
| c11-054 | A36 | result.tex | three seeds scored 18.27 and their ensemble 19.77 (best published 9.95) | 3, 18.27, 19.77, 9.95 |
| c11-055 | A36 | result.tex | best member (18.69; $+1.07$, $-0.55$ to $+2.69$) | 18.69, +1.07, -0.55, +2.69 |
| c11-056 | A37 | result.tex | lowered validation \gls{bleu}-4 from 17.72 to 16.84 | 17.72, 16.84 |
| c11-057 | no saved output | result.tex | none of the 10,327 checkable YouTube-ASL videos comes from the programme behind \textsc{Auslan-Daily}, but 769 are offline | 10,327, 769 |
| c11-058 | covered by A14 (A1-A35 mapping, not reviewed) | result.tex | 230 of the 800 clips (28.75\%) have a reference that also occurs verbatim in the training set | 230, 800, 28.75 |
| c11-059 | A74 | result.tex | seen-target subset (29.7 to 30.2) is 5.2 to 6.1 times that on the unseen-target subset (4.88 to 5.81) | 29.7, 30.2, 5.2, 6.1, 4.88, 5.81 |
| c11-060 | covered by A3 (A1-A35 mapping, not reviewed) | result.tex | The final ensemble scores 6.60 on the unseen-target subset, against 11.80 on all 800 clips | 6.60, 11.80, 800 |
| c11-061 | A57 | result.tex | exactly on 0 of the 564 unseen-target clips that contain content words, against 19.8\% of seen-target clips | 0, 564, 19.8 |
| c11-062 | covered by A3 (A1-A35 mapping, not reviewed) | result.tex | all 800 clips, the 230 seen-target clips and the 570 unseen-target clips | 800, 230, 570 |
| c11-063 | covered by A4 (A1-A35 mapping, not reviewed) | result.tex | Baseline model (DWPose + \gls{ctc}) & 9.52 & 29.7 & 4.88 | 9.52, 29.7, 4.88 |
| c11-064 | covered by A4 (A1-A35 mapping, not reviewed) | result.tex | 40-epoch pre-trained model & 10.27 & 29.9 & 5.31 | 40, 10.27, 29.9, 5.31 |
| c11-065 | covered by A4 (A1-A35 mapping, not reviewed) | result.tex | \gls{rgb}-fusion model & 10.33 & 30.2 & \textbf{5.81} | 10.33, 30.2, 81 |
| c11-066 | A74 | result.tex | about 30 \gls{bleu}-4 (word overlap with reference, higher better) on sentences seen verbatim in training (light, 230 clips), about 5 on unseen ones (hatched, 570) | 30, 230, 5, 570 |
| c11-067 | A74 | result.tex | Official: all 800 \textsc{Auslan-Daily} Communication test clips; beam width 5 | 800, 5 |
| c11-068 | check not yet built | result.tex | 40 epochs of vision--language pretraining | 40 |
| c11-069 | needs the cluster | result.tex | video-disjoint split (6.39, one run) | 6.39, 1 |
| c11-070 | A80 | result.tex | training set (11,341 clips from 73 videos) and its test set (1,254 clips from 8 videos) | 11,341, 73, 1,254, 8 |
| c11-071 | needs the cluster | result.tex | It scored 6.39 \gls{bleu}-4 on that test set (one run), against 9.63 on the official test set (mean of three seeds) | 6.39, 9.63, one, three |
| c11-072 | needs the cluster | result.tex | which adds the 13,760 duration-modelled stitched | 13,760 |
| c11-073 | A80 | result.tex | raised mean beam-5 test \gls{bleu}-4 from 6.14 to 7.08 ($+0.94$; | 5, 6.14, 7.08, 0.94 |
| c11-074 | covered by A27 (A1-A35 mapping, not reviewed) | result.tex | 95\% $t$-interval over the five paired seeds, $+0.16$ to $+1.72$; higher in all five seeds | 95, 5, +0.16, +1.72, 5 |
| c11-075 | A80 | result.tex | 29.3\% of its test clips have a reference that occurs verbatim in its training text | 29.3 |
| c11-076 | A80 | result.tex | On the 886 test clips whose reference never occurs in that training text | 886 |
| c11-077 | needs the cluster | result.tex | \gls{bleu}-4 from 3.04 to 3.97 ($+0.93$; $t$-interval over five seeds, $+0.41$ to $+1.46$; | 3.04, 3.97, 0.93, 0.41, 1.46, five |
| c11-078 | A80 | result.tex | A six-model ensemble trained on this split scored 7.12 on its test set, against 7.29 for the best single model ($-0.16$; 95\% paired-bootstrap interval, $-0.86$ to $+0.57$) | 7.12, 7.29, -0.16, 95, -0.86, 0.57, six |
| c11-079 | A61 | result.tex | a null model that outputs one five-word training sentence for every clip, chosen against the test references, scores 4.41 \gls{bleu}-4 on the 800-clip official test set: 18 test references are exactly that sentence | 4.41, 800, 18, one, five |
| c11-081 | A61 | result.tex | the same sentence scores 1.09, against 4.88 to 6.60 for the trained systems | 1.09, 4.88, 6.60 |
| c11-083 | covered by A23 (A1-A35 mapping, not reviewed) | result.tex | 14.39 for a presenter, 9.97 for a costumed character and 3.12 for the 61 clips of other signers | 14.39, 9.97, 3.12, 61 |
| c11-084 | A77 | result.tex | All nine test signers appear in training | 9 |
| c11-085 | A77 | result.tex | baseline models trained on the presenter and the costumed character only (three seeds) scored 1.29 \gls{bleu}-4 on the 1,087 clips of the 47 other signers, against 1.83 | 1.29, 1,087, 47, 1.83, three |
| c11-086 | A77 | result.tex | The difference ($+0.55$) | +0.55 |
| c11-087 | covered by A23 (A1-A35 mapping, not reviewed) | result.tex | scores 14.39 \gls{bleu}-4 for the presenter but 3.12 for the 61 clips of other signers | 14.39, 3.12, 61 |
| c11-088 | A65 | result.tex | six averaged translators | 6 |
| c12-001 | A65 | result.tex | paired-bootstrap 95\% intervals for \gls{bleu}-4 differences between systems on the 800-sentence official test set | 95, 4, 800 |
| c12-002 | A53 | result.tex | only the final ensemble's $+1.47$ over its \gls{rgb}-fusion member (11.80 against 10.33) is resolvable | +1.47, 11.80, 10.33 |
| c12-003 | A65 | result.tex | 95\% interval from 2,000 paired resamples of the 800 test sentences | 95, 2,000, 800 |
| c12-004 | A53 | result.tex | Final ensemble (11.80) vs.\ its \gls{rgb}-fusion member (10.33) & $+1.47$ & $[+0.40, +2.56]$ & 99.4\% | 11.80, 10.33, +1.47, +0.40, +2.56, 99.4 |
| c12-005 | A53 | result.tex | Final ensemble (11.80) vs.\ its strongest member (80-epoch pre-trained model, 10.80), same decoder & $+1.00$ & $[-0.13, +2.11]$ & 96.0\% | 11.80, 80, 10.80, +1.00, -0.13, +2.11, 96.0 |
| c12-006 | A65 | result.tex | \gls{rgb}-fusion model (10.33) vs.\ baseline model (9.52) & $+0.81$ & $[-0.34, +1.97]$ & 91.5\% | 10.33, 9.52, +0.81, -0.34, +1.97, 91.5 |
| c12-007 | A65 | result.tex | 40-epoch pre-trained model (10.27) vs.\ baseline model (9.52) & $+0.75$ & $[-0.22, +1.71]$ & 93.2\% | 40, 10.27, 9.52, +0.75, -0.22, +1.71, 93.2 |
| c12-008 | covered by A7 (A1-A35 mapping, not reviewed) | result.tex | Ensemble with stitched-data members (0.55:1; 12.11) vs.\ final ensemble (11.80) & $+0.31$ & $[-0.89, +1.50]$ & 69.3\% | 0.55:1, 12.11, 11.80, +0.31, -0.89, +1.50, 69.3 |
| c12-009 | covered by A8 (A1-A35 mapping, not reviewed) | result.tex | Ensemble with stitched-data members (1.1:1; 12.84) vs.\ final ensemble (11.80) & $+1.03$ & $[-0.10, +2.15]$ & 96.2\% | 1.1:1, 12.84, 11.80, +1.03, -0.10, +2.15, 96.2 |
| c12-010 | covered by A5 (A1-A35 mapping, not reviewed) | result.tex | Only the final ensemble's gain over its \gls{rgb}-fusion member (blue, +1.47 \gls{bleu}-4) has a 95\% interval excluding zero | +1.47, 4, 95 |
| c12-011 | no saved output | result.tex | Stitched-data ensemble: three models trained with stitched synthetic sentences at the given synthetic-to-real ratio plus two \gls{rgb}-fusion models (row 5: six such models at 0.55:1 plus two) | three, two, row 5, six, 0.55:1, two |
| c12-012 | A65 | result.tex | 800 \textsc{Auslan-Daily} Communication test sentences, 2,000 paired resamples. Rows 5 and 7 are not in Table | 800, 2,000, 5, 7 |
| c12-013 | A53 | result.tex | five of the six members scored below the final ensemble with intervals that exclude zero | five, six |
| c12-014 | A53 | result.tex | the $+1.00$ over the strongest member, the 80-epoch pre-trained model (10.80), is not resolvable | +1.00, 80, 10.80 |
| c12-015 | A53 | result.tex | On 20\% of test clips no member produced the ensemble's sentence | 20 |
| c12-016 | A53 | result.tex | above every member's on 36 clips and below all on 13 (sign test, $p = 0.001$) | 36, 13, 0.001 |
| c12-017 | A51 | result.tex | An ablation over all 63 subsets of the six members, decoded on validation | 63, six |
| c12-018 | covered by A34 (A1-A35 mapping, not reviewed) | result.tex | Mean \gls{bleu}-4 rises with each added member, from 10.31 for one to 12.47 for six | 10.31, one, 12.47, six |
| c12-019 | covered by A34 (A1-A35 mapping, not reviewed) | result.tex | A member's Shapley value, its contribution averaged over every order of adding members, lies between 1.87 and 2.39 | 1.87, 2.39 |
| c12-020 | covered by A34 (A1-A35 mapping, not reviewed) | result.tex | Leaving a member out costs 0.19 to 0.69; the interval of this loss excludes zero for two members, for one of them only just (lower bound $+0.02$) | 0.19, 0.69, two, one, +0.02 |
| c12-021 | A51 | result.tex | summed Shapley values 6.51 against 5.96; difference $+0.56$, 95\% interval $-0.57$ to $+1.97$ | 6.51, 5.96, +0.56, 95, -0.57, +1.97 |
| c12-022 | covered by A34 (A1-A35 mapping, not reviewed) | result.tex | One of the 62 smaller subsets scored above the six on validation (12.68) | one, 62, six, 12.68 |
| c12-023 | A44 | result.tex | no power mean of the members' probabilities (exponents $-1$ to 2) was distinguishable from the geometric mean | -1, 2 |
| c12-024 | covered by A34 (A1-A35 mapping, not reviewed) | result.tex | validation BLEU-4 (800 clips, beam width 10, length penalty 0.3) | 63, six, 800, 10, 0.3 |
| c12-025 | covered by A34 (A1-A35 mapping, not reviewed) | result.tex | 95\% intervals from 500 resamples of validation sentences | 95, 500 |
| c12-026 | covered by A34 (A1-A35 mapping, not reviewed) | result.tex | 80-epoch pre-trained & 9.78 & $+0.44$ ($+0.02$, $+0.91$) & $+0.20$ ($-0.08$, $+0.48$) & 1.87 (1.51, 2.29) | 9.78, +0.44, +0.02, +0.91, +0.20, -0.08, +0.48, 1.87, 1.51, 2.29 |
| c12-027 | covered by A34 (A1-A35 mapping, not reviewed) | result.tex | \gls{rgb}-fusion, run 1 & 10.63 & $+0.33$ ($-0.15$, $+0.83$) & $+0.34$ ($-0.01$, $+0.73$) & 2.09 (1.64, 2.58) | 10.63, +0.33, -0.15, +0.83, +0.34, -0.01, +0.73, 2.09, 1.64, 2.58 |
| c12-028 | covered by A34 (A1-A35 mapping, not reviewed) | result.tex | 40-epoch pre-trained, run 1 & 10.69 & $+0.19$ ($-0.04$, $+0.42$) & $+0.52$ ($+0.26$, $+0.88$) & 2.20 (1.82, 2.63) | 10.69, +0.19, -0.04, +0.42, +0.52, +0.26, +0.88, 2.20, 1.82, 2.63 |
| c12-029 | covered by A34 (A1-A35 mapping, not reviewed) | result.tex | 40-epoch pre-trained, run 2 & 10.00 & $+0.35$ ($-0.10$, $+0.82$) & $+0.22$ ($-0.04$, $+0.51$) & 1.89 (1.49, 2.36) | 10.00, +0.35, -0.10, +0.82, +0.22, -0.04, +0.51, 1.89, 1.49, 2.36 |
| c12-030 | covered by A34 (A1-A35 mapping, not reviewed) | result.tex | \gls{rgb}-fusion, run 2 & 10.30 & $+0.36$ ($-0.19$, $+0.98$) & $+0.33$ ($-0.13$, $+0.90$) & 2.04 (1.46, 2.67) | 10.30, +0.36, -0.19, +0.98, +0.33, -0.13, +0.90, 2.04, 1.46, 2.67 |
| c12-031 | covered by A34 (A1-A35 mapping, not reviewed) | result.tex | \gls{rgb}-fusion, run 3 & 10.43 & $+0.69$ ($+0.12$, $+1.36$) & $+0.75$ ($+0.37$, $+1.16$) & 2.39 (1.87, 2.90) | 10.43, +0.69, +0.12, +1.36, +0.75, +0.37, +1.16, 2.39, 1.87, 2.90 |
| c12-032 | covered by A34 (A1-A35 mapping, not reviewed) | result.tex | Three pose-only members together & 11.22 & & & 5.96 (sum) | three, 11.22, 5.96 |
| c12-033 | covered by A34 (A1-A35 mapping, not reviewed) | result.tex | Three \gls{rgb}-fusion members together & 11.42 & & & 6.51 (sum) | three, 11.42, 6.51 |
| c12-034 | covered by A34 (A1-A35 mapping, not reviewed) | result.tex | All six members & \textbf{12.47} & & & 12.47 (sum) | six, 12.47, 12.47 |
| c12-035 | covered by A8 (A1-A35 mapping, not reviewed) | result.tex | Three ensembles scored above the final ensemble's 11.80 \gls{bleu}-4 on the official test set | Three, 11.80, 4 |
| c12-036 | covered by A7 (A1-A35 mapping, not reviewed) | result.tex | members trained on duration-modelled stitched data at 0.55:1, scored 12.11 and was withdrawn | 0.55:1, 12.11 |
| c12-037 | A51 | result.tex | A larger variant scored higher than it on validation (13.97) and lower on test (11.56) | 13.97, 11.56 |
| c12-038 | A51 | result.tex | Across the four ensemble member sets scored on both, validation and test \gls{bleu}-4 were uncorrelated ($r = -0.014$) | four, 4, -0.014 |
| c12-039 | covered by A8 (A1-A35 mapping, not reviewed) | result.tex | The second ensemble, pre-registered, used members trained at 1.1:1 and scored 12.84 (7.44 against 6.60 on the unseen-target subset) | 1.1:1, 12.84, 7.44, 6.60 |
| c12-040 | A51 | result.tex | The third, rebuilt from three models of the adopted configuration and two \gls{rgb}-fusion models, scored 12.73 | three, two, 12.73 |
| c12-041 | A51 | result.tex | ahead on the other half in 91.5\% of 1,000 halvings (mean $+0.71$; $-0.29$ to $+1.74$), below the 95\% fixed in advance | 91.5, 1,000, +0.71, -0.29, +1.74, 95 |
| c12-042 | covered by A1 (A1-A35 mapping, not reviewed) | result.tex | The main result remains 11.80. | 11.80 |
| c12-044 | covered by A20 (A1-A35 mapping, not reviewed) | result.tex | Empty output (all deletions) & 1.0000 & 0.00 | 1.0000, 0.00 |
| c12-045 | A83 | result.tex | Constant output \texttt{"hello ."} & 0.9988 & 0.00 | 0.9988, 0.00 |
| c12-046 | A83 | result.tex | Constant output \texttt{"it is sally ."} & 0.9504 & 0.38 | 0.9504, 0.38 |
| c12-047 | covered by A20 (A1-A35 mapping, not reviewed) | result.tex | \gls{rgb}-fusion model, beam width 5 (best saved single model) & 0.9497 & 10.33 | 5, 0.9497, 10.33 |
| c12-048 | covered by A1 (A1-A35 mapping, not reviewed) | result.tex | Final ensemble (beam width 10, length penalty 0.3) & 0.9069 & 11.80 | 10, 0.3, 0.9069, 11.80 |
| c12-049 | covered by A20 (A1-A35 mapping, not reviewed) | result.tex | \gls{rgb}-fusion model, output truncated to 3 words & 0.8999 & 4.87 | 3, 0.8999, 4.87 |
| c12-050 | A83 | result.tex | Official Communication test set (800 clips) | 800 |
| c12-051 | A83 | result.tex | a constant sentence scores 0.950, the \gls{rgb}-fusion model 0.950 and the final ensemble 0.907, and cutting the \gls{rgb}-fusion output to three words gives 0.900 | 0.950, 0.950, 0.907, three, 0.900 |
| c12-052 | A53 | result.tex | Oracle: the best of six saved outputs per sentence | six |
| c12-053 | A83 | result.tex | the constant output \texttt{"it is sally ."} has a \gls{wer} of 0.9504 and the \gls{rgb}-fusion model has 0.9497, a difference of 0.0007 | 800, 0.9504, 0.9497, 0.0007 |
| c12-054 | A83 | result.tex | their \gls{bleu}-4 scores are 0.38 and 10.33 | 0.38, 10.33 |
| c12-055 | A83 | result.tex | lowers its \gls{wer} from 0.9497 to 0.8999, below the 0.9069 of the final ensemble, while its \gls{bleu}-4 falls from 10.33 to 4.87 | first three, 0.9497, 0.8999, 0.9069, 10.33, 4.87 |
| c12-056 | covered by A20 (A1-A35 mapping, not reviewed) | result.tex | reports \gls{wer} with the null-model floor (0.950) and the empty-output \gls{wer} (1.000) | 0.950, 1.000 |
| c12-057 | A47 | result.tex | the saved outputs of the baseline model score 16.56 ROUGE-L when hypotheses and references are detokenised and 36.22 when they are whitespace-tokenised (both at $\beta = 1.0$) | 800, 16.56, 36.22, 1.0 |
| c12-058 | A47 | result.tex | more than 19 points apart with no change to the model | 19 |
| c12-059 | covered by A21 (A1-A35 mapping, not reviewed) | result.tex | With whitespace tokens and $\beta = 1.2$, the \gls{rgb}-fusion model scores 37.08 ROUGE-L | 1.2, 37.08 |
| c12-062 | A79 | result.tex | All 14,041 Communication clips were re-extracted with a person tracker | 14,041 |
| c12-063 | A79 | result.tex | on at least one frame in 23.8\% of training clips and on 5.7\% of training frames | 23.8, 5.7 |
| c12-064 | A79 | result.tex | Two raters judged 160 frames without knowing which skeleton came from which method | Two, 160 |
| c12-065 | A79 | result.tex | the tracker was on the signer and the original extraction was not in 87 of 96 cases (91\%) | 87, 96, 91 |
| c12-066 | A79 | result.tex | trained on the tracked and on the original keypoints at five seeds | five |
| c12-067 | covered by A28 (A1-A35 mapping, not reviewed) | result.tex | the tracked keypoints scored a mean of $+0.46$ \gls{bleu}-4 higher (95\% $t$-interval over five seeds, $+0.00$ to $+0.91$) | +0.46, 95, five, +0.00, +0.91 |
| c12-068 | disagrees | result.tex | On the 800-clip official test set the mean difference was $-0.06$ ($-0.66$ to $+0.55$), and tracking scored lower in one of the two stage-2 seeds | 800, -0.06, -0.66, +0.55, one, two, 2 |
| c12-069 | A79 | result.tex | Because 77\% of evaluation clips contain no switched frame | 77 |
| c12-070 | A79 | result.tex | A second pre-registered test used a later tracker, which chooses the signer over the whole clip, and five new seeds | second, five |
| c12-071 | A79 | result.tex | On the 64 test clips with more than 30\% switched frames, tracking raised \gls{bleu}-4 by 3.52 more than on clips without switches | 64, 30, 3.52 |
| c12-072 | A79 | result.tex | 95\% $t$-interval over seeds $+1.44$ to $+5.61$, all five seeds positive; validation $+4.74$ | 95, +1.44, +5.61, five, +4.74 |
| c12-073 | A79 | result.tex | the whole test set did not change resolvably ($+0.80$, $-0.54$ to $+2.13$) | +0.80, -0.54, +2.13 |
| c13-001 | A75 | result.tex | scores 17.42 \gls{bleu}-4 on the 800-clip official Communication test set | 17.42, 800 |
| c13-002 | A75 | result.tex | (16.13 with the cross-entropy term alone) | 16.13 |
| c13-003 | A75 | result.tex | against 9.52 for all 800 clips | 9.52, 800 |
| c13-004 | A75 | result.tex | at beam width 5 | 5 |
| c13-005 | A75 | result.tex | on the validation set the scores are 18.42 and 9.64 | 18.42, 9.64 |
| c13-006 | A75 | result.tex | A control that answers the clips with the longest outputs instead scores only 10.48 at 50\% coverage | 10.48, 50 |
| c13-007 | A75 | result.tex | The rise also holds on the 570 unseen-target test clips, whose more confident half scores 7.92 against 4.88 for all 570 | 570, 7.92, 4.88, 570 |
| c13-008 | A75 | result.tex | from 9.52 to 17.42 at 50\% coverage | 9.52, 17.42, 50 |
| c13-009 | covered by A14 (A1-A35 mapping, not reviewed) | result.tex | \textsc{Auslan-Daily} Communication test (800 clips) | 800 |
| c13-010 | covered by A3 (A1-A35 mapping, not reviewed) | result.tex | Unseen-target clips (570 in test) | 570 |
| c13-011 | covered by A30 (A1-A35 mapping, not reviewed) | result.tex | the ensemble answers 51.9\% of the 800 official test clips and scores 19.07 \gls{bleu}-4 on them, against 11.80 on all 800 | 51.9, 800, 19.07, 11.80, 800 |
| c13-012 | A75 | result.tex | validation threshold for 50\% coverage, the ensemble answers 51.9\% of the 800 official test | 50, 51.9, 800 |
| c13-013 | covered by A30 (A1-A35 mapping, not reviewed) | result.tex | On the 570 unseen-target clips it answers 55.1\% and scores 8.95, against 6.60 on all 570 | 570, 55.1, 8.95, 6.60, 570 |
| c13-014 | covered by A30 (A1-A35 mapping, not reviewed) | result.tex | \textbf{Target coverage} & \textbf{100\%} & \textbf{90\%} & \textbf{75\%} & \textbf{50\%} | 100, 90, 75, 50 |
| c13-015 | covered by A30 (A1-A35 mapping, not reviewed) | result.tex | Test, all targets     & 11.80 & 12.90 (89\%) & 14.61 (76\%) & \textbf{19.07} (52\%) | 11.80, 12.90, 89, 14.61, 76, 19.07, 52 |
| c13-016 | covered by A30 (A1-A35 mapping, not reviewed) | result.tex | Test, unseen targets  & 6.60  & 7.18 (89\%)  & 7.71 (77\%)  & \textbf{8.95} (55\%) | 6.60, 7.18, 89, 7.71, 77, 8.95, 55 |
| c13-017 | covered by A30 (A1-A35 mapping, not reviewed) | result.tex | Test, random abstention & 11.80 & 11.83 & 11.70 & 11.78 | 11.80, 11.83, 11.70, 11.78 |
| c13-018 | A75 | result.tex | at 25\% coverage, 98\% of the answered outputs of the baseline model are verbatim training sentences | 25, 98 |
| c13-019 | A53 | result.tex | both perform poorly on the same clips ($r = 0.78$) | 0.78 |
| c13-020 | needs the cluster | result.tex | runs at 21.7\,frames per second on one V100 GPU, against 25 frames per second in the source video | 21.7, 25 |
| c13-021 | A52 | result.tex | On a laptop CPU (Apple M3 Max) it runs at 5.2\,frames per second with the detector run on every frame | 5.2 |
| c13-022 | A52 | result.tex | the pose model alone runs at 25.4\,frames per second on the same CPU | 25.4 |
| c13-023 | A52 | result.tex | decodes one clip in 36\,ms at beam width 5 | 36, 5 |
| c13-024 | A52 | result.tex | the six-member ensemble at beam width 10 takes a median of 269\,ms per clip | 6, 10, 269 |
| c13-025 | A52 | result.tex | so 5.2 frames per second applies to it | 5.2 |
| c13-026 | A52 | result.tex | six models at beam width 10 reach 12.47 validation \gls{bleu}-4 in a median 269\,ms per clip | 10, 12.47, 269 |
| c13-027 | covered by A14 (A1-A35 mapping, not reviewed) | result.tex | 800 \textsc{Auslan-Daily} Communication validation clips | 800 |
| c13-028 | A46 | result.tex | over all 800 official Communication test clips, with the 40-epoch pre-trained pose-only model at beam width 5 | 800, 40, 5 |
| c13-029 | A46 | result.tex | estimated from each clip's length at 25.4 frames per second | 25.4 |
| c13-030 | A46 | result.tex | start of speech is 2.75\,s and the 95th percentile is 5.27\,s | 2.75, 95, 5.27 |
| c13-031 | A46 | result.tex | Pose estimation contributes 71.7\% of the median delay, speech synthesis 24.1\% and translation 3.3\% | 71.7, 24.1, 3.3 |
| c13-032 | A46 | result.tex | With the detector run on every frame (5.2 frames per second), pose estimation alone would take about 9.6\,s for the median clip | 5.2, 9.6 |
| c13-033 | A46 | result.tex | Time per utterance over the 800 official test clips on a laptop CPU; pose estimation is estimated from clip length at 25.4 frames per second | 800, 25.4 |
| c13-034 | A46 | result.tex | the batched beam search of Section~\ref{sec:res_throughput} takes 36\,ms | 36 |
| c13-035 | A46 | result.tex | Pose estimation (estimated) & 1968.5 & 4448.8 & 71.7\% | 1968.5, 4448.8, 71.7 |
| c13-036 | A46 | result.tex | Translation (beam width 5)  &   91.8 &  217.7 &  3.3\% | 5, 91.8, 217.7, 3.3 |
| c13-037 | A46 | result.tex | Speech synthesis         &  660.8 &  697.1 & 24.1\% | 660.8, 697.1, 24.1 |
| c13-038 | A46 | result.tex | Translation + synthesis  &  752.6 &  901.5 & 27.4\% | 752.6, 901.5, 27.4 |
| c13-039 | A46 | result.tex | \textbf{Whole chain}     & \textbf{2746.7} & \textbf{5271.0} & 100\% | 2746.7, 5271.0, 100 |
| c13-040 | A46 | result.tex | Pose estimation takes 71.7\% of the median 2.75\,s from end of signing to speech; translation takes 92\,ms with this benchmark's own beam search, which calls the decoder once per hypothesis at each step (36\,ms with the batched search | 71.7, 2.75, 92, 36 |
| c13-041 | covered by A14 (A1-A35 mapping, not reviewed) | result.tex | 800 \textsc{Auslan-Daily} Communication test clips | 800 |
| c13-042 | A46 | result.tex | this model's voiced output scores 19.19 \gls{bleu}-4 at 50\% coverage, against 10.27 when all 800 clips are voiced | 19.19, 50, 10.27, 800 |
| c13-043 | A45 | result.tex | the 847 such crops from the test videos | 847 |
| c13-044 | A45 | result.tex | At the threshold fitted for 50\% coverage, the system voiced 53.5\% of the test-set signers and 53.9\% of the non-signers, a ratio of 1.01 | 50, 53.5, 53.9, 1.01 |
| c13-045 | A45 | result.tex | (95\% confidence interval 0.92 to 1.10, clustered by video | 95, 0.92, 1.10 |
| c13-046 | A45 | result.tex | area under the receiver operating characteristic curve is 0.51, where 0.50 is chance | 0.51, 0.50 |
| c13-047 | A45 | result.tex | the threshold voiced 51.9\% of the signers and 43.8\% of the non-signers, a ratio of 0.84 (0.75 to 0.94; area under the curve 0.56) | 51.9, 43.8, 0.84, 0.75, 0.94, 0.56 |
| c13-048 | A45 | result.tex | more than four non-signing people in ten are still voiced | 4, 10 |
| c13-049 | A45 | result.tex | 53.9\% against 53.5\% (ratio 1.01, 95\% interval 0.92 to 1.10); the final ensemble's voices 43.8\% against 51.9\% (ratio 0.84, 0.75 to 0.94 | 53.9, 53.5, 1.01, 95, 0.92, 1.10, 43.8, 51.9, 0.84, 0.75, 0.94 |
| c13-050 | A45 | result.tex | threshold fitted on validation for 50\% coverage | 50 |
| c13-051 | A45 | result.tex | the 847 such crops from the test videos were processed | 847 |
| c13-052 | covered by A1 (A1-A35 mapping, not reviewed) | result.tex | scores 11.80 \gls{bleu}-4 and a \gls{wer} of 0.907 on the 800-clip official \textsc{Auslan-Daily} Communication test set | 11.80, 0.907, 800 |
| c13-053 | A65 | result.tex | 1.85 \gls{bleu}-4 above the strongest published pose-input baseline (GASLT, 9.95) | 1.85, 9.95 |
| c13-055 | A68 | result.tex | scores 18.21, or 20.60 as an ensemble | 18.21, 20.60 |
| c14-002 | covered by A32 (A1-A35 mapping, not reviewed) | discussion.tex | DWPose against AlphaPose on the signer's keypoints, mean $+0.27$ over three seeds ($-1.03$ to $+1.57$) | +0.27, 3, -1.03, +1.57 |
| c14-003 | A63 | discussion.tex | doubled real pairs +2.86 | +2.86 |
| c14-004 | needs the cluster | discussion.tex | stitched synthetic data 9.49 to 10.97 (+1.48) | 9.49, 10.97, +1.48 |
| c14-005 | A65 | discussion.tex | ensemble 11.80, +1.47 over \gls{rgb}-fusion member | 11.80, +1.47 |
| c14-006 | covered by A13 (A1-A35 mapping, not reviewed) | discussion.tex | No correct content word for 66.9\% of 786 clips | 66.9, 786 |
| c14-007 | covered by A13 (A1-A35 mapping, not reviewed) | discussion.tex | content-word recall 18.7\% in-order against 18.8\% order-free | 18.7, 18.8 |
| c14-008 | no saved output | discussion.tex | 45-sign identification 42.8\% citation against 18.1\% conversational (chance 2.2\%) | 45, 42.8, 18.1, 2.2 |
| c14-009 | A83 | discussion.tex | a difference of about one \gls{bleu}-4 point lies within the variation of the measurement | one |
| c14-010 | A74 | discussion.tex | 230 of 800 test references occur verbatim in training | 230, 800 |
| c14-011 | A61 | discussion.tex | constant sentence scored 4.41, 1.09 on unseen targets | 4.41, 1.09 |
| c14-012 | covered by A5 (A1-A35 mapping, not reviewed) | discussion.tex | only the ensemble's +1.47 is resolvable by paired bootstrap | +1.47 |
| c14-013 | covered by A27 (A1-A35 mapping, not reviewed) | discussion.tex | Pose estimation 5.2 frames per second (detector every frame) | 5.2 |
| c14-014 | A46 | discussion.tex | 71.7\% of 2.75 s median delay (translation 3.3\%) | 71.7, 2.75, 3.3 |
| c14-015 | covered by A30 (A1-A35 mapping, not reviewed) | discussion.tex | ensemble scored 19.07 on 51.9\% answered | 19.07, 51.9 |
| c14-016 | A45 | discussion.tex | threshold voiced 53.9\% non-signers against 53.5\% signers | 53.9, 53.5 |
| c14-017 | retired (text no longer in the dissertation) | discussion.tex | no change to the translator's architecture, training objective or decoder approached the effect of using the signer's keypoints |  |
| c14-018 | A55 | discussion.tex | DWPose scored $+0.27$ above AlphaPose (10.35 against 10.08, three seeds; $-1.03$ to $+1.57$ | +0.27, 10.35, 10.08, 3, -1.03, +1.57 |
| c14-019 | retired (text no longer in the dissertation) | discussion.tex | the wrong person in most clips cost about 5 \gls{bleu}-4 | 5 |
| c14-020 | disagrees | discussion.tex | produced no resolvable change in test \gls{bleu}-4 ($-0.06$; 95\% $t$-interval over five seeds, $-0.66$ to $+0.55$) | -0.06, 95, -0.66, 0.55, five |
| c14-021 | A55 | discussion.tex | Two estimators and three seeds neither rank estimators nor exclude a difference of about one point | 2, 3, one |
| c14-022 | needs the cluster | discussion.tex | raised the three-seed mean from 9.49 for the concurrent real-clips-only controls to 10.97 ($+1.48$; higher in all three seeds) | 3, 9.49, 10.97, +1.48, 3 |
| c14-023 | needs the cluster | discussion.tex | mean validation \gls{bleu}-4 over two paired seeds was 9.04, against 9.38 for the control and 9.70 with correct signs | 2, 9.04, 9.38, 9.70 |
| c14-024 | needs the cluster | discussion.tex | Over these two seeds the stitched data helped only when showing the correct sign | 2 |
| c14-025 | covered by A25 (A1-A35 mapping, not reviewed) | discussion.tex | Pose augmentation beat its concurrent controls in all three seeds (mean 11.34 against 10.64, $+0.70$) | 3, 11.34, 10.64, +0.70 |
| c14-026 | A66 | discussion.tex | the 95\% $t$-interval over the seeds includes zero ($-0.18$ to $+1.57$) | 95, -0.18, +1.57 |
| c14-027 | A63 | discussion.tex | doubling the real training pairs ($+2.86$) | +2.86 |
| c14-028 | covered by A4 (A1-A35 mapping, not reviewed) | discussion.tex | \gls{rgb} fusion (10.27 to 10.33) | 10.27, 10.33 |
| c14-029 | A73 | discussion.tex | neither rank estimators nor exclude a difference of about one point | one |
| c14-030 | covered by A1 (A1-A35 mapping, not reviewed) | discussion.tex | The final ensemble of six translators scored 11.80 | 6, 11.80 |
| c14-031 | A65 | discussion.tex | 1.47 above its RGB-fusion member (10.33; 95\% paired-bootstrap interval over the 800 test sentences, $+0.40$ to $+2.56$) | 1.47, 10.33, 95, 800, +0.40, +2.56 |
| c14-032 | covered by A1 (A1-A35 mapping, not reviewed) | discussion.tex | 1.00 above its strongest member (not resolvable) | 1.00 |
| c14-033 | A65 | discussion.tex | 1.85 above GASLT | 1.85 |
| c14-034 | covered by A1 (A1-A35 mapping, not reviewed) | discussion.tex | Other member sets scored between 11.54 and 12.84 on the same test set | 11.54, 12.84 |
| c14-035 | A80 | discussion.tex | a six-model ensemble did not exceed its best single model (7.12 against 7.29) | 6, 7.12, 7.29 |
| c14-036 | covered by A13 (A1-A35 mapping, not reviewed) | discussion.tex | no correct content word for 66.9\% of the 786 test clips | 66.9, 786 |
| c14-037 | covered by A13 (A1-A35 mapping, not reviewed) | discussion.tex | 18.7\% when words had to appear in reference order and 18.8\% when order was ignored | 18.7, 18.8 |
| c14-038 | A57 | discussion.tex | words seen more than 50 times in training are 78\% of the reference content tokens | 50, 78 |
| c14-039 | A57 | discussion.tex | 85\% occur somewhere in the outputs of the 40-epoch pre-trained model, but for the right clip only 20.8\% of the time | 85, 40, 20.8 |
| c14-040 | check not yet built | discussion.tex | scored 7.66 (mT5) and 8.97 (mBART), against 9.63 for the baseline | 7.66, 8.97, 9.63 |
| c14-041 | covered by A28 (A1-A35 mapping, not reviewed) | discussion.tex | ($-0.06$; 95\% $t$-interval over five seeds, $-0.66$ to $+0.55$) | -0.06, 95, 5, -0.66, +0.55 |
| c14-042 | disagrees | discussion.tex | same on correctly and incorrectly identified spans (0.814 against 0.816) | 0.814, 0.816 |
| c14-043 | no saved output | discussion.tex | Hand keypoints identified 45 signs with 42.8\% top-1 accuracy in citation form (signs produced alone, as in a dictionary) and 18.1\% in conversational signing (chance 2.2\%) | 45, 42.8, 1, 18.1, 2.2 |
| c14-044 | no saved output | discussion.tex | hands move at 3.2 times citation speed per frame (2.7 times per second) | 3.2, 2.7 |
| c14-045 | no saved output | discussion.tex | not the 9.86 times obtained from first-detection keypoints | 9.86 |
| c14-046 | covered by A1 (A1-A35 mapping, not reviewed) | discussion.tex | they move at 1.10 times citation speed | 1.10 |
| c14-047 | A60 | discussion.tex | the 45 shared signs are atypical | 45 |
| c14-048 | needs the cluster | discussion.tex | joint training with the News subset scored 8.89, against 10.27 for Communication alone ($-1.38$; one run each | 8.89, 10.27, -1.38, 1 |
| c14-049 | A63 | discussion.tex | doubling the real training pairs from 6,241 to 12,439 raised mean \gls{bleu}-4 on the 570 unseen-target test clips from 3.15 to 5.00 ($+1.84$; 95\% $t$-interval over five seeds, $+0.61$ to $+3.08$ | 6,241, 12,439, 570, 3.15, 5.00, +1.84, 95, 5, +0.61, +3.08 |
| c14-050 | A36 | discussion.tex | fine-tuned ensemble scored 16.39 against 6.60 for the 11.80 ensemble | 570, 16.39, 6.60, 11.80 |
| c14-052 | A60 | discussion.tex | citation-form recordings identify conversational signs at 4.5\% top-1, against 19.9\% from isolated clips | 4.5, 19.9 |
| c14-070 | A60 | discussion.tex | 337 conversational spans of 45 signs; 630 MM-WLAuslan citation recordings (14 per sign); 379 isolated Auslan-Daily clips of these signs (mean 7.5 same-sign references per query) | 337, 45, 630, 14, 379, 7.5 |
| c14-071 | A60 | discussion.tex | 45 signs (chance 2.2\%) | 45, 2.2 |
| c14-072 | A60 | discussion.tex | Top-1: spans 18.1\%, isolated clips 19.9\%, citation recordings 4.5\% | 18.1, 19.9, 4.5 |
| c14-073 | A73 | discussion.tex | a difference of about one \gls{bleu}-4 point lies within the variation of the measurement | one |
| c14-074 | A74 | discussion.tex | 230 of the 800 test references occur verbatim in training | 230, 800 |
| c14-075 | A74 | discussion.tex | Three single models scored 29.7 to 30.2 on these seen-target clips and 4.88 to 5.81 on the other 570 | 3, 29.7, 30.2, 4.88, 5.81, 570 |
| c14-076 | covered by A1 (A1-A35 mapping, not reviewed) | discussion.tex | The final ensemble scored 6.60 on the unseen-target clips | 6.60 |
| c14-077 | A61 | discussion.tex | A null model, whose output does not depend on the video, scored 4.41 with one constant sentence, almost half of each published baseline; on the unseen-target clips it scored 1.09 | 4.41, 1.09, one |
| c14-078 | needs the cluster | discussion.tex | raised the five-seed mean on the 886 test clips whose reference never occurs in training from 3.04 to 3.97 ($+0.93$; 95\% $t$-interval over five seeds, $+0.41$ to $+1.46$) | 5, 886, 3.04, 3.97, +0.93, 95, 5, +0.41, +1.46 |
| c14-079 | A83 | discussion.tex | an empty output scores 1.000, the null-model floor (the lowest null-model \gls{wer}) is 0.950 and the final ensemble scores 0.907 | 1.000, 0.950, 0.907 |
| c14-080 | covered by A20 (A1-A35 mapping, not reviewed) | discussion.tex | Truncating the RGB-fusion outputs to three words lowered \gls{wer} from 0.9497 to 0.8999, while \gls{bleu}-4 fell from 10.33 to 4.87 | 3, 0.9497, 0.8999, 10.33, 4.87 |
| c14-081 | A47 | discussion.tex | ROUGE-L of one set of outputs is 16.56 or 36.22 | 16.56, 36.22 |
| c14-082 | covered by A21 (A1-A35 mapping, not reviewed) | discussion.tex | the RGB-fusion model (37.08) is level with them | 37.08 |
| c14-083 | covered by A5 (A1-A35 mapping, not reviewed) | discussion.tex | only the final ensemble's $+1.47$ over its RGB-fusion member is resolvable, meaning its 95\% paired-bootstrap interval excludes zero | 1.47, 95 |
| c14-084 | A51 | discussion.tex | Three ensembles that scored above 11.80 (12.11, 12.84 and 12.73) | 3, 11.80, 12.11, 12.84, 12.73 |
| c14-085 | A58 | discussion.tex | The pose-estimation stage, not the translator, limits speed. |  |
| c14-086 | A52 | discussion.tex | 5.2 frames per second with the person detector on every frame and at 25.4 with the detector run once per clip, against 25 in the source video | 5.2, 25.4, 25 |
| c14-087 | A46 | discussion.tex | The median delay from the end of signing to the start of speech over the 800 test clips was 2.75\,s | 800, 2.75 |
| c14-088 | A46 | discussion.tex | at 25.4 frames per second, is 71.7\% of this delay and translation time 3.3\% | 25.4, 71.7, 3.3 |
| c14-089 | A52 | discussion.tex | At 5.2 frames per second, pose estimation alone would take about 9.6\,s for the median clip | 5.2, 9.6 |
| c14-090 | covered by A30 (A1-A35 mapping, not reviewed) | discussion.tex | answered 51.9\% of the 800 test clips and scored 19.07 on them, against 11.80 on all clips | 51.9, 800, 19.07, 11.80 |
| c14-091 | covered by A30 (A1-A35 mapping, not reviewed) | discussion.tex | On unseen-target clips it answered 55.1\% and scored 8.95, against 6.60 | 55.1, 8.95, 6.60 |
| c14-092 | A75 | discussion.tex | At 50\% coverage (the proportion of clips answered) the signer must repeat one utterance in two | 50, one, two |
| c14-093 | A45 | discussion.tex | voiced 53.9\% of non-signing people and 53.5\% of signers (ratio 1.01; 95\% confidence interval 0.92 to 1.10, clustered by video | 50, 53.9, 53.5, 1.01, 95, 0.92, 1.10 |
| c14-094 | A45 | discussion.tex | The final ensemble's threshold voiced 43.8\% and 51.9\% (ratio 0.84; 0.75 to 0.94) | 43.8, 51.9, 0.84, 0.75, 0.94 |
| c14-095 | A78 | discussion.tex | round-trip \gls{wer} 2.8\% and 3.0\%, against 3.8\% | 2.8, 3.0, 3.8 |
| c14-097 | A77 | discussion.tex | all nine test signers appear in training | 9 |
| c14-098 | covered by A23 (A1-A35 mapping, not reviewed) | discussion.tex | The final ensemble scored 3.12 on the 61 test clips of other signers, against 14.39 and 9.97 for the two principal signers | 3.12, 61, 14.39, 9.97 |
| c14-099 | check not yet built | discussion.tex | The 10.71 of the 80-epoch pre-trained model was not reproduced in two later runs (9.27 and 9.55) or in a three-seed re-run | 10.71, 80, 2, 9.27, 9.55, 3 |
| c14-100 | no saved output | discussion.tex | three choices (the synthetic-to-real ratio, the \gls{ctc} weight and the training epochs) rest on test scores | three |
| c14-101 | A53 | discussion.tex | members pre-trained for 40 and 80 epochs | 40, 80 |
| c14-102 | covered by A27 (A1-A35 mapping, not reviewed) | discussion.tex | on which the adopted configuration still gained $+0.94$ | +0.94 |
| c14-103 | covered by A1 (A1-A35 mapping, not reviewed) | discussion.tex | the authors' code came within 1.0 of it on the signer's keypoints | 1.0 |
| c14-104 | A64 | discussion.tex | raised mean test \gls{bleu}-4 from 1.80 to 2.14, 0.03 above SL-Transformer (2.11) and below GASLT (2.72) and SL-Luong (2.81) | 1.80, 2.14, 0.03, 2.11, 2.72, 2.81 |
| c15-003 | A56 | appendix_supp.tex | AlphaPose's 136-keypoint HALPE models, whose layout \textsc{Auslan-Daily} uses | 136 |
| c15-009 | A49 | appendix_supp.tex | Sign-spotting instances & 1,069 occurrences | 1,069 |
| c15-010 | A49 | appendix_supp.tex | Isolated gloss segments & 3,209 (610 glosses) | 3,209, 610 |
| c15-011 | A49 | appendix_supp.tex | Fingerspelling spans & 1,343 spans | 1,343 |
| c15-012 | A49 | appendix_supp.tex | Subtitle alignment file & timings, 14,041 rows | 14,041 |
| c15-013 | A49 | appendix_supp.tex | English sentence per clip & 14,041 clips | 14,041 |
| c15-014 | needs the cluster | appendix_supp.tex | \textsc{Auslan-Daily} is 25\,fps and MM-WLAuslan is 30\,fps | 25, 30 |
| c15-015 | needs the cluster | appendix_supp.tex | an error that alone lengthened every synthetic clip $1.2\times$ | 1.2 |
| c15-016 | A82 | appendix_supp.tex | All three synthetic rows remove 15\% of the frames at each end of every citation clip | 15 |
| c15-017 | A50 | appendix_supp.tex | Real \textsc{Auslan-Daily} clip (a whole sentence, 25\,fps) & 73.1 | 25, 73.1 |
| c15-018 | A50 | appendix_supp.tex | Real continuous sign, per sign (25\,fps) & 22.5 | 25, 22.5 |
| c15-019 | disagrees | appendix_supp.tex | MM-WLAuslan citation clip, one sign (30\,fps) & 79.4 | 30, 79.4 |
| c15-020 | needs the cluster | appendix_supp.tex | Synthetic clip, no rate or duration correction & 246.1 | 246.1 |
| c15-021 | needs the cluster | appendix_supp.tex | Synthetic clip, frame-rate corrected only & 205.9 | 205.9 |
| c15-022 | needs the cluster | appendix_supp.tex | Synthetic clip, duration-modelled & \textbf{108.4} | 108.4 |
| c16-001 | check not yet built | appendix_supp.tex | Official Communication test set (800 clips), beam width 5, one run per point | 800, 5 |
| c16-002 | check not yet built | appendix_supp.tex | The points suggest a maximum near 80 epochs, and the two longest pretraining runs score below the 40-epoch run | 80, 40 |
| c16-003 | check not yet built | appendix_supp.tex | the points come from three job submissions and the 80-epoch point is unreplicated | 3, 80 |
| c16-005 | check not yet built | appendix_supp.tex | values are BLEU-4 on the official Communication test set (800 clips) at beam width 5 from one run | 800, 5 |
| c16-006 | check not yet built | appendix_supp.tex | Bracketed pairs are 95\% $t$-intervals over seeds | 95 |
| c16-008 | check not yet built | appendix_supp.tex | Learned skeleton-graph encoder & 8.78 vs.\ 9.40 (2 layers) & $-0.62$ | 8.78, 9.40, 2, -0.62 |
| c16-009 | check not yet built | appendix_supp.tex | Frequency-weighted \gls{ctc} & 8.50 vs.\ 9.18 ($\alpha$\,0.5); 6.36 at $\alpha$\,1.0 & $-0.68$ | 8.50, 9.18, 0.5, 6.36, 1.0, -0.68 |
| c16-010 | check not yet built | appendix_supp.tex | \gls{bpe} subword vocabulary & 8.96 vs.\ $\approx$9.2 & $-0.2$ | 8.96, 9.2, -0.2 |
| c16-011 | check not yet built | appendix_supp.tex | R-Drop consistency regularisation & 9.17 vs.\ 9.18 ($\alpha$\,1.0; control from a concurrent job) & $-0.01$ | 9.17, 9.18, 1.0, -0.01 |
| c16-012 | check not yet built | appendix_supp.tex | Multi-reference paraphrase augmentation & 9.51 vs.\ 9.72 & $-0.21$ | 9.51, 9.72, -0.21 |
| c16-013 | check not yet built | appendix_supp.tex | Regularisation sweep (dropout, embedding dropout, label smoothing) & 10.26 vs.\ 10.21 & $+0.05$ | 10.26, 10.21, +0.05 |
| c16-014 | check not yet built | appendix_supp.tex | Temporal frame pooling ($k=2$) & 9.65 vs.\ 9.77 & $-0.12$ | 2, 9.65, 9.77, -0.12 |
| c16-015 | check not yet built | appendix_supp.tex | \gls{islr} encoder transfer & 9.98 vs.\ 9.63 & $+0.35$ | 9.98, 9.63, +0.35 |
| c16-016 | check not yet built | appendix_supp.tex | Pretrained language-model decoder (mT5 with low-rank adapters, LoRA) & 7.66 vs.\ 9.63 & $-1.97$ | 7.66, 9.63, -1.97 |
| c16-017 | check not yet built | appendix_supp.tex | Pretrained language-model decoder (mBART with LoRA) & 8.97, and 9.18 with an alignment stage, vs.\ 9.63 & $-0.45$ | 8.97, 9.18, 9.63, -0.45 |
| c16-018 | check not yet built | appendix_supp.tex | Discrete pose tokens with a language-model decoder & 0.00 (codebook collapse) | 0.00 |
| c16-019 | check not yet built | appendix_supp.tex | Reinforcement learning (group relative policy optimisation, 2 configurations) | 2 |
| c16-020 | check not yet built | appendix_supp.tex | Averaging the weights of independently seeded models & 0.16 (different loss basins) | 0.16 |
| c16-021 | A71 | appendix_supp.tex | 1.03 and 7.11 with greedy decoding | 1.03, 7.11 |
| c16-022 | disagrees | appendix_supp.tex | the larger model used 60 epochs at learning rate $3\times10^{-4}$ against 40 at $5\times10^{-4}$ | 60, 3, -4, 40, 5 |
| c16-023 | check not yet built | appendix_supp.tex | Each hand expressed around its own wrist at palm scale (3 seeds) & pre-registered replication rule met in 1/3 seeds; mean validation difference $-0.29$ | 3, 1/3, -0.29 |
| c16-024 | needs the cluster | appendix_supp.tex | Motion-aware sign resampling (3 seeds) & 0/3 seeds vs.\ duration-only stitching on greedy test, 1/3 at beam width 5 (required: 3/3) & $-0.13$ | 3, 0/3, 5, 1/3, 3/3, -0.13 |
| c16-025 | check not yet built | appendix_supp.tex | Test-time augmentation over temporal views & validation difference $+0.46$ on 120 clips and $-0.04$ on the full validation set | +0.46, 120, -0.04 |
| c16-026 | no saved output | appendix_supp.tex | 2.72 on first-listed AlphaPose keypoints (not repeated on the signer's), vs.\ 10.27 (1.47 excluding same-video neighbours) | 2.72, 10.27, 1.47 |
| c16-027 | no saved output | appendix_supp.tex | Learned per-sentence ensemble selector & 8.95 vs.\ 10.33 for the best member & $-1.38$ | 8.95, 10.33, -1.38 |
| c16-028 | needs the cluster | appendix_supp.tex | Synthetic pretraining on 25,000--300,000 new stitched sentences | 25,000, 300,000 |
| c16-029 | needs the cluster | appendix_supp.tex | 200,000 sentences, 5 seeds: test $+0.04$ $[-0.52, +0.60]$, validation $+0.25$ (3/5); 100,000 and 300,000 abandoned at stage~1 | 200,000, 5, +0.04, -0.52, +0.60, +0.25, 3/5, 100,000, 300,000, 1 |
| c16-030 | needs the cluster | appendix_supp.tex | $+0.04$ \\ | +0.04 |
| c16-031 | no saved output | appendix_supp.tex | Synthetic corpus with a 1.9$\times$ wider sign vocabulary, mixed with the real clips at 1.1:1 (5 seeds) | 1.9, 1.1:1, 5 |
| c16-032 | A67 | appendix_supp.tex | validation $-0.19$ (2/5), test $+0.09$ $[-0.45, +0.63]$; at twice the amount, abandoned at stage~1 (validation $-0.80$, 0/3) | -0.19, 2/5, +0.09, -0.45, +0.63, 1, -0.80, 0/3 |
| c16-033 | A67 | appendix_supp.tex | Not signing English function words in the stitched corpus (32\% of stitched signs dropped; 5 seeds) | 32, 5 |
| c16-034 | A67 | appendix_supp.tex | validation $+0.18$ (4/5), test $-0.09$ $[-0.76, +0.59]$ (2/5; both additional seeds negative); content-word precision $-0.95$ (0/5); a length-matched condition (2 seeds) | +0.18, 4/5, -0.09, -0.76, +0.59, 2/5, -0.95, 0/5, 2 |
| c16-035 | needs the cluster | appendix_supp.tex | News pretraining, then Communication training (2 seeds) & $-0.72$ and $-2.08$ against the concurrent control | 2, -0.72, -2.08 |
| c16-036 | needs the cluster | appendix_supp.tex | Adding the News subset (22,088 clips)} & \textbf{8.89 vs.\ 10.27} & $\mathbf{-1.38}$ | 22,088, 89, 10.27, 1.38 |
| c16-037 | check not yet built | appendix_supp.tex | negative} with greedy decoding (8.46 vs.\ 8.92) and below every pose-only \gls{vlp} pretraining variant (10.21--10.71) | 8.46, 8.92, 10.21, 10.71 |
| c16-038 | A47 | appendix_supp.tex | Official Communication test set (800 clips), beam width 5, the saved outputs of one run per model | 800, 5 |
| c16-040 | A47 | appendix_supp.tex | Baseline model (DWPose + \gls{ctc}) & 16.56 & 36.22 & 35.70 | 16.56, 36.22, 35.70 |
| c16-041 | A47 | appendix_supp.tex | 40-epoch pre-trained model & 17.74 & 36.68 & 36.27 | 40, 17.74, 36.68, 36.27 |
| c16-042 | covered by A21 (A1-A35 mapping, not reviewed) | appendix_supp.tex | \gls{rgb}-fusion model & 17.84 & 37.67 & \textbf{37.08} | 17.84, 37.67, 7.08 |
| c16-043 | A47 | appendix_supp.tex | $\beta = 1.0$ & $\beta = 1.0$ & $\beta = 1.2$ | 1.0, 1.0, 1.2 |
| c16-044 | A75 | appendix_supp.tex | on the official Communication test and validation sets (800 clips each) and the 570 unseen-target test clips | 800, 570 |
| c16-046 | A75 | appendix_supp.tex | Test & 9.52 & 10.37 & 12.05 & \textbf{17.42} | 100, 90, 75, 50, 9.52, 10.37, 12.05, 17.42 |
| c16-047 | A75 | appendix_supp.tex | Validation & 9.64 & 10.37 & 12.44 & \textbf{18.42} | 9.64, 10.37, 12.44, 18.42 |
| c16-048 | A75 | appendix_supp.tex | Test, unseen targets & 4.88 & --- & --- & 7.92 | 4.88, 7.92 |
| c16-049 | A52 | appendix_supp.tex | The first one, three and six members of the final ensemble, in a fixed order, on the 800 validation clips at batch size 1 on a laptop CPU (Apple M3 Max, 12 threads) | 1, 3, 6, 800, 1, 12 |
| c16-050 | A52 | appendix_supp.tex | 1 & 5 & 9.53 & 35 & 76 | 1, 5, 9.53, 35, 76 |
| c16-051 | A52 | appendix_supp.tex | 3 & 5 & 11.31 & 103 & 228 | 3, 5, 11.31, 103, 228 |
| c16-052 | A52 | appendix_supp.tex | 6 & 5 & 12.19 & 207 & 460 | 6, 5, 12.19, 207, 460 |
| c16-053 | A52 | appendix_supp.tex | 1 & 10 & 9.78 & 44 & 101 | 1, 10, 9.78, 44, 101 |
| c16-054 | A52 | appendix_supp.tex | 3 & 10 & 11.47 & 133 & 311 | 3, 10, 11.47, 133, 311 |
| c16-055 | A52 | appendix_supp.tex | 6 & 10 & \textbf{12.47} & 269 & 613 | 6, 10, 12.47, 269, 613 |
| c16-056 | A46 | appendix_supp.tex | 40-epoch pre-trained pose-only model, beam width 5, 800 official Communication test clips | 40, 5, 800 |
| c16-058 | A46 | appendix_supp.tex | 100\% & 800 & --- & 10.27 | 100, 800, 10.27 |
| c16-059 | A46 | appendix_supp.tex | 90\% & 720 & 0.941 & 11.27 | 90, 720, 0.941, 11.27 |
| c16-060 | A46 | appendix_supp.tex | 75\% & 600 & 0.826 & 12.88 | 75, 600, 0.826, 12.88 |
| c16-061 | A46 | appendix_supp.tex | 50\% & 400 & 0.681 & 19.19 | 50, 400, 0.681, 19.19 |
| c16-062 | A46 | appendix_supp.tex | 25\% & 200 & 0.523 & 33.54 | 25, 200, 0.523, 33.54 |
| c17-001 | A72 | appendix_supp.tex | values are BLEU-4 on the 800-clip official | 800 |
| c17-002 | A72 | appendix_supp.tex | Communication test set at beam width 5 | 5 |
| c17-003 | A72 | appendix_supp.tex | bracketed pairs are 95\% intervals | 95 |
| c17-005 | A56 | appendix_supp.tex | ($384\times288$ instead of $256\times192$) | 384, 288, 256, 192 |
| c17-006 | no saved output | appendix_supp.tex | hand top-1 accuracy 18.4\% against 18.7\% ($p = 1.0$ | 18.4, 18.7, 1.0 |
| c17-007 | no saved output | appendix_supp.tex | the standard model gives 18.7\%, against 18.1\% with the original extraction | 18.7, 18.1 |
| c17-008 | needs the cluster | appendix_supp.tex | median difference 0.8\% of body span (8.1\,px), 2.5\% of face span, about 9\% of hand span | 0.8, 8.1, 2.5, 9 |
| c17-009 | A54 | appendix_supp.tex | 25.95\,M-parameter model 2.29 | 25.95, 2.29 |
| c17-010 | A54 | appendix_supp.tex | SignJoey-compatible re-implementation 1.03 (9 distinct sentences) | 1.03, 9 |
| c17-011 | A54 | appendix_supp.tex | after the bounding-box correction 2.13 | 2.13 |
| c17-012 | A79 | appendix_supp.tex | Pose jumps in 2,128 locally stored clips | 2,128 |
| c17-013 | disagrees | appendix_supp.tex | at least one jump in 25.2\% of clips and on 2.31\% of frame transitions | 25.2, 2.31 |
| c17-014 | no saved output | appendix_supp.tex | of 552 sampled jumps, 87.5\% identity switches and 8.5\% camera cuts | 552, 87.5, 8.5 |
| c17-015 | A79 | appendix_supp.tex | $+4.4$ on 55 validation clips (all five seeds) | +4.4, 55, five |
| c17-016 | A79 | appendix_supp.tex | $+0.6$ on 64 test clips (negative in two seeds) | +0.6, 64, two |
| c17-017 | A79 | appendix_supp.tex | validation $-0.02$, stopped at stage 1 | -0.02 |
| c17-018 | A79 | appendix_supp.tex | test $-0.04$ ($-1.79$ to $+1.71$) | -0.04, -1.79, +1.71 |
| c17-019 | A79 | appendix_supp.tex | test $+3.52$ more than on clips without switches ($+1.44$ to $+5.61$; all five seeds) | +3.52, +1.44, +5.61, five |
| c17-020 | A79 | appendix_supp.tex | validation $+4.74$; interval over clips $-0.13$ to $+8.86$ | +4.74, -0.13, +8.86 |
| c17-021 | A79 | appendix_supp.tex | whole test set $+0.80$ ($-0.54$ to $+2.13$) | +0.80, -0.54, +2.13 |
| c17-022 | A79 | appendix_supp.tex | the untracked model decoded on tracked keypoints already gives $+4.49$ | +4.49 |
| c17-023 | no saved output | appendix_supp.tex | Input of 69 keypoints (9 body, 21 per hand, 18 face; | 69, 9, 21, 18 |
| c17-024 | A40 | appendix_supp.tex | three seeds in each of two job submissions | three, two |
| c17-025 | A40 | appendix_supp.tex | first submission: validation $+0.35$ ($-0.85$ to $+1.56$; higher in two of three seeds) | +0.35, -0.85, +1.56, two, three |
| c17-026 | A40 | appendix_supp.tex | test $+0.86$ ($-2.75$ to $+4.48$) | +0.86, -2.75, +4.48 |
| c17-027 | A40 | appendix_supp.tex | second: validation $+0.64$ ($-0.10$ to $+1.37$; all three seeds) | +0.64, -0.10, +1.37, three |
| c17-028 | A40 | appendix_supp.tex | test $+0.02$ ($-1.79$ to $+1.82$) | +0.02, -1.79, +1.82 |
| c17-029 | A40 | appendix_supp.tex | the same configuration scored up to 1.66 apart on test between the two submissions | 1.66, two |
| c17-030 | A40 | appendix_supp.tex | unseen-target validation $+0.81$ and $+1.14$ | +0.81, +1.14 |
| c17-031 | A40 | appendix_supp.tex | validation $+0.60$ and $+0.74$ (all three seeds each) | +0.60, +0.74, three |
| c17-032 | A40 | appendix_supp.tex | five seeds (the three above plus two fresh seeds | five, three, two |
| c17-033 | A40 | appendix_supp.tex | validation $+0.94$ ($-0.28$ to $+2.15$), higher in four of five seeds | 0.94, -0.28, 2.15, four, five |
| c17-034 | A40 | appendix_supp.tex | test $+0.94$ ($-0.35$ to $+2.23$), four of five | +0.94, -0.35, +2.23, four, five |
| c17-035 | A40 | appendix_supp.tex | unseen-target validation $+1.18$ and test $+1.29$ | +1.18, +1.29 |
| c17-036 | A40 | appendix_supp.tex | about 0.35 of the 0.94 test gain comes from longer output | 0.35, 0.94 |
| c17-037 | A40 | appendix_supp.tex | one-sided test $p = 0.057$ | 0.057 |
| c17-038 | A40 | appendix_supp.tex | validation $-0.63$ ($-1.01$ to $-0.25$), lower in all three seeds | -0.63, -1.01, -0.25, three |
| c17-039 | A40 | appendix_supp.tex | test $+0.16$ ($-2.33$ to $+2.65$) | +0.16, -2.33, +2.65 |
| c17-040 | A40 | appendix_supp.tex | 15.7\,M against 10.6\,M parameters; three seeds | 15.7, 10.6, three |
| c17-042 | A40 | appendix_supp.tex | validation $-1.50$ ($-2.56$ to $-0.43$) and $-1.48$ ($-1.60$ to $-1.36$), lower in all three seeds | -1.50, -2.56, -0.43, -1.48, -1.60, -1.36, three |
| c17-043 | A40 | appendix_supp.tex | test $-1.20$ and $-1.06$ | -1.20, -1.06 |
| c17-044 | A40 | appendix_supp.tex | the skeleton graph over the identity graph $+0.02$ on validation | +0.02 |
| c17-045 | A40 | appendix_supp.tex | against the 133-keypoint control $-0.86$ and $-0.84$ | 133, -0.86, -0.84 |
| c17-046 | A40 | appendix_supp.tex | validation $+0.09$ ($-3.08$ to $+3.27$; positive in one seed of three) | +0.09, -3.08, +3.27, one, three |
| c17-047 | A40 | appendix_supp.tex | by more than 10\% in only 14 of 200 sampled clips | 10, 14, 200 |
| c17-048 | A67 | appendix_supp.tex | 3.45\% of the signs (pre-registered, three seeds) | 3.45, three |
| c17-049 | A67 | appendix_supp.tex | validation $+0.12$ ($-1.57$ to $+1.80$), test $-0.03$ ($-1.48$ to $+1.43$) | +0.12, -1.57, +1.80, -0.03, -1.48, +1.43 |
| c17-050 | A82 | appendix_supp.tex | without the 15\% boundary trim (three seeds) | 15, three |
| c17-051 | A82 | appendix_supp.tex | $-0.44$ ($-0.90$ to $+0.01$); lower in all three seeds | -0.44, -0.90, +0.01, three |
| c17-052 | A39 | appendix_supp.tex | five seeds; both hands drop between signs 4.68 times per stitched clip, against 0.25 in real clips with tracked keypoints and 0.67 as extracted | five, 4.68, 0.25, 0.67 |
| c17-053 | A39 | appendix_supp.tex | deeper cut (12.6\% of frames removed; 0.28 such pauses per clip left) | 12.6, 0.28 |
| c17-054 | A39 | appendix_supp.tex | test $+0.29$ ($+0.13$ to $+0.45$), validation $+0.19$ ($-0.36$ to $+0.74$) | +0.29, +0.13, +0.45, +0.19, -0.36, +0.74 |
| c17-055 | A39 | appendix_supp.tex | shallower cut (8.8\%; 1.91 left) | 8.8, 1.91 |
| c17-056 | A39 | appendix_supp.tex | test $+0.49$ ($-0.03$ to $+1.02$), validation $+0.18$ ($-0.54$ to $+0.90$) | +0.49, -0.03, +1.02, +0.18, -0.54, +0.90 |
| c17-057 | A39 | appendix_supp.tex | test higher in all five seeds for both | five |
| c17-058 | A39 | appendix_supp.tex | length-adjusted test gains are $+0.20$ and $+0.34$ | +0.20, +0.34 |
| c17-059 | A82 | appendix_supp.tex | $-0.29$ ($-1.15$ to $+0.57$) | -0.29, -1.15, +0.57 |
| c17-060 | A82 | appendix_supp.tex | Two stitched realisations of each of half as many sentences (five seeds) | Two, half, five |
| c17-061 | needs the cluster | appendix_supp.tex | 80-epoch pre-trained model (one run) & $+1.50$ | 80, +1.50 |
| c17-062 | needs the cluster | appendix_supp.tex | rise in decoder cross-entropy 0.906 to 0.931, against 0.785 for their controls | 0.906, 0.931, 0.785 |
| c17-063 | A67 | appendix_supp.tex | validation $-0.69$, test $-0.68$ | -0.69, -0.68 |
| c17-064 | A67 | appendix_supp.tex | validation $+0.45$, test $-0.12$ | +0.45, -0.12 |
| c17-065 | A41 | appendix_supp.tex | Three training ideas, three seeds each in one job | Three, three |
| c17-066 | no saved output | appendix_supp.tex | every stitched sign re-timed by its own random factor of 0.7 to 1.3 | 0.7, 1.3 |
| c17-067 | A41 | appendix_supp.tex | a ten-epoch stage that rewards a higher reference probability | ten |
| c17-068 | A41 | appendix_supp.tex | against a ten-epoch continuation | ten |
| c17-069 | A41 | appendix_supp.tex | sign targets: validation $+0.28$ ($-0.65$ to $+1.20$) | +0.28, -0.65, +1.20 |
| c17-070 | A41 | appendix_supp.tex | test $-0.31$ ($-1.80$ to $+1.18$) | -0.31, -1.80, +1.18 |
| c17-071 | A41 | appendix_supp.tex | with five seeds, validation $+0.19$ ($-0.18$ to $+0.56$) and test $+0.33$ ($-1.02$ to $+1.68$) | five, +0.19, -0.18, +0.56, +0.33, -1.02, +1.68 |
| c17-072 | A41 | appendix_supp.tex | Re-timing: validation $+0.26$ ($-1.81$ to $+2.33$), test $+0.15$ ($-0.61$ to $+0.92$) | +0.26, -1.81, +2.33, +0.15, -0.61, +0.92 |
| c17-073 | A41 | appendix_supp.tex | test content-word recall $+1.27$ ($-1.16$ to $+3.70$) points | +1.27, -1.16, +3.70 |
| c17-074 | A41 | appendix_supp.tex | validation higher in one seed of three | one, three |
| c17-075 | A41 | appendix_supp.tex | with five seeds, validation $+0.21$ ($-0.56$ to $+0.98$) and test $+0.24$ ($-0.48$ to $+0.97$) | five, +0.21, -0.56, +0.98, +0.24, -0.48, +0.97 |
| c17-076 | A41 | appendix_supp.tex | Preference stage: validation $-0.31$ ($-2.07$ to $+1.44$), test $-0.15$ ($-1.36$ to $+1.06$), recall $-0.34$ points | -0.31, -2.07, +1.44, -0.15, -1.36, +1.06, -0.34 |
| c17-077 | A41 | appendix_supp.tex | 603 annotated sign spans: 47.3\% hit against 26.2\% for random placement | 603, 47.3, 26.2 |
| c17-078 | A41 | appendix_supp.tex | gate failed (needs 50\% and twice random) | 50, twice |
| c17-079 | A67 | appendix_supp.tex | (8.7\% of stitched signs) removed (three seeds) | 8.7, three |
| c17-080 | A67 | appendix_supp.tex | validation $-0.10$, test $+0.03$ | -0.10, +0.03 |
| c17-081 | no saved output | appendix_supp.tex | twice and at half the adopted strength | twice, half |
| c17-082 | needs the cluster | appendix_supp.tex | validation $-0.22$, $-0.39$ and $-0.48$ | -0.22, -0.39, -0.48 |
| c17-083 | check not yet built | appendix_supp.tex | about 15\% of frames per epoch (three seeds) | 15, three |
| c17-084 | check not yet built | appendix_supp.tex | validation $-0.50$ | -0.50 |
| c17-085 | check not yet built | appendix_supp.tex | 60 training epochs instead of 40 (five seeds) | 60, 40, five |
| c17-086 | check not yet built | appendix_supp.tex | validation $+0.08$ at stage 1 | +0.08 |
| c17-087 | check not yet built | appendix_supp.tex | test $-0.02$ ($-0.74$ to $+0.69$) | -0.02, -0.74, +0.69 |
| c17-088 | check not yet built | appendix_supp.tex | both stage-2 seeds lower ($-0.66$, $-0.44$) | -0.66, -0.44 |
| c17-089 | check not yet built | appendix_supp.tex | 80 training epochs instead of 40 | 80, 40 |
| c17-090 | check not yet built | appendix_supp.tex | baseline model $+0.74$ (two seeds; validation preferred 150 epochs) | +0.74, two, 150 |
| c17-091 | check not yet built | appendix_supp.tex | with stitched data $-0.96$ (one seed) | -0.96, one |
| c17-092 | A42 | appendix_supp.tex | A first training cycle of 92 epochs on the adopted corpus before the 40-epoch training (pre-registered, five new seeds) | 92, 40, five |
| c17-093 | A42 | appendix_supp.tex | test $-0.56$ ($-1.41$ to $+0.28$), higher in one seed of five | -0.56, -1.41, +0.28, one, five |
| c17-094 | A42 | appendix_supp.tex | validation $+0.26$ ($-0.12$ to $+0.65$) | +0.26, -0.12, +0.65 |
| c17-095 | A42 | appendix_supp.tex | one-sided $p = 0.93$ | 0.93 |
| c17-096 | A42 | appendix_supp.tex | the 13,760 synthetic clips alone | 13,760 |
| c17-097 | A42 | appendix_supp.tex | test $-0.07$ ($-0.59$ to $+0.45$) against no first cycle | -0.07, -0.59, +0.45 |
| c17-098 | A42 | appendix_supp.tex | $+0.49$ ($-0.26$ to $+1.25$) against the joint one | +0.49, -0.26, +1.25 |
| c17-099 | needs the cluster | appendix_supp.tex | A first stage of 40 epochs with only the pseudo-gloss CTC loss | 40 |
| c17-100 | no saved output | appendix_supp.tex | against 40 further epochs of the full loss | 40 |
| c17-101 | A70 | appendix_supp.tex | three seeds | three |
| c17-102 | A70 | appendix_supp.tex | validation $+0.39$ ($-0.81$ to $+1.60$; higher in two of three seeds) | +0.39, -0.81, +1.60, two, three |
| c17-103 | A70 | appendix_supp.tex | test $+0.38$ ($-0.34$ to $+1.10$; all three) | +0.38, -0.34, +1.10, three |
| c17-104 | A70 | appendix_supp.tex | test $+0.32$ ($-0.10$ to $+0.75$) | +0.32, -0.10, +0.75 |
| c17-105 | A70 | appendix_supp.tex | test $+0.70$ ($+0.10$ to $+1.31$) | +0.70, +0.10, +1.31 |
| c17-106 | A70 | appendix_supp.tex | the control differed from an earlier job by 0.62 at one seed | 0.62, one |
| c17-107 | no saved output | appendix_supp.tex | 16,800 updates on 44,010 unlabelled dictionary clips | 16,800, 44,010 |
| c17-108 | no saved output | appendix_supp.tex | the 12,439 unlabelled training clips added | 12,439 |
| c17-109 | A81 | appendix_supp.tex | (pre-registered, three seeds, validation as criterion) | three |
| c17-110 | A81 | appendix_supp.tex | all nine pretraining runs met their pre-registered training check | nine |
| c17-111 | A63 | appendix_supp.tex | at equal size (five seeds; difference between two differences) | five, two |
| c17-112 | A63 | appendix_supp.tex | $-0.78$ ($-2.06$ to $+0.49$) on all affected test clips | -0.78, -2.06, +0.49 |
| c17-113 | A63 | appendix_supp.tex | $-0.03$ ($-0.98$ to $+0.93$) on the unseen-target clips | -0.03, -0.98, +0.93 |
| c18-001 | A53 | appendix_supp.tex | $+0.35$ ($-0.05$ to $+0.75$); higher in five of seven | +0.35, -0.05, +0.75, five, seven |
| c18-002 | A53 | appendix_supp.tex | seven job submissions | seven |
| c18-003 | A53 | appendix_supp.tex | Ensemble of three seeds | three |
| c18-004 | A51 | appendix_supp.tex | Ensemble of three pose-only and two RGB-fusion models on the video-disjoint split | three, two |
| c18-005 | A51 | appendix_supp.tex | 7.84 against 7.29 for the best single model ($+0.55$; simultaneous 95\% interval $-0.23$ to $+1.34$) | 7.84, 7.29, +0.55, 95, -0.23, +1.34 |
| c18-006 | A51 | appendix_supp.tex | Ensemble with three pose-augmented members | three |
| c18-007 | A51 | appendix_supp.tex | 12.73 against 11.80 for the final ensemble ($+0.93$; $-0.28$ to $+2.11$) | 12.73, 11.80, +0.93, -0.28, +2.11 |
| c18-008 | A51 | appendix_supp.tex | unseen-target subset 7.89 | 7.89 |
| c18-009 | A53 | appendix_supp.tex | 10.69 and WER 0.956, against 11.80 and 0.907 | 10.69, 0.956, 11.80, 0.907 |
| c18-010 | disagrees | appendix_supp.tex | $+0.73$ ($-0.43$ to $+1.96$) | +0.73, -0.43, +1.96 |
| c18-011 | covered by A29 (A1-A35 mapping, not reviewed) | appendix_supp.tex | strongest member's (10.80) | 10.80 |
| c18-012 | covered by A29 (A1-A35 mapping, not reviewed) | appendix_supp.tex | 11.34 and 11.26 | 11.34, 11.26 |
| c18-013 | A83 | appendix_supp.tex | substitutions are 56--58\% of word edits and equal 0.53 of the reference length | 56, 58, 0.53 |
| c18-014 | A83 | appendix_supp.tex | an oracle over six sets of outputs reaches 0.807 | six, 0.807 |
| c18-015 | A83 | appendix_supp.tex | ratio of output to reference length 0.79--0.86 | 0.79, 0.86 |
| c18-017 | A47 | appendix_supp.tex | 1.119 to 1.151 for three single models | 1.119, 1.151, three |
| c18-018 | A80 | appendix_supp.tex | five seeds | five |
| c18-019 | needs the cluster | appendix_supp.tex | content-word precision 14.1\% to 18.5\% ($+4.38$ points; $+2.95$ to $+5.81$) | 14.1, 18.5, +4.38, +2.95, +5.81 |
| c18-020 | needs the cluster | appendix_supp.tex | recall 12.1\% to 14.4\% ($+2.30$; $+0.86$ to $+3.73$) | 12.1, 14.4, +2.30, +0.86, +3.73 |
| c18-021 | needs the cluster | appendix_supp.tex | WER 1.002 to 0.958 | 1.002, 0.958 |
| c18-022 | A76 | appendix_supp.tex | graded higher on 148 clips and lower on 134 | 148, 134 |
| c18-023 | disagrees | appendix_supp.tex | $+2.2$ to $+4.1$ ($p$ between 0.02 and 0.29) | +2.2, +4.1, 0.02, 0.29 |
| c18-024 | disagrees | appendix_supp.tex | unseen-target clips $+1.1$ to $-0.3$ | +1.1, -0.3 |
| c18-025 | covered by A24 (A1-A35 mapping, not reviewed) | appendix_supp.tex | 1,088 held-out clips against 12,953 for training | 1,088, 12,953 |
| c18-026 | covered by A24 (A1-A35 mapping, not reviewed) | appendix_supp.tex | 19.3\% verbatim reference overlap | 19.3 |
| c18-027 | A77 | appendix_supp.tex | two signers supply 75.6\% of the held-out clips | 75.6 |
| c18-028 | A77 | appendix_supp.tex | baseline model, three seeds, beam width 5; trained on 12,024 clips | three, 5, 12,024 |
| c18-029 | A77 | appendix_supp.tex | 1.29 BLEU-4 on the 1,087 held-out clips and 0.87 on the 869 with unseen references | 1.29, 1,087, 0.87, 869 |
| c18-030 | A77 | appendix_supp.tex | with other clips of the held-out signers in training 1.83 | 1.83 |
| c18-031 | A77 | appendix_supp.tex | difference $+0.55$ (bootstrap over clips $+0.23$ to $+0.91$; $t$-interval over seeds $-0.43$ to $+1.53$) | +0.55, +0.23, +0.91, -0.43, +1.53 |
| c18-032 | A77 | appendix_supp.tex | (thresholds 0.5 and 1.0) | 0.5, 1.0 |
| c18-033 | A77 | appendix_supp.tex | a constant sentence chosen on validation scores 0.44 | 0.44 |
| c18-034 | A77 | appendix_supp.tex | 59\% of the held-out clips are one signer's | 59 |
| c18-035 | A75 | appendix_supp.tex | 16.13 on test and 17.06 on validation, against 17.42 and 18.42 for the two-term score | 16.13, 17.06, 17.42, 18.42 |
| c18-036 | A75 | appendix_supp.tex | at 50\% coverage | 50 |
| c18-037 | A75 | appendix_supp.tex | Longest-output control at 25\% coverage & 7.22 | 25, 7.22 |
| c18-038 | A75 | appendix_supp.tex | $+7.30$ ($+4.55$ to $+10.17$) at 51.9\% coverage | +7.30, +4.55, +10.17, 51.9 |
| c18-039 | A75 | appendix_supp.tex | unseen-target subset $+2.36$ ($+0.47$ to $+4.42$) at 55.1\% | 2.36, 0.47, 4.42, 55.1 |
| c18-040 | A45 | appendix_supp.tex | area under the ROC curve 0.51 (0.48 to 0.54) | 0.51, 0.48, 0.54 |
| c18-041 | A45 | appendix_supp.tex | non-signers who hardly move voiced in 50.3\% of cases | 50.3 |
| c18-042 | A45 | appendix_supp.tex | frozen at one frame voiced in 46.5\% | 46.5 |
| c18-043 | A45 | appendix_supp.tex | 831 test and 805 validation crops | 831, 805 |
| c18-044 | A45 | appendix_supp.tex | ratio 0.85 (0.77 to 0.93) | 0.85, 0.77, 0.93 |
| c18-045 | A45 | appendix_supp.tex | threshold for 75\% coverage: 0.91 (0.85 to 0.98) | 75, 0.91, 0.85, 0.98 |
| c18-046 | A45 | appendix_supp.tex | person-tracked pose: 0.84 (0.75 to 0.93) | 0.84, 0.75, 0.93 |
| c18-047 | A45 | appendix_supp.tex | area under the ROC curve 0.56 (0.53 to 0.60) | 0.56, 0.53, 0.60 |
| c18-048 | A45 | appendix_supp.tex | voiced non-signer outputs 99\% verbatim training sentences | 99 |
| c18-049 | A45 | appendix_supp.tex | threshold for 50\% coverage; 831 test crops | 50, 831 |
| c18-050 | A45 | appendix_supp.tex | ratio of non-signers voiced to signers voiced 0.95 (0.86 to 1.04) against 0.84 for confidence | 0.95, 0.86, 1.04, 0.84 |
| c18-051 | A45 | appendix_supp.tex | area under the ROC curve 0.52 against 0.56 (difference $-0.05$, $-0.08$ to $-0.01$) | 0.52, 0.56, -0.05, -0.08, -0.01 |
| c18-052 | A45 | appendix_supp.tex | 40-epoch pre-trained model 1.01 (0.91 to 1.10) against 1.01 | 40, 1.01, 0.91, 1.10, 1.01 |
| c18-053 | A45 | appendix_supp.tex | true video beats a mismatched one on 97 to 99\% of clips | 97, 99 |
| c18-054 | A78 | appendix_supp.tex | $-1.5$ percentage points ($-2.8$ to $-0.1$) | -1.5, -2.8, -0.1 |
| c18-055 | A52 | appendix_supp.tex | 55.0 frames per second | 55.0 |
| c18-056 | A52 | appendix_supp.tex | Three ensemble members at beam width 10 | Three, 10 |
| c18-057 | A52 | appendix_supp.tex | 11.47 on validation at a median of 133\,ms per clip | 11.47, 133 |
| c18-058 | A51 | appendix_supp.tex | members added in the recorded order: 9.78, 11.08, 11.47, 11.57, 11.78, 12.47 | 9.78, 11.08, 11.47, 11.57, 11.78, 12.47 |
| c18-059 | covered by A34 (A1-A35 mapping, not reviewed) | appendix_supp.tex | $+0.20$ ($-0.99$ to $+1.33$, bootstrap over validation sentences) | +0.20, -0.99, +1.33 |
| c18-060 | A44 | appendix_supp.tex | exponent $-1$, 0.5, 1 and 2 | -1, 0.5, 1, 2 |
| c18-061 | A44 | appendix_supp.tex | 98.75\% intervals | 98.75 |
| c18-062 | A44 | appendix_supp.tex | $-0.68$ ($-1.79$ to $+0.10$), $0.00$ ($-0.59$ to $+0.60$), $+0.13$ ($-0.53$ to $+0.74$), $+0.23$ ($-0.60$ to $+1.05$) | -0.68, -1.79, +0.10, 0.00, -0.59, +0.60, +0.13, -0.53, +0.74, +0.23, -0.60, +1.0 |
| c18-063 | A44 | appendix_supp.tex | brevity penalty 0.80 for the geometric mean against 0.84 and 0.92 for exponents 1 and 2 | 0.80, 0.84, 0.92, 1, 2 |
| c18-064 | A43 | appendix_supp.tex | three seeds | three |
| c18-065 | A43 | appendix_supp.tex | both together $+0.53$ ($-0.40$ to $+1.45$) | +0.53, -0.40, +1.45 |
| c18-066 | A43 | appendix_supp.tex | stitched data alone $+0.21$ ($-0.84$ to $+1.27$) | 0.21, -0.84, 1.27 |
| c18-067 | A43 | appendix_supp.tex | augmentation added to it $+0.31$ ($-0.50$ to $+1.12$) | +0.31, -0.50, +1.12 |
| c18-068 | A43 | appendix_supp.tex | augmentation alone, a later arm of three seeds (descriptive; two seeds are compared with controls from an earlier job submission) $+0.19$ ($-0.50$ to $+0.88$) | three, two, +0.19, -0.50, +0.88 |
| c18-069 | A69 | appendix_supp.tex | 256-frame against 512-frame | 256, 512 |
| c18-070 | needs the cluster | appendix_supp.tex | 1.64 against 2.03 on test; 1.94 against 1.87 on validation | 1.64, 2.03, 1.94, 1.87 |
| c18-071 | needs the cluster | appendix_supp.tex | three unseen words produced in 699 test clips, none correct | three, 699 |
| c18-072 | A64 | appendix_supp.tex | (five seeds) | five |
| c18-073 | A64 | appendix_supp.tex | 2.14 to 2.33 ($+0.19$; $-0.01$ to $+0.39$) | 2.14, 2.33, +0.19, -0.01, +0.39 |
| c18-074 | A64 | appendix_supp.tex | brevity penalty 0.73 to 0.94 | 0.73, 0.94 |
| c18-075 | A64 | appendix_supp.tex | (five seeds) | five |
| c18-076 | A64 | appendix_supp.tex | $+1.16$ percentage points for words that the synthetic data can sign; $+0.60$ for words that it cannot | +1.16, +0.60 |
| c18-077 | A60 | appendix_supp.tex | top-1 19.9\% against 18.1\% ($p = 0.33$) | 19.9, 18.1, 0.33 |
| c18-078 | A60 | appendix_supp.tex | citation-form recordings of the same signs 4.5\% | 4.5 |
| c18-079 | A60 | appendix_supp.tex | top-1 11.3\% against 16.3\% ($p = 0.014$) | 11.3, 16.3, 0.014 |
| c18-080 | A60 | appendix_supp.tex | depth alone 5.9\% (above a permutation null of 3.9\%) | 5.9, 3.9 |
| c18-081 | check not yet built | appendix_supp.tex | CTR-GCN with MS-TCN, 1,024 dimensions | 1,024 |
| c18-082 | check not yet built | appendix_supp.tex | one run | one |
| c18-083 | check not yet built | appendix_supp.tex | 1.60 and WER 1.281 (validation 1.90) | 1.60, 1.281, 1.90 |
| c18-084 | covered by A29 (A1-A35 mapping, not reviewed) | appendix_supp.tex | 10.42 for one member (mean), rising to 16.27 for six | 10.42, one, 16.27, six |
| c18-085 | covered by A29 (A1-A35 mapping, not reviewed) | appendix_supp.tex | the ensemble's 11.80 is 18.3\% of the way from its strongest member (10.80) to that oracle | 11.80, 18.3, 10.80 |
| c18-086 | no saved output | appendix_supp.tex | content-word recall 0.15 to 0.66 | 0.15, 0.66 |
| c18-087 | no saved output | appendix_supp.tex | WER 0.9497 to 0.8685 and BLEU-4 10.33 to 12.45 | 0.9497, 0.8685, 10.33, 12.45 |
| c18-088 | A83 | appendix_supp.tex | across six sets of outputs the correlation of output length with WER is $+0.990$ | six, +0.990 |
| c18-089 | check not yet built | appendix_supp.tex | in 140 saved runs | 140 |
| c18-090 | check not yet built | appendix_supp.tex | 35 runs peaked in the final 5\% of their schedule | 35, 5 |
| c18-091 | A63 | appendix_supp.tex | 16.56\% content-word recall for the full training set against 16.47\% observed | 16.56, 16.47 |
| c18-092 | no saved output | appendix_supp.tex | 0.6319 against 0.0872 (0.14 times) | 0.6319, 0.0872, 0.14 |
| c18-093 | check not yet built | appendix_supp.tex | beam width 5 | 5 |
| c18-094 | check not yet built | appendix_supp.tex | validation at beam width 5 $+0.713$ against greedy validation $+0.483$ (12 ratio runs) | +0.713, +0.483, 12 |
| c18-095 | check not yet built | appendix_supp.tex | $+0.257$ against $-0.543$ (6 training-length runs) | +0.257, -0.543, 6 |
| c18-096 | A51 | appendix_supp.tex | three models | three |
| c18-097 | A51 | appendix_supp.tex | validation 12.472 against 12.470 for the final ensemble; test 11.54 against 11.80 | 12.472, 12.470, 11.54, 11.80 |
| c18-098 | A64 | appendix_supp.tex | ensemble of five seeds (pre-registered validation gate of $+0.10$) | five, +0.10 |
| c18-099 | A64 | appendix_supp.tex | validation 2.948 against 2.872 for the best single model ($+0.076$) | 2.948, 2.872, +0.076 |
| c19-001 | covered by A17 (A1-A35 mapping, not reviewed) | appendix_supp.tex | 12,439 training clips | 12,439 |
| c19-002 | covered by A14 (A1-A35 mapping, not reviewed) | appendix_supp.tex | 800 official test clips | 800 |
| c19-003 | covered by A3 (A1-A35 mapping, not reviewed) | appendix_supp.tex | the 570 test clips whose reference is absent from training | 570 |
| c19-007 | no saved output | appendix_supp.tex | ten-point behavioural description | ten |
| c19-008 | A37 | appendix_supp.tex | within $1.9\times10^{-5}$ on 20 validation clips | 1.9, -5, 20 |
| c19-009 | A68 | appendix_supp.tex | The port itself scores 14.37 on the How2Sign test set, where the authors' released output scores 14.50 | 14.37, 14.50 |
| c19-010 | A37 | appendix_supp.tex | a test mean of 17.86 against the port's 18.21, with 81 to 85\% of outputs identical | 17.86, 18.21, 81, 85 |
| c19-011 | A36 | appendix_supp.tex | each ensemble used two test decodes | two |
| c19-014 | A36 | appendix_supp.tex | beam width 5, length penalty 1.0 | 5, 1.0 |
| c19-015 | A36 | appendix_supp.tex | beam width 10, length penalty 0.8 | 10, 0.8 |
| c19-016 | A36 | appendix_supp.tex | max-$T$, family of four | four |
| c19-017 | A65 | appendix_supp.tex | over the 800 test sentences | 800 |
| c19-018 | A36 | appendix_supp.tex | predicted power of at least 0.5 | 0.5 |
| c19-019 | A68 | appendix_supp.tex | the pre-trained model, fine-tuned (private port) & 5 & 17.37 & 18.21 & 13.73 | 5, 17.37, 18.21, 13.73 |
| c19-020 | A68 | appendix_supp.tex | Smaller translator of this thesis, from scratch, same job & 5 & 10.52 & 10.37 & 5.47 | 5, 10.52, 10.37, 5.47 |
| c19-021 | A68 | appendix_supp.tex | the pre-trained model network, random pose-side weights (mT5 pre-trained) & 5 & 7.06 & 6.60 & 3.21 | 5, 7.06, 6.60, 3.21 |
| c19-022 | A37 | appendix_supp.tex | Same five fine-tuned models, decoded with our code & 5 & 17.35 & 17.86 & 13.49 | 5, 17.35, 17.86, 13.49 |
| c19-023 | A36 | appendix_supp.tex | Ensemble of the five, E0 & --- & 19.54 & 20.20 & 16.11 | 19.54, 20.20, 16.11 |
| c19-024 | A36 | appendix_supp.tex | Ensemble of the five, E* & --- & 19.88 & 20.60 & 16.39 | 19.88, 20.60, 16.39 |
| c19-025 | A37 | appendix_supp.tex | the pre-trained model, fine-tuned with our code & 3 & 17.72 & 18.27 & 13.49 | 3, 17.72, 18.27, 13.49 |
| c19-026 | A36 | appendix_supp.tex | Ensemble of the three, E0 & --- & --- & 19.65 & 14.81 | 19.65, 14.81 |
| c19-027 | A36 | appendix_supp.tex | Ensemble of the three, E* & --- & --- & 19.77 & 14.71 | 19.77, 14.71 |
| c19-028 | A37 | appendix_supp.tex | plus stitched data and pose augmentation & 3 & 16.84 & 17.63 & 13.21 | 3, 16.84, 17.63, 13.21 |
| c19-029 | A68 | appendix_supp.tex | Fine-tuned minus from scratch, same job (5 seeds) & $+7.84$ ($+7.04$ to $+8.64$); validation $+6.85$ ($+6.67$ to $+7.02$) | 5, +7.84, +7.04, +8.64, +6.85, +6.67, +7.02 |
| c19-030 | disagrees | appendix_supp.tex | positive in 5 of 5 seeds; $+5.79$ ($+4.82$ to $+6.77$) after adjusting for output length | 5, 5, +5.79, +4.82, +6.77 |
| c19-031 | A68 | appendix_supp.tex | Fine-tuned minus random pose-side weights (5 seeds) & $+11.61$ ($+10.09$ to $+13.13$) | 5, +11.61, +10.09, +13.13 |
| c19-032 | A68 | appendix_supp.tex | one random-weight run produced 30 distinct outputs, below the guard of 100 | 30, 100 |
| c19-033 | A36 | appendix_supp.tex | Five-model ensemble E*, minus member mean (17.86) & $+2.75$ ($+1.54$ to $+3.95$) | 17.86, +2.75, +1.54, +3.95 |
| c19-034 | A36 | appendix_supp.tex | Claim rule met (predicted power 0.99) | 0.99 |
| c19-035 | A36 | appendix_supp.tex | E*, minus member best on validation (17.43) & $+3.18$ ($+1.43$ to $+4.92$) | 17.43, +3.18, +1.43, +4.92 |
| c19-036 | A36 | appendix_supp.tex | Claim rule met (predicted power 0.72) | 0.72 |
| c19-037 | A36 | appendix_supp.tex | $+2.34$ ($+1.17$ to $+3.52$); $+2.78$ ($+1.08$ to $+4.47$) | +2.34, +1.17, +3.52, +2.78, +1.08, +4.47 |
| c19-038 | A36 | appendix_supp.tex | Claim rule met (power 0.63) | 0.63 |
| c19-039 | A36 | appendix_supp.tex | Three-model ensemble E*, minus member mean (18.27); minus member best on validation (18.69) & $+1.50$ ($+0.38$ to $+2.62$); $+1.07$ ($-0.55$ to $+2.69$) | 18.27, 18.69, +1.50, +0.38, +2.62, +1.07, -0.55, +2.69 |
| c19-040 | A37 | appendix_supp.tex | Validation $-0.88$ ($-2.99$ to $+1.23$); test $-0.64$ ($-3.16$ to $+1.89$) | -0.88, -2.99, +1.23, -0.64, -3.16, +1.89 |
| c19-041 | A37 | appendix_supp.tex | Stopped by the futility rule (1 of 3 seeds positive on validation) | 1, 3 |
| c19-042 | A37 | appendix_supp.tex | Validation $+0.35$; test $+0.06$ | +0.35, +0.06 |
| c19-045 | no saved output | appendix_supp.tex | Of 11,096 listed identifiers, 10,327 could be looked up | 11,096, 10,327 |
| c19-046 | no saved output | appendix_supp.tex | 769 are deleted or private | 769 |
| c19-047 | no saved output | appendix_supp.tex | Of the 570 unseen test references, 13 short phrases | 570, 13 |
| c19-048 | no saved output | appendix_supp.tex | scored 4.37 on them and 13.75 on the other 557 | 4.37, 13.75, 557 |
| c19-049 | A68 | appendix_supp.tex | $p = 0.094$ | 0.094 |
| c19-050 | no saved output | appendix_supp.tex | the 769 offline identifiers | 769 |
| c19-052 | no saved output | appendix_supp.tex | 200 epochs at a constant learning rate of $5\times10^{-5}$ and batch size 64 | 200, 5, -5, 64 |
| c19-053 | A38 | appendix_supp.tex | the project's scoring function gives 8.66 and 8.69 for the paper-literal cells | 8.66, 8.69 |
| c19-054 | check not yet built | appendix_supp.tex | the 40-epoch control for seed 101 scored 10.05, 9.90 and 9.72 in three jobs | 40, 101, 10.05, 9.90, 9.72 |
| c19-055 | check not yet built | appendix_supp.tex | differences below about 0.3 are within job-to-job scatter | 0.3 |
| c19-056 | A71 | appendix_supp.tex | scored 2 to 5 \gls{bleu}-4 | 2, 5 |
| c19-057 | A55 | appendix_supp.tex | 12,411 of the 14,041 clips | 12,411, 14,041 |
| c19-059 | A38 | appendix_supp.tex | reproduction tolerance 1.0 | 1.0 |
| c19-060 | no saved output | appendix_supp.tex | the authors' code on 53 DWPose points | 53 |
| c19-061 | no saved output | appendix_supp.tex | the project's translator (40 epochs) | 40 |
| c19-062 | A71 | appendix_supp.tex | the usual 133-point input, beam width 5 | 133, 5 |
| c19-063 | no saved output | appendix_supp.tex | the signer's AlphaPose keypoints (56 points) | 56 |
| c19-064 | A38 | appendix_supp.tex | SL-Transformer, keypoints as distributed & 8.77 & 8.25 to 9.29 | 8.77, 8.25, 9.29 |
| c19-065 | A38 | appendix_supp.tex | SL-Transformer, per-frame standardised & 8.87 & 7.50 to 10.23 | 8.87, 7.50, 10.23 |
| c19-066 | A38 | appendix_supp.tex | GASLT, keypoints as distributed & 9.16 & 7.36 to 10.96 | 9.16, 7.36, 10.96 |
| c19-067 | A38 | appendix_supp.tex | GASLT, per-frame standardised & 9.67 & 9.23 to 10.11 | 9.67, 9.23, 10.11 |
| c19-068 | A38 | appendix_supp.tex | SL-Transformer, 3 seeds (8.44 against 10.30) & $-1.86$ & $-2.71$ to $-1.02$ | 3, 8.44, 10.30, -1.86, -2.71, -1.02 |
| c19-069 | A38 | appendix_supp.tex | GASLT, 3 seeds (9.70 against 10.30) & $-0.60$ & $-1.44$ to $+0.24$ | 3, 9.70, 10.30, -0.60, -1.44, +0.24 |
| c19-070 | A38 | appendix_supp.tex | SL-Transformer, 5 seeds (8.63 against 10.43) & $-1.80$ & $-2.27$ to $-1.34$ | 5, 8.63, 10.43, -1.80, -2.27, -1.34 |
| c19-071 | A38 | appendix_supp.tex | GASLT, 5 seeds (9.58 against 10.43) & $-0.85$ & $-1.32$ to $-0.39$ | 5, 9.58, 10.43, -0.85, -1.32, -0.39 |
| c19-072 | A38 | appendix_supp.tex | & $+0.07$; $+0.49$ & $-0.86$ to $+0.99$; $-1.35$ to $+2.34$ | +0.07, +0.49, -0.86, +0.99, -1.35, +2.34 |
| c19-073 | A71 | appendix_supp.tex | O53 (9.91) minus their SL-Transformer retrained in the same job (8.44) & $+1.48$ & $+0.63$ to $+2.33$ | 9.91, 8.44, +1.48, +0.63, +2.33 |
| c19-074 | A71 | appendix_supp.tex | Above (3 of 3 seeds) | 3, 3 |
| c19-075 | A71 | appendix_supp.tex | O53 minus their GASLT retrained in the same job (9.47) & $+0.44$ & $-0.16$ to $+1.05$ | 9.47, +0.44, -0.16, +1.05 |
| c19-076 | A71 | appendix_supp.tex | Ocwb (9.63) minus Edw (10.20), beam width 5 & $-0.57$ & $-1.55$ to $+0.42$ | 9.63, 10.20, 5, -0.57, -1.55, +0.42 |
| c19-077 | A71 | appendix_supp.tex | Not resolved (0 of 3 seeds positive) | 0, 3 |
| c19-078 | A71 | appendix_supp.tex | Ocwb minus O53 (9.80), beam width 5 & $-0.17$ & $-1.14$ to $+0.80$ | 9.80, 5, -0.17, -1.14, +0.80 |
| c20-001 | A64 | appendix_supp.tex | Trained on the Communication and News subsets together (22,088 training clips against 12,439) | 22,088, 12,439 |
| c20-002 | needs the cluster | appendix_supp.tex | the configuration with 40 epochs of vision--language pretraining scored 8.89 on the official Communication test set at beam width 5, against 10.27 when trained on Communication alone | 40, 8.89, 5, 10.27 |
| c20-003 | needs the cluster | appendix_supp.tex | ($-1.38$; one run each, in different job submissions) | -1.38 |
| c20-004 | needs the cluster | appendix_supp.tex | Pretraining on News before training on Communication also scored below its concurrent control in both of two seeds ($-0.72$ and $-2.08$) | -0.72, -2.08, 2 |
| c20-005 | needs the cluster | appendix_supp.tex | distant frames in the trained translators were 3.4 to 6.6 times \emph{farther} apart than adjacent frames | 3.4, 6.6 |
| c20-006 | needs the cluster | appendix_supp.tex | Sign spotting by nearest-prototype matching, with prototypes from an \gls{islr} encoder, reached recall 0.05 | 0.05 |
| c20-007 | needs the cluster | appendix_supp.tex | and precision 0.01 on continuous clips | 0.01 |
| c20-008 | A76 | appendix_supp.tex | Qwen2.5-3B-Instruct) graded every output of four systems on the official test set for adequacy on a 0--4 scale | 2.5, 3, 4, 0, 4 |
| c20-009 | A76 | appendix_supp.tex | from 1.155 for the baseline model to 1.311 for the final ensemble | 1.155, 1.311 |
| c20-010 | A65 | appendix_supp.tex | between the final ensemble (1.311) and its \gls{rgb}-fusion member (1.259) is not resolvable in adequacy (95\% confidence interval $-0.01$ to $+0.11$; paired bootstrap over the 800 test clips) | 1.311, 1.259, 95, -0.01, 0.11, 800 |
| c20-011 | A74 | appendix_supp.tex | paired bootstrap over the 800 test clips | 800 |
| c20-012 | A78 | appendix_supp.tex | Under strict scoring it was 2.8\% for the output of the 40-epoch pre-trained model and 3.0\% for the final ensemble, against 3.8\% for the human reference sentences | 2.8, 40, 3.0, 3.8 |
| c20-013 | A78 | appendix_supp.tex | Scrambling the word order of the references raised it from 3.8\% to 21.3\% | 3.8, 21.3 |
| c20-014 | disagrees | appendix_supp.tex | the round trip added at most half a percentage point more \gls{wer} to the translator's output than to human text | 0.5 |
| c20-015 | A78 | appendix_supp.tex | 800 official test clips, 95\% bootstrap intervals in brackets | 800, 95 |
| c20-016 | A78 | appendix_supp.tex | Output of the 40-epoch pre-trained model (voiced) & 2.8\% [1.7, 4.1] & 1.1\% [0.7, 1.5] | 40, 2.8, 1.7, 4.1, 1.1, 0.7, 1.5 |
| c20-017 | disagrees | appendix_supp.tex | Final ensemble output & 3.0\% [1.8, 4.5] & 1.6\% [1.0, 2.6] | 3.0, 1.8, 4.5, 1.6, 1.0, 2.6 |
| c20-018 | A78 | appendix_supp.tex | Human reference sentences & 3.8\% [2.8, 5.0] & 2.2\% [1.6, 3.1] | 3.8, 2.8, 5.0, 2.2, 1.6, 3.1 |
| c20-019 | A78 | appendix_supp.tex | References with scrambled word order & 21.3\% [19.6, 23.0] & 19.4\% [17.9, 21.0] | 21.3, 19.6, 23.0, 19.4, 17.9, 21.0 |
| c20-020 | check not yet built | appendix_supp.tex | The 80-epoch pre-trained model scored 10.71 \gls{bleu}-4 on the 800-clip official Communication test set at beam width 5 in one run | 80, 10.71, 800, 5 |
| c20-021 | check not yet built | appendix_supp.tex | (10.89 with checkpoint averaging) | 10.89 |
| c20-022 | check not yet built | appendix_supp.tex | In two later job submissions the same training configuration scored 9.27 and 9.55 | 2, 9.27, 9.55 |
| c20-023 | needs the cluster | appendix_supp.tex | The 10.97 of duration-modelled synthetic data, a three-seed mean that exceeds its paired-seed control in every seed | 10.97, 3 |
| c20-024 | check not yet built | appendix_supp.tex | Two pairs of runs with identical seed, configuration, data and GPU node differed by 0.33 and 0.49 test \gls{bleu}-4 at beam width 5 | 2, 0.33, 0.49, 5 |
| c20-025 | needs the cluster | appendix_supp.tex | with deterministic kernels, two three-epoch runs with the same seed on the same node produced bit-identical weights | 2, 3 |
| c20-026 | check not yet built | appendix_supp.tex | Of three pairs of identical deterministic runs submitted separately, one was bit-identical and two were not (test \gls{bleu}-4 at beam width 5 differing by 0.62 and 0.43) | 3, 1, 2, 5, 0.62, 0.43 |
| c20-027 | covered by A25 (A1-A35 mapping, not reviewed) | appendix_supp.tex | Six runs of the baseline configuration with one seed in one job submission, with default kernels, had a standard deviation of 0.22 test \gls{bleu}-4 | 6, 0.22 |
| c20-028 | covered by A25 (A1-A35 mapping, not reviewed) | appendix_supp.tex | the pooled within-seed standard deviation of the adopted configuration across five job submissions was 0.42 | 5, 0.42 |
| c20-029 | check not yet built | appendix_supp.tex | the largest gap between two submissions of one configuration at one seed was 1.18 | 1.18 |
| c20-030 | covered by A25 (A1-A35 mapping, not reviewed) | appendix_supp.tex | The difference between two single runs has a 95\% range of about $\pm0.6$ within one job submission and $\pm1.2$ across job submissions | 95, 0.6, 1.2 |
| c20-031 | not a measurement | appendix_supp.tex | differences below about one point are treated as noise and the three-seed rule is read as consistency of sign | 1, 3 |
| c20-032 | needs the cluster | appendix_supp.tex | re-decoding the baseline model, the two pre-trained models, the \gls{rgb}-fusion model and the final ensemble gave each recorded test \gls{bleu}-4 to within 0.002 | 2, 0.002 |
| c20-033 | A73 | appendix_supp.tex | on the 800-clip official Communication test set at beam width 5 | 800, 5 |
| c20-034 | check not yet built | appendix_supp.tex | Dashed line and shaded band: mean and $\pm0.39$ standard deviation of six runs of the 1.1:1 configuration without augmentation | 0.39, 6, 1.1 |
| c20-035 | A73 | appendix_supp.tex | three-seed means; horizontal lines: range of the three seeds | 3 |
| c20-036 | A73 | appendix_supp.tex | The second and third rows are one configuration in two job submissions | 2, 3 |
| c21-001 | A37 | abstract.tex | fine-tuned with it, three seeds scored 18.27 | 18.27, three |
| c21-002 | A68 | abstract.tex | against 10.37 for a smaller translator of this thesis trained from scratch on real clips in the same jobs | 10.37 |
| c21-003 | A83 | abstract.tex | truncating one model's outputs to three words lowered it from 0.9497 to 0.8999 while BLEU-4 fell from 10.33 to 4.87 | 0.9497, 0.8999, 10.33, 4.87, one, three |
| c21-004 | A47 | abstract.tex | ROUGE-L of one set of outputs was 16.56 or 36.22 depending only on tokenisation | 16.56, 36.22, one |
| c21-005 | A68 | conclusion.tex | scored 18.21 after fine-tuning and 20.60 as an ensemble | 18.21, 20.60 |
| c21-006 | A68 | conclusion.tex | yielded 18.21 \gls{bleu}-4 (five-seed mean), while its five-model ensemble scored 20.60 | 18.21, 20.60, five, five |
| c21-007 | A65 | discussion.tex | The final ensemble of six translators scored 11.80, 1.47 above its RGB-fusion member (10.33; 95\% paired-bootstrap interval over the 800 test sentences, $+0.40$ to $+2.56$), 1.00 above its strongest member (not resolvable) and 1.85 above GA | 11.80, 1.47, 10.33, 95, 800, 0.40, 2.56, 1.00, 1.85, six |
| c21-008 | A57 | discussion.tex | share of the 2,043 reference content tokens | 2,043 |
| c21-009 | A57 | discussion.tex | Content words: words outside a 71-word function list. 786 official \textsc{Auslan-Daily} Communication test clips with a reference content word; beam width 5 (a, c) and 10 (b) | 71, 786, 5, 10 |
| c21-010 | no saved output | discussion.tex | 0.6319 for real signing in \textsc{Auslan-Daily}, 0.0872 for stitched signs (0.14 times) | 0.6319, 0.0872, 0.14 |
| c21-011 | A39 | discussion.tex | 4.68 in stitched clips, 0.25 in real clips with tracked keypoints | 4.68, 0.25 |
| c21-012 | A68 | intro.tex | scored 18.21 \gls{bleu}-4 on the official test set after fine-tuning (mean of five seeds), against 10.37 for a smaller translator of this thesis trained from scratch | 18.21, 10.37, five |
| c21-013 | A36 | intro.tex | and 20.60 as an ensemble of five | 20.60, five |
| c21-015 | A56 | method.tex | For 12,409 of the 12,411 multi-person clips (two lack a DWPose file) | 12,409, 12,411, two |
| c21-016 | A74 | method.tex | official 12,441 training, 800 validation and 800 test clips (230 seen-target, 570 unseen-target) | 12,441, 800, 800, 230, 570 |
| c21-017 | A77 | method.tex | signer-disjoint run 12,024 training clips of two signers, 1,087 held-out clips of 47 others | 12,024, 1,087, 47, two |
| c21-018 | A46 | method.tex | raises \gls{bleu}-4 of what is spoken from 10.27 to 19.19 at half coverage | 10.27, 19.19 |
| c21-019 | A72 | method.tex | trained at seeds 42, 101 and 202 (stage~1) and, if the change survives, at seeds 303 and 404 (stage~2) | 42, 101, 202, 303, 404 |
| c21-020 | check not yet built | method.tex | first 6,880, first 13,760 and all 20,608 clips give nested synthetic-to-real ratios of 0.55:1, 1.1:1 and 1.66:1 against the 12,439 real clips | 6,880, 13,760, 20,608, 0.55:1, 1.1:1, 1.66:1, 12,439 |
| c21-021 | needs the cluster | result.tex | Superseded runs on the first-listed person's keypoints gave 5.58 against 10.03 | 5.58, 10.03 |
| c21-022 | A55 | result.tex | DWPose higher in two of three seeds | two, three |
| c21-023 | A65 | result.tex | but its interval (9.87 to 13.89) includes 9.95 | 9.87, 13.89, 9.95 |
| c21-024 | A45 | result.tex | the 831 analysed crops of non-signing people | 831 |
| c21-025 | A65 | result.tex | scores 11.80 \gls{bleu}-4 and a \gls{wer} of 0.907 on the 800-clip official \textsc{Auslan-Daily} Communication test set, 1.85 \gls{bleu}-4 above the strongest published pose-input baseline (GASLT, 9.95) | 11.80, 0.907, 800, 1.85, 9.95 |
| c21-026 | A40 | appendix_supp.tex | about 0.35 of the 0.94 test gain comes from longer output; one-sided test $p = 0.057$ | 0.35, 0.94, 0.057, one |

## Where the saved data disagree with the dissertation

- **c04-011** (A49, review.tex): & Auslan & 9,665 / 700 / 700 & 18/17/17 & 12,346$^\dagger$  
  DISAGREES: dissertation says 9,665 News training clips, 18/17/17 signers, vocabulary 12,346 (the Auslan-Daily paper's Table 2 figure), the released spreadsheets give 9,669 clips (700/700 dev/test agree), 22/21/21 signers, vocabulary 18,893 (lower-cased whitespace tokens)
- **c04-023** (A49, review.tex): contains 9,665, 700 and 700 clips, signed by 18, 17 and 17 signers  
  DISAGREES: dissertation says 9,665 News training clips, 18/17/17 signers, vocabulary 12,346 (the Auslan-Daily paper's Table 2 figure), the released spreadsheets give 9,669 clips (700/700 agree), 22/21/21 signers, vocabulary 18,893
- **c04-025** (A49, review.tex): 25,106 clips, all splits pooled  
  DISAGREES: dissertation says 25,106 clips (the Auslan-Daily paper's Table 2 figure), the released spreadsheets give 25,110 (14,041 Communication + 11,069 News)
- **c04-037** (A49, review.tex): All 25,106 clips (22,106 in the training splits) are annotated with English sentences.  
  DISAGREES: dissertation says 25,106 clips, 22,106 in the training splits (the Auslan-Daily paper's Table 2 figure), the released spreadsheets give 25,110 clips, 22,110 in the training splits
- **c05-028** (A49, review.tex): it can use all 22,106 \textsc{Auslan-Daily} training clips  
  DISAGREES: dissertation says 22,106 training clips (the Auslan-Daily paper's Table 2 figure), the released spreadsheets give 22,110 training clips (12,441 + 9,669)
- **c06-021** (A56, method.tex): On 200 audited Communication clips  
  DISAGREES (not reproduced): the dissertation says normalised coordinates have a standard deviation of 87 under the xyxy reading. The audited sample IS reproduced (200 clips, 15,231 frames, 14,910 / 6,271 valid boxes, 88.9% / 14.4% inside), but the 5 Aug 2026 script that defined 'normalised coordinates' is lost; the natural reading (x/y divided by box size, all 136 keypoints, positive boxes) gives std 52.41 for xyxy and 0.28 for xywh, printed untagged by check.py. About 40 candidate definitions failed to match in the earlier numpy audit and a brute-force search here (valid-frame rule, centring, x/y/confidence pooling, joint subset) did not match either.
- **c06-023** (A56, method.tex): \texttt{xyxy} (assumed) & 14.4\% & 87 & $-5{,}036$ / $+12{,}912$  
  DISAGREES (partly): the 14.4% (xyxy, confident joints inside) is reproduced exactly on the identified audit sample (printed under c06-022), but std 87 and extrema -5,036 / +12,912 are not: the natural reading gives std 52.41 and extrema -45.5 / +6252.2. The original definition of the normalised coordinates is not recorded (script lost).
- **c06-024** (A56, method.tex): \texttt{xywh} (correct) & \textbf{88.9\%} & \textbf{0.60} & $-2.1$ / $+2.6$  
  DISAGREES (partly): the 88.9% (xywh, confident joints inside) is reproduced exactly on the identified audit sample (printed under c06-022), but std 0.60 and extrema -2.1 / +2.6 are not: the natural reading gives std 0.28 and extrema -1.0 / +1.9. The original definition of the normalised coordinates is not recorded (script lost).
- **c07-031** (A48, method.tex): One citation clip (mean 79.4 frames) is about as long as a real sentence clip (73.1 frames)  
  DISAGREES: dissertation says citation mean 79.4 frames; the shipped MM-WLAuslan pose (45,010 Train+Valid clips) gives 80.27. 73.1 for Auslan-Daily is reproduced (73.12). 79.4 may come from the DWPose re-extraction, which is not saved locally (the author's working notes lists its provenance as open).
- **c07-032** (A48, method.tex): Over the 45 glosses present in both datasets, signs in continuous signing were 72\% shorter in seconds than their citation forms  
  DISAGREES: dissertation says 72% shorter (3.60 times) over 45 shared glosses; pooled spans give 65% (2.89 times). The original paired per-gloss computation is not saved.
- **c07-038** (A48, method.tex): lasts 6.4 times as long in citation form (median 3.08\,s, $n=12$) as in conversation (median 0.48\,s, $n=6$)  
  DISAGREES on n only: medians 3.08 s and 0.48 s and the ratio 6.4 reproduce, but the saved data give n=14 citation clips and n=10 spotted spans, the dissertation says n=12 and n=6.
- **c08-010** (A65, result.tex): A 95\% paired-bootstrap interval over the 800 test sentences has a half-width of about 1.2 \gls{bleu}-4  
  DISAGREES: dissertation says the 95% paired-bootstrap half-width is about 1.2 BLEU-4; recomputed half-widths of the six rows of Table res_bootstrap are 1.08, 1.12, 1.15, 0.96, 1.19 and 1.13 (mean 1.11, range 0.96 to 1.19); with the two extra rows of Figure res_bootstrap the mean is 1.09. Only the largest (1.19) rounds to 1.2, so 'about 1.2' is the upper end and 'about 1.1' would be the typical value: partially consistent, not exact. The half-width lines are printed without a claim tag so that the claim is not counted as covered.
- **c09-075** (A66, result.tex): all three seeds: from 10.34 to 11.44, from 10.93 to 11.38 and from 10.66 to 11.19 (means 10.64  
  DISAGREES: dissertation says the second seed rose to 11.38; the saved hypotheses and the saved job output beam5_test.json give 11.3749, which rounds to 11.37. All other values (10.34, 11.44, 10.93, 10.66, 11.19, mean 10.64) reproduce.
- **c09-078** (A66, result.tex): Content-word precision rose by 1.6 to 2.3 points in every seed  
  DISAGREES: the dissertation says the pooled content-word precision rose by 1.6 to 2.3 points; with the 71-word content list its method chapter defines (stitch_lexicon.STOP) the saved hypotheses give rises of about +2.38, +1.72, +1.73 (the 75-word train_ctc.STOP list is printed as a labelled second line).
- **c12-068** (A79, result.tex): On the 800-clip official test set the mean difference was $-0.06$ ($-0.66$ to $+0.55$), and tracking scored lower in one of the two stage-2 seeds  
  DISAGREES (rounding): dissertation says the interval is -0.66 to +0.55; the saved five-seed decodes give -0.658 to +0.5447, which prints as +0.54. Mean -0.06, 2 of 5 seeds positive, and the Stage-2 seed result (one of two lower) agree.
- **c14-020** (A79, discussion.tex): produced no resolvable change in test \gls{bleu}-4 ($-0.06$; 95\% $t$-interval over five seeds, $-0.66$ to $+0.55$)  
  DISAGREES (rounding): dissertation says the interval is -0.66 to +0.55; the saved five-seed decodes give -0.658 to +0.5447, which prints as +0.54. Mean -0.06, 2 of 5 seeds positive, and the Stage-2 seed result (one of two lower) agree.
- **c14-042** (A59, discussion.tex): same on correctly and incorrectly identified spans (0.814 against 0.816)  
  DISAGREES: dissertation says hand confidence 0.814 on correctly and 0.816 on incorrectly identified spans; the saved per-span probe output (pred_dwpose.csv, hands descriptor, 334 scored spans) with hand_stats on the same spans gives 0.817 correct against 0.814 wrong (correlation +0.019). Same conclusion (no difference), different digits; the 0.814/0.816 pair came from an earlier probe run that is not saved.
- **c15-019** (A50, appendix_supp.tex): MM-WLAuslan citation clip, one sign (30\,fps) & 79.4  
  DISAGREES: dissertation says 79.4 mean frames per MM-WLAuslan clip, the shipped pose file gives 80.27 over 45,010 clips; the video-extracted frame counts of 8,581 of those recordings (stitching sidecar) equal the pose-file counts (78.66 vs 78.74), so no local source gives 79.4.
- **c16-022** (A71, appendix_supp.tex): the larger model used 60 epochs at learning rate $3\times10^{-4}$ against 40 at $5\times10^{-4}$  
  DISAGREES: the dissertation says the larger model used 60 epochs at learning rate 3e-4 against 40 at 5e-4. The metrics.json written by the job of the 7.11 run (d=512, 6+6 layers, DWPose) records 40 epochs at 5e-4 (cosine, batch 32), the same epochs and learning rate as the 7.92 default-size run (batch 64). 60 epochs at 3e-4 (cosine, batch 16 x 2) is recorded for a different run, the 25.9 M-class 512-d 3+3 model on first-listed AlphaPose keypoints (greedy test 2.29, comm_slt_1024_512_3x3_accum2), not for the 1.03 run (checkpoint settings: plateau schedule, 1e-3, 10 of 60 epochs run) and not for the 7.11 run. A submit script (train_dwpose_scaled.sbatch, 60 epochs, 3e-4) exists but no saved output of a run from it was found, so the sentence may mix two runs; the check prints all four recorded settings.
- **c17-013** (A79, appendix_supp.tex): at least one jump in 25.2\% of clips and on 2.31\% of frame transitions  
  DISAGREES: dissertation says 25.2% of clips and 2.31% of frame transitions; the jump definition of pose/tracker_v2/research/alphapose_identity_check.py applied to the same 2,128 clips gives 25.0% (531 clips) and 2.39% (3116/130353). The original statistic (recorded in the numpy audit; its script was not kept) used the larger of the x and y shift, so the definitions differ.
- **c18-010** (A53, appendix_supp.tex): $+0.73$ ($-0.43$ to $+1.96$)  
  DISAGREES: dissertation says +0.73 (-0.43 to +1.96); the saved decodes give +0.7286 [-0.4322, +1.9545] with the original tool's draws (2,000 resamples, seed 0), i.e. +1.95 at two decimals, not +1.96. The point value and lower bound agree.
- **c18-023** (A74, appendix_supp.tex): $+2.2$ to $+4.1$ ($p$ between 0.02 and 0.29)  
  DISAGREES: dissertation says +2.2 to +4.1 (p 0.02 to 0.29); saved decodes give +2.147 to +4.085 (RGB-fusion gap 2.147 rounds to +2.1; the +2.15 in the project record is a rounding of it) and a 2000-draw within-video permutation gives p 0.03 to 0.30 (original 300-draw run, seed not kept, is not recoverable). Differences are rounding and permutation noise.
- **c18-024** (A74, appendix_supp.tex): unseen-target clips $+1.1$ to $-0.3$  
  DISAGREES: dissertation says unseen-target gap +1.1 to -0.3 for the single models; the three single models give +0.61 to -0.28. The +1.11 is the S6 ensemble's gap, which is not a single model.
- **c19-030** (A68, appendix_supp.tex): positive in 5 of 5 seeds; $+5.79$ ($+4.82$ to $+6.77$) after adjusting for output length  
  DISAGREES (rounding only): dissertation says the length-adjusted interval is +4.82 to +6.77; saved decodes give +5.79 [+4.8187, +6.7648], which rounds to +6.76. All other numbers of the claim (5 of 5 seeds, +5.79, +4.82) agree.
- **c20-014** (A78, appendix_supp.tex): the round trip added at most half a percentage point more \gls{wer} to the translator's output than to human text  
  DISAGREES: appendix says that in every variant the round trip added at most half a percentage point to the translator's output; recomputed upper bound is +0.5 for the 40-epoch output (A) but +0.8 for the final ensemble (A6, strict scoring, 757 scramblable clips); the original job's recorded file gives the same +0.8
- **c20-017** (A78, appendix_supp.tex): Final ensemble output & 3.0\% [1.8, 4.5] & 1.6\% [1.0, 2.6]  
  DISAGREES: dissertation says strict interval [1.8, 4.5] for the final ensemble; recomputed upper bound is 4.446 (4.4); the original job's recorded file also has 0.04446. Other numbers of the row agree (3.0, 1.8, 1.6, 1.0, 2.6)
