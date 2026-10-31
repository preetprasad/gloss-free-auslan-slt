# Sources

All files in `data/` are written by `prepare_data.py` (run once in the private tree, not by make) from the release's tables. No subtitle text is saved.

| file | made from | notes |
|---|---|---|
| `data/news_counts.json` | `data/auslan-daily/translation/Auslan-Daily_News.xlsx` | clip, signer and vocabulary counts per split |
| `data/spotting.csv` | `Dataset Split Table/Sign Spotting/Sign_Spotting.xlsx` | clip id, query gloss, split |
| `data/fingerspelling.csv` | `Dataset Split Table/Fingerspelling Detection/Fingerspelling_Detection.xlsx` | clip id and split only (the spelled strings are dropped) |
| `data/isolated.csv` | `Dataset Split Table/Isolated Sign Language Recognition/Isolated_Sign_Language_Recognition.xlsx` | gloss, split |
| `data/mmwl_gloss_counts.csv` | MM-WLAuslan `Annotation/Labels & Split/Train.json` and `Valid.json` | clips per gloss |
| `data/dwpose_train_clip_ids.txt` | `results/real_data_learning_curve/manifests/R100_s42.tsv` (training list of Kaya job 21734, clip names only) | the 12,439-clip list used for the DWPose input |
| `data/alignment_rows.csv` | `Dataset Split Table/Sign Language Alignment/Auslan_Daily_Communication.xlsx`, joined to the Communication translation table on (video, subtitle) pairs unique in both | begin/end frame, clip name, signer; no text |
| `data/dwpose_clip_frames.csv` | frame counts of `pose/extraction/dwpose_clips/*.npz` (657 locally held clips) | for the extraction-exactness check |

The Communication translation table (`data/auslan-daily/translation/Auslan-Daily_Communication.xlsx`) is read directly (restored by `make data`).
`output.txt` is the output of `check.py` as run.
