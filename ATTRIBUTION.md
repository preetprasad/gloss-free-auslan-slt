# Data sources and attribution

This repository contains no video, pose, subtitle or annotation file of the datasets named below, as released. Some saved outputs under `results/checks/` do hold extracts of their annotations; these are listed under "Annotation extracts" below. The datasets are the work of their authors and must be obtained from them under their terms.

## Auslan-Daily

- **Citation.** X. Shen, S. Yuan, H. Sheng, H. Du and X. Yu, "Auslan-Daily: Australian Sign Language Translation for Daily Communication and News", Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023.
- **Project page.** <https://uq-cvlab.github.io/Auslan-Daily-Dataset/>
- **Official release.** <https://drive.google.com/drive/folders/17E5wgq1ig7-WynNskG-8tTMoOnGlVh4V>
- **Licence.** The dataset's download page stated CC BY 4.0 when it was checked on 25 September 2026; the paper names CC BY-NC-ND 4.0. Users must follow the terms set by the dataset's authors.
- **Use in this work.** Auslan-Daily is the training and evaluation data of every model. The reported scores are computed against its English subtitles.
- **What this repository holds.** No file of the dataset as released, and no subtitle of four or more words. Extracts of its annotations are listed under "Annotation extracts" below. A script that needs a particular sentence (an example in a figure, a constant baseline) names it by its SHA-1 digest and reads the text from the spreadsheet. Everyday phrases of up to three words used as examples in code can coincide with subtitles. The saved outputs under `results/` hold the models' hypotheses, with each reference sentence replaced by its SHA-1 digest, and `results/train_target_hashes.txt` holds digests of the training targets. A hypothesis that coincides with a subtitle is the model's own output.
- **How the data is obtained.** `make data` copies `Dataset Split Table/Sign Language Translation/Ausan-Daily Communication/Auslan-Daily_Communication.xlsx` (the folder name is spelt this way in the release) from the official folder with rclone and restores the reference sentences locally. Video and pose files are fetched from the same folder by the job scripts in `jobs/slurm/`.

## MM-WLAuslan

- **Citation.** X. Shen et al., "MM-WLAuslan: Multi-View Multi-Modal Word-Level Australian Sign Language Recognition Dataset", Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2024, arXiv:2410.19488.
- **Official release.** <https://drive.google.com/drive/folders/1EQ1Nh3lidEcu1QLFw0IjRN7YqEq1N48q>
- **Licence.** CC BY-NC-SA 4.0, as stated by the dataset's authors.
- **Use in this work.** Front-view recordings of isolated signs serve as the lexicon from which synthetic training sentences are stitched. No result is evaluated on this dataset, and no file of it is included here as released. Extracts of its labels are listed under "Annotation extracts" below.

## Annotation extracts

Some checks re-derive their numbers from tables under `results/checks/` that hold values copied from the datasets' annotation files. The tables were selected and reformatted from the released files. They hold no video, no subtitles and no reference sentences. They are credited to the datasets' authors as above, are not covered by this repository's MIT licence, and remain under the terms of the dataset each comes from, as stated above.

| Dataset | What the tables hold | Tables under `results/checks/` |
|---|---|---|
| Auslan-Daily | Video and clip names, split labels, subtitle start and end frames, signer numbers | `dataset_statistics/data/alignment_rows.csv`, `duration_and_frame_rates/alignment_frames.csv`, `duration_and_frame_rates/clip_lengths.csv`, `citation_vs_conversational_signs/alignment_frames_seconds.csv`, `citation_vs_conversational_signs/auslan_daily_clip_frames.csv` |
| Auslan-Daily | Sign-spotting spans and gloss labels | `dataset_statistics/data/spotting.csv`, `citation_vs_conversational_signs/spotting.csv`, `duration_and_frame_rates/spotting_frames.csv`, `sign_identification_probes/spans_safe.csv`, `sign_identification_probes/spans45.csv`, `hand_speed_dwpose_frontend/data/spans45.csv` |
| Auslan-Daily | Isolated-sign gloss labels and their located frame spans | `dataset_statistics/data/isolated.csv`, `sign_identification_probes/located.csv` |
| Auslan-Daily | Clip names and split labels of the fingerspelling annotation | `dataset_statistics/data/fingerspelling.csv` |
| Auslan-Daily | Distributed keypoints of one clip (70 frames, 11 keypoints), beside this work's DWPose values | `extraction_error_audit/registration_video_11_6.csv` |
| MM-WLAuslan | Clip identifiers, gloss labels, frame counts and gloss counts | `citation_vs_conversational_signs/mmwl_clip_frames.csv`, `duration_and_frame_rates/mmwl_lengths.csv`, `duration_and_frame_rates/mmwl_video_frames.csv`, `dataset_statistics/data/mmwl_gloss_counts.csv` |

Clip names, split labels and gloss labels also appear beside this work's own results in other tables, such as the `pred_*.csv` probe predictions. Tables that hold only statistics computed by this work are not listed.

## Model weights

The weights of the live demonstrator were trained on both datasets. They are published separately at <https://huggingface.co/preetprasad/auslan-slt-live> under CC BY-NC-SA 4.0, with the same attribution.

## BibTeX

```bibtex
@inproceedings{shen2023auslandaily,
  title     = {{Auslan-Daily}: {Australian} Sign Language Translation for Daily Communication and News},
  author    = {Shen, Xin and Yuan, Shaozu and Sheng, Hongwei and Du, Heming and Yu, Xin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year      = {2023}
}

@inproceedings{shen2024mmwlauslan,
  title     = {{MM-WLAuslan}: Multi-View Multi-Modal Word-Level {A}ustralian Sign Language Recognition Dataset},
  author    = {Shen, Xin and Du, Heming and Sheng, Hongwei and Wang, Shuyun and Chen, Hui and Chen, Huiqiang and Wu, Zhuojie and Du, Xiaobiao and Ying, Jiaying and Lu, Ruihan and Xu, Qingzheng and Yu, Xin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year      = {2024},
  note      = {arXiv:2410.19488}
}
```
