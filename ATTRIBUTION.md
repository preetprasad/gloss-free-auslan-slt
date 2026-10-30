# Data sources and attribution

This repository contains no video, pose, subtitle or annotation file of the datasets named below. The datasets are the work of their authors and must be obtained from them under their terms.

## Auslan-Daily

- **Citation.** X. Shen, S. Yuan, H. Sheng, H. Du and X. Yu, "Auslan-Daily: Australian Sign Language Translation for Daily Communication and News", Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023.
- **Project page.** <https://uq-cvlab.github.io/Auslan-Daily-Dataset/>
- **Official release.** <https://drive.google.com/drive/folders/17E5wgq1ig7-WynNskG-8tTMoOnGlVh4V>
- **Licence.** The dataset's download page stated CC BY 4.0 when it was checked on 25 September 2026; the paper names CC BY-NC-ND 4.0. Users must follow the terms set by the dataset's authors.
- **Use in this work.** Auslan-Daily is the training and evaluation data of every model. The reported scores are computed against its English subtitles.
- **What this repository holds.** No file of the dataset, and no subtitle of four or more words. A script that needs a particular sentence (an example in a figure, a constant baseline) names it by its SHA-1 digest and reads the text from the spreadsheet. Everyday phrases of up to three words used as examples in code can coincide with subtitles. The saved outputs under `results/` hold the models' hypotheses, with each reference sentence replaced by its SHA-1 digest, and `results/train_target_hashes.txt` holds digests of the training targets. A hypothesis that coincides with a subtitle is the model's own output.
- **How the data is obtained.** `make data` copies `Dataset Split Table/Sign Language Translation/Ausan-Daily Communication/Auslan-Daily_Communication.xlsx` (the folder name is spelt this way in the release) from the official folder with rclone and restores the reference sentences locally. Video and pose files are fetched from the same folder by the job scripts in `jobs/slurm/`.

## MM-WLAuslan

- **Citation.** X. Shen et al., "MM-WLAuslan: Multi-View Multi-Modal Word-Level Australian Sign Language Recognition Dataset", Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2024, arXiv:2410.19488.
- **Official release.** <https://drive.google.com/drive/folders/1EQ1Nh3lidEcu1QLFw0IjRN7YqEq1N48q>
- **Licence.** CC BY-NC-SA 4.0, as stated by the dataset's authors.
- **Use in this work.** Front-view recordings of isolated signs serve as the lexicon from which synthetic training sentences are stitched. No result is evaluated on this dataset, and no file of it is included here.

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
