"""Prep (run once in the private tree, not by make): small numeric extras for the News check.
 data/news_clip_frames.csv  one column 'frames' = End_Frame - Begin_Frame of every row of the News alignment spreadsheet (numbers only, no text)
 data/real_train_count.json number of lines of the real Communication training list of job 21734 (the 12,439-clip DWPose list)
 data/ens5_val.log          byte copy of the five-member ensemble decode log of job 21744"""
import json, shutil
from pathlib import Path
import openpyxl
ROOT = Path(__file__).resolve().parents[3]; HERE = Path(__file__).resolve().parent
ws = openpyxl.load_workbook(ROOT / "data/raw/Auslan-Daily/Dataset Split Table/Sign Language Alignment/Auslan_Daily_News.xlsx", read_only=True).worksheets[0]
it = ws.iter_rows(values_only=True); h = list(next(it)); b, e = h.index("Begin_Frame"), h.index("End_Frame")
fr = [int(r[e]) - int(r[b]) for r in it if r[b] is not None]
(HERE / "data/news_clip_frames.csv").write_text("frames\n" + "\n".join(map(str, fr)) + "\n")
n = sum(1 for l in open(ROOT / "results/real_data_learning_curve/manifests/R100_s42.tsv") if l.strip())
json.dump({"clips": n}, open(HERE / "data/real_train_count.json", "w"))
shutil.copy(ROOT / "results/news_decoding_length_check/news_ens/ens5_val.log", HERE / "data/ens5_val.log")
print(len(fr), n)
