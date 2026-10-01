"""Do spotted Auslan signs occur in the same order as their English words in the subtitle? (cited by
results/PREREG_WAVE10_SIGNORDER_2026-10-01.md)
Pairs of spotted signs in one clip; each gloss must be spotted once and match exactly one word position in the subtitle;
pairs with the same Begin_Frame are skipped. Output on 2026-10-01: 16 clips, 15 pairs, 12 in English order, 3 reversed
(video_29_177 POT BOTTOM, video_61_142 FOOTPRINT MUDDY, video_83_90 PIE ... IF). Several English-order pairs are
single-letter glosses or overlapping spans, and the 120 spotted glosses skew to findable content signs, so this shows
that the order sometimes differs, not how often. Run from the repo root with pandas + openpyxl."""
import glob, re, itertools
import pandas as pd
base = "data/raw/Auslan-Daily/Dataset Split Table/"
s = pd.read_excel(base + "Sign Spotting/Sign_Spotting.xlsx")
s["clip"] = s["Video_Name_Clip"].str.replace(r"_signer\.mp4$", "", regex=True)
comm = pd.read_excel(glob.glob(base + "Sign Language Translation/Ausan-Daily Communication/*.xlsx")[0])
sub = dict(zip(comm["Video_Clip_Name"].astype(str), comm["Subtitle"].astype(str)))
agree = disagree = skipped = clips = 0
examples = []
for clip, g in s.groupby("clip"):
    if len(g) < 2 or clip not in sub:
        continue
    words = re.findall(r"[a-z']+", sub[clip].lower())
    pos = {}
    for _, r in g.iterrows():
        gl = str(r["Query_Gloss"]).lower().strip()
        hits = [i for i in range(len(words)) if " ".join(words[i:i + len(gl.split())]) == gl]
        if len(hits) == 1:
            pos.setdefault(gl, []).append((r["Begin_Frame"], hits[0]))
    items = [v[0] for v in pos.values() if len(v) == 1]   # gloss spotted once, word appears once
    if len(items) < 2:
        skipped += 1; continue
    clips += 1
    for (f1, w1), (f2, w2) in itertools.combinations(items, 2):
        if f1 == f2: continue
        if (f1 < f2) == (w1 < w2): agree += 1
        else:
            disagree += 1
            if len(examples) < 8: examples.append((clip, sub[clip], sorted(pos.items(), key=lambda kv: kv[1][0][0])))
print(f"clips with >=2 usable spotted signs: {clips} (skipped {skipped}); sign pairs: {agree + disagree}")
print(f"same order as English: {agree}  reversed: {disagree}")
for e in examples:
    print(e[0], "|", e[1], "| signed order:", [k for k, _ in e[2]])
