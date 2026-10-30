"""Wave 12f, a local check from LABELS ONLY (no pose, no accuracy; not run on Kaya).

Arm A2 of measurement 1 excludes the reference clips cut from the query's OWN CLIP.  Communication clips of one
episode can overlap in time.  If a sign lies in the overlap of two clips and Sign_Spotting annotates it in both, the
second clip holds the SAME signing as a reference: a self-match that the own-clip exclusion would not remove.
This counts such pairs.

Position of a span in its episode = Begin_Frame of its clip (alignment table) + the span's frame in the clip.  The
alignment table has no clip name; a clip is joined on (episode, subtitle) where that pair is unique in both tables.

  python tools/w12f_overlap.py <spans_safe.csv written by `w12f_probe.py spans`> [<Dataset Split Table folder>]
"""
import collections
import csv
import os
import re
import sys

from w12f_probe import GLOSSES45


def overlap(b1, e1, b2, e2):
    """Frames shared by [b1, e1) and [b2, e2)."""
    return max(0, min(e1, e2) - max(b1, b2))


assert overlap(0, 10, 10, 20) == 0 and overlap(0, 10, 7, 20) == 3 and overlap(5, 6, 0, 100) == 1


def rows(path):
    import openpyxl
    r = list(openpyxl.load_workbook(path, read_only=True).active.iter_rows(values_only=True))
    col = {str(c): i for i, c in enumerate(r[0]) if c is not None}
    return [{k: x[i] for k, i in col.items()} for x in r[1:] if any(v is not None for v in x)]


def main():
    root = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/raw/Auslan-Daily/Dataset Split Table")
    tr = rows(f"{root}/Sign Language Translation/Ausan-Daily Communication/Auslan-Daily_Communication.xlsx")
    al = rows(f"{root}/Sign Language Alignment/Auslan_Daily_Communication.xlsx")
    key = lambda r: (str(r["Video_Name"]).strip(), str(r["Subtitle"]).strip())
    ct, ca = collections.Counter(map(key, tr)), collections.Counter(map(key, al))
    al_by = {key(r): r for r in al if ca[key(r)] == 1}
    begin = {str(r["Video_Clip_Name"]).strip(): int(al_by[key(r)]["Begin_Frame"]) for r in tr if ct[key(r)] == 1 and key(r) in al_by}
    signer = {str(r["Video_Clip_Name"]).strip(): r["Signer_ID"] for r in tr}
    print(f"clips {len(tr)} | placed in their episode by a unique (episode, subtitle): {len(begin)}")
    spans = [(re.sub(r"_signer\.mp4$", "", r["Video_Name_Clip"].strip()), r["Query_Gloss"].strip().lower(),
              int(r["Begin_Frame"]), int(r["End_Frame"])) for r in csv.DictReader(open(sys.argv[1]))]
    for title, sel in (("all spotting glosses", spans), ("45-gloss set", [s for s in spans if s[1] in GLOSSES45])):
        by = collections.defaultdict(list)
        for c, g, b, e in sel:
            if c in begin:
                by[(c.rsplit("_", 1)[0], g)].append((c, begin[c] + b, begin[c] + e))
        pairs = [(g, a, b, overlap(a[1], a[2], b[1], b[2]), signer.get(a[0]), signer.get(b[0]))
                 for (_, g), lst in by.items() for i, a in enumerate(lst) for b in lst[i + 1:]
                 if a[0] != b[0] and overlap(a[1], a[2], b[1], b[2])]
        print(f"{title}: spans {len(sel)}, placed {sum(len(v) for v in by.values())} | pairs of spans of one sign in "
              f"DIFFERENT clips that share episode frames: {len(pairs)}")
        for p in pairs:
            print("   sign %s: %s and %s share %d frames (Signer_ID %s and %s)" % p)


if __name__ == "__main__":
    main()
