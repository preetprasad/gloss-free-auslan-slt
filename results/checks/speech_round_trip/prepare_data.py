"""One-off: copy the minimal saved speech round-trip data into this folder (private tree only).
Run from the repository root of the private tree:  python3 results/checks/speech_round_trip/prepare_data.py
Transcripts of spoken REFERENCE text (arms B, C) are stored as character-level edits against the spoken text, so no
dataset sentence is stored outside the references themselves."""
import difflib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src" / "tools"))
import speech_roundtrip as S  # noqa: E402

SRC = ROOT / "results" / "speech_roundtrip"
d = json.load(open(SRC / "arms.json"))
tr = {a: [(SRC / a / f"{i:03d}.wav.txt").read_text().strip() for i in range(800)] for a in S.ARMS}


def delta(spoken, raw):
    ops = difflib.SequenceMatcher(None, spoken, raw, autojunk=False).get_opcodes()
    return [[t, i1, i2] if t == "equal" else [t, i1, i2, raw[j1:j2]] for t, i1, i2, j1, j2 in ops]


def apply(spoken, ops):
    return "".join(spoken[o[1]:o[2]] if o[0] == "equal" else (o[3] if o[0] != "delete" else "") for o in ops)


dl = {"B": [], "C": []}
for a in ("B", "C"):
    for i in range(800):
        ops = delta(d["arms"][a][i], tr[a][i])
        assert apply(d["arms"][a][i], ops) == tr[a][i]
        dl[a].append(ops)
with open(HERE / "items.jsonl", "w") as f:
    for i in range(800):
        f.write(json.dumps({"id": d["clips"][i], "hyp_A": d["arms"]["A"][i], "hyp_A6": d["arms"]["A6"][i],
                            "tr_A": tr["A"][i], "tr_A6": tr["A6"][i]}) + "\n")
json.dump(dl, open(HERE / "transcript_edits.json", "w"), separators=(",", ":"))
rec = json.load(open(SRC / "roundtrip_results.json"))
for k, v in rec.items():
    if isinstance(v, dict):
        v.pop("subs_top20", None)
json.dump(rec, open(HERE / "recorded_results.json", "w"), indent=1)
json.dump(json.load(open(SRC / "transcribe_metal.json")), open(HERE / "transcribe_metal.json", "w"), indent=1)
json.dump(json.load(open(SRC / "g2_pass.json")), open(HERE / "g2_pass.json", "w"), indent=1)
print("ok")
