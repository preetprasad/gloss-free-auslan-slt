"""One-off preparation (NOT run by make): shrink the saved per-clip decodes and grounding scores to the fields the check needs.
usage: python prep_data.py <private results dir>
 decodes_single.json / decodes_ens.json : per set {clip: ce} (+ 'frozen' ce for the single model's signers; non-signer crops as [ce, hyp])
 ground_single.json / ground_ens.json   : per set {clip: G} (sig_test and neg: [G, X1_logratio]) from the wave-18 per-clip scores
 b3.json                                : the B3 same-person flagged crop ids (read from the saved job summary)
 """
import json, sys
from pathlib import Path
R = Path(sys.argv[1]); here = Path(__file__).parent
for name, src in (("single", R / "nonsigner/decodes.json"), ("ens", R / "nonsigner_s6/decodes.json")):
    D = json.load(open(src)); out = {}
    for s in ("sig_test", "sig_val") + (("trk_test", "trk_val") if name == "ens" else ()):
        out[s] = {c: v["ce"] for c, v in D[s].items()}
    out["neg"] = {c: [v["ce"], v["hyp"]] for c, v in D["neg"].items()}
    if name == "single":
        out["frozen"] = {s: {c: v["frozen"] for c, v in D[s].items()} for s in ("sig_test", "sig_val")}
    out["zero"] = D["zero"]
    json.dump(out, open(here / f"decodes_{name}.json", "w"), separators=(",", ":"))
    sc = json.load(open(R / "wave18_171529/kaya" / f"scores_{name}.json"))
    g = {s: {c: ([v["G"], v["X1_logratio"]] if s in ("sig_test", "neg") else v["G"]) for c, v in sc[s].items()} for s in ("sig_val", "sig_test", "neg")}
    json.dump(g, open(here / f"ground_{name}.json", "w"), separators=(",", ":"))
rs = json.load(open(R / "nonsigner_s6/nonsigner_gate_s6_results.json"))
json.dump({"flagged_ids": rs["B3"]["flagged_ids"]}, open(here / "b3.json", "w"))
