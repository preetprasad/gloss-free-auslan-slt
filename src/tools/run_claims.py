#!/usr/bin/env python3
"""Run the checks written in REPRODUCING.md and compare each with the output recorded there.

    python src/tools/run_claims.py REPRODUCING.md            # every check that needs only this repository
    python src/tools/run_claims.py REPRODUCING.md --only A4  # one check, with its full output
    python src/tools/run_claims.py --self_test

A check is a section headed "### A<n>". It is run when its "**Runs with:**" line names this repository and the
dataset spreadsheet (fetched by `make data`); a section that needs other files is listed as skipped. For a check that is run, every
```bash block is executed from src/ and must exit 0, and every number printed in the section's
```text block(s) (the recorded output) must occur in the new output. Exit status is non-zero if any check fails.
"""
import argparse, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
RESULTS = os.path.join(os.path.dirname(CODE), "results")
RUNNABLE = "**Runs with:** this repository and the dataset spreadsheet"
NUM = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w]|\.\d)")


def sections(md):
    """Yield (id, title, text) for each '### A<n>' section."""
    parts = re.split(r"^### (A\d+)\b(.*)$", md, flags=re.M)
    for i in range(1, len(parts), 3):
        yield parts[i], parts[i + 1].strip(" —-"), parts[i + 2].split("\n## ")[0]


def blocks(text, lang):
    return re.findall(r"^```" + lang + r"\n(.*?)^```", text, flags=re.M | re.S)


def numbers(text):
    return {m.group(0).lstrip("+") for m in NUM.finditer(text)}


def run(script):
    script = re.sub(r"\bpython3?\b", lambda m: f'"{sys.executable}"', script)
    r = subprocess.run(["bash", "-e", "-o", "pipefail", "-c", script], cwd=CODE, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONWARNINGS="ignore"))
    return r.returncode, r.stdout + r.stderr


def self_test():
    md = ("### A1 — toy\n" + RUNNABLE + "\n```bash\npython3 -c \"print('x 1.50 [-0.20, +3.00] n=7')\"\n```\n"
          "```text\nx 1.50 [-0.20, +3.00]\n```\n### A2 — skipped\n**Runs with:** the dataset\n```bash\nfalse\n```\n")
    secs = list(sections(md))
    assert [s[0] for s in secs] == ["A1", "A2"]
    assert numbers("x 1.50 [-0.20, +3.00] v2 A9") == {"1.50", "-0.20", "3.00"}
    rc, out = run(blocks(secs[0][2], "bash")[0])
    assert rc == 0 and numbers(blocks(secs[0][2], "text")[0]) <= numbers(out)
    assert numbers("9.99") - numbers(out) == {"9.99"}          # a number that is absent is reported as missing
    assert RUNNABLE not in secs[1][2]
    print("run_claims self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doc", nargs="?")
    ap.add_argument("--only", default="", help="comma-separated check ids; prints the full output")
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--strict-output", action="store_true", help="compare labelled output lines in order, rather than a bag of numbers")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if os.path.exists(os.path.join(RESULTS, "SHA256SUMS")) and not os.path.exists(os.path.join(RESULTS, ".references_restored")):
        sys.exit("The reference sentences have not been restored yet. Run `make data` first (README, section Data).")
    only = {x for x in a.only.split(",") if x}
    passed, failed, skipped = [], [], []
    selected = list(sections(open(a.doc, encoding="utf-8").read()))
    unknown = only - {s[0] for s in selected}
    if unknown:
        sys.exit("Unknown check ids: " + ", ".join(sorted(unknown)))
    if not selected:
        sys.exit("No A-numbered checks found; refusing an empty success")
    for cid, title, text in selected:
        if only and cid not in only:
            continue
        if RUNNABLE not in text:
            skipped.append(cid); print(f"{cid:4s} SKIP  needs files that are not distributed  | {title[:70]}"); continue
        t0 = time.time(); out_all, ok = "", bool(blocks(text, "bash")) and bool(blocks(text, "text"))
        for b in blocks(text, "bash"):
            rc, out = run(b); out_all += out; ok &= rc == 0
        want = set().union(*[numbers(b) for b in blocks(text, "text")]) if blocks(text, "text") else set()
        missing = sorted(want - numbers(out_all))
        ok &= not missing
        if a.strict_output:
            from audit_dissertation_claims import missing_lines
            wrong_lines = missing_lines("\n".join(blocks(text, "text")), out_all)
            ok &= not wrong_lines
            if wrong_lines:
                print(f"{cid}: labelled output mismatch: {wrong_lines[:3]}")
        (passed if ok else failed).append(cid)
        print(f"{cid:4s} {'OK  ' if ok else 'FAIL'}  {len(want) - len(missing)}/{len(want)} recorded numbers reproduced"
              f"  {time.time() - t0:6.1f}s | {title[:60]}" + (f"\n       missing: {missing[:12]}" if missing else ""), flush=True)
        if only or not ok:
            print("       " + "\n       ".join(out_all.strip().split("\n")[-25 if not only else None:]))
    print(f"\n{len(passed)} reproduced, {len(failed)} failed, {len(skipped)} skipped (need files that are not distributed)")
    sys.exit(1 if failed or not passed else 0)


if __name__ == "__main__":
    main()
