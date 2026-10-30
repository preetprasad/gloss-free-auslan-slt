"""Write a deterministic manifest-level audit for pseudo-content labels."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slt.pseudo_labels import label_statistics


def texts(manifest: str) -> list[str]:
    with open(manifest, encoding="utf-8") as handle:
        return [line.rstrip("\n").split("\t", 1)[1] for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = label_statistics(texts(args.manifest))
    report["manifest"] = os.path.abspath(args.manifest)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
