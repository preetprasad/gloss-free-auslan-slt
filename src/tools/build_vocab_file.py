"""Build ONE fixed vocabulary (itos JSON) over the text of several manifests, with the trainer's own Vocab.build.

Used with `train_ctc.py --vocab_file` so that a pre-train and a fine-tune on different manifests index every word the
same way (Vocab.build orders words by frequency in the manifest it is given). Usage:
  python tools/build_vocab_file.py --out vocab.json --manifests real_train.tsv joint_13760.tsv syn_300k.tsv
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config  # noqa: E402
from slt.data import Vocab  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifests", nargs="+", required=True)
    a = ap.parse_args()
    cfg, texts = Config(), []
    for m in a.manifests:
        rows = [ln.rstrip("\n").split("\t")[1] for ln in open(m) if "\t" in ln]
        print(f"{m}: {len(rows)} rows")
        texts += rows
    v = Vocab.build(texts, cfg.min_freq, cfg.text_tokenization, cfg.sentencepiece_model)
    json.dump(v.itos, open(a.out, "w"))
    print(f"wrote {len(v.itos)} entries to {a.out} (tokenization {cfg.text_tokenization}, min_freq {cfg.min_freq})")


if __name__ == "__main__":
    main()
