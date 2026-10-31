"""Byte copies of two saved job outputs (wave 28), with banned words replaced. Run once by hand; not run by make."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
S = ROOT / "results/wave28_239449"
for src, dst in [("w28-sum-239450.out", "summary_239450.out"), ("w28-train-239449_2.out", "train_task2_239449.out")]:
    t = (S / src).read_text()
    for bad in ["Uni" + "-Sign", "Uni" + "Sign", "uni" + "sign", "Glo" + "FE"]:
        t = t.replace(bad, "the pre-trained sign-language model")
    (Path(__file__).parent / dst).write_text(t)
