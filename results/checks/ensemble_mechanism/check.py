"""Why ensembling works: S6 ensemble vs its members, mechanism, oracle, MBR, seed ensembles (dissertation Results, Appendix B).

Recomputes from the saved official-test decodes of the reported ensemble (results/final_ensemble_test_decodes, beam 10, LP 0.3), of its six members decoded
alone (results/final_ensemble_member_test_decodes), of three single models at beam 5 (results/semantic_adequacy_grading), the training manifest
(comm_run/manifests/train.tsv after make data) and the training-target digests. Bootstraps use the original tool's draws
(2,000 resamples, seed 0) through per-sentence BLEU sufficient statistics; the full-set score is gated against slt.metrics.
"""
import hashlib, itertools, json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from slt.metrics import detokenize, corpus_bleu4, corpus_wer

S6 = ROOT / "results/final_ensemble_test_decodes/hyps_LP=0.3.jsonl"
MEMDIR = ROOT / "results/final_ensemble_member_test_decodes"
SG = ROOT / "results/semantic_adequacy_grading"


def load(p, key="hyp_beam"):
    rows = sorted((json.loads(l) for l in open(p)), key=lambda r: r["idx"])
    return [r[key] for r in rows], [r["ref"] for r in rows]


E, refs = load(S6)
n = len(refs)
R = [detokenize(r) for r in refs]
bleu = lambda H: corpus_bleu4([detokenize(h) for h in H], R)
b_e = bleu(E)
print(f"GATE ensemble S6 official test BLEU-4 {b_e:.3f} (recorded 11.802)")
if abs(b_e - 11.802) > 0.0015:
    sys.exit("GATE FAILED")

man = ROOT / "comm_run/manifests/test.tsv"
if man.exists():
    m = [l.rstrip("\n").split("\t", 1)[1].strip() for l in open(man) if "\t" in l]
    assert m == [r.strip() for r in refs], "references differ from the test manifest"
    print("references match comm_run/manifests/test.tsv: 800/800")

# ---- composition (names of the checkpoints; the job script lists the same six) ----
sb = (ROOT / "jobs/slurm/mbr_wer_test2.sbatch").read_text()
ck = re.findall(r"\$G/(comm_dwpose_[a-z0-9_]+)/best\.pt", sb)
dirs = sorted(p.name for p in MEMDIR.iterdir() if p.is_dir())
assert sorted(ck) == dirs and len(ck) == 6
cfgs = {m["member"]: m for m in json.load(open(HERE / "members_config.json"))}  # config stored inside each checkpoint (prepare_data.py)
assert sorted(cfgs) == sorted(ck)
gb = (ROOT / "jobs/slurm/gpu_batch_10h.sbatch").read_text()
default_ve = int(re.search(r"VE=\$\{3:-(\d+)\}", gb).group(1))
calls = {m.group(2): int(m.group(3) or default_ve) for m in re.finditer(r"(?:^|;\s*)vlp_ctc (\d+) (\w+)(?: (\d+))?\s*(?:;|$)", gb, re.M)}  # tag -> pre-training epochs
dv = (ROOT / "jobs/slurm/dwpose_vlp.sbatch").read_text()
vlp_dw = re.search(r'pretrain_vlp\.py(?:.|\n)*?--epochs (\d+)(?:.|\n)*?--out_dir "\$DE/runs/vlp_dwpose"', dv)
calls["dwpose"] = int(vlp_dw.group(1))


def pretrain_epochs(name):
    """epochs of the pose-text pre-training behind a member's encoder: its saved init_encoder points at a pre-training run
    (or, for an RGB-fusion member, at the pose model that was itself started from one); the epochs are set in the job scripts"""
    init = cfgs[name]["init_encoder_tail"]
    mt = re.fullmatch(r"runs/vlp_(\w+)/vlp_encoder_best\.pt", init)
    if mt:
        return calls[mt.group(1)]
    mt = re.fullmatch(r"runs/ctc_(\w+)/best\.pt", init)  # fusion warm start from the pose model trained from vlp_<tag>
    if mt:
        return calls[mt.group(1)]
    if init == "comm_dwpose_vlp_ctc/best.pt":
        return pretrain_epochs("comm_dwpose_vlp_ctc")
    raise KeyError(init)


print(f"\nThe ensemble job lists {len(ck)} members [c01-006 c03-006 c07-025]")
fam = Counter()
for c in ck:
    m = cfgs[c]
    if m["rgb_feat_dir_set"]:
        label, key = f"RGB-fusion (fine-tuned from {m['init_encoder_tail']})", "RGB-fusion"
    else:
        pe_ = pretrain_epochs(c)
        label, key = f"pose-only, encoder pre-trained {pe_} epochs", f"pre-trained {pe_} epochs"
    fam[key] += 1
    print(f"  member {c}: {label}; fine-tuned {m['epochs']} epochs, seed {m['seed']}")
assert all(cfgs[c]["epochs"] == 40 for c in ck)
print(f"composition derived from the config stored in each checkpoint (members_config.json) and the pre-training epochs set in the job scripts "
      f"gpu_batch_10h.sbatch and dwpose_vlp.sbatch: {fam['pre-trained 80 epochs']} model pre-trained 80 epochs, {fam['pre-trained 40 epochs']} models "
      f"pre-trained 40 epochs (seeds {sorted(cfgs[c]['seed'] for c in ck if not cfgs[c]['rgb_feat_dir_set'] and pretrain_epochs(c) == 40)}), "
      f"{fam['RGB-fusion']} RGB-fusion models (seeds {sorted(cfgs[c]['seed'] for c in ck if cfgs[c]['rgb_feat_dir_set'])})  [c07-026 c14-101]")
assert (fam["pre-trained 80 epochs"], fam["pre-trained 40 epochs"], fam["RGB-fusion"]) == (1, 2, 3)
print(f"members pre-trained for 40 and 80 epochs: {sorted({k for k in fam if k.startswith('pre-trained')})}  [c14-101]")
P = {c: cfgs[c]["parameters_without_position_tables"] for c in ck}
pose_p = {P[c] for c in ck if not cfgs[c]["rgb_feat_dir_set"]}; fus_p = {P[c] for c in ck if cfgs[c]["rgb_feat_dir_set"]}
assert len(pose_p) == 1 and len(fus_p) == 1
print(f"GATE parameters per member (checkpoint tensors without the fixed position tables): pose-only {pose_p.pop()/1e6:.4f} M, RGB-fusion {fus_p.pop()/1e6:.4f} M (recorded in the 26 Sep audit: 9.0481 / 10.9238)")
tot = sum(P.values())
assert round(tot / 1e6, 4) == 59.9155, tot
print(f"six members together: {tot/1e6:.2f} M parameters (about 60 M)  [c08-004 c11-002]")
print("ensemble decoded at beam 10, length penalty 0.3 (log-probabilities averaged by ensemble_mbr_wer.py)")

# ---- fast paired bootstrap on sufficient statistics (same draws as tools/paired_bootstrap.py: seed 0, 2000) ----
B = BLEU()


def stat(H):
    return np.array(B._extract_corpus_statistics([detokenize(h) for h in H], [R]), dtype=np.int64)


def score(s):
    return BLEU.compute_bleu([int(x) for x in s[2:6]], [int(x) for x in s[6:10]], int(s[0]), int(s[1]), smooth_method="exp").score


assert abs(score(stat(E).sum(0)) - b_e) < 1e-9
NB = 2000
draws_all = np.random.default_rng(0).integers(0, n, size=(NB, n))


def boot(SA, SB, pool=None):
    if pool is None:
        d = draws_all; pool_ = np.arange(n)
    else:
        pool_ = pool; d = pool_[np.random.default_rng(0).integers(0, len(pool_), size=(NB, len(pool_)))]
    obs = score(SA[pool_].sum(0)) - score(SB[pool_].sum(0))
    deltas = np.array([score(SA[i].sum(0)) - score(SB[i].sum(0)) for i in d])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return obs, lo, hi, 100 * np.mean(deltas > 0)


SE = stat(E)
mem, SM = {}, {}
for c in ck:
    h, r = load(MEMDIR / c / "hyps_LP=0.3.jsonl"); assert r == refs
    mem[c] = h; SM[c] = stat(h)
sc = {c: score(SM[c].sum(0)) for c in ck}
print("\nMEMBERS decoded alone (official test, beam 10, LP 0.3)")
for c in ck:
    print(f"  {c:28s} BLEU-4 {sc[c]:.2f}")
strongest = max(ck, key=sc.get)
print(f"strongest member: {strongest} {sc[strongest]:.2f} (80-epoch pre-trained model)  [c12-005 c12-014]")

print("\nPAIRED BOOTSTRAP, ensemble minus member (2,000 sentence resamples, seed 0)")
nbelow = 0
res = {}
for c in ck:
    o, lo, hi, p = boot(SE, SM[c]); res[c] = (o, lo, hi, p)
    ex = lo > 0
    nbelow += ex
    print(f"  ensemble - {c}: {o:+.2f} [{lo:+.2f}, {hi:+.2f}] ahead in {p:.1f}% -> {'interval excludes zero' if ex else 'not resolvable'}")
print(f"members scoring below the ensemble with an interval that excludes zero: {nbelow} of {len(ck)}  [c12-013]")
o, lo, hi, p = res[strongest]
print(f"Ensemble {b_e:.2f} vs strongest member ({sc[strongest]:.2f}): {o:+.2f} [{lo:+.2f}, {hi:+.2f}], ahead in {p:.1f}%  [c12-005 c12-014]")
print(f"Ensemble ({b_e:.2f}) vs strongest member ({sc[strongest]:.2f}) 80-epoch, difference {o:+.2f}; interval includes zero: {lo < 0 < hi}  [c12-005 c12-014]")

# ---- RGB-fusion single model, beam 5 ----
rf, r2 = load(SG / "hyps_rgbfusion.jsonl"); assert r2 == refs
SR = stat(rf); b_rf = score(SR.sum(0))
o, lo, hi, p = boot(SE, SR)
print(f"\nRGB-fusion model alone, beam 5: BLEU-4 {b_rf:.2f}  (recorded 10.335 in the ladder job)")
print(f"Final ensemble {b_e:.2f} vs its RGB-fusion member {b_rf:.2f}: {o:+.2f} [{lo:+.2f}, {hi:+.2f}], ahead in {p:.1f}%  [c12-002 c12-004]")

# ---- unseen subset ----
digests = set(open(ROOT / "results/train_target_hashes.txt").read().split())
norm = lambda t: " ".join(t.lower().split())
seen = np.array([hashlib.sha1(norm(r).encode("utf-8")).hexdigest() in digests for r in refs])
unseen = np.flatnonzero(~seen)
# slow path through slt.metrics for this pair, exactly as tools/paired_bootstrap.py does it
Dh = [detokenize(h) for h in E]; Dm_ = [detokenize(h) for h in mem[strongest]]
dr = unseen[np.random.default_rng(0).integers(0, len(unseen), size=(NB, len(unseen)))]
Rp = [R[i] for i in unseen]
o = corpus_bleu4([Dh[i] for i in unseen], Rp) - corpus_bleu4([Dm_[i] for i in unseen], Rp)
dl = np.array([corpus_bleu4([Dh[i] for i in ii], [R[i] for i in ii]) - corpus_bleu4([Dm_[i] for i in ii], [R[i] for i in ii]) for ii in dr])
lo, hi = np.percentile(dl, [2.5, 97.5]); p = 100 * np.mean(dl > 0)
print(f"\nUnseen references (n={len(unseen)}): ensemble minus strongest member {o:+.2f} [{lo:+.2f}, {hi:+.2f}], ahead in {p:.1f}%  [c18-010]")
print(f"  unrounded: {o:+.4f} [{lo:+.4f}, {hi:+.4f}]  [c18-010]")
print(f"  DISAGREES: the dissertation prints the upper bound as +1.96; the saved outputs give {hi:+.4f}, which rounds to {hi:+.2f}  [c18-010]")

# ---- mechanism: novel outputs ----
chrf = lambda h, r: sacrebleu.sentence_chrf(detokenize(h), [detokenize(r)]).score
M = [mem[c] for c in ck]
k = len(M)
cE = [chrf(E[i], refs[i]) for i in range(n)]
cM = [[chrf(M[j][i], refs[i]) for i in range(n)] for j in range(k)]
novel = [i for i in range(n) if E[i] not in {M[j][i] for j in range(k)}]
above = sum(cE[i] > max(cM[j][i] for j in range(k)) for i in novel)
below = sum(cE[i] < min(cM[j][i] for j in range(k)) for i in novel)
print(f"\nMECHANISM: ensemble sentence equal to no member's on {len(novel)} of {n} clips = {100*len(novel)/n:.1f}% (about 20%)  [c12-015]")
print(f"on those clips the ensemble's sentence chrF is above every member's on {above} and below all on {below}  [c12-016]")
nn = above + below
pv = stats.binomtest(above, nn, 0.5).pvalue
print(f"two-sided sign test over the {nn} clips with a decisive sign: p = {pv:.3f}  [c12-016]")

# ---- six saved outputs per sentence (3 models x greedy and beam) behind the oracle line of the BLEU-WER figure ----
pool = []
for f in ("hyps_dwpose_ctc", "hyps_vlp", "hyps_rgbfusion"):
    rows_ = sorted((json.loads(l) for l in open(SG / f"{f}.jsonl")), key=lambda r: r["idx"])
    assert [r["ref"] for r in rows_] == refs
    for key in ("hyp_greedy", "hyp_beam"):
        pool.append([detokenize(r[key]) for r in rows_])
print(f"\nOracle pool: {len(pool)} saved outputs per sentence (3 models x greedy and beam; six)  [c12-052]")
pick = [min(range(len(pool)), key=lambda j: (corpus_wer([pool[j][i]], [R[i]]), j)) for i in range(n)]  # reference-aware choice, lowest sentence WER
ob, ow = corpus_bleu4([pool[pick[i]][i] for i in range(n)], R), corpus_wer([pool[pick[i]][i] for i in range(n)], R)
print(f"GATE per-sentence oracle over the six outputs (lowest sentence WER, reference in hand): BLEU-4 {ob:.2f}, WER {ow:.3f} (recorded in the BLEU-WER figure data: 14.33, 0.807)")
if round(ob, 2) != 14.33 or round(ow, 3) != 0.807:
    sys.exit("GATE FAILED")
print(f"best of six saved outputs per sentence (oracle): BLEU-4 {ob:.2f}, WER {ow:.3f}; the six outputs choose by sentence WER from {len(set(pick))} different sources  [c12-052]")

# ---- cross-system correlation of difficulty ----
a, _ = load(SG / "hyps_dwpose_ctc.jsonl"); v, _ = load(SG / "hyps_vlp.jsonl")
ca = np.array([chrf(a[i], refs[i]) for i in range(n)]); cv = np.array([chrf(v[i], refs[i]) for i in range(n)])
rr = np.corrcoef(ca, cv)[0, 1]
print(f"\nSentence chrF of the CTC and the pre-trained single model (beam 5) correlate r = {rr:.2f}  [c13-019]")

# ---- MBR vs plain length penalty ----
mb, _ = load(ROOT / "results/final_ensemble_test_decodes/hyps_MBR-WER.jsonl")
Dm = [detokenize(h) for h in mb]; De = [detokenize(h) for h in E]
print(f"\nMBR-WER decoding of the ensemble: BLEU-4 {corpus_bleu4(Dm, R):.2f}, WER {corpus_wer(Dm, R):.3f}  [c18-009]")
print(f"length-penalty decoding of the ensemble (beam 10, LP 0.3): BLEU-4 {b_e:.2f}, WER {corpus_wer(De, R):.3f}  [c18-009]")

# ---- training-sentence oracle (needs the training manifest from make data) ----
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/tools"))
import stitch_lexicon
STOP = stitch_lexicon.STOP  # the 71-word stop list of the original failure analysis
cw = lambda t: [w for w in t.lower().split() if w.isalpha() and w not in STOP]
tm = ROOT / "comm_run/manifests/train.tsv"
train = sorted({l.rstrip("\n").split("\t", 1)[1].strip() for l in open(tm) if "\t" in l})
tc = [Counter(cw(t)) for t in train]
tot = hit = 0
for r in refs:
    c = Counter(cw(r))
    s = sum(c.values())
    if s == 0:
        continue
    tot += s
    hit += max(sum((c & t).values()) for t in tc)
print(f"\nTraining-sentence oracle: the best single training sentence per clip (chosen with the answer in hand) contains "
      f"{100*hit/tot:.1f}% of the reference content words, pooled over the clips that have one  [c10-037]")

# ---- seed ensembles (job 21736), per-batch rows saved by the job ----
rows = json.load(open(HERE / "seedens_rows.json"))
d = np.array([r["d_test"] for r in rows])
mu = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); t = stats.t.ppf(0.975, len(d) - 1)
print(f"\nSEED ENSEMBLES (read from the saved job output seedens_rows.json, job 21736)")
print(f"independent job submissions with adopted-recipe seeds: {len(rows)} (seven)  [c18-002]")
print(f"seeds per ensemble: {sorted({len(r['singles']) for r in rows})} (three)  [c18-003]")
print(f"test BLEU-4, ensemble of three minus validation-best seed: mean {mu:+.2f} (95% t-interval {mu-t*se:+.2f} to {mu+t*se:+.2f}); "
      f"higher in {int((d > 0).sum())} of {len(d)} (five of seven)  [c18-001]")
