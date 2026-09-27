"""Independent adversarial verification of the wave-6 readout (jobs 16435 + 25863).

Reads the raw eval_beam JSON dumps only (never analyse_wave6.py / summary_*.json / family.json output), re-scores every
beam-5 decode through slt.metrics (detokenize + corpus_bleu4 13a), cross-checks the 2-decimal value printed in each
decode log, gates on the Stage-1 values recorded in PREREG_WAVE6 before printing anything new, then applies
PREREG_WAVE6 section 4 as written.  Run from src:
  ~/.pyenv/versions/3.14.3/bin/python ../results/wider_corpus_joint_training/verify_wave6.py
"""
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path('.')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'tools'))
from slt.metrics import corpus_bleu4, corpus_wer, detokenize  # noqa: E402
from stitch_lexicon import STOP  # noqa: E402

RUNS = ROOT / 'results/wider_corpus_joint_training/runs'
KEY = 'tokenize=13a;detok=True'
SEEDS = {'A': [42, 101, 202, 303, 404], 'B': [42, 101, 202, 303, 404], 'C': [42, 101, 202]}
T975 = {2: 4.303, 3: 3.182, 4: 2.776}

# ---------- load + guard every dump ----------
D, refs_by_split = {}, {}
for arm, seeds in SEEDS.items():
    for s in seeds:
        for split in ('val', 'test'):
            for ck, ckfile in (('best', 'best.pt'), ('soup_last5', 'soup_last5.pt')):
                p = RUNS / f'{arm}_s{s}' / f'beam5_{split}_{ck}.json'
                j = json.load(open(p))
                assert j['checkpoint'].endswith(f'/wave6/runs/{arm}_s{s}/{ckfile}'), (p, j['checkpoint'])
                assert j['manifest'].endswith(f'/{split}.tsv'), (p, j['manifest'])
                assert j['beam'] == 5 and abs(j['length_penalty'] - 0.6) < 1e-9, p
                assert j['beam_result']['n'] == 800 == len(j['beam_hyps']) == len(j['refs']), p
                H = [detokenize(h) for h in j['beam_hyps']]
                R = [detokenize(r) for r in j['refs']]
                b = corpus_bleu4(H, R, '13a')
                assert abs(b - j['beam_result']['metrics'][KEY]) < 1e-6, (p, b, j['beam_result']['metrics'][KEY])
                # third source: the eval log's printed 13a value after the [beam5] header
                log = p.with_suffix('.log').read_text()
                m = re.search(r'\[beam5\].*?tokenize=13a\s+detok=True\s+BLEU-4=([\d.]+)', log, re.S)
                assert m and abs(float(m.group(1)) - b) < 0.006, (p, m and m.group(1), b)
                refs_by_split.setdefault(split, j['refs'])
                assert j['refs'] == refs_by_split[split], f'{p}: refs differ from other runs'
                D[(arm, s, split, ck)] = dict(H=H, R=R, raw_h=j['beam_hyps'], raw_r=j['refs'], bleu=b)
assert refs_by_split['val'] != refs_by_split['test']
test_manifest_txt = [l.rstrip('\n').split('\t', 1)[1] for l in open(ROOT / 'results/real_data_learning_curve/test.tsv') if '\t' in l]
assert test_manifest_txt == refs_by_split['test'], 'test refs != wave8 test.tsv text column'
print(f'loaded {len(D)} dumps; all re-score through slt.metrics and match JSON (1e-6) and log (2dp); '
      f'test refs == wave8 test.tsv 800/800; val refs identical across runs')

bleu = lambda a, s, sp, ck='best': D[(a, s, sp, ck)]['bleu']

# ---------- GATE: recorded Stage-1 values (PREREG_WAVE6 Stage-1 outcome section) ----------
rec = {('A', 42): (11.307, 11.711), ('A', 101): (11.939, 11.366), ('A', 202): (11.291, 10.839),
       ('B', 42): (11.446, 11.672), ('B', 101): (11.314, 11.236)}
for (a, s), (v, t) in rec.items():
    assert round(bleu(a, s, 'val'), 3) == v and round(bleu(a, s, 'test'), 3) == t, (a, s, bleu(a, s, 'val'), bleu(a, s, 'test'))
rd = {('B', 'val'): [0.138, -0.625, 0.516], ('B', 'test'): [-0.039, -0.131, 0.811],
      ('C', 'val'): [-0.159, -1.325], ('C', 'test'): [-1.59, -1.28]}
for (a, sp), vals in rd.items():
    got = [bleu(a, s, sp) - bleu('A', s, sp) for s in (42, 101, 202)[:len(vals)]]
    tol = 0.006 if a == 'C' and sp == 'test' else 0.0005   # prereg quotes C test at 2dp (-1.28 = double-rounded -1.2749)
    assert all(abs(g - v) <= tol for g, v in zip(got, vals)), (a, sp, got)
assert round(np.mean([bleu('B', s, 'val') - bleu('A', s, 'val') for s in (42, 101, 202)]), 3) == 0.010
print('GATE PASSED: every recorded Stage-1 value reproduces (A, B absolutes; B-A val/test; C-A val/test at s42/s101)')

def summ(x, name):
    x = np.asarray(x, float); n = len(x); m = x.mean(); sd = x.std(ddof=1); se = sd / np.sqrt(n)
    tq = stats.t.ppf(0.975, n - 1); assert abs(tq - T975[n - 1]) < 1e-3
    t = m / se; p1 = stats.t.sf(t, n - 1)
    print(f'  {name:26s} ' + ' '.join(f'{v:+.3f}' for v in x) + f' | {int((x > 0).sum())}/{n} pos | mean {m:+.3f} '
          f'95% t(df{n-1}) [{m - tq*se:+.3f}, {m + tq*se:+.3f}] | one-sided t {t:+.3f} p {p1:.3f}')
    return m, p1

# ---------- criterion deltas ----------
print('\nAbsolute beam-5 best.pt val/test')
for a, ss in SEEDS.items():
    print(f'  {a}: ' + '  '.join(f's{s} {bleu(a, s, "val"):.3f}/{bleu(a, s, "test"):.3f}' for s in ss))
print('\nPaired deltas vs in-batch A (best.pt)')
dl = {(a, sp): [bleu(a, s, sp) - bleu('A', s, sp) for s in SEEDS[a]] for a in 'BC' for sp in ('val', 'test')}
cval_m, _ = summ(dl[('C', 'val')], 'C-A val (Stage-1 criterion)')
summ(dl[('C', 'test')], 'C-A test (descriptive)')
bval3 = np.mean(dl[('B', 'val')][:3])
print(f'  B Stage-1 3-seed val mean {bval3:+.4f} -> {"CONTINUE" if bval3 > 0 else "STOP"};  '
      f'C 3-seed val mean {cval_m:+.4f} -> {"CONTINUE" if cval_m > 0 else "STOP"};  '
      f'C_s202 val delta {dl[("C","val")][2]:+.3f} vs survival threshold > +1.484 (= 0 - (-0.159 - 1.325))')
bval_m, _ = summ(dl[('B', 'val')], 'B-A val 5 seeds')
btest_m, bp = summ(dl[('B', 'test')], 'B-A test 5 seeds')

# ---------- Holm m=2, section 4 as written ----------
P = {'B': bp, 'C': 1.0 if cval_m <= 0 else None}
order = sorted(P, key=P.get); rejected = {}; alive = True
for k, arm in enumerate(order):
    lvl = 0.05 / (len(order) - k); rej = alive and P[arm] < lvl; alive = rej; rejected[arm] = rej
    print(f'  Holm rank {k+1}: {arm} p {P[arm]:.3f} level {lvl:.4f} -> {"REJECT" if rej else "retain"}')
fresh = dl[('B', 'test')][3:]
print(f'  B conditions: holm_reject={rejected["B"]} val5_mean>0={bval_m > 0} ({bval_m:+.3f}) '
      f'fresh s303/s404 test {fresh[0]:+.3f}/{fresh[1]:+.3f} both>0={all(f > 0 for f in fresh)}'
      f' -> {"ADOPT" if rejected["B"] and bval_m > 0 and all(f > 0 for f in fresh) else "NOT adopted"}')

# ---------- mandated readouts (descriptive), own implementation of the R2 definitions ----------
norm = lambda t: ' '.join(t.lower().split())
train_txt = [l.rstrip('\n').split('\t', 1)[1] for l in open(ROOT / 'results/real_data_learning_curve/train.tsv') if '\t' in l]
assert len(train_txt) == 12439
train = {norm(t) for t in train_txt}
unseen = [i for i, r in enumerate(refs_by_split['test']) if norm(r) not in train]
assert len(unseen) == 570, len(unseen)
cw = lambda t: [w for w in t.lower().split() if w.isalpha() and w not in STOP]
def ro(a, s):
    d = D[(a, s, 'test', 'best')]; hit = nh = nr = 0
    for h, r in zip(d['raw_h'], d['raw_r']):
        ch, cr = cw(h), cw(r); hit += sum((Counter(ch) & Counter(cr)).values()); nh += len(ch); nr += len(cr)
    return dict(unseen=corpus_bleu4([d['H'][i] for i in unseen], [d['R'][i] for i in unseen], '13a'),
                cprec=100 * hit / nh, crec=100 * hit / nr, wer=corpus_wer(d['H'], d['R']),
                lenr=sum(len(h.split()) for h in d['H']) / sum(len(r.split()) for r in d['R']), distinct=len(set(d['H'])))
RO = {(a, s): ro(a, s) for a in SEEDS for s in SEEDS[a]}
print('\nAbsolute unseen-570 BLEU: ' + '  '.join(f'{a}_s{s} {RO[(a, s)]["unseen"]:.3f}' for a in SEEDS for s in SEEDS[a]))
print('Readout deltas on test best.pt (descriptive only, never criteria)')
for a in 'BC':
    for k in ('unseen', 'cprec', 'crec', 'wer', 'lenr', 'distinct'):
        summ([RO[(a, s)][k] - RO[('A', s)][k] for s in SEEDS[a]], f'{a}-A {k}')

print('Last-5 soup (never a criterion)')
for a in 'BC':
    for sp in ('val', 'test'):
        summ([bleu(a, s, sp, 'soup_last5') - bleu('A', s, sp, 'soup_last5') for s in SEEDS[a]], f'{a}-A soup {sp}')

# ---------- section 5 descriptive: hypothesis tokens outside the AD real-train token vocabulary ----------
vocab = {w for t in train_txt for w in t.split()}
print(f'\nSection-5 check: AD real-train whitespace token types = {len(vocab)}')
for a in SEEDS:
    for s in SEEDS[a]:
        toks = [w for h in D[(a, s, 'test', 'best')]['raw_h'] for w in h.split()]
        oov = [w for w in toks if w not in vocab]
        print(f'  {a}_s{s}: {len(oov)}/{len(toks)} OOV tokens {sorted(set(oov))}')

# ---------- R2 rule 7 item not in the readout: pooled sentence bootstrap, "test sampling only" ----------
Rt = D[('A', 42, 'test', 'best')]['R']
def pooled(a, ii):
    Rk = [Rt[i] for i in ii]
    return np.mean([corpus_bleu4([D[(a, s, 'test', 'best')]['H'][i] for i in ii], Rk, '13a')
                    - corpus_bleu4([D[('A', s, 'test', 'best')]['H'][i] for i in ii], Rk, '13a') for s in SEEDS[a]])
rng = np.random.default_rng(0)
for a in 'BC':
    obs = pooled(a, np.arange(800))
    draws = rng.integers(0, 800, size=(1000, 800))
    d = np.array([pooled(a, ii) for ii in draws]); lo, hi = np.percentile(d, [2.5, 97.5])
    print(f'  pooled sentence bootstrap {a}-A test (TEST SAMPLING ONLY, 1000 draws, seed 0): '
          f'obs {obs:+.3f} 95% [{lo:+.3f}, {hi:+.3f}] P(>0) {100*np.mean(d > 0):.1f}%')
