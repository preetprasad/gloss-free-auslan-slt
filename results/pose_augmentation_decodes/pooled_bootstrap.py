"""Pooled paired bootstrap across seeds for job 1215231 (pose augmentation).
Resample the 800 official test sentences with replacement ONCE per draw and average, over the three seeds,
the paired difference BLEU(aug_s) - BLEU(ctrl_s) on that same resample. Scored exactly as tools/paired_bootstrap.py
(slt.metrics.detokenize on hyps and refs, corpus_bleu4 13a). Seen = reference verbatim in the real train manifest."""
import hashlib, json, os, sys, numpy as np
D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(D))
sys.path.insert(0, os.path.join(REPO, 'src'))
from slt.metrics import corpus_bleu4, detokenize
def load(arm):
    rows = sorted((json.loads(l) for l in open(f'{D}/{arm}/hyps_beam5.jsonl')), key=lambda r: r['idx'])
    return [r['hyp_beam'] for r in rows], [r['ref'] for r in rows]
seeds = (42, 101, 202)
H, refs0 = {}, None
for s in seeds:
    for k in ('ctrl', 'aug'):
        h, r = load(f'pa_{k}_s{s}')
        refs0 = refs0 or r
        assert r == refs0, 'reference mismatch'
        H[(k, s)] = [detokenize(x) for x in h]
R = [detokenize(x) for x in refs0]
norm = lambda t: ' '.join(t.lower().split())
MANIFEST = os.path.join(REPO, 'results', '_kaya_train_manifest.tsv')            # training subtitles, when on disk
HASHES = os.path.join(REPO, 'results', 'train_target_hashes.txt')               # otherwise their SHA-1 digests
if os.path.exists(MANIFEST):
    train = {norm(l.split('\t', 1)[1]) for l in open(MANIFEST, encoding='utf-8') if '\t' in l}
    seen = np.array([norm(r) in train for r in refs0])
else:
    digests = set(open(HASHES).read().split())
    seen = np.array([hashlib.sha1(norm(r).encode('utf-8')).hexdigest() in digests for r in refs0])
def mean_delta(ii):
    Rk = [R[i] for i in ii]
    return np.mean([corpus_bleu4([H[('aug', s)][i] for i in ii], Rk, '13a') - corpus_bleu4([H[('ctrl', s)][i] for i in ii], Rk, '13a') for s in seeds])
rng = np.random.default_rng(0)
for name, pool in (('all', np.arange(len(R))), ('unseen', np.flatnonzero(~seen))):
    obs = mean_delta(pool)
    draws = pool[rng.integers(0, len(pool), size=(2000, len(pool)))]
    d = np.array([mean_delta(ii) for ii in draws])
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f'[pooled over seeds 42/101/202, {name} n={len(pool)}] mean(aug - ctrl) observed {obs:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(>0) {100*np.mean(d>0):.1f}%  -> {"significant" if lo > 0 else "not resolvable"}', flush=True)
