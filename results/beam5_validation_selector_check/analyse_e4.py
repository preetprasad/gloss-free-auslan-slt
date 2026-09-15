"""E4 — does BEAM-5 validation select arms better than GREEDY validation? (exploratory, post-hoc; no test re-selection)

Inputs (all produced this session, nothing hand-typed):
  results/beam5_validation_selector_check/<name>.json      tools/eval_beam.py --output_json on the 800-clip local validation manifest
  results/beam5_validation_selector_check/kaya_recorded.jsonl   per-run progress.json (val_best = greedy validation at best epoch)
  results/beam5_validation_selector_check/kaya_beam5_test_logs.txt  "=== ARM <name>" blocks with greedy / [beam5] test BLEU lines
  results/pose_augmentation_decodes/<arm>/beam5_test.json  beam-5 test for the pose-aug arms
Rules (pre-stated in the research-round-1 plan, E4):
  (0) harness: greedy validation from the re-decode must reproduce the recorded val_best within 0.006
  (i) "beam-5 validation agrees on ratio" only if r111 > r055 AND r111 > r166 in 3/3 seeds
  (ii) beam-5 validation is the better selector only if its Spearman correlation with beam-5 test exceeds greedy
       validation's in BOTH the ratio family and the comm-long family
"""
import json, os, re
import numpy as np

D = './results/beam5_validation_selector_check'
PA = './results/pose_augmentation_decodes'


def metric(block):
    m = block.get('metrics', {}) if isinstance(block, dict) else {}
    return m.get('tokenize=13a;detok=True')


# ---- local validation decodes ----
val = {}
for f in sorted(os.listdir(D)):
    if not f.endswith('.json') or f.startswith('kaya'):
        continue
    b = json.load(open(os.path.join(D, f)))
    if not isinstance(b, dict) or 'beam_result' not in b:   # skip this script's own e4_table.json and any non-decode file
        continue
    g, bm = metric(b.get('greedy', {})), metric(b.get('beam_result', {}))
    assert b.get('beam') == 5, (f, b.get('beam'))
    assert os.path.basename(b.get('manifest', '')) == '_local_val_manifest.tsv', (f, b.get('manifest'))
    val[f[:-5]] = {'greedy_val': g, 'beam5_val': bm}

# alias names from the first E4 pass
alias = {'r111_s42': 'sc_r111_s42', 'r111_s101': 'sc_r111_s101', 'r111_s202': 'sc_r111_s202',
         'aug_s42': 'pa_aug_s42', 'aug_s101': 'pa_aug_s101', 'aug_s202': 'pa_aug_s202'}
for a, c in alias.items():
    if a in val and c in val:
        same = abs(val[a]['beam5_val'] - val[c]['beam5_val']) < 1e-9 and abs(val[a]['greedy_val'] - val[c]['greedy_val']) < 1e-9
        print(f'determinism check {a} vs {c}: {"identical" if same else "DIFFERENT"}')
    if a in val and c not in val:
        val[c] = val[a]
    val.pop(a, None)

# ---- recorded greedy val_best per Kaya run dir ----
rec = {}
for line in open(os.path.join(D, 'kaya_recorded.jsonl')):
    line = line.strip()
    if not line.startswith('{'):
        continue
    r = json.loads(line)
    d = r['dir']; name = os.path.basename(d)
    key = name + ('_batch1' if d.startswith('stitch_scale/') and name in ('sc_ctrl_s101', 'sc_r055_s101', 'sc_r166_s101') else '')
    rec[key] = r

# ---- beam-5 test from job logs ----
test = {}
job = None; arm = None; seen_beam = False
for line in open(os.path.join(D, 'kaya_beam5_test_logs.txt')):
    m = re.match(r'#####\s+(\S+)', line)
    if m:
        job = m.group(1); continue
    m = re.search(r'=== ARM (\S+)', line)
    if m:
        arm = m.group(1); seen_beam = False; continue
    if '[beam5]' in line:
        seen_beam = True; continue
    m = re.search(r'tokenize=13a\s+detok=True\s+BLEU-4=\s*([\d.]+)', line)
    if m and arm and seen_beam:
        key = arm + ('_batch1' if job.startswith('stitch-scale-') and arm.endswith('_s101') else '')
        test[key] = float(m.group(1)); seen_beam = False
for a in ('pa_ctrl_s42', 'pa_ctrl_s101', 'pa_ctrl_s202', 'pa_aug_s42', 'pa_aug_s101', 'pa_aug_s202'):
    p = os.path.join(PA, a, 'beam5_test.json')
    if os.path.exists(p):
        test[a] = metric(json.load(open(p))['beam_result'])

# ---- table + harness ----
rows = []
print(f"\n{'checkpoint':22s} {'rec val_best':>12s} {'greedy val':>10s} {'beam5 val':>9s} {'beam5 test':>10s}  harness")
for k in sorted(val):
    rv = rec.get(k, {}).get('val_best')
    g, bv, t = val[k]['greedy_val'], val[k]['beam5_val'], test.get(k)
    ok = rv is not None and g is not None and abs(g - rv) < 0.006
    rows.append(dict(name=k, rec_val=rv, greedy_val=g, beam5_val=bv, beam5_test=t, harness=ok))
    print(f"{k:22s} {rv if rv is not None else float('nan'):12.3f} {g:10.3f} {bv:9.3f} {t if t is not None else float('nan'):10.2f}  {'OK' if ok else 'MISMATCH'}")
json.dump(rows, open(os.path.join(D, 'e4_table.json'), 'w'), indent=1)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


by = {r['name']: r for r in rows}
fam = {
    'ratio (in-batch per seed)': [n for n in by if re.match(r'sc_(ctrl|r055|r111|r166)_s(42|101|202)$', n)],
    'comm-long epochs': [n for n in by if n.startswith('cl_e')],
    'pose-aug': [n for n in by if n.startswith('pa_')],
}
print()
for fname, names in fam.items():
    names = [n for n in names if by[n]['beam5_test'] is not None]
    if len(names) < 3:
        print(f'{fname}: only {len(names)} checkpoints with test numbers so far'); continue
    t = [by[n]['beam5_test'] for n in names]
    sg = spearman([by[n]['greedy_val'] for n in names], t)
    sb = spearman([by[n]['beam5_val'] for n in names], t)
    print(f'{fname}: n={len(names)}  Spearman(greedy val, beam5 test)={sg:+.3f}  Spearman(beam5 val, beam5 test)={sb:+.3f}')

print('\nRule (i): within-seed ratio ordering on beam-5 validation (and on beam-5 test for reference)')
agree = 0; complete = 0
for s in ('42', '101', '202'):
    need = [f'sc_{a}_s{s}' for a in ('r055', 'r111', 'r166')]
    if not all(n in by for n in need):
        print(f'  s{s}: incomplete'); continue
    complete += 1
    v = {n.split('_')[1]: by[n]['beam5_val'] for n in need}
    te = {n.split('_')[1]: by[n]['beam5_test'] for n in need}
    ok = v['r111'] > v['r055'] and v['r111'] > v['r166']; agree += ok
    print(f"  s{s}: beam5 val r055 {v['r055']:.2f} r111 {v['r111']:.2f} r166 {v['r166']:.2f} -> r111 best: {ok} | "
          f"beam5 test r055 {te['r055']:.2f} r111 {te['r111']:.2f} r166 {te['r166']:.2f}")
print(f'  rule (i): {agree}/{complete} seeds' + ('' if complete == 3 else ' (incomplete)'))
