import json, os, statistics as st
R='/group/pmc099/pprasad/wave5/runs'; K="tokenize=13a;detok=True"
T975={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571}
def b5(d,split,ck):
    return json.load(open(f"{d}/beam5_{split}_{ck}.json"))["beam_result"]["metrics"][K]
rec={'C':[11.307,10.522,11.338,10.996,11.424],'PJ':[11.403,11.195,12.200,11.900,12.800]}
S=[42,101,202,303,404]; rows={}
for arm in ('C','PJ'):
    for i,s in enumerate(S):
        d=f"{R}/{arm}_s{s}"; p=json.load(open(f"{d}/progress.json"))
        h=[r for r in p["history"] if "bleu4" in r]; mx=max(r["bleu4"] for r in h)
        r=dict(bv=b5(d,'val','best'),bt=b5(d,'test','best'),sv=b5(d,'val','soup_last5'),st=b5(d,'test','soup_last5'),
               ep=p["best"]["epoch"],near=sum(r["bleu4"]>=mx-0.3 for r in h),n=len(h))
        assert abs(r['bv']-rec[arm][i])<0.002, (arm,s,r['bv'],rec[arm][i])
        r['I']=(r['bv']-r['sv'])-(r['bt']-r['st']); rows[(arm,s)]=r
        print(f"{arm} s{s}: best v/t {r['bv']:.3f}/{r['bt']:.3f} soup v/t {r['sv']:.3f}/{r['st']:.3f} ep {r['ep']} near {r['near']}/{r['n']} I {r['I']:+.3f}")
def summ(name,d):
    m=st.mean(d); hw=T975[len(d)-1]*st.stdev(d)/len(d)**.5
    print(f"{name}: {' '.join(f'{x:+.3f}' for x in d)} | mean {m:+.3f} [{m-hw:+.3f}, {m+hw:+.3f}] | pos {sum(x>0 for x in d)}/{len(d)}")
    return m
print('GATE OK: best.pt beam-5 val reproduces PREREG_WAVE5 s11 for all 10 runs')
dI=[rows[('PJ',s)]['I']-rows[('C',s)]['I'] for s in S]; mI=summ('I(PJ)-I(C)',dI)
earlier=sum(rows[('PJ',s)]['ep']<rows[('C',s)]['ep'] for s in S); print('PJ best epoch earlier than C:',earlier,'/5')
ov=sum(rows[('PJ',s)]['bv']-rows[('PJ',s)]['sv']>=0.3 and rows[('PJ',s)]['bt']-rows[('PJ',s)]['st']>=0.3 for s in S)
print('PJ best beats PJ soup by >=0.3 on both val and test:',ov,'/5')
summ('PJ-C soup val',[rows[('PJ',s)]['sv']-rows[('C',s)]['sv'] for s in S])
summ('PJ-C soup test',[rows[('PJ',s)]['st']-rows[('C',s)]['st'] for s in S])
summ('PJ-C best val',[rows[('PJ',s)]['bv']-rows[('C',s)]['bv'] for s in S])
summ('PJ-C best test',[rows[('PJ',s)]['bt']-rows[('C',s)]['bt'] for s in S])
summ('PJ val-test gap (best)',[rows[('PJ',s)]['bv']-rows[('PJ',s)]['bt'] for s in S])
summ('C  val-test gap (best)',[rows[('C',s)]['bv']-rows[('C',s)]['bt'] for s in S])
art = mI>=0.4 and sum(x>0 for x in dI)>=4 and earlier>=4
print('READING:', 'SELECTION ARTEFACT' if art else ('EARLY-PEAK OVERFIT' if ov>=4 else 'UNRESOLVED at n=5'))
