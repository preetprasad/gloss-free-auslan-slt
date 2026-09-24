import json, statistics as st, sys
sys.path.insert(0, '/group/pmc099/pprasad/wave5/code')
from slt.metrics import detokenize, corpus_bleu4
R='/group/pmc099/pprasad/wave5/runs'; M='/group/pmc099/pprasad/scratch_backup/dwpose/manifests'
T975={4:2.776}
train={" ".join(l.split("\t",1)[1].split()) for l in open(f"{M}/train.tsv") if "\t" in l}
def bleu(h,r): return corpus_bleu4([detokenize(x) for x in h],[detokenize(x) for x in r], tokenize="13a")
S=[42,101,202,303,404]; out={}
for arm in ('C','PJ'):
  for s in S:
    for split in ('val','test'):
      for ck in ('best','soup_last5'):
        j=json.load(open(f"{R}/{arm}_s{s}/beam5_{split}_{ck}.json")); H,Rf=j["beam_hyps"],j["refs"]
        seen=[" ".join(r.split()) in train for r in Rf]
        allb=bleu(H,Rf); assert abs(allb-j["beam_result"]["metrics"]["tokenize=13a;detok=True"])<0.01,(arm,s,split,ck,allb)
        sb=bleu([h for h,z in zip(H,seen) if z],[r for r,z in zip(Rf,seen) if z])
        ub=bleu([h for h,z in zip(H,seen) if not z],[r for r,z in zip(Rf,seen) if not z])
        out[(arm,s,split,ck)]=(allb,sb,ub,sum(seen))
print("seen counts val/test:", out[('C',42,'val','best')][3], out[('C',42,'test','best')][3])
def summ(name,d):
    m=st.mean(d); hw=T975[len(d)-1]*st.stdev(d)/len(d)**.5
    print(f"{name:34s} {' '.join(f'{x:+.2f}' for x in d)} | mean {m:+.3f} [{m-hw:+.3f}, {m+hw:+.3f}] pos {sum(x>0 for x in d)}/5")
for ck in ('best','soup_last5'):
  for split in ('val','test'):
    for k,nm in ((1,'SEEN'),(2,'UNSEEN')):
      summ(f"PJ-C {ck} {split} {nm}",[out[('PJ',s,split,ck)][k]-out[('C',s,split,ck)][k] for s in S])
