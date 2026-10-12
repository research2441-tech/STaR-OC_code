import numpy as np, pickle, json
split={int(k):v for k,v in json.load(open('splits.json')).items()}
out={}
for s in 'ABCDE':
    P=pickle.load(open(f'prep/pats_{s}.pkl','rb'))
    nv=np.array([len(p['visits']) for p in P]); ev=np.array([p['event'] for p in P]); fu=np.array([p['time'] for p in P])
    gaps=np.concatenate([np.diff(p['visits']) for p in P if len(p['visits'])>1])
    lt=np.bincount([p['ltype'] for p in P],minlength=5)/len(P); gr=np.bincount([p['grade'] for p in P],minlength=4)/len(P)
    sp={}
    for p in P: sp[split[p['pid']]]=sp.get(split[p['pid']],0)+1
    out[s]=dict(n=len(P),visits=int(nv.sum()),med_visits=float(np.median(nv)),events=int(ev.sum()),event_rate=float(ev.mean()),
       med_fu=float(np.median(fu)),med_gap=float(np.median(gaps)),age=float(np.mean([p['age'] for p in P])),female=float(np.mean([p['female'] for p in P])),
       smoke=float(np.mean([p['smoke'] for p in P])),areca=float(np.mean([p['areca'] for p in P])),ltype=lt.tolist(),grade=gr.tolist(),split=sp,
       single=float(np.mean(nv==1)))
    print(s,{k:(round(v,3) if isinstance(v,float) else v) for k,v in out[s].items()})
tot={}
for spn in ['train','val','cal','test','extD','extE']:
    n=0;e=0;v=0
    for s in 'ABCDE':
        for p in pickle.load(open(f'prep/pats_{s}.pkl','rb')):
            if split[p['pid']]==spn: n+=1;e+=p['event'];v+=len(p['visits'])
    tot[spn]=(n,e,v); print(spn,n,e,v)
out['splits']=tot
json.dump(out,open('results/cohort.json','w'),indent=1)
