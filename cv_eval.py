import numpy as np, metrics as MT, os, json
res={}
for cfg in ['gated_attn','gated_none','concat_attn','concat_none']:
    Ps=[np.load(f'runs_cv/{cfg}_f{f}/pred_val.npy') for f in range(5) if os.path.exists(f'runs_cv/{cfg}_f{f}/pred_val.npy')]
    if not Ps: continue
    per=[MT.auc_t(P[:,6],P[:,3],P[:,4],12.0,MT.km_censor(P[:,3],P[:,4])) for P in Ps]
    P=np.concatenate(Ps); G=MT.km_censor(P[:,3],P[:,4])
    res[cfg]=dict(folds=len(Ps),auc12_mean=float(np.mean(per)),auc12_sd=float(np.std(per)),auc12_pooled=float(MT.auc_t(P[:,6],P[:,3],P[:,4],12.0,G)),
                  ibs=float(MT.ibs(P,G)),c24=float(MT.uno_c(P[:,8],P[:,3],P[:,4],24.0,G)))
    print(cfg,{k:round(v,4) for k,v in res[cfg].items()})
json.dump(res,open('results/cv_select.json','w'),indent=1)
