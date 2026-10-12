"""Refit the three landmark Cox baselines (same data/settings as cox_baselines.py) and apply them
to the stress-test cohorts. Usage: python cox_robust.py cond1 cond2 ..."""
import numpy as np, pickle, sys, os
import pandas as pd
from lifelines import CoxPHFitter
import cox_baselines as CB
from build_dataset import meta_vector

HOR = [6, 12, 18, 24]


def rows_dir(ddir, keep=1.0, seed=0):
    rng = np.random.default_rng(seed)
    R = []
    for s in "ABC":
        pats = pickle.load(open(f"{ddir}/pats_{s}.pkl", "rb"))
        hc = np.load(f"{ddir}/hc_{s}.npy")
        for p in pats:
            idx = list(range(len(p["visits"])))
            if keep < 1 and len(idx) > 1:
                idx = [0] + [j for j in idx[1:] if rng.random() < keep]
            H = hc[np.array(p["img_idx"])[idx]]
            t = np.array(p["visits"])[idx]
            m = meta_vector(p)
            for k in range(len(t)):
                chg = H[k] - H[0]
                if k >= 1:
                    tt = t[:k + 1] - t[:k + 1].mean()
                    slope = (tt[:, None] * (H[:k + 1] - H[:k + 1].mean(0))).sum(0) / ((tt ** 2).sum() + 1e-9) * 12
                else:
                    slope = np.zeros(7)
                R.append(dict(pid=p["pid"], split="rob", k=k, tk=t[k], u=p["time"] - t[k], d=p["event"], m=m, cur=H[k], chg=chg, slope=slope, nprior=k))
    return R


if __name__ == "__main__":
    R = CB.rows()
    models = {}
    for kind, name in [("clin", "cox_clin"), ("asym", "cox_asym"), ("long", "cox_long")]:
        tr = [r for r in R if r["split"] == "train"]
        dtr = CB.frame(tr, kind)
        mu, sd = dtr.iloc[:, :-2].mean(), dtr.iloc[:, :-2].std() + 1e-9
        dtr.iloc[:, :-2] = (dtr.iloc[:, :-2] - mu) / sd
        models[name] = (kind, mu, sd, CoxPHFitter(penalizer=0.05).fit(dtr, duration_col="u", event_col="d"))
    for cond in sys.argv[1:]:
        ddir, keep = (f"robust/{cond}", 1.0)
        if cond.startswith("keep"):
            ddir, keep = "robust/base", float(cond[4:])
        rs = rows_dir(ddir, keep)
        for name, (kind, mu, sd, cph) in models.items():
            df = CB.frame(rs, kind)
            df.iloc[:, :-2] = (df.iloc[:, :-2] - mu) / sd
            S = cph.predict_survival_function(df.iloc[:, :-2], times=HOR).values.T
            Sprev = np.concatenate([np.ones((len(S), 1)), S[:, :-1]], 1)
            h = np.clip(1 - S / np.clip(Sprev, 1e-9, None), 1e-6, 1 - 1e-6)
            P = np.column_stack([[r["pid"] for r in rs], [r["k"] for r in rs], [r["tk"] for r in rs], [r["u"] for r in rs],
                                 [r["d"] for r in rs], 1 - S, np.log(h / (1 - h)), np.ones(len(rs))])
            np.save(f"runs/{name}/pred_rob_{cond}.npy", P)
        print(cond, "done", flush=True)
