"""Landmark Cox baselines: clinical-only, cross-sectional asymmetry, longitudinal asymmetry
(change from first visit and least-squares slope). Writes predictions in the common format."""
import numpy as np, pickle, json, os, sys
import pandas as pd
from lifelines import CoxPHFitter
from build_dataset import meta_vector

SITES = ["A", "B", "C", "D", "E"]
HOR = [6, 12, 18, 24]
PREP = sys.argv[1] if (__name__ == "__main__" and len(sys.argv) > 1) else "prep"
OUT = sys.argv[2] if (__name__ == "__main__" and len(sys.argv) > 2) else "runs"


def rows():
    split = {int(k): v for k, v in json.load(open("splits.json")).items()}
    R = []
    for s in SITES:
        pats = pickle.load(open(f"{PREP}/pats_{s}.pkl", "rb"))
        hc = np.load(f"{PREP}/hc_{s}.npy")
        for p in pats:
            H = hc[p["img_idx"]]
            t = np.array(p["visits"])
            m = meta_vector(p)
            for k in range(len(t)):
                cur = H[k]
                chg = H[k] - H[0]
                if k >= 1:
                    tt = t[:k + 1] - t[:k + 1].mean()
                    slope = (tt[:, None] * (H[:k + 1] - H[:k + 1].mean(0))).sum(0) / ((tt ** 2).sum() + 1e-9) * 12
                else:
                    slope = np.zeros(7)
                R.append(dict(pid=p["pid"], split=split[p["pid"]], k=k, tk=t[k], u=p["time"] - t[k], d=p["event"],
                              m=m, cur=cur, chg=chg, slope=slope, nprior=k))
    return R


def frame(R, kind):
    X = []
    for r in R:
        v = list(r["m"][[0, 1, 2, 3, 4]]) + list(r["m"][6:10]) + list(r["m"][11:14]) + list(r["m"][15:18]) + [r["m"][18]]
        if kind in ("asym", "long"):
            v += list(r["cur"])
        if kind == "long":
            v += list(r["chg"]) + list(r["slope"]) + [r["nprior"] / 5.0]
        X.append(v)
    X = np.array(X, float)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["u"] = np.clip([r["u"] for r in R], 1e-3, None)
    df["d"] = [r["d"] for r in R]
    return df


if __name__ == "__main__":
    R = rows()
    for kind, name in [("clin", "cox_clin"), ("asym", "cox_asym"), ("long", "cox_long")]:
        tr = [r for r in R if r["split"] == "train"]
        dtr = frame(tr, kind)
        mu, sd = dtr.iloc[:, :-2].mean(), dtr.iloc[:, :-2].std() + 1e-9
        dtr.iloc[:, :-2] = (dtr.iloc[:, :-2] - mu) / sd
        cph = CoxPHFitter(penalizer=0.05)
        cph.fit(dtr, duration_col="u", event_col="d")
        os.makedirs(f"{OUT}/{name}", exist_ok=True)
        for sp in ["val", "cal", "test", "extD", "extE"]:
            rs = [r for r in R if r["split"] == sp]
            df = frame(rs, kind)
            df.iloc[:, :-2] = (df.iloc[:, :-2] - mu) / sd
            S = cph.predict_survival_function(df.iloc[:, :-2], times=HOR).values.T  # (n,4)
            F = 1 - S
            Sprev = np.concatenate([np.ones((len(S), 1)), S[:, :-1]], 1)
            h = np.clip(1 - S / np.clip(Sprev, 1e-9, None), 1e-6, 1 - 1e-6)
            lg = np.log(h / (1 - h))
            P = np.column_stack([[r["pid"] for r in rs], [r["k"] for r in rs], [r["tk"] for r in rs],
                                 [r["u"] for r in rs], [r["d"] for r in rs], F, lg, np.ones(len(rs))])
            np.save(f"{OUT}/{name}/pred_{sp}.npy", P)
        json.dump(dict(nparam=int(len(cph.params_))), open(f"{OUT}/{name}/info.json", "w"))
        print(name, "done", flush=True)
