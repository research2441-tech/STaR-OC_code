"""Signal-recovery bounds. For every landmark, the noise-free lesion thermal excess is the mean
over the lesion disc of T(prog) - T(prog = 0), rendered with nominal session conditions and no
camera, registration or inflammation. A landmark Cox model on clinical variables plus this excess
gives the attainable ceiling; clinical variables alone give the floor. Writes results/ceiling.json."""
import numpy as np, pickle, json, os
import pandas as pd
from types import SimpleNamespace
from lifelines import CoxPHFitter
import oralthermsim as S
import metrics as MT
from build_dataset import meta_vector

CACHE = "results/clean_excess.pkl"


def nominal_session(site):
    c = S.SITE_CFG[site]
    return dict(tenv=c["amb"], evap=1.1, grad=0.0, sfield=0.0, inflam=[])


def clean_excess(d, site, rng):
    p = SimpleNamespace(**d)
    q = SimpleNamespace(**d)
    q.tstar, q.benign_drift = 1e9, 0.0
    disk = ((S.xx - p.lx) ** 2 + (S.yy - p.ly) ** 2) < (p.r0 * 1.2) ** 2
    out = []
    for t in p.visits:
        T1, _, _ = S.render_canonical(p, t, rng, site, nominal_session(site))
        T0, _, _ = S.render_canonical(q, t, rng, site, nominal_session(site))
        out.append((T1 - T0)[disk].mean())
    return np.array(out)


def main():
    split = {int(k): v for k, v in json.load(open("splits.json")).items()}
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    rng = np.random.default_rng(0)
    for s in "ABCDE":
        for d in pickle.load(open(f"data/pats_{s}.pkl", "rb")):
            sp = split[d["pid"]]
            keep = sp in ("train", "test", "extD", "extE")
            if keep and d["pid"] not in cache:
                cache[d["pid"]] = (sp, clean_excess(d, s, rng), meta_vector(d), d["time"], d["event"], np.array(d["visits"]))
    pickle.dump(cache, open(CACHE, "wb"))
    R = []
    for pid, (sp, ex, m, T, E, v) in cache.items():
        for k in range(len(v)):
            mm = np.concatenate([m[[0, 1, 2, 3, 4]], m[6:10], m[11:14], m[15:18], [m[18]]])  # reference levels dropped, as in Cox-Clin
            R.append((sp, np.concatenate([mm, [ex[k]]]), max(T - v[k], 1e-3), E))
    res = {}
    for name, cols in [("clinical_floor", list(range(16))), ("noise_free_ceiling", list(range(17)))]:
        def frame(rs):
            X = np.array([r[1][cols] for r in rs])
            df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
            df["u"] = [r[2] for r in rs]; df["d"] = [r[3] for r in rs]
            return df
        dtr = frame([r for r in R if r[0] == "train"])
        mu, sd = dtr.iloc[:, :-2].mean(), dtr.iloc[:, :-2].std() + 1e-9
        dtr.iloc[:, :-2] = (dtr.iloc[:, :-2] - mu) / sd
        cph = CoxPHFitter(penalizer=0.05).fit(dtr, "u", "d")
        res[name] = {}
        for sp in ("test", "extD", "extE"):
            df = frame([r for r in R if r[0] == sp])
            df.iloc[:, :-2] = (df.iloc[:, :-2] - mu) / sd
            sc = cph.predict_partial_hazard(df.iloc[:, :-2]).values
            u, dd = df["u"].values, df["d"].values
            res[name][sp] = float(MT.auc_t(sc, u, dd, 12.0, MT.km_censor(u, dd)))
    ex = np.concatenate([c[1] for c in cache.values()])
    res["excess_percentiles_50_90_99"] = [float(x) for x in np.percentile(ex, [50, 90, 99])]
    json.dump(res, open("results/ceiling.json", "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
