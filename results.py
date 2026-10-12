"""Aggregate all experiments into results/*.json used by the manuscript tables and figures."""
import numpy as np, json, os, sys
import metrics as MT
import analysis as A
from scipy import stats

os.makedirs("results", exist_ok=True)
METHODS = ["cox_clin", "cox_asym", "cox_long", "cs", "siam", "tlstm", "tlstm_twin", "final"]
NAMES = {"cox_clin": "Cox-Clin", "cox_asym": "Cox-Asym", "cox_long": "Cox-Long", "cs": "CS-Net", "siam": "Siamese-Δ",
         "tlstm": "T-LSTM", "tlstm_twin": "T-LSTM+Twin", "final": "STaR-OC"}
ABL = ["ab_notwin", "ab_twinonly", "ab_nodrift", "ab_norobust", "ab_noreg", "ab_moment", "ab_nonorm",
       "ab_nometa", "ab_notime", "ab_polar", "ab_cnn", "abl_concat", "abl_notemporal", "staroc"]
SPLITS = ["test", "extD", "extE"]
CONDS = ["base", "const0.5", "const2", "inflam0", "inflam2", "sess2", "pose2", "noise2", "keep0.5"]
B = int(os.environ.get("BOOT", "1000"))


def per_seed(prefix, split):
    out = []
    runs = [prefix] if prefix.startswith("cox") else A.seeds(prefix)
    for r in runs:
        P = A.load(r, split)
        Pv = A.load(r, "val")
        thr = MT.youden_thr(Pv[:, 6], Pv[:, 3], Pv[:, 4], 12.0, MT.km_censor(Pv[:, 3], Pv[:, 4]))
        out.append(MT.all_metrics(P, None, thr))
    return out


def summarise(lst):
    return {k: [float(np.mean([o[k] for o in lst])), float(np.std([o[k] for o in lst]))] for k in lst[0]}


def ens(prefix, split):
    if prefix.startswith("cox"):
        return A.load(prefix, split), np.zeros(1)
    return A.ensemble(prefix, split)


res = {}
# ---- 1. main and external tables (per-seed mean +- sd) and ensemble metrics
main = {}
for mth in METHODS:
    main[mth] = {}
    for sp in SPLITS:
        main[mth][sp] = summarise(per_seed(mth, sp))
        P, _ = ens(mth, sp)
        Pv, _ = ens(mth, "val")
        thr = MT.youden_thr(Pv[:, 6], Pv[:, 3], Pv[:, 4], 12.0, MT.km_censor(Pv[:, 3], Pv[:, 4]))
        main[mth][sp + "_ens"] = {k: float(v) for k, v in MT.all_metrics(P, None, thr).items()}
    print("main", mth, {k: round(v[0], 3) for k, v in main[mth]["test"].items()}, flush=True)
res["main"] = main

# ---- 2. paired bootstrap (seed-ensembles) STaR-OC vs each comparator
tests = {}
for sp in SPLITS:
    PA, _ = ens("final", sp)
    tests[sp] = {}
    for mth in METHODS[:-1]:
        PB, _ = ens(mth, sp)
        assert np.allclose(PA[:, :2], PB[:, :2])
        tests[sp][mth] = {n: list(A.paired_boot(PA, PB, f, B=B, seed=1)) for n, f in
                          [("AUC12", A.f_auc12), ("C24", A.f_c24), ("IBS", A.f_ibs)]}
    for n in ("AUC12", "C24", "IBS"):
        adj = A.holm([tests[sp][m][n][3] for m in METHODS[:-1]])
        for m, a in zip(METHODS[:-1], adj):
            tests[sp][m][n].append(float(a))
    print("tests", sp, {m: [round(x, 4) for x in tests[sp][m]["AUC12"]] for m in METHODS[:-1]}, flush=True)
res["tests"] = tests

# ---- 3. discrimination by number of prior visits and by horizon (ensembles, internal test)
byk = {}
for mth in METHODS:
    P, _ = ens(mth, "test")
    G = MT.km_censor(P[:, 3], P[:, 4])
    d = {}
    for lo, hi, lab in [(0, 0, "1"), (1, 1, "2"), (2, 2, "3"), (3, 3, "4"), (4, 99, "5+")]:
        msk = (P[:, 1] >= lo) & (P[:, 1] <= hi)
        Q = P[msk]
        Gq = MT.km_censor(Q[:, 3], Q[:, 4])
        d[lab] = [float(MT.auc_t(Q[:, 6], Q[:, 3], Q[:, 4], 12.0, Gq)), int(msk.sum()),
                  int(((Q[:, 3] <= 12) & (Q[:, 4] == 1)).sum())]
    d["horizon"] = {str(t): float(MT.auc_t(P[:, MT.COL[t]], P[:, 3], P[:, 4], t, G)) for t in (6.0, 12.0, 18.0, 24.0)}
    byk[mth] = d
res["byk"] = byk
# bootstrap CI of the by-visit AUC difference STaR-OC minus T-LSTM and Cox-Long
byk_tests = {}
PA, _ = ens("final", "test")
for other in ("tlstm", "cox_long"):
    PB, _ = ens(other, "test")
    byk_tests[other] = {}
    for lo, hi, lab in [(0, 0, "1"), (1, 1, "2"), (2, 2, "3"), (3, 3, "4"), (4, 99, "5+")]:
        msk = (PA[:, 1] >= lo) & (PA[:, 1] <= hi)
        byk_tests[other][lab] = list(A.paired_boot(PA[msk], PB[msk], A.f_auc12, B=B, seed=2))
res["byk_tests"] = byk_tests

# ---- 4. ablations (per-seed mean +- sd, ensemble deltas with bootstrap p)
abl = {}
Pfull, _ = ens("final", "test")
for a in ABL:
    if not A.seeds(a):
        continue
    abl[a] = {"test": summarise(per_seed(a, "test"))}
    Pa, _ = ens(a, "test")
    abl[a]["delta"] = {n: list(A.paired_boot(Pa, Pfull, f, B=B, seed=3)) for n, f in
                       [("AUC12", A.f_auc12), ("C24", A.f_c24), ("IBS", A.f_ibs)]}
    for sp in ("extD", "extE"):
        Pe, _ = ens(a, sp)
        abl[a][sp] = float(A.f_auc12(Pe))
    print("abl", a, round(abl[a]["test"]["AUC12"][0], 3), [round(x, 4) for x in abl[a]["delta"]["AUC12"]], flush=True)
res["abl"] = abl

# ---- 5. stress tests
rob = {}
for cond in CONDS:
    rob[cond] = {}
    for mth in METHODS + ["ab_notwin"]:
        runs = [mth] if mth.startswith("cox") else A.seeds(mth)
        f = [f"runs/{r}/pred_rob_{cond}.npy" for r in runs]
        if not all(os.path.exists(x) for x in f):
            continue
        Ps = [np.load(x) for x in f]
        vals = [A.f_auc12(P) for P in Ps]
        P = Ps[0].copy(); P[:, 5:9] = np.mean([Q[:, 5:9] for Q in Ps], 0)
        rob[cond][mth] = dict(auc_mean=float(np.mean(vals)), auc_sd=float(np.std(vals)), auc_ens=float(A.f_auc12(P)),
                              ibs_ens=float(A.f_ibs(P)), c24_ens=float(A.f_c24(P)))
    print("rob", cond, {m: round(v["auc_ens"], 3) for m, v in rob[cond].items()}, flush=True)
res["rob"] = rob

# ---- 6. Friedman / Wilcoxon over the 11 settings (ensemble AUC12)
settings = {}
for sp in SPLITS:
    settings[sp] = {m: res["main"][m][sp + "_ens"]["AUC12"] for m in METHODS}
for cond in CONDS[1:]:
    if all(m in rob[cond] for m in METHODS):
        settings[cond] = {m: rob[cond][m]["auc_ens"] for m in METHODS}
mat = np.array([[settings[s][m] for m in METHODS] for s in settings])
ranks = np.array([stats.rankdata(-row) for row in mat])
fr = stats.friedmanchisquare(*[mat[:, j] for j in range(len(METHODS))])
best_other = METHODS[int(np.argmin(ranks[:, :-1].mean(0)))]
wil = stats.wilcoxon(mat[:, -1], mat[:, METHODS.index(best_other)])
res["friedman"] = dict(settings=list(settings), mean_rank={m: float(r) for m, r in zip(METHODS, ranks.mean(0))},
                       chi2=float(fr.statistic), p=float(fr.pvalue), best_other=best_other,
                       wilcoxon_stat=float(wil.statistic), wilcoxon_p=float(wil.pvalue),
                       wins=int((mat[:, -1] > mat[:, METHODS.index(best_other)]).sum()))
print("friedman", res["friedman"], flush=True)

# ---- 7. calibration curves and KM tertiles (ensembles)
cal = {}
km = {}
for mth in ["cox_long", "tlstm", "final"]:
    for sp in SPLITS:
        P, _ = ens(mth, sp)
        G = MT.km_censor(P[:, 3], P[:, 4])
        cal[f"{mth}|{sp}"] = MT.reliability(P[:, 6], P[:, 3], P[:, 4], 12.0, G).tolist()
for sp in SPLITS:
    P, _ = ens("final", sp)
    Pv, _ = ens("final", "val")
    q1, q2 = np.quantile(Pv[:, 6], [1 / 3, 2 / 3])
    grp = np.digitize(P[:, 6], [q1, q2])
    curves = {}
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    for g in range(3):
        kmf = KaplanMeierFitter().fit(P[grp == g, 3], P[grp == g, 4])
        tl = kmf.survival_function_.index.values
        sv = kmf.survival_function_.values[:, 0]
        curves[str(g)] = [tl[tl <= 24].tolist(), sv[tl <= 24].tolist(), int((grp == g).sum())]
    lr = multivariate_logrank_test(np.minimum(P[:, 3], 24), grp, P[:, 4] * (P[:, 3] <= 24))
    # 12-month cumulative incidence per tertile
    inc = []
    for g in range(3):
        kmf = KaplanMeierFitter().fit(P[grp == g, 3], P[grp == g, 4])
        inc.append(float(1 - kmf.predict(12.0)))
    km[sp] = dict(curves=curves, logrank_p=float(lr.p_value), logrank_stat=float(lr.test_statistic), inc12=inc)
res["cal"] = cal
# ---- 8. subgroups (internal test + both external sites pooled), seed ensembles
import pickle
meta = {}
for site in "ABCDE":
    for p in pickle.load(open(f"prep/pats_{site}.pkl", "rb")):
        meta[p["pid"]] = (p["lsite"], p["ltype"], p["grade"])
sub = {}
for mth in ["cox_asym", "cox_long", "cs", "tlstm", "final"]:
    P = np.concatenate([ens(mth, sp)[0] for sp in SPLITS])
    sub[mth] = {}
    groups = {"buccal/tongue": lambda m: m[0] in (0, 1), "floor/palate (near midline)": lambda m: m[0] in (2, 3),
              "homogeneous leukoplakia": lambda m: m[1] == 0, "non-homogeneous leukoplakia/erythroplakia": lambda m: m[1] in (1, 2),
              "submucous fibrosis": lambda m: m[1] == 3, "lichen planus": lambda m: m[1] == 4,
              "no/mild dysplasia": lambda m: m[2] <= 1, "moderate/severe dysplasia": lambda m: m[2] >= 2}
    for gname, fn in groups.items():
        msk = np.array([fn(meta[int(p)]) for p in P[:, 0]])
        Q = P[msk]
        sub[mth][gname] = [float(A.f_auc12(Q)), int(len(np.unique(Q[:, 0]))), int(((Q[:, 3] <= 12) & (Q[:, 4] == 1)).sum())]
res["sub"] = sub
res["km"] = km
json.dump(res, open("results/results.json", "w"), indent=1, default=float)
print("saved")
