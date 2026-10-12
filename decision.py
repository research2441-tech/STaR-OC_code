"""Reliability-aware surveillance decisions: censoring-weighted, case-conditional split conformal
clearance with ensemble and twin-evidence inflation. Writes results/decision.json."""
import numpy as np, json, sys, os
import metrics as MT
import analysis as A

ALPHAS = [0.05, 0.10, 0.20]


def scores(P, sig, kappa, gamma):
    return A.logit(P[:, 6]) + kappa * sig + gamma * P[:, 13]


def tune(Pv, sv, Gv, alpha):
    """Choose (kappa, gamma) on validation: minimise escalation at the validation conformal
    threshold of level alpha (validation used for tuning only)."""
    best, sel = -1, (0.0, 0.0)
    for kappa in [0.0, 0.5, 1.0, 2.0]:
        for gamma in [0.0, 0.5, 1.0, 2.0]:
            s = scores(Pv, sv, kappa, gamma)
            thr = A.conformal_threshold(s, Pv, alpha, Gv)
            r = A.decision_eval(s, Pv, thr, Gv)
            if r["cleared"] > best:
                best, sel = r["cleared"], (kappa, gamma)
    return sel


def fixed_threshold(Pv, Gv, alpha):
    """Non-conformal rule: largest validation threshold keeping weighted sensitivity >= 1-alpha."""
    s = A.logit(Pv[:, 6])
    best = -np.inf
    for thr in np.quantile(s, np.linspace(0.01, 0.9, 120)):
        r = A.decision_eval(s, Pv, thr, Gv)
        if r["miss"] <= alpha:
            best = max(best, thr)
    return best


def run(prefix, splits=("test", "extD", "extE"), local_n=150, seed=0):
    out = {}
    Pv, sv = A.ensemble(prefix, "val")
    Pc, sc = A.ensemble(prefix, "cal")
    Gv, Gc = MT.km_censor(Pv[:, 3], Pv[:, 4]), MT.km_censor(Pc[:, 3], Pc[:, 4])
    for alpha in ALPHAS:
        kg = tune(Pv, sv, Gv, alpha)
        fthr = fixed_threshold(Pv, Gv, alpha)
        rules = {
            "fixed": (lambda P, s: A.logit(P[:, 6]), fthr),
            "conf_point": (lambda P, s: A.logit(P[:, 6]), None),
            "conf_ens": (lambda P, s: scores(P, s, kg[0], 0.0), None),
            "conf_full": (lambda P, s: scores(P, s, kg[0], kg[1]), None),
        }
        for name, (fn, thr) in rules.items():
            if thr is None:
                thr = A.conformal_threshold(fn(Pc, sc), Pc, alpha, Gc)
            for sp in splits:
                Pt, st = A.ensemble(prefix, sp)
                Gt = MT.km_censor(Pt[:, 3], Pt[:, 4])
                r = A.decision_eval(fn(Pt, st), Pt, thr, Gt)
                out[f"{alpha}|{name}|{sp}"] = r
        out[f"{alpha}|kg"] = kg
        # site-local recalibration for external sites: first local_n patients calibrate, rest test
        for sp in ("extD", "extE"):
            Pt, st = A.ensemble(prefix, sp)
            rng = np.random.default_rng(seed)
            pids = np.unique(Pt[:, 0]); rng.shuffle(pids)
            cal_ids = set(pids[:local_n].tolist())
            mc = np.array([p in cal_ids for p in Pt[:, 0]])
            for name in ("conf_point", "conf_full"):
                fn = rules[name][0]
                Gl = MT.km_censor(Pt[mc, 3], Pt[mc, 4])
                thr = A.conformal_threshold(fn(Pt[mc], st[mc]), Pt[mc], alpha, Gl)
                Gt = MT.km_censor(Pt[~mc, 3], Pt[~mc, 4])
                out[f"{alpha}|{name}_local|{sp}"] = A.decision_eval(fn(Pt[~mc], st[~mc]), Pt[~mc], thr, Gt)
    return out


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    res = {p: run(p) for p in sys.argv[1:]}
    json.dump(res, open("results/decision.json", "w"), indent=1, default=float)
    for p, r in res.items():
        for k, v in r.items():
            if k.startswith("0.1|"):
                print(p, k, v)
