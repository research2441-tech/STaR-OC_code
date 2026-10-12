"""Shared analysis helpers: seed aggregation, paired cluster bootstrap, decision layer."""
import numpy as np, os, json
import metrics as MT

RUNS = "runs"


def load(run, split):
    return np.load(f"{RUNS}/{run}/pred_{split}.npy")


def seeds(prefix, n=5):
    return [f"{prefix}_s{i}" for i in range(n) if os.path.exists(f"{RUNS}/{prefix}_s{i}/pred_extE.npy")]


def seed_metrics(prefix, split, n=5, G=None):
    out = []
    for r in (seeds(prefix, n) if not prefix.startswith("cox") else [prefix]):
        P = load(r, split)
        Pv = load(r, "val")
        thr = MT.youden_thr(Pv[:, 6], Pv[:, 3], Pv[:, 4], 12.0, MT.km_censor(Pv[:, 3], Pv[:, 4]))
        out.append(MT.all_metrics(P, G, thr))
    keys = out[0].keys()
    return {k: (np.mean([o[k] for o in out]), np.std([o[k] for o in out]), len(out)) for k in keys}


def ensemble(prefix, split, n=5):
    rs = seeds(prefix, n) if not prefix.startswith("cox") else [prefix]
    Ps = [load(r, split) for r in rs]
    P = Ps[0].copy()
    for Q in Ps[1:]:
        assert np.allclose(Q[:, :2], P[:, :2])
    Fs = np.stack([Q[:, 5:9] for Q in Ps])
    P[:, 5:9] = Fs.mean(0)
    lg = np.log(np.clip(Fs[..., 1], 1e-6, 1 - 1e-6) / np.clip(1 - Fs[..., 1], 1e-6, 1))
    sig = lg.std(0) if len(Ps) > 1 else np.zeros(len(P))
    return P, sig


def paired_boot(PA, PB, fn, B=1000, seed=0):
    """Two-sided paired patient-cluster bootstrap for fn(P) difference A - B."""
    rng = np.random.default_rng(seed)
    pids = PA[:, 0]
    obs = fn(PA) - fn(PB)
    diffs = []
    for idx in MT.cluster_bootstrap_ids(pids, B, rng):
        diffs.append(fn(PA[idx]) - fn(PB[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return obs, lo, hi, max(p, 1.0 / B)


def holm(ps):
    ps = np.array(ps)
    order = np.argsort(ps)
    adj = np.zeros_like(ps)
    m = len(ps)
    run = 0
    for i, j in enumerate(order):
        run = max(run, (m - i) * ps[j])
        adj[j] = min(run, 1.0)
    return adj


def f_auc12(P):
    return MT.auc_t(P[:, 6], P[:, 3], P[:, 4], 12.0, MT.km_censor(P[:, 3], P[:, 4]))


def f_ibs(P):
    return MT.ibs(P, MT.km_censor(P[:, 3], P[:, 4]))


def f_c24(P):
    return MT.uno_c(P[:, 8], P[:, 3], P[:, 4], 24.0, MT.km_censor(P[:, 3], P[:, 4]))


# ---------------- decision layer ----------------
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def case_weights(P, G, t=12.0):
    u, d = P[:, 3], P[:, 4]
    case, ctrl, w = MT.ipcw_weights(u, d, t, G)
    return case, ctrl, w


def per_patient_weight(P):
    pids = P[:, 0]
    _, inv, cnt = np.unique(pids, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def conformal_threshold(score, P, alpha, G, rng=None, draws=0):
    """Censoring-weighted Mondrian (case-conditional) split-conformal threshold.
    Landmarks are down-weighted by 1/K_p so that every calibration patient counts once."""
    case, ctrl, w = case_weights(P, G)
    w = w * per_patient_weight(P)
    s, ww = score[case], w[case]
    o = np.argsort(s)
    s, ww = s[o], ww[o]
    W = ww.sum() + ww.max()
    cum = np.concatenate([[0], np.cumsum(ww)])  # weight strictly below s[j]
    ok = np.where(cum[:-1] / W <= alpha)[0]
    return s[ok[-1]] if len(ok) else -np.inf


def decision_eval(score, P, thr, G):
    case, ctrl, w = case_weights(P, G)
    pw = per_patient_weight(P)
    clear = score < thr
    miss = (w * pw * case * clear).sum() / max((w * pw * case).sum(), 1e-12)
    clr = (pw * clear).sum() / pw.sum()
    ctrl_clear = (w * pw * ctrl * clear).sum() / max((w * pw * ctrl).sum(), 1e-12)
    return dict(miss=miss, cleared=clr, ctrl_cleared=ctrl_clear)
