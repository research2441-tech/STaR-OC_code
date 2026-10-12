"""Censoring-aware evaluation of landmark risk predictions.
Prediction array columns: pid, k, t_k, u (months from landmark to event/censoring), delta,
F6, F12, F18, F24, logit_h1..logit_h4, evidence."""
import numpy as np

HOR = [6.0, 12.0, 18.0, 24.0]
COL = {6.0: 5, 12.0: 6, 18.0: 7, 24.0: 8}


def km_censor(u, d):
    """Kaplan-Meier estimate of the censoring survival G(t) = P(C > t). Returns step function."""
    u = np.asarray(u, float); c = 1 - np.asarray(d)
    times, ncen = np.unique(u[c == 1], return_counts=True)
    us = np.sort(u)
    at_risk = len(us) - np.searchsorted(us, times, side="left")
    vals = np.cumprod(1 - ncen / np.maximum(at_risk, 1))

    def G(t, left=False):
        t = np.atleast_1d(t)
        j = np.searchsorted(times, t, side="left" if left else "right") - 1
        out = np.where(j >= 0, vals[np.clip(j, 0, None)] if len(vals) else 1.0, 1.0)
        return np.clip(out, 0.05, 1.0)
    return G


def ipcw_weights(u, d, t, G):
    case = (u <= t) & (d == 1)
    ctrl = u > t
    w = np.zeros_like(u)
    w[case] = 1.0 / G(u[case], left=True)
    w[ctrl] = 1.0 / G(np.full(ctrl.sum(), t))
    return case, ctrl, w


def auc_t(score, u, d, t, G):
    case, ctrl, w = ipcw_weights(u, d, t, G)
    if case.sum() == 0 or ctrl.sum() == 0:
        return np.nan
    sc, wc = score[case], w[case]
    so, wo = score[ctrl], w[ctrl]
    order = np.argsort(so)
    so, wo = so[order], wo[order]
    cw = np.concatenate([[0], np.cumsum(wo)])
    lo = np.searchsorted(so, sc, side="left")
    hi = np.searchsorted(so, sc, side="right")
    num = (wc * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))).sum()
    return num / (wc.sum() * wo.sum())


def uno_c(score, u, d, tau, G):
    """Uno's truncated C-index (IPCW)."""
    idx = np.where((d == 1) & (u < tau))[0]
    if len(idx) == 0:
        return np.nan
    w = 1.0 / G(u[idx], left=True) ** 2
    num = den = 0.0
    for s in range(0, len(idx), 256):
        ii = idx[s:s + 256]
        comp = u[None, :] > u[ii, None]
        gt = (score[ii, None] > score[None, :]) & comp
        eq = (score[ii, None] == score[None, :]) & comp
        ww = w[s:s + 256]
        num += (ww * (gt.sum(1) + 0.5 * eq.sum(1))).sum()
        den += (ww * comp.sum(1)).sum()
    return num / den if den > 0 else np.nan


def brier_t(F, u, d, t, G):
    case, ctrl, w = ipcw_weights(u, d, t, G)
    return (w * (case * (1 - F) ** 2 + ctrl * F ** 2)).sum() / len(u)


def ibs(P, G, tmax=24.0):
    u, d = P[:, 3], P[:, 4]
    ts = [0.0] + HOR
    bs = [0.0] + [brier_t(P[:, COL[t]], u, d, t, G) for t in HOR]
    return np.trapezoid(bs, ts) / tmax


def ece_t(F, u, d, t, G, nb=10):
    case, ctrl, w = ipcw_weights(u, d, t, G)
    known = case | ctrl
    F, y, w = F[known], case[known].astype(float), w[known]
    edges = np.quantile(F, np.linspace(0, 1, nb + 1))
    b = np.clip(np.searchsorted(edges, F, side="right") - 1, 0, nb - 1)
    e = 0.0
    for j in range(nb):
        m = b == j
        if w[m].sum() > 0:
            e += w[m].sum() * abs((w[m] * y[m]).sum() / w[m].sum() - (w[m] * F[m]).sum() / w[m].sum())
    return e / w.sum()


def reliability(F, u, d, t, G, nb=10):
    case, ctrl, w = ipcw_weights(u, d, t, G)
    known = case | ctrl
    F, y, w = F[known], case[known].astype(float), w[known]
    edges = np.quantile(F, np.linspace(0, 1, nb + 1))
    b = np.clip(np.searchsorted(edges, F, side="right") - 1, 0, nb - 1)
    out = []
    for j in range(nb):
        m = b == j
        if w[m].sum() > 0:
            out.append(((w[m] * F[m]).sum() / w[m].sum(), (w[m] * y[m]).sum() / w[m].sum(), m.sum()))
    return np.array(out)


def binary_at(F, u, d, t, G, thr):
    case, ctrl, w = ipcw_weights(u, d, t, G)
    pos = F >= thr
    tp = (w * case * pos).sum(); fn = (w * case * ~pos).sum()
    tn = (w * ctrl * ~pos).sum(); fp = (w * ctrl * pos).sum()
    sens = tp / max(tp + fn, 1e-9); spec = tn / max(tn + fp, 1e-9)
    prec = tp / max(tp + fp, 1e-9)
    f1 = 2 * prec * sens / max(prec + sens, 1e-9)
    return sens, spec, f1, prec


def youden_thr(F, u, d, t, G):
    best, bt = -1, 0.5
    for thr in np.quantile(F, np.linspace(0.02, 0.98, 97)):
        s, sp, _, _ = binary_at(F, u, d, t, G, thr)
        if s + sp - 1 > best:
            best, bt = s + sp - 1, thr
    return bt


def all_metrics(P, G=None, thr12=None):
    u, d = P[:, 3], P[:, 4]
    G = G or km_censor(u, d)
    out = {}
    for t in HOR[:3]:
        out[f"AUC{int(t)}"] = auc_t(P[:, COL[t]], u, d, t, G)
    out["C24"] = uno_c(P[:, COL[24.0]], u, d, 24.0, G)
    out["IBS"] = ibs(P, G)
    out["Brier12"] = brier_t(P[:, COL[12.0]], u, d, 12.0, G)
    out["ECE12"] = ece_t(P[:, COL[12.0]], u, d, 12.0, G)
    if thr12 is not None:
        s, sp, f1, pr = binary_at(P[:, COL[12.0]], u, d, 12.0, G, thr12)
        out.update(Sens12=s, Spec12=sp, F1_12=f1, PPV12=pr)
    return out


def cluster_bootstrap_ids(pids, B, rng):
    up = np.unique(pids)
    groups = {p: np.where(pids == p)[0] for p in up}
    for _ in range(B):
        samp = rng.choice(up, len(up), replace=True)
        yield np.concatenate([groups[p] for p in samp])
