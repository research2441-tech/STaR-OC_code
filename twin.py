"""Patient-specific thermal twin: pixel-wise robust local-linear-trend state-space model on the
registered asymmetry field and the session-normalised temperature field of the lesion crop.

State per pixel and channel: level b (personal thermal reference) and drift d (deg C / month).
Transient inflammation is absorbed by a Huber-weighted measurement update, so single-visit
excursions do not move the personal reference, while persistent change accumulates in d.
"""
import numpy as np

CH = 2  # asym, normalised temperature


def group_key(p):
    return (p["lsite"], p["ltype"])


def population_prior(raw, pats):
    """Group-wise mean and pixel-wise variance of first-visit crops (training patients only)."""
    groups = {}
    allfirst = []
    for p in pats:
        x = raw[p["img_idx"][0]].astype(np.float32)
        groups.setdefault(group_key(p), []).append(x)
        allfirst.append(x)
    allfirst = np.stack(allfirst)
    gmean = {k: np.mean(v, 0) if len(v) >= 15 else allfirst.mean(0) for k, v in groups.items()}
    resid = np.stack([raw[p["img_idx"][0]].astype(np.float32) - gmean[group_key(p)] for p in pats])
    tau2 = resid.var(0) + 1e-4
    return dict(gmean=gmean, gall=allfirst.mean(0), tau2=tau2)


def run_patient(Y, t, mu0, tau2, hp, robust=True, drift=True):
    """Y: (K, CH, H, W) observations; t: (K,) months. Returns (K, 4*CH, H, W) twin maps and
    evidence sufficiency per visit."""
    K = Y.shape[0]
    qb, qd, R, sd0, c = hp["qb"], hp["qd"], hp["R"], hp["sd0"], hp["c"]
    if not drift:
        qd, sd0 = 0.0, 1e-6
    sd0a = np.array(sd0).reshape(-1, 1, 1)
    qb = np.array(qb).reshape(-1, 1, 1); qd = np.array(qd).reshape(-1, 1, 1); R = np.array(R).reshape(-1, 1, 1)
    b = mu0.copy()
    d = np.zeros_like(mu0)
    P11 = tau2.copy()
    P12 = np.zeros_like(mu0)
    P22 = np.ones_like(mu0) * sd0a ** 2
    out = np.zeros((K, 4 * CH) + Y.shape[2:], np.float32)
    evid = np.zeros(K, np.float32)
    b_ref = None
    for k in range(K):
        if k > 0:
            dt = t[k] - t[k - 1]
            b = b + d * dt
            P11 = P11 + 2 * dt * P12 + dt * dt * P22 + qb * dt
            P12 = P12 + dt * P22
            P22 = P22 + qd * dt
        S = P11 + R
        nu = Y[k] - b
        z = nu / np.sqrt(S)
        w = np.minimum(1.0, c / (np.abs(z) + 1e-9)) if robust else np.ones_like(z)
        Se = S / w
        K1 = P11 / Se
        K2 = P12 / Se
        b = b + K1 * nu
        d = d + K2 * nu
        P11n = P11 - K1 * P11
        P12n = P12 - K1 * P12
        P22n = P22 - K2 * P12
        P11, P12, P22 = P11n, P12n, np.maximum(P22n, 1e-10)
        if b_ref is None:
            b_ref = b.copy()
        sdd = np.sqrt(P22)
        out[k, 0:CH] = np.clip(z, -6, 6)
        out[k, CH:2 * CH] = np.clip(d / sdd, -6, 6) if drift else 0.0
        out[k, 2 * CH:3 * CH] = b - b_ref
        out[k, 3 * CH:4 * CH] = sdd / sd0a if drift else 0.0
        h, w_ = Y.shape[2], Y.shape[3]
        evid[k] = float(np.mean((sdd / sd0a)[0, h // 2 - 4:h // 2 + 4, w_ // 2 - 4:w_ // 2 + 4])) if drift else 1.0
    return out, evid


def loglik(raw, pats, prior, hp, sub=slice(10, 22)):
    ll = 0.0
    for p in pats:
        if len(p["img_idx"]) < 3:
            continue
        Y = raw[p["img_idx"]][:, :, sub, sub].astype(np.float32)
        t = np.array(p["visits"])
        mu0 = prior["gmean"].get(group_key(p), prior["gall"])[:, sub, sub]
        tau2 = prior["tau2"][:, sub, sub]
        qb, qd, R, sd0 = [np.array(hp[k]).reshape(-1, 1, 1) for k in ("qb", "qd", "R", "sd0")]
        b, d = mu0.copy(), np.zeros_like(mu0)
        P11, P12, P22 = tau2.copy(), np.zeros_like(mu0), np.ones_like(mu0) * sd0 ** 2
        for k in range(len(t)):
            if k > 0:
                dt = t[k] - t[k - 1]
                b = b + d * dt
                P11 = P11 + 2 * dt * P12 + dt * dt * P22 + qb * dt
                P12 = P12 + dt * P22
                P22 = P22 + qd * dt
            S = P11 + R
            nu = Y[k] - b
            if k > 0:
                ll += float((-0.5 * (np.log(2 * np.pi * S) + nu ** 2 / S)).sum())
            K1, K2 = P11 / S, P12 / S
            b, d = b + K1 * nu, d + K2 * nu
            P11, P12, P22 = P11 - K1 * P11, P12 - K1 * P12, np.maximum(P22 - K2 * P12, 1e-10)
    return ll


def fit_hyper(raw, pats, prior, rng=np.random.default_rng(0)):
    """Empirical-Bayes (type-II maximum likelihood) coordinate search, per channel."""
    sel = [p for p in pats if len(p["img_idx"]) >= 3]
    sel = [sel[i] for i in rng.choice(len(sel), min(400, len(sel)), replace=False)]
    hp = dict(qb=[1e-3, 1e-3], qd=[1e-5, 1e-5], R=[0.01, 0.01], sd0=[0.01, 0.01], c=2.5)
    grids = dict(qb=np.logspace(-5, -1, 9), qd=np.logspace(-8, -3, 11), R=np.logspace(-3, 0, 10), sd0=np.logspace(-4, -1, 7))
    for sweep in range(2):
        for ch in range(2):
            for key in ("R", "qb", "sd0", "qd"):
                best, bv = -np.inf, None
                for v in grids[key]:
                    h = {k: (list(vv) if isinstance(vv, list) else vv) for k, vv in hp.items()}
                    h[key][ch] = v
                    ll = loglik(raw, sel, prior, h)
                    if ll > best:
                        best, bv = ll, v
                hp[key][ch] = float(bv)
        print("sweep", sweep, hp, flush=True)
    return hp


def run_all(raw, pats, prior, hp, robust=True, drift=True):
    n = raw.shape[0]
    tw = np.zeros((n, 4 * CH) + raw.shape[2:], np.float16)
    ev = np.zeros(n, np.float32)
    for p in pats:
        ii = p["img_idx"]
        mu0 = prior["gmean"].get(group_key(p), prior["gall"])
        o, e = run_patient(raw[ii].astype(np.float32), np.array(p["visits"]), mu0, prior["tau2"], hp, robust, drift)
        tw[ii] = o.astype(np.float16)
        ev[ii] = e
    return tw, ev


def fit_drift_prior(raw, pats, prior, hp, grid_sd0=(0.005, 0.01, 0.02, 0.04, 0.08), grid_qd=(1e-7, 1e-6, 1e-5, 1e-4)):
    """Drift-prior selection on training patients. Type-II likelihood is dominated by the
    non-progressing majority and drives the drift prior to zero, so (sd0, qd) are chosen to
    maximise the 12-month IPCW AUC of the lesion-centre drift statistic instead."""
    import metrics as MT
    best, sel = -1, None
    rows_u, rows_d = [], []
    for p in pats:
        for k in range(len(p["img_idx"])):
            rows_u.append(p["time"] - p["visits"][k]); rows_d.append(p["event"])
    u, d = np.array(rows_u), np.array(rows_d)
    G = MT.km_censor(u, d)
    table = []
    for sd0 in grid_sd0:
        for qd in grid_qd:
            h = dict(hp); h["sd0"] = [sd0, sd0]; h["qd"] = [qd, qd]
            stat = []
            for p in pats:
                mu0 = prior["gmean"].get(group_key(p), prior["gall"])
                o, _ = run_patient(raw[p["img_idx"]].astype(np.float32), np.array(p["visits"]), mu0, prior["tau2"], h)
                stat.extend(o[:, CH:2 * CH, 12:20, 12:20].mean((2, 3)).sum(1))
            a = MT.auc_t(np.array(stat), u, d, 12.0, G)
            table.append((sd0, qd, a))
            if a > best:
                best, sel = a, (sd0, qd)
    hp = dict(hp); hp["sd0"] = [sel[0], sel[0]]; hp["qd"] = [sel[1], sel[1]]
    print("drift prior", sel, round(best, 4), flush=True)
    return hp, table
