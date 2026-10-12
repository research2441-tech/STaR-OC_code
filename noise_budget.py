"""Per-visit noise budget of the lesion thermal contrast in non-progressing patients.
The same visits are rendered with acquisition factors switched on cumulatively, and the
visit-to-visit standard deviation of the lesion contrast (core mean minus perilesional ring, in
the session-normalised temperature channel) is measured at each stage. Writes
results/noise_budget.json."""
import numpy as np, json
import oralthermsim as S
import preprocess as PP

g = np.arange(PP.CROP) - (PP.CROP - 1) / 2
RAD = np.sqrt(g[:, None] ** 2 + g[None, :] ** 2)
STAGES = ["physiology", "session", "inflammation", "camera", "pose_registration"]


def contrast(field, large, lx, ly):
    Tn, An = PP.normalise(field, lx, ly)
    cx, cy = PP.C0 + lx * PP.C0, PP.C0 + ly * PP.C0
    rc = 4.5 if large else 3.0
    core, ring = RAD <= rc, (RAD > rc + 2) & (RAD <= rc + 6)
    t = PP.crop(Tn, cx, cy, lx < 0)
    a = PP.crop(An, cx, cy, lx < 0)
    return t[core].mean() - t[ring].mean(), a[core].mean()


def budget(site, n_pat=80, seed=11):
    rng = np.random.default_rng(seed)
    fpn = S.SITE_CFG[site]["fpn"] * S.smooth_field(np.random.default_rng(999), 10.0)
    per = {k: [] for k in STAGES}
    n = 0
    while n < n_pat:
        p = S.sample_patient(rng, n, site)
        if p.event == 1 or len(p.visits) < 4:
            continue
        n += 1
        lx, ly = p.lx + rng.normal(0, 1.5 / PP.C0), p.ly + rng.normal(0, 1.5 / PP.C0)
        vals = {k: [] for k in STAGES}
        ref = None
        for t in p.visits[:6]:
            sess = S.session_params(p, rng, site, t)
            nominal = dict(tenv=S.SITE_CFG[site]["amb"], evap=1.1, grad=0.0, sfield=0.0, inflam=[])
            no_infl = dict(sess); no_infl["inflam"] = []
            T0, _, _ = S.render_canonical(p, t, rng, site, nominal)
            T1, _, _ = S.render_canonical(p, t, rng, site, no_infl)
            T2, _, _ = S.render_canonical(p, t, rng, site, sess)
            still = dict(sess); still.update(rot=0.0, scale=1.0, open=1.0, tx=0.0, ty=0.0, elastic=0.0)
            I3 = S.camera(T2, rng, site, still, fpn)
            I4 = S.camera(T2, rng, site, sess, fpn)
            af = PP.moment_affine(I4)
            C4 = PP.refine(I4, af, ref) if ref is not None else PP.to_canonical(I4, af)
            ref = C4 if ref is None else ref
            for k, f in zip(STAGES, (T0, T1, T2, I3, C4)):
                vals[k].append(contrast(f, p.large, lx, ly))
        for k in STAGES:
            v = np.array(vals[k])
            per[k].append(np.std(np.diff(v, axis=0), axis=0) / np.sqrt(2))
    return {k: [float(x) for x in np.sqrt(np.mean(np.square(per[k]), 0))] for k in STAGES}


if __name__ == "__main__":
    res = {s: budget(s) for s in "ADE"}
    json.dump(res, open("results/noise_budget.json", "w"), indent=1)
    for s, r in res.items():
        print(s, {k: [round(x, 4) for x in v] for k, v in r.items()})
