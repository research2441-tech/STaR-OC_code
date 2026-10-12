"""Registration to the canonical oral frame, session normalisation, lesion-centred crops
and handcrafted asymmetry features. Output is consumed by twin.py and the models."""
import numpy as np, pickle, sys, os
from scipy import ndimage as ndi

N = 64
C0 = (N - 1) / 2
A0, B0, YOFF = 0.60 * C0, 0.42 * C0, 0.03 * C0
CROP = 32


def cavity_mask(img):
    thr = 0.5 * (np.percentile(img, 30) + np.percentile(img, 97))
    m = img > thr
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = lab == (1 + np.argmax(sizes))
    return ndi.binary_fill_holes(m)


def moment_affine(img):
    """Affine (canonical -> image) from second moments of the warm cavity region."""
    m = cavity_mask(img)
    iy, ix = np.nonzero(m)
    cy, cx = iy.mean(), ix.mean()
    cov = np.cov(np.vstack([ix - cx, iy - cy]))
    w, v = np.linalg.eigh(cov)
    major = v[:, 1]
    th = np.arctan2(major[1], major[0])
    if th > np.pi / 2:
        th -= np.pi
    if th < -np.pi / 2:
        th += np.pi
    a = 2 * np.sqrt(w[1])
    b = 2 * np.sqrt(w[0])
    return dict(cx=cx, cy=cy, th=th, sa=a / A0, sb=b / B0)


def to_canonical(img, af, dx=0.0, dy=0.0):
    iy, ix = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    u = (ix - C0) * af["sa"]
    v = (iy - C0 - YOFF) * af["sb"]
    c, s = np.cos(af["th"]), np.sin(af["th"])
    X = c * u - s * v + af["cx"] + dx
    Y = s * u + c * v + af["cy"] + dy
    return ndi.map_coordinates(img, [Y, X], order=1, mode="nearest")


iyc, ixc = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
XN = (ixc - C0) / C0
YN = (iyc - C0) / C0
CAV = ((XN / 0.60) ** 2 + ((YN - 0.03) / 0.42) ** 2) < 0.85


def ncc(a, b, m):
    a = a[m] - a[m].mean()
    b = b[m] - b[m].mean()
    return (a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9)


def refine(img, af, ref):
    best, bd = -2, (0.0, 0.0)
    for dx in np.arange(-2, 2.01, 0.5):
        for dy in np.arange(-2, 2.01, 0.5):
            c = to_canonical(img, af, dx, dy)
            s = ncc(c, ref, CAV)
            if s > best:
                best, bd = s, (dx, dy)
    return to_canonical(img, af, *bd)


def crop(field, cx, cy, flip):
    out = ndi.map_coordinates(field, [cy - CROP / 2 + 0.5 + np.arange(CROP)[:, None] + 0 * np.arange(CROP)[None, :],
                                      cx - CROP / 2 + 0.5 + np.arange(CROP)[None, :] + 0 * np.arange(CROP)[:, None]],
                              order=1, mode="nearest")
    return out[:, ::-1].copy() if flip else out


def normalise(Tc, lx, ly):
    """Session normalisation: remove global offset and antisymmetric linear gradient."""
    excl = ((XN - lx) ** 2 + (YN - ly) ** 2 < 0.25 ** 2) | ((XN + lx) ** 2 + (YN - ly) ** 2 < 0.25 ** 2)
    ref = CAV & ~excl
    Tn = Tc - np.median(Tc[ref])
    A = Tc - Tc[:, ::-1]
    g = (A[ref] * XN[ref]).sum() / ((XN[ref] ** 2).sum() + 1e-9)
    An = A - g * XN
    return Tn, An


def handcrafted(An, Tn, lx, ly, large):
    r = (0.13 if large else 0.08) + 0.03
    disk = CAV & ((XN - lx) ** 2 + (YN - ly) ** 2 < r ** 2)
    if disk.sum() < 3:
        disk = (XN - lx) ** 2 + (YN - ly) ** 2 < r ** 2
    a = An[disk]
    t = Tn[disk]
    hist, _ = np.histogram(a, bins=10, range=(-1.5, 1.5))
    pr = hist / max(hist.sum(), 1)
    ent = -(pr[pr > 0] * np.log(pr[pr > 0])).sum()
    return np.array([a.mean(), a.max(), np.percentile(a, 90), a.std(), t.mean(), t.std(), ent], dtype=np.float32)


HC_NAMES = ["asym_mean", "asym_max", "asym_p90", "asym_sd", "tnorm_mean", "tnorm_sd", "asym_entropy"]


def process_site(site, ddir, mode="full", rng=None):
    imgs = np.load(f"{ddir}/imgs_{site}.npy")
    pats = pickle.load(open(f"{ddir}/pats_{site}.pkl", "rb"))
    rng = rng or np.random.default_rng(7)
    n = len(imgs)
    raw = np.zeros((n, 2, CROP, CROP), np.float32)   # [asym, normalised temperature]
    hc = np.zeros((n, 7), np.float32)
    for p in pats:
        # clinician-annotated lesion centre (canonical frame, ~1.5 px annotation error)
        lx = p["lx"] + rng.normal(0, 1.5 / C0)
        ly = p["ly"] + rng.normal(0, 1.5 / C0)
        p["ann"] = (lx, ly)
        ref = None
        for k, ii in enumerate(p["img_idx"]):
            img = imgs[ii]
            if mode == "none":
                Tc = img
            else:
                af = moment_affine(img)
                if mode == "full" and ref is not None:
                    Tc = refine(img, af, ref)
                else:
                    Tc = to_canonical(img, af)
                if ref is None:
                    ref = Tc
            if mode == "nonorm":
                Tn = Tc - np.median(Tc[CAV])
                An = Tc - Tc[:, ::-1]
            else:
                Tn, An = normalise(Tc, lx, ly)
            cx = C0 + lx * C0
            cy = C0 + ly * C0
            flip = lx < 0
            raw[ii, 0] = crop(An, cx, cy, flip)
            raw[ii, 1] = crop(Tn, cx, cy, flip)
            hc[ii] = handcrafted(An, Tn, lx, ly, p["large"])
    return pats, raw, hc


if __name__ == "__main__":
    ddir, odir, mode = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "full")
    os.makedirs(odir, exist_ok=True)
    for site in ["A", "B", "C", "D", "E"]:
        pats, raw, hc = process_site(site, ddir, mode, np.random.default_rng(ord(site)))
        np.save(f"{odir}/raw_{site}.npy", raw.astype(np.float16))
        np.save(f"{odir}/hc_{site}.npy", hc)
        pickle.dump(pats, open(f"{odir}/pats_{site}.pkl", "wb"))
        print(site, raw.shape, flush=True)
