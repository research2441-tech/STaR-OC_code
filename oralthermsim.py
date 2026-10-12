"""OralThermSim-L: physics-based longitudinal intraoral thermography cohort simulator.

Each virtual patient carries an oral potentially malignant disorder (OPMD) followed over
irregular surveillance visits. Surface temperature is obtained from a steady-state 2-D
effective Pennes bioheat balance on a canonical oral-cavity frame, then passed through a
pose, optics and detector model of the site camera.

Sources of variation that make single-visit thermal asymmetry ambiguous:
  * patient-specific constitutive perfusion asymmetry (stable over time)
  * lesion-type specific baseline signatures (keratin insulation, erythema, fibrosis)
  * transient benign inflammation episodes (single visit)
  * session effects: ambient temperature, evaporative cooling, calibration offset,
    smooth asymmetric gradients
  * pose / mouth-opening variation and small elastic deformation
Malignant progression is a gradual angiogenic and metabolic increase that starts
6-18 months before the latent transformation time.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter, map_coordinates

N = 64
T_ART = 37.0
D_PX = 2.5  # effective lateral diffusion (pixel^2 units)

LESION_TYPES = ["hom_leuko", "nonhom_leuko", "erythro", "osf", "olp"]
SITES = ["buccal", "tongue", "floor", "palate"]
GRADES = ["none", "mild", "moderate", "severe"]

SITE_CFG = {
    # name: psf sigma(px), NETD(C), pose scale, ambient mean, ambient sd, calib sd, fixed-pattern amp, areca prev, osf boost
    "A": dict(psf=0.8, netd=0.04, pose=1.0, amb=23.0, amb_sd=1.0, cal=0.20, fpn=0.04, areca=0.30, osf=1.0, accl=1.0),
    "B": dict(psf=1.1, netd=0.05, pose=1.0, amb=24.0, amb_sd=1.2, cal=0.25, fpn=0.05, areca=0.35, osf=1.0, accl=1.0),
    "C": dict(psf=0.9, netd=0.05, pose=1.2, amb=23.5, amb_sd=1.0, cal=0.25, fpn=0.05, areca=0.35, osf=1.0, accl=1.0),
    # external, unseen: D = low-cost camera (blur, noise); E = rural population + short acclimatisation
    "D": dict(psf=1.7, netd=0.10, pose=1.3, amb=24.5, amb_sd=1.5, cal=0.35, fpn=0.08, areca=0.35, osf=1.0, accl=1.0),
    "E": dict(psf=1.0, netd=0.05, pose=1.5, amb=26.0, amb_sd=2.0, cal=0.30, fpn=0.05, areca=0.65, osf=2.2, accl=1.8),
}

# robustness multipliers (all 1.0 for the main cohorts)
FACT = dict(const=1.0, inflam=1.0, sess=1.0, pose=1.0, noise=1.0)

yy, xx = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), indexing="ij")


def _laplacian(n):
    e = np.ones(n)
    L1 = sp.diags([e[:-1], -2 * e, e[:-1]], [-1, 0, 1]).tolil()
    L1[0, 0] = -1
    L1[-1, -1] = -1  # Neumann
    L1 = L1.tocsr()
    I = sp.identity(n, format="csr")
    return (sp.kron(I, L1) + sp.kron(L1, I)).tocsr()


LAP = _laplacian(N)


def smooth_field(rng, sigma, shape=(N, N)):
    f = gaussian_filter(rng.standard_normal(shape), sigma, mode="reflect")
    return f / (f.std() + 1e-9)


def solve_pennes(omega, h, q, t_env):
    """D lap T + omega (Ta - T) + h (Tenv - T) + q = 0."""
    A = D_PX * LAP - sp.diags((omega + h).ravel())
    b = -(omega * T_ART + h * t_env + q).ravel()
    return spla.spsolve(A.tocsc(), b).reshape(N, N)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Patient:
    pass


def sample_patient(rng, pid, site):
    cfg = SITE_CFG[site]
    p = Patient()
    p.pid, p.site = pid, site
    p.age = float(np.clip(rng.normal(52, 11), 25, 85))
    p.female = int(rng.random() < 0.3)
    p.smoke = int(rng.random() < 0.40)
    p.areca = int(rng.random() < cfg["areca"])
    p.alcohol = int(rng.random() < 0.30)
    w = np.array([0.40, 0.20, 0.05, 0.15 * cfg["osf"], 0.20])
    if p.areca:
        w[3] *= 1.8
    w /= w.sum()
    p.ltype = int(rng.choice(5, p=w))
    gw = np.array([0.35, 0.30, 0.22, 0.13])
    if p.ltype in (1, 2):
        gw = np.array([0.15, 0.30, 0.30, 0.25])
    p.grade = int(rng.choice(4, p=gw))
    p.lsite = int(rng.choice(4, p=[0.50, 0.30, 0.10, 0.10]))
    p.large = int(rng.random() < 0.35)
    p.side = 1 if rng.random() < 0.5 else -1
    # transformation hazard (log-HR), literature-directed magnitudes
    lp = (0.015 * (p.age - 52) + 0.2 * p.female + 0.3 * p.smoke + 0.4 * p.areca + 0.2 * p.alcohol
          + [0.0, 0.8, 1.4, 0.5, -0.6][p.ltype] + [0.0, 0.5, 1.0, 1.6][p.grade]
          + [0.0, 0.5, 0.6, -0.2][p.lsite] + 0.5 * p.large)
    p.frailty = rng.normal(0, 0.8)
    lam = 0.0022  # per month scale (enriched surveillance cohort, ~29% observed 48-month events)
    shape = 1.1
    u = rng.random()
    p.tstar = (-np.log(u) / (lam * np.exp(lp + p.frailty - 1.0))) ** (1 / shape) / 1.0
    p.lp_clin = lp
    # progression thermal signature
    p.lead = rng.uniform(6, 18)
    p.ampl = rng.uniform(0.20, 0.60)
    # benign slow change for some non-progressors
    p.benign_drift = rng.uniform(0.05, 0.18) if rng.random() < 0.25 else 0.0
    p.benign_tau = rng.uniform(12, 36)
    # anatomy
    p.ca = rng.normal(0.60, 0.035)
    p.cb = rng.normal(0.42, 0.035)
    p.const_field = smooth_field(rng, 5.0)
    p.const_amp = abs(rng.normal(0.18, 0.06)) * FACT["const"]
    p.lip_tilt = rng.normal(0, 0.03)
    # lesion geometry
    sx = {0: 0.45, 1: 0.33, 2: 0.12, 3: 0.10}[p.lsite]
    sy = {0: 0.00, 1: 0.27, 2: 0.36, 3: -0.24}[p.lsite]
    p.lx = p.side * (sx + rng.normal(0, 0.03))
    p.ly = sy + rng.normal(0, 0.03)
    p.r0 = rng.uniform(0.11, 0.16) if p.large else rng.uniform(0.06, 0.10)
    p.harm = rng.normal(0, 1, size=(4, 2))
    p.hot_spots = rng.uniform(-1, 1, size=(4, 2)) * 0.6
    # schedule
    base_int = 4.0 if (p.grade >= 2 or p.ltype == 2) else 6.0
    t = 0.0
    visits = []
    dropout = rng.exponential(12 / 0.08)
    cens = min(dropout, 48.0)
    end = min(cens, p.tstar)
    while t < end and len(visits) < 10:
        if len(visits) == 0 or rng.random() > 0.12:
            visits.append(t)
        t += base_int * rng.uniform(0.8, 1.3)
    p.visits = np.array(visits)
    p.cens = cens
    p.time = min(cens, p.tstar)
    p.event = int(p.tstar <= cens)
    return p


def lesion_mask(p, t, prog):
    dx, dy = xx - p.lx, yy - p.ly
    rho = np.sqrt(dx ** 2 + dy ** 2)
    th = np.arctan2(dy, dx)
    irr = 0.10 + 0.25 * prog
    rr = p.r0 * (1 + 0.30 * prog)
    pert = sum(p.harm[k, 0] * np.cos((k + 2) * th) + p.harm[k, 1] * np.sin((k + 2) * th) for k in range(4))
    rt = rr * (1 + irr * 0.35 * pert)
    return sigmoid((rt - rho) / 0.012)


def render_canonical(p, t, rng, site, sess):
    """Return canonical-frame true surface temperature map at month t."""
    cav = sigmoid((1 - np.sqrt((xx / p.ca) ** 2 + ((yy - 0.03) / p.cb) ** 2)) / 0.03)
    lipring = sigmoid((1.18 - np.sqrt((xx / p.ca) ** 2 + ((yy - 0.03) / p.cb) ** 2)) / 0.03) - cav
    tongue = sigmoid((1 - np.sqrt((xx / 0.40) ** 2 + ((yy - 0.27) / 0.20) ** 2)) / 0.05) * cav
    skin = 1 - cav - lipring
    omega = 0.35 * skin + 0.55 * lipring + 1.0 * cav + 0.12 * tongue
    q = 0.02 * skin + 0.05 * cav
    hmuc = 0.12 * sess["evap"]
    h = 0.12 * skin + 0.20 * lipring + hmuc * cav
    # constitutive (patient-specific, stable) asymmetry
    omega = omega * (1 + p.const_amp * p.const_field * (cav + 0.5 * lipring))
    # osf: diffuse reduced perfusion of buccal mucosa (bilateral)
    if p.ltype == 3:
        buc = sigmoid((np.abs(xx) - 0.30) / 0.05) * cav
        omega = omega * (1 - 0.20 * buc)
    # lesion
    prog = float(sigmoid((t - (p.tstar - p.lead)) / (p.lead / 4.0)))
    m = lesion_mask(p, t, prog) * cav
    base_dw = {0: 0.0, 1: 0.06, 2: 0.20, 3: -0.10, 4: 0.08}[p.ltype]
    benign = p.benign_drift * (1 - np.exp(-t / p.benign_tau))
    omega = omega * (1 + (base_dw + benign) * m)
    if prog > 1e-3:
        spots = np.zeros_like(xx)
        for (ax, ay) in p.hot_spots:
            spots += np.exp(-(((xx - p.lx - ax * p.r0) ** 2 + (yy - p.ly - ay * p.r0) ** 2) / (2 * (0.35 * p.r0) ** 2)))
        het = 0.6 + 0.8 * np.clip(spots, 0, 1.5) / 1.5
        omega = omega * (1 + p.ampl * prog * m * het)
        q = q + 0.15 * p.ampl * prog * m
    # transient inflammation episodes
    for (ix, iy, ir, ia) in sess["inflam"]:
        g = np.exp(-(((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * ir ** 2))) * cav
        omega = omega * (1 + ia * g)
    T = solve_pennes(omega, h, q, sess["tenv"])
    # keratin insulation (leukoplakia) lowers surface temperature
    kd = {0: 0.25, 1: 0.12, 2: 0.0, 3: 0.05, 4: 0.0}[p.ltype]
    T = T - kd * m
    # smooth asymmetric session gradient (head tilt / airflow)
    T = T + sess["grad"] * xx * cav + sess["sfield"] * cav
    return T, prog, m


def session_params(p, rng, site, t):
    cfg = SITE_CFG[site]
    s = {}
    s["tenv"] = cfg["amb"] + FACT["sess"] * (rng.normal(cfg["amb"], cfg["amb_sd"]) - cfg["amb"])
    s["evap"] = (1.1 + FACT["sess"] * (rng.uniform(0.8, 1.4) - 1.1)) * (1 + 0.15 * (cfg["accl"] - 1) * rng.random())
    s["grad"] = rng.normal(0, 0.10 * cfg["accl"]) * FACT["sess"]
    s["sfield"] = 0.03 * cfg["accl"] * FACT["sess"] * smooth_field(rng, 6.0)
    pinf = {0: 0.15, 1: 0.18, 2: 0.20, 3: 0.15, 4: 0.35}[p.ltype] + 0.05 * p.smoke
    pinf *= FACT["inflam"]
    infl = []
    if rng.random() < pinf:
        if rng.random() < 0.45:  # at or near the lesion
            ix, iy = p.lx + rng.normal(0, 0.05), p.ly + rng.normal(0, 0.05)
        else:
            ix, iy = rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.35)
        infl.append((ix, iy, rng.uniform(0.06, 0.13), rng.uniform(0.25, 0.7)))
    s["inflam"] = infl
    s["calib"] = rng.normal(0, cfg["cal"])
    sc = cfg["pose"] * FACT["pose"]
    s["rot"] = np.deg2rad(rng.normal(0, 3.5 * sc))
    s["scale"] = rng.normal(1.0, 0.035 * sc)
    s["open"] = rng.normal(1.0, 0.05 * sc)
    s["tx"] = rng.normal(0, 1.5 * sc)
    s["ty"] = rng.normal(0, 1.5 * sc)
    s["elastic"] = 0.7 * sc
    return s


def camera(Tc, rng, site, sess, fpn):
    cfg = SITE_CFG[site]
    c = (N - 1) / 2
    # image coord -> canonical coord (inverse warp)
    iy, ix = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    X = (ix - c - sess["tx"]) / sess["scale"]
    Y = (iy - c - sess["ty"]) / (sess["scale"] * sess["open"])
    cr, sr = np.cos(-sess["rot"]), np.sin(-sess["rot"])
    Xc = cr * X - sr * Y + c
    Yc = sr * X + cr * Y + c
    ex = sess["elastic"] * smooth_field(rng, 8.0)
    ey = sess["elastic"] * smooth_field(rng, 8.0)
    img = map_coordinates(Tc, [Yc + ey, Xc + ex], order=1, mode="nearest")
    img = gaussian_filter(img, cfg["psf"])
    img = img + sess["calib"] + fpn + rng.normal(0, cfg["netd"] * FACT["noise"], size=img.shape)
    return np.round(img, 2)


def simulate_cohort(n, site, seed, pid0=0):
    rng = np.random.default_rng(seed)
    fpn = SITE_CFG[site]["fpn"] * smooth_field(np.random.default_rng(seed + 999), 10.0)
    pats, imgs = [], []
    for i in range(n):
        p = sample_patient(rng, pid0 + i, site)
        if len(p.visits) == 0:
            continue
        p.img_idx = []
        p.prog = []
        for t in p.visits:
            sess = session_params(p, rng, site, t)
            Tc, prog, m = render_canonical(p, t, rng, site, sess)
            img = camera(Tc, rng, site, sess, fpn)
            p.img_idx.append(len(imgs))
            p.prog.append(prog)
            imgs.append(img)
        pats.append(p)
    return pats, np.array(imgs, dtype=np.float32)
