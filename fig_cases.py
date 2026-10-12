"""Fig. 2 (example sequences) and Fig. 3 (twin trajectories) for one progressor and one
non-progressor with a transient warm episode, both from the internal test split."""
import sys, json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import twin as TW

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif", "font.size": 7})
W = "./"
import os
os.makedirs("figs", exist_ok=True)
split = {int(k): v for k, v in json.load(open(W + "splits.json")).items()}
pats = pickle.load(open(W + "prep/pats_A.pkl", "rb"))
raw = np.load(W + "prep/raw_A.npy").astype(np.float32)
imgs = np.load(W + "data/imgs_A.npy")
prior = pickle.load(open(W + "twin_full/twin_prior.pkl", "rb"))
hp = json.load(open(W + "twin_full/twin_hp.json"))
g = np.arange(32) - 15.5
disc = (g[:, None] ** 2 + g[None, :] ** 2) <= 16


def roi(x):
    return x[..., disc].mean(-1)


def pick():
    prog, non = None, None
    for p in pats:
        if split[p["pid"]] != "test" or len(p["visits"]) < 5:
            continue
        r = raw[p["img_idx"]]
        a = roi(r[:, 0])
        if p["event"] and prog is None and p["prog"][-1] > 0.9 and p["prog"][0] < 0.05 and a[-1] - a[0] > 0.25 and p["lsite"] == 0:
            prog = p
        if not p["event"] and non is None and p["time"] > 40 and p["const_amp"] > 0.2 and p["lsite"] == 0:
            jumps = np.abs(np.diff(a))
            if jumps.max() > 0.3 and np.argmax(jumps) in (1, 2):
                non = p
    return prog, non


prog, non = pick()
print("progressor", prog["pid"], prog["visits"], "nonprog", non["pid"], non["visits"])


def twin_run(p):
    mu0 = prior["gmean"].get(TW.group_key(p), prior["gall"])
    o, e = TW.run_patient(raw[p["img_idx"]], np.array(p["visits"]), mu0, prior["tau2"], hp)
    return o, e


# ---------------- Fig. 2: image panels
cases = [(prog, "Progressor"), (non, "Non-progressor")]
nv = 4
fig, axes7 = plt.subplots(7, nv, figsize=(3.5, 6.0), gridspec_kw=dict(hspace=0.12, wspace=0.06, height_ratios=[1, 1, 1, 0.32, 1, 1, 1]))
for a in axes7[3]:
    a.axis("off")
axes = np.concatenate([axes7[:3], axes7[4:]])
ims = []
for ci, (p, name) in enumerate(cases):
    K = len(p["visits"])
    sel = np.linspace(0, K - 1, nv).round().astype(int)
    o, _ = twin_run(p)
    amax = float(np.ceil(np.percentile(np.abs(raw[[p["img_idx"][k] for k in sel], 0]), 99) * 10) / 10)
    for j, k in enumerate(sel):
        ii = p["img_idx"][k]
        a0 = axes[3 * ci, j]; a1 = axes[3 * ci + 1, j]; a2 = axes[3 * ci + 2, j]
        im0 = a0.imshow(imgs[ii], cmap="gray", vmin=32.5, vmax=36.5)
        im1 = a1.imshow(raw[ii, 0], cmap="gray", vmin=-amax, vmax=amax)
        im2 = a2.imshow(o[k, 2], cmap="gray", vmin=-3, vmax=3)
        for a in (a0, a1, a2):
            a.set_xticks([]); a.set_yticks([])
        a0.set_title("month %.0f" % p["visits"][k], fontsize=6.5, pad=2)
        if j == 0:
            a0.set_ylabel("%s\nthermogram" % name, fontsize=6.3)
            a1.set_ylabel("asymmetry\ncrop $A_k$", fontsize=6.3)
            a2.set_ylabel("drift $z$\n(twin)", fontsize=6.3)
    ims.append((im0, im1, im2))
for ci in range(2):
    for r, (im, lab) in enumerate(zip(ims[ci], ["deg C", "deg C", "z"])):
        pos = axes[3 * ci + r, nv - 1].get_position()
        cax = fig.add_axes([pos.x1 + 0.025, pos.y0 + 0.1 * pos.height, 0.018, 0.8 * pos.height])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=5.2)
        cb.set_label(lab, fontsize=5.4)
fig.savefig(W + "figs/fig2_cases.png", dpi=400, bbox_inches="tight")

# ---------------- Fig. 3: twin trajectories at the lesion centre
fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.3), sharex="col", gridspec_kw=dict(hspace=0.22, wspace=0.42))
for ci, (p, name) in enumerate(cases):
    t = np.array(p["visits"])
    r = raw[p["img_idx"]]
    obs = roi(r[:, 0])
    o, e = twin_run(p)
    ax = axes[0, ci]
    ax.plot(t, obs - obs[0], "o", ms=3, mfc="white", mec="black", label="observed change in ROI asymmetry")
    ax.plot(t, roi(o[:, 4]), "-", color="black", lw=1.0, label="twin level change $\\Delta b$")
    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    ax.set_title(name, fontsize=7, pad=3)
    ax.set_ylabel("change (deg C)" if ci == 0 else "", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax2 = axes[1, ci]
    ax2.plot(t, roi(o[:, 2]), "-s", color="black", ms=2.5, lw=0.9, label="drift $z$ (asymmetry)")
    ax2.plot(t, roi(o[:, 3]), "--^", color="0.45", ms=2.5, lw=0.9, label="drift $z$ (temperature)")
    ax2.plot(t, roi(o[:, 0]), ":", color="0.3", lw=0.9, label="innovation $z$")
    ax2.axhline(0, color="0.7", lw=0.5)
    ax2.set_xlabel("months since enrolment", fontsize=6.5)
    ax2.set_ylabel("z" if ci == 0 else "")
    ax2.tick_params(labelsize=6)
    if p["event"]:
        ax.axvline(p["tstar"] - p["lead"], color="0.5", lw=0.6, ls="--", label="latent angiogenic onset")
        ax2.axvline(p["tstar"] - p["lead"], color="0.5", lw=0.6, ls="--")
h1, l1 = axes[0, 0].get_legend_handles_labels()
h2, l2 = axes[1, 0].get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=2, fontsize=5.6, frameon=False, bbox_to_anchor=(0.5, -0.17))
fig.savefig(W + "figs/fig3_twin.png", dpi=400, bbox_inches="tight")
json.dump(dict(prog=int(prog["pid"]), non=int(non["pid"]), prog_tstar=float(prog["tstar"]), prog_onset=float(prog["tstar"] - prog["lead"])),
          open(W + "results/fig_cases.json", "w"))
