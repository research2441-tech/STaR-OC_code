"""Result figures (grayscale): Fig. 4-9."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                     "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
W = "./"
import os
os.makedirs("figs", exist_ok=True)
res = json.load(open(W + "results/results.json"))
dec = json.load(open(W + "results/decision.json"))
NAMES = {"cox_clin": "Cox-Clin", "cox_asym": "Cox-Asym", "cox_long": "Cox-Long", "cs": "CS-Net", "siam": "Siamese-Δ",
         "tlstm": "T-LSTM", "tlstm_twin": "T-LSTM+Twin", "final": "STaR-OC", "ab_notwin": "STaR-OC w/o twin"}
STY = {"cox_clin": ("0.65", ":", "x"), "cox_asym": ("0.55", "--", "v"), "cox_long": ("0.35", "-.", "s"),
       "cs": ("0.6", "--", "o"), "siam": ("0.45", ":", "D"), "tlstm": ("0.3", "--", "^"),
       "tlstm_twin": ("0.2", ":", "P"), "final": ("0.0", "-", "o"), "ab_notwin": ("0.5", "-", "o")}


def style(m, ax_plot, x, y, **kw):
    c, ls, mk = STY[m]
    mfc = "white" if m in ("cs", "tlstm", "cox_asym", "ab_notwin") else c
    ax_plot.plot(x, y, color=c, ls=ls, marker=mk, ms=3.6, lw=1.1 if m == "final" else 0.8, mfc=mfc, mec=c,
                 label=NAMES[m], **kw)


# ---------------- Fig. 4: by prior visits and horizon
ms = ["cox_long", "cs", "siam", "tlstm", "tlstm_twin", "final"]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw=dict(wspace=0.28))
labs = ["1", "2", "3", "4", "5+"]
for m in ms:
    style(m, axs[0], range(5), [res["byk"][m][l][0] for l in labs])
axs[0].set_xticks(range(5)); axs[0].set_xticklabels(labs)
axs[0].set_xlabel("visits available at the landmark (k)")
axs[0].set_ylabel("12-month AUC")
axs[0].set_title("(a) discrimination by follow-up depth", fontsize=7)
hz = ["6.0", "12.0", "18.0", "24.0"]
for m in ms:
    style(m, axs[1], [6, 12, 18, 24], [res["byk"][m]["horizon"][h] for h in hz])
axs[1].set_xticks([6, 12, 18, 24])
axs[1].set_xlabel("prediction horizon (months)")
axs[1].set_ylabel("time-dependent AUC")
axs[1].set_title("(b) discrimination by horizon", fontsize=7)
for a in axs:
    a.grid(alpha=0.25, lw=0.4)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=6.3, bbox_to_anchor=(0.5, -0.12))
fig.savefig(W + "figs/fig4_byk.png", dpi=400, bbox_inches="tight")

# ---------------- Fig. 5: reliability
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw=dict(wspace=0.3))
for ax, sp, t in zip(axs, ["test", "extD", "extE"], ["(a) internal test", "(b) external site D", "(c) external site E"]):
    ax.plot([0, 0.6], [0, 0.6], color="0.7", lw=0.7, ls="-")
    for m in ["cox_long", "tlstm", "final"]:
        r = np.array(res["cal"][f"{m}|{sp}"])
        style(m, ax, r[:, 0], r[:, 1])
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.6)
    ax.set_xlabel("predicted 12-month risk")
    ax.set_title(t, fontsize=7)
    ax.grid(alpha=0.25, lw=0.4)
axs[0].set_ylabel("observed 12-month incidence")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.2))
fig.savefig(W + "figs/fig5_calib.png", dpi=400, bbox_inches="tight")

# ---------------- Fig. 6: decision layer (alpha = 0.10)
D = dec["final"]
fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.45), gridspec_kw=dict(wspace=0.55))
rules = [("fixed", "validation threshold", "white", "////"), ("conf_full", "conformal, calibration split", "0.7", ""),
         ("conf_full_local", "conformal, site-local", "0.2", "")]
sites = [("test", "internal"), ("extD", "site D"), ("extE", "site E")]
wbar = 0.25
for j, (rk, rl, col, hat) in enumerate(rules):
    idx = [i for i, (s_, _) in enumerate(sites) if f"0.1|{rk}|{s_}" in D]
    xs = np.array(idx) + (j - 1) * wbar
    axs[0].bar(xs, [D[f"0.1|{rk}|{sites[i][0]}"]["miss"] for i in idx], wbar, color=col, edgecolor="black", lw=0.5, hatch=hat, label=rl)
    axs[1].bar(xs, [D[f"0.1|{rk}|{sites[i][0]}"]["cleared"] for i in idx], wbar, color=col, edgecolor="black", lw=0.5, hatch=hat)
axs[0].axhline(0.10, color="black", ls="--", lw=0.7)
axs[0].text(-0.45, 0.104, "$\\alpha$ = 0.10", fontsize=6, va="bottom", ha="left")
axs[0].set_ylim(0, 0.16); axs[1].set_ylim(0, 0.55)
for a_, t, yl in zip(axs, ["(a) missed progressors", "(b) cleared patients"], ["share of 12-month progressors cleared", "share of patients cleared"]):
    a_.set_xticks(range(3)); a_.set_xticklabels([s_ for _, s_ in sites], fontsize=6.3)
    a_.set_title(t, fontsize=7); a_.set_ylabel(yl, fontsize=6.3)
    a_.grid(axis="y", alpha=0.25, lw=0.4)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=6.0, bbox_to_anchor=(0.5, -0.2))
fig.savefig(W + "figs/fig6_decision.png", dpi=400, bbox_inches="tight")

# ---------------- Fig. 7: stress tests
conds = ["base", "const0.5", "const2", "inflam0", "inflam2", "sess2", "pose2", "noise2", "keep0.5"]
clab = {"base": "reference", "const0.5": "constitutive x0.5", "const2": "constitutive x2", "inflam0": "no inflammation",
        "inflam2": "inflammation x2", "sess2": "session effects x2", "pose2": "pose variation x2", "noise2": "detector noise x2",
        "keep0.5": "50% visits dropped"}
mm = ["cox_long", "cs", "tlstm", "tlstm_twin", "ab_notwin", "final"]
fig, ax = plt.subplots(figsize=(3.5, 3.0))
off = np.linspace(-0.3, 0.3, len(mm))
for i, m in enumerate(mm):
    ys = [res["rob"][c][m]["auc_ens"] for c in conds]
    c, ls, mk = STY[m]
    mfc = "white" if m in ("cs", "tlstm", "ab_notwin") else c
    ax.plot(ys, np.arange(len(conds)) + off[i], ls="none", marker=mk, ms=3.6, color=c, mfc=mfc, mec=c, label=NAMES[m])
ax.set_yticks(range(len(conds))); ax.set_yticklabels([clab[c] for c in conds], fontsize=6.3)
ax.invert_yaxis()
for y in range(len(conds) - 1):
    ax.axhline(y + 0.5, color="0.85", lw=0.4)
ax.set_xlabel("12-month AUC (seed ensemble)")
ax.grid(axis="x", alpha=0.25, lw=0.4)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.0, bbox_to_anchor=(0.55, -0.12))
fig.savefig(W + "figs/fig7_stress.png", dpi=400, bbox_inches="tight")

# ---------------- Fig. 8: KM by risk tertile
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True, gridspec_kw=dict(wspace=0.12))
gl = {"0": ("low", "0.6", ":"), "1": ("intermediate", "0.35", "--"), "2": ("high", "0.0", "-")}
for ax, sp, t in zip(axs, ["test", "extD", "extE"], ["(a) internal test", "(b) external site D", "(c) external site E"]):
    k = res["km"][sp]
    for g in ("0", "1", "2"):
        tl, sv, n = k["curves"][g]
        ax.step(tl, sv, where="post", color=gl[g][1], ls=gl[g][2], lw=1.0, label="%s-risk tertile" % gl[g][0])
    ax.set_xlim(0, 24); ax.set_ylim(0.55, 1.01)
    ax.set_xlabel("months after landmark visit")
    p = k["logrank_p"]
    ax.text(0.5, 0.575, "log-rank $p$ %s" % ("< 0.001" if p < 0.001 else "= %.3f" % p), fontsize=6.3)
    ax.set_title(t, fontsize=7)
    ax.grid(alpha=0.25, lw=0.4)
axs[0].set_ylabel("transformation-free probability")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.2))
fig.savefig(W + "figs/fig8_km.png", dpi=400, bbox_inches="tight")

# ---------------- Fig. 9: ablation forest
AB = [("ab_notwin", "without twin maps"), ("ab_twinonly", "twin maps only"), ("ab_nodrift", "level-only twin (no drift)"),
      ("ab_norobust", "Gaussian update (no Huber)"), ("ab_noreg", "no registration"), ("ab_moment", "moment registration only"),
      ("ab_nonorm", "no session normalisation"), ("ab_nometa", "no clinical variables"), ("ab_notime", "no visit-time encoding"),
      ("ab_polar", "polar branch only"), ("ab_cnn", "CNN branch only"), ("abl_concat", "+ time-decay attention"),
      ("abl_notemporal", "gated cross-modal fusion"), ("staroc", "gated fusion + attention")]
AB = [a for a in AB if a[0] in res["abl"]]
fig, ax = plt.subplots(figsize=(3.5, 3.4))
for i, (k, lab) in enumerate(AB):
    o, lo, hi, p = res["abl"][k]["delta"]["AUC12"][:4]
    ax.plot([lo, hi], [i, i], color="black", lw=0.8)
    ax.plot([o], [i], marker="s", ms=3.6, color="black" if p < 0.05 else "white", mec="black", ls="none")
ax.axvline(0, color="0.5", lw=0.7, ls="--")
ax.set_yticks(range(len(AB))); ax.set_yticklabels([l for _, l in AB], fontsize=6.3)
ax.invert_yaxis()
ax.set_xlabel("change in 12-month AUC vs. STaR-OC (95% CI)")
ax.grid(axis="x", alpha=0.25, lw=0.4)
ax.set_title("filled markers: paired bootstrap $p$ < 0.05", fontsize=6.0, loc="right", pad=4)
fig.savefig(W + "figs/fig9_ablation.png", dpi=400, bbox_inches="tight")
print("figures done")
