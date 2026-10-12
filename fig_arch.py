"""Fig. 1: STaR-OC architecture (grayscale, non-overlapping blocks)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys, os
os.makedirs("figs", exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif", "font.size": 6.6})
W, H = 7.32, 4.85
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

BW, BH = 1.2, 0.66
XS = [0.05, 1.5, 2.95, 4.4, 5.85]
Y1, Y2, Y3 = 3.72, 1.98, 0.42
FILL = {"data": "#ffffff", "twin": "#d9d9d9", "learn": "#f0f0f0", "dec": "#bdbdbd", "loss": "#ffffff"}
boxes = {}


def box(key, x, y, text, kind, w=BW, h=BH, bold_title=None, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06", lw=0.8,
                       ec="black", fc=FILL[kind], ls=ls)
    ax.add_patch(p)
    if bold_title:
        ax.text(x + w / 2, y + h - 0.13, bold_title, ha="center", va="center", fontsize=6.4, fontweight="bold")
        ax.text(x + w / 2, y + (h - 0.2) / 2, text, ha="center", va="center", fontsize=6.0, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.2, linespacing=1.25)
    boxes[key] = (x, y, w, h)


def arrow(p0, p1, ls="-", lw=0.9, cs="arc3,rad=0", label=None, lpos=None, ha="center"):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=lw, color="black", ls=ls,
                        connectionstyle=cs, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if label:
        ax.text(*lpos, label, fontsize=5.8, ha=ha, va="center", style="italic")


def line(pts, ls="-", lw=0.9):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="black", lw=lw, ls=ls, solid_capstyle="butt")


# ---------------- row 1: acquisition and twin (left -> right)
box("img", XS[0], Y1, "intraoral thermogram\n$I_k$ at month $t_k$", "data", bold_title="Visit $k$")
box("reg", XS[1], Y1, "moment affine to\ncanonical frame + NCC\nrefinement to visit 1", "data", bold_title="Registration")
box("norm", XS[2], Y1, "offset removal, mirror\nasymmetry $A_k$, gradient\ncorrection, $\\tilde{T}_k$", "data", bold_title="Session normalisation")
box("twin", XS[3], Y1, "per-pixel level $b$ and\ndrift $d$; Huber update;\npopulation prior", "twin", bold_title="Patient thermal twin")
box("maps", XS[4], Y1, "innovation, drift-$z$,\nlevel change, drift s.d.;\nevidence $e_k$", "twin", bold_title="Twin maps")
for a, b in [("img", "reg"), ("reg", "norm"), ("norm", "twin"), ("twin", "maps")]:
    xa, ya, wa, ha_ = boxes[a]; xb, yb, wb, hb = boxes[b]
    arrow((xa + wa, ya + ha_ / 2), (xb, yb + hb / 2))
# twin recursion: posterior carried to the next visit (self-loop above the twin block)
xt, yt, wt, ht = boxes["twin"]
arrow((xt + wt - 0.3, yt + ht), (xt + 0.3, yt + ht), cs="arc3,rad=0.55", lw=0.8)
ax.text(xt + wt / 2, yt + ht + 0.27, "state $\\mathbf{s}_{k|k}$ carried to visit $k{+}1$", ha="center", fontsize=5.8, style="italic")

# ---------------- row 2: learning (right -> left)
X2 = [0.05, 1.983, 3.917, 5.85]
box("crop", X2[3], Y2, "32 x 32 crops: 2 registered\n+ 8 twin channels", "data", bold_title="Lesion-centred input")
box("enc", X2[2], Y2, "polar ring descriptor\n+ strided CNN branch;\nvisit-time encoding", "learn", bold_title="Visit encoder")
box("fuse", X2[1], Y2, "concatenation with\nclinical embedding", "learn", bold_title="Clinical fusion")
box("head", X2[0], Y2, "$h_{j}$ for 0-6, 6-12,\n12-18, 18-24 months;\n$F(\\tau_j)$", "learn", bold_title="Time-aware hazard")
xm, ym, wm, hm = boxes["maps"]; xc, yc, wc, hc = boxes["crop"]
arrow((xm + wm / 2, ym), (xc + wc / 2, yc + hc))
for a, b in [("crop", "enc"), ("enc", "fuse"), ("fuse", "head")]:
    xa, ya, wa, ha_ = boxes[a]; xb, yb, wb, hb = boxes[b]
    arrow((xa, ya + ha_ / 2), (xb + wb, yb + hb / 2))
# clinical metadata above the fusion block
box("clin", X2[1] - 0.06, 2.96, "9 variables: age, sex, tobacco,\nareca, alcohol, lesion type,\ngrade, site, size", "data", w=BW + 0.12, h=0.6, bold_title="Clinical metadata")
xq, yq, wq, hq = boxes["clin"]; xf, yf, wf, hf = boxes["fuse"]
arrow((xq + wq / 2, yq), (xf + wf / 2, yf + hf))
xh, yh, wh, hh = boxes["head"]
line([(xq, yq + hq / 2), (xh + wh / 2, yq + hq / 2)])
arrow((xh + wh / 2, yq + hq / 2), (xh + wh / 2, yh + hh))
ax.text((xq + xh + wh / 2) / 2, yq + hq / 2 + 0.08, "linear offset", fontsize=5.6, ha="center", style="italic")

# trainable group outline (dashed)
gx0, gx1 = X2[0] - 0.03, X2[2] + BW + 0.03
gy0, gy1 = Y2 - 0.1, Y2 + BH + 0.07
ax.add_patch(FancyBboxPatch((gx0, gy0), gx1 - gx0, gy1 - gy0, boxstyle="round,pad=0,rounding_size=0.05",
                            fill=False, ls=(0, (4, 2)), lw=0.7, ec="black"))
ax.text(gx1 - 0.04, gy0 - 0.09, "trainable parameters $\\theta$", ha="right", fontsize=5.8, style="italic")

# ---------------- row 3: training loss and inference outputs (left -> right)
box("loss", XS[0], Y3, "censored discrete-time\nlikelihood on all\nlandmark visits", "loss", bold_title="Training loss $\\mathcal{L}$")
box("ens", XS[1], Y3, "$M=5$ members: mean\nrisk and logit s.d. $\\sigma$", "learn", bold_title="Deep ensemble")
box("conf", XS[2], Y3, "score $s=\\mathrm{logit}F_{12}$\n$+\\kappa\\sigma+\\xi e$; clear\nif $s<\\hat{t}_\\alpha$", "dec", bold_title="Conformal clearance")
box("dec", XS[3], Y3, "cleared: routine\nsurveillance interval", "dec", bold_title="Automated decision")
box("ref", XS[4], Y3, "not cleared: specialist\nreview or biopsy", "dec", bold_title="Specialist referral")
xl, yl, wl, hl = boxes["loss"]
# forward (training) arrow: head -> loss
arrow((xh + 0.3, yh), (xl + 0.3, yl + hl))
ax.text(xh + 0.25, (yh + yl + hl) / 2 + 0.05, "training", fontsize=5.6, ha="right", style="italic", rotation=90, va="center")
# reverse-mode gradient: loss -> trainable group (dashed, upward)
arrow((xl + 0.72, yl + hl), (xl + 0.72, gy0), ls=(0, (3, 2)), lw=0.9)
ax.text(xl + 0.79, (yl + hl + gy0) / 2 + 0.02, "$\\nabla_{\\theta}\\mathcal{L}$", fontsize=6.2, ha="left", va="center")
# inference arrow: head -> ensemble (elbow)
xe, ye, we, he = boxes["ens"]
line([(xh + wh - 0.12, yh), (xh + wh - 0.12, 1.47), (xe + we / 2, 1.47)])
arrow((xe + we / 2, 1.47), (xe + we / 2, ye + he))
ax.text(xe + we / 2 + 0.05, 1.56, "inference", fontsize=5.6, ha="left", style="italic")
for a, b in [("ens", "conf"), ("conf", "dec")]:
    xa, ya, wa, ha_ = boxes[a]; xb, yb, wb, hb = boxes[b]
    arrow((xa + wa, ya + ha_ / 2), (xb, yb + hb / 2))
xc2, yc2, wc2, hc2 = boxes["conf"]; xr, yr, wr, hr = boxes["ref"]
line([(xc2 + wc2 / 2, yc2), (xc2 + wc2 / 2, 0.22), (xr + wr / 2, 0.22)])
arrow((xr + wr / 2, 0.22), (xr + wr / 2, yr))
# evidence score e_k from the twin maps to the clearance rule (drawn connection, right margin)
xm2, ym2, wm2, hm2 = boxes["maps"]
ye_ = 1.58
line([(xm2 + wm2, ym2 + 0.18), (xm2 + wm2 + 0.16, ym2 + 0.18), (xm2 + wm2 + 0.16, ye_), (xc2 + wc2 / 2, ye_)], lw=0.8)
arrow((xc2 + wc2 / 2, ye_), (xc2 + wc2 / 2, yc2 + hc2), lw=0.8)
ax.text(xm2 + wm2 + 0.08, ye_ - 0.1, "evidence score $e_k$ from the twin", fontsize=5.6, ha="right", style="italic")
out = sys.argv[1] if len(sys.argv) > 1 else "figs/fig1_arch.png"
fig.savefig(out, dpi=400)
