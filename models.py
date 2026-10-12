"""STaR-OC and deep baselines. All models emit discrete-time hazards for the intervals
(0,6], (6,12], (12,18], (18,24] months after every visit (landmark) of a patient sequence."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

J = 4  # hazard intervals
META_SLICES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 10), (10, 14), (14, 18), (18, 19)]


class VisitEncoder(nn.Module):
    """Lesion-centred visit encoder shared by every deep model.
    Polar branch: ring-wise mean (5 concentric rings) and spread (3 inner rings) of every input
    map around the annotated lesion centre, a rotation-invariant lesion descriptor.
    Morphology branch: a small strided CNN with global pooling for shape and texture."""
    RADII = (2, 4, 6, 9, 13)

    def __init__(self, cin, d=64, drop=0.2, mode="both"):
        super().__init__()
        self.mode = mode
        g = torch.arange(32).float() - 15.5
        rad = torch.sqrt(g.view(-1, 1) ** 2 + g.view(1, -1) ** 2)
        lo = (0,) + self.RADII[:-1]
        rings = torch.stack([((rad > l) & (rad <= h)).float() for l, h in zip(lo, self.RADII)])
        self.register_buffer("rings", rings / rings.sum((1, 2), keepdim=True))
        nf = cin * (len(self.RADII) + 3)
        self.polar = nn.Sequential(nn.LayerNorm(nf), nn.Linear(nf, 96), nn.GELU(), nn.Dropout(drop), nn.Linear(96, d))
        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.GroupNorm(8, o), nn.GELU())
        self.cnn = nn.Sequential(blk(cin, 16, 2), blk(16, 32, 2), blk(32, 48, 2))
        self.cproj = nn.Sequential(nn.Dropout(drop), nn.Linear(96, d))

    def descriptor(self, x):
        m = torch.einsum("nchw,rhw->ncr", x, self.rings)
        m2 = torch.einsum("nchw,rhw->ncr", x * x, self.rings[:3])
        sd = torch.sqrt(torch.clamp(m2 - m[:, :, :3] ** 2, min=1e-6))
        return torch.cat([m, sd], 2).flatten(1)

    def forward(self, x):
        h = 0
        if self.mode in ("both", "polar"):
            h = h + self.polar(self.descriptor(x))
        if self.mode in ("both", "cnn"):
            f = self.cnn(x)
            h = h + self.cproj(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        return F.gelu(h)


def time_features(t, dt, d):
    """Continuous-time encoding of visit time t and preceding interval dt (months)."""
    half = d // 2
    freqs = torch.exp(torch.linspace(math.log(1 / 64), math.log(1 / 2), half, device=t.device))
    ang = t.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


class TimeDecayAttention(nn.Module):
    """Causal multi-head attention over visits with a learned per-head penalty on the
    elapsed time between visits (time-aware temporal mixing)."""
    def __init__(self, d=64, heads=4, time_aware=True, drop=0.1):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.time_aware = time_aware
        self.lam = nn.Parameter(torch.full((heads,), -1.0))
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, t, mask):
        B, K, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, K, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dk)          # (B,h,K,K)
        if self.time_aware:
            gap = (t.unsqueeze(2) - t.unsqueeze(1)).abs() / 12.0          # years
            logits = logits - F.softplus(self.lam).view(1, -1, 1, 1) * gap.unsqueeze(1)
        causal = torch.tril(torch.ones(K, K, dtype=torch.bool, device=x.device))
        allow = causal.unsqueeze(0) & mask.unsqueeze(1)                  # (B,K,K)
        logits = logits.masked_fill(~allow.unsqueeze(1), -1e9)
        a = torch.softmax(logits, -1)
        self.last_attn = a.detach()
        y = (a @ v).transpose(1, 2).reshape(B, K, d)
        x = x + self.drop(self.o(y))
        return x + self.drop(self.ff(self.n2(x)))


class GatedCrossModal(nn.Module):
    """Image representation queries one token per clinical variable; a sigmoid gate decides
    how much clinical context enters each visit representation."""
    def __init__(self, d=64):
        super().__init__()
        self.tok = nn.ModuleList([nn.Linear(b - a, d) for a, b in META_SLICES])
        self.vtype = nn.Parameter(torch.randn(len(META_SLICES), d) * 0.02)
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.g = nn.Linear(2 * d, d)

    def tokens(self, m):
        return torch.stack([lin(m[:, a:b]) for lin, (a, b) in zip(self.tok, META_SLICES)], 1) + self.vtype

    def forward(self, h, m):
        E = self.tokens(m)                                             # (B,P,d)
        att = torch.softmax(self.q(h) @ self.k(E).transpose(1, 2) / math.sqrt(h.shape[-1]), -1)  # (B,K,P)
        self.last_attn = att.detach()
        c = att @ self.v(E)
        g = torch.sigmoid(self.g(torch.cat([h, c], -1)))
        self.last_gate = g.detach()
        return h + g * c


class HazardHead(nn.Module):
    """Time-aware discrete-time hazard head. The visit representation is reduced to a scalar
    thermal-progression score that loads on each horizon interval with its own non-negative
    weight, on top of interval-specific baseline log-odds."""
    def __init__(self, d=64, extra=0, drop=0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d + extra, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))
        self.alpha = nn.Parameter(torch.full((J,), -3.0))
        self.load = nn.Parameter(torch.zeros(J))

    def forward(self, z):
        s = self.mlp(z)                                   # (..., 1)
        return self.alpha + F.softplus(self.load + 0.5413) * s   # softplus(0.5413) = 1


class STaROC(nn.Module):
    """use_twin/use_raw choose input channels (handled by the data loader);
    fusion: gated | concat | none; temporal: attn | none."""
    def __init__(self, cin, d=64, fusion="gated", temporal="attn", time_aware=True, drop=0.2, meta_dim=19, enc="both"):
        super().__init__()
        self.enc = VisitEncoder(cin, d, drop, enc)
        self.tproj = nn.Linear(d + 2, d)
        self.temporal = temporal
        self.time_aware = time_aware
        if temporal == "attn":
            self.att = TimeDecayAttention(d, 4, time_aware, drop)
        self.fusion = fusion
        if fusion == "gated":
            self.fuse = GatedCrossModal(d)
            extra = 0
        elif fusion == "concat":
            self.mproj = nn.Sequential(nn.Linear(meta_dim, d), nn.GELU())
            extra = d
        else:
            extra = 0
        self.head = HazardHead(d, extra, drop)
        self.clin = nn.Linear(meta_dim, 1)   # clinical log-odds offset shared by all intervals
        self.d = d

    def forward(self, x, t, dt, mask, m):
        B, K = mask.shape
        h = self.enc(x.flatten(0, 1)).view(B, K, -1)
        tf = time_features(t, dt, self.d) if self.time_aware else torch.zeros(B, K, self.d, device=x.device)
        h = self.tproj(torch.cat([h + tf, (t / 48).unsqueeze(-1), (dt / 12).unsqueeze(-1)], -1)) if self.time_aware \
            else self.tproj(torch.cat([h, torch.zeros(B, K, 2, device=x.device)], -1))
        if self.temporal == "attn":
            h = self.att(h, t, mask)
        mm = m.unsqueeze(1).expand(B, K, m.shape[-1])
        if self.fusion == "gated":
            z = self.fuse(h, m)
        elif self.fusion == "concat":
            z = torch.cat([h, self.mproj(mm)], -1)
        else:
            z = h
        out = self.head(z)
        if self.fusion != "none":
            out = out + self.clin(m).unsqueeze(1)
        return out


class TLSTMCell(nn.Module):
    """Time-aware LSTM: short-term memory discounted by g(dt) = 1/log(e + dt)."""
    def __init__(self, din, d):
        super().__init__()
        self.lstm = nn.LSTMCell(din, d)
        self.wd = nn.Linear(d, d)

    def forward(self, x, hc, dt):
        h, c = hc
        cs = torch.tanh(self.wd(c))
        g = 1.0 / torch.log(math.e + dt.unsqueeze(-1))
        c = c - cs + cs * g
        return self.lstm(x, (h, c))


class TLSTMModel(nn.Module):
    def __init__(self, cin, d=64, drop=0.2, meta_dim=19):
        super().__init__()
        self.enc = VisitEncoder(cin, d, drop)
        self.cell = TLSTMCell(d, d)
        self.mproj = nn.Sequential(nn.Linear(meta_dim, d), nn.GELU())
        self.head = HazardHead(d, d, drop)
        self.clin = nn.Linear(meta_dim, 1)

    def forward(self, x, t, dt, mask, m):
        B, K = mask.shape
        e = self.enc(x.flatten(0, 1)).view(B, K, -1)
        h = torch.zeros(B, e.shape[-1], device=x.device)
        c = torch.zeros_like(h)
        outs = []
        for k in range(K):
            h, c = self.cell(e[:, k], (h, c), dt[:, k])
            outs.append(h)
        H = torch.stack(outs, 1)
        mm = self.mproj(m).unsqueeze(1).expand(B, K, -1)
        return self.head(torch.cat([H, mm], -1)) + self.clin(m).unsqueeze(1)


class CrossSectional(nn.Module):
    """Single-visit CNN (or Siamese difference model when cin includes first-visit maps)."""
    def __init__(self, cin, d=64, drop=0.2, meta_dim=19):
        super().__init__()
        self.enc = VisitEncoder(cin, d, drop)
        self.mproj = nn.Sequential(nn.Linear(meta_dim, d), nn.GELU())
        self.head = HazardHead(d, d, drop)
        self.clin = nn.Linear(meta_dim, 1)

    def forward(self, x, t, dt, mask, m):
        B, K = mask.shape
        e = self.enc(x.flatten(0, 1)).view(B, K, -1)
        mm = self.mproj(m).unsqueeze(1).expand(B, K, -1)
        return self.head(torch.cat([e, mm], -1)) + self.clin(m).unsqueeze(1)


def surv_loss(logits, u, delta, valid, edges=(6.0, 12.0, 18.0, 24.0)):
    """Censored discrete-time likelihood. u: time from landmark to event/censoring (months)."""
    edges = torch.tensor(edges, device=logits.device)
    starts = torch.cat([torch.zeros(1, device=logits.device), edges[:-1]])
    mids = (starts + edges) / 2
    u_ = u.unsqueeze(-1)
    d_ = delta.unsqueeze(-1)
    # interval containing the event
    ev_in = (d_ > 0) & (u_ > starts) & (u_ <= edges)
    # survived interval: fully passed, or censored after its midpoint
    surv = (u_ > edges) | ((d_ == 0) & (u_ > mids) & (u_ <= edges))
    logh = F.logsigmoid(logits)
    log1mh = F.logsigmoid(-logits)
    ll = (surv.float() * log1mh + ev_in.float() * logh).sum(-1)
    return -(ll * valid).sum() / valid.sum().clamp(min=1)


def cum_risk(logits):
    h = torch.sigmoid(logits)
    return 1 - torch.cumprod(1 - h, -1)
