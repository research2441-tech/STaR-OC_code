"""Train one model (one seed) and write landmark predictions for val/cal/test/external sites.
Example: python train.py --model staroc --chan both --twin twin_full --seed 0 --out runs/staroc_s0
"""
import argparse, json, os, pickle, time
import numpy as np
import torch
import models as M
from build_dataset import meta_vector

SITES = ["A", "B", "C", "D", "E"]
KMAX = 10


def load(prep, twin_dir, chan, model, perturb=None, rng=None):
    split = {int(k): v for k, v in json.load(open("splits.json")).items()}
    seqs = []
    for s in SITES:
        pats = pickle.load(open(f"{prep}/pats_{s}.pkl", "rb"))
        raw = np.load(f"{prep}/raw_{s}.npy").astype(np.float32)
        tw = np.load(f"{twin_dir}/twin_{s}.npy").astype(np.float32) if chan in ("both", "twin") else None
        ev = np.load(f"{twin_dir}/evid_{s}.npy") if os.path.exists(f"{twin_dir}/evid_{s}.npy") else None
        for p in pats:
            ii = p["img_idx"]
            r = raw[ii]
            if model == "siam":
                x = np.concatenate([r, r - r[:1]], 1)
            elif chan == "raw":
                x = r
            elif chan == "twin":
                x = tw[ii]
            else:
                x = np.concatenate([r, tw[ii]], 1)
            t = np.array(p["visits"], np.float32)
            dt = np.concatenate([[0.0], np.diff(t)]).astype(np.float32)
            seqs.append(dict(pid=p["pid"], site=s, split=split[p["pid"]], x=x, t=t, dt=dt,
                             m=meta_vector(p), time=p["time"], event=p["event"],
                             evid=(ev[ii] if ev is not None else np.ones(len(ii), np.float32)),
                             prog=np.array(p["prog"], np.float32)))
    return seqs


def batchify(seqs, mu, sd, aug=False, rng=None):
    B = len(seqs)
    C = seqs[0]["x"].shape[1]
    X = np.zeros((B, KMAX, C, 32, 32), np.float32)
    T = np.zeros((B, KMAX), np.float32); DT = np.zeros((B, KMAX), np.float32)
    MASK = np.zeros((B, KMAX), bool); U = np.zeros((B, KMAX), np.float32); D = np.zeros((B, KMAX), np.float32)
    Mt = np.stack([s["m"] for s in seqs])
    for i, s in enumerate(seqs):
        K = len(s["t"])
        x = (s["x"] - mu) / sd
        if aug:
            for k in range(K):
                dy, dx = rng.integers(-1, 2, 2)
                x[k] = np.roll(x[k], (dy, dx), (1, 2))
        X[i, :K] = x; T[i, :K] = s["t"]; DT[i, :K] = s["dt"]; MASK[i, :K] = True
        U[i, :K] = s["time"] - s["t"]; D[i, :K] = s["event"]
    f = torch.from_numpy
    return f(X), f(T), f(DT), f(MASK), f(Mt), f(U), f(D)


def build(args, cin):
    if args.model == "staroc":
        return M.STaROC(cin, fusion=args.fusion, temporal=args.temporal, time_aware=not args.no_time, enc=getattr(args, "enc", "both"))
    if args.model == "tlstm":
        return M.TLSTMModel(cin)
    return M.CrossSectional(cin)


def predict(model, seqs, mu, sd, bs=64):
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            ch = seqs[i:i + bs]
            X, T, DT, MASK, Mt, U, D = batchify(ch, mu, sd)
            lg = model(X, T, DT, MASK, Mt)
            F = M.cum_risk(lg).numpy()
            for b, s in enumerate(ch):
                for k in range(len(s["t"])):
                    rows.append([s["pid"], k, s["t"][k], s["time"] - s["t"][k], s["event"], *F[b, k], *lg[b, k].numpy(), s["evid"][k]])
    return np.array(rows, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="staroc")
    ap.add_argument("--chan", default="both")
    ap.add_argument("--prep", default="prep")
    ap.add_argument("--twin", default="twin_full")
    ap.add_argument("--fusion", default="gated")
    ap.add_argument("--temporal", default="attn")
    ap.add_argument("--no_time", action="store_true")
    ap.add_argument("--enc", default="both")
    ap.add_argument("--fast_mult", type=float, default=1.0)
    ap.add_argument("--cv_fold", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(int(os.environ.get("NT", "1")))
    rng = np.random.default_rng(args.seed)
    seqs = load(args.prep, args.twin, args.chan, args.model)
    if args.cv_fold >= 0:
        # internal 5-fold cross-validation on development training+validation patients
        dev = sorted({s["pid"] for s in seqs if s["split"] in ("train", "val")})
        fold = {p: i % 5 for i, p in enumerate(np.random.default_rng(123).permutation(dev))}
        for s in seqs:
            if s["split"] in ("train", "val"):
                s["split"] = "val" if fold[s["pid"]] == args.cv_fold else "train"
            else:
                s["split"] = "skip"
    tr = [s for s in seqs if s["split"] == "train"]
    va = [s for s in seqs if s["split"] == "val"]
    allx = np.concatenate([s["x"] for s in tr])
    mu = allx.mean((0, 2, 3), keepdims=True)[0]
    sd = allx.std((0, 2, 3), keepdims=True)[0] + 1e-6
    cin = allx.shape[1]
    model = build(args, cin)
    nparam = sum(p.numel() for p in model.parameters())
    fast = [p for n, p in model.named_parameters() if n.startswith("clin") or n.startswith("head.alpha") or n.startswith("head.load")]
    slow = [p for n, p in model.named_parameters() if not (n.startswith("clin") or n.startswith("head.alpha") or n.startswith("head.load"))]
    opt = torch.optim.AdamW([dict(params=slow, lr=args.lr, weight_decay=1e-3),
                             dict(params=fast, lr=args.lr * args.fast_mult, weight_decay=0.0)])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, bad, hist = 1e9, 0, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(tr))
        tl = 0
        for i in range(0, len(tr), 32):
            X, T, DT, MASK, Mt, U, D = batchify([tr[j] for j in perm[i:i + 32]], mu, sd, aug=True, rng=rng)
            lg = model(X, T, DT, MASK, Mt)
            loss = M.surv_loss(lg, U, D, MASK.float())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tl += loss.item() * len(perm[i:i + 32])
        sched.step()
        model.eval()
        with torch.no_grad():
            vl, vn = 0, 0
            for i in range(0, len(va), 64):
                X, T, DT, MASK, Mt, U, D = batchify(va[i:i + 64], mu, sd)
                vl += M.surv_loss(model(X, T, DT, MASK, Mt), U, D, MASK.float()).item() * MASK.sum().item()
                vn += MASK.sum().item()
        vl /= vn
        hist.append((ep, tl / len(tr), vl))
        if vl < best - 1e-4:
            best, bad = vl, 0
            torch.save(model.state_dict(), f"{args.out}/model.pt")
        else:
            bad += 1
            if bad >= 6:
                break
    train_time = time.time() - t0
    model.load_state_dict(torch.load(f"{args.out}/model.pt"))
    t1 = time.time()
    for sp in (["val"] if args.cv_fold >= 0 else ["val", "cal", "test", "extD", "extE"]):
        P = predict(model, [s for s in seqs if s["split"] == sp], mu, sd)
        np.save(f"{args.out}/pred_{sp}.npy", P)
    infer_time = time.time() - t1
    np.savez(f"{args.out}/norm.npz", mu=mu, sd=sd)
    json.dump(dict(args=vars(args), nparam=nparam, train_time=train_time, infer_time=infer_time,
                   epochs=len(hist), best_val=best, hist=hist), open(f"{args.out}/info.json", "w"))
    print(args.out, "params", nparam, "epochs", len(hist), "val", round(best, 4), "time", round(train_time), flush=True)


if __name__ == "__main__":
    main()
