"""Inference of a trained run on a (possibly perturbed) cohort. The twin is recomputed from the
raw crops so that visit subsampling and alternative twin settings are honoured.
Usage: python infer.py RUN_DIR DATA_DIR SITES OUT_NAME [--keep 1.0] [--twin_hp file] [--mc 0]"""
import argparse, json, pickle, os
import numpy as np
import torch
import models as M
import twin as TW
from build_dataset import meta_vector
from train import batchify, build


def load_cohort(ddir, sites, chan, model, prior, hp, keep=1.0, seed=0, robust=True, drift=True):
    rng = np.random.default_rng(seed)
    seqs = []
    for s in sites:
        pats = pickle.load(open(f"{ddir}/pats_{s}.pkl", "rb"))
        raw = np.load(f"{ddir}/raw_{s}.npy").astype(np.float32)
        for p in pats:
            ii = np.array(p["img_idx"])
            t = np.array(p["visits"], np.float32)
            if keep < 1.0 and len(ii) > 1:
                sel = [0] + [j for j in range(1, len(ii)) if rng.random() < keep]
                ii, t = ii[sel], t[sel]
            r = raw[ii]
            mu0 = prior["gmean"].get(TW.group_key(p), prior["gall"])
            tw, ev = TW.run_patient(r, t, mu0, prior["tau2"], hp, robust, drift)
            if model == "siam":
                x = np.concatenate([r, r - r[:1]], 1)
            elif chan == "raw":
                x = r
            elif chan == "twin":
                x = tw
            else:
                x = np.concatenate([r, tw], 1)
            dt = np.concatenate([[0.0], np.diff(t)]).astype(np.float32)
            seqs.append(dict(pid=p["pid"], x=x, t=t, dt=dt, m=meta_vector(p), time=p["time"], event=p["event"], evid=ev,
                             site=s))
    return seqs


def run(model, seqs, mu, sd, mc=0, bs=64):
    rows = []
    model.eval()
    if mc:
        for mod in model.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.train()
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            ch = seqs[i:i + bs]
            X, T, DT, MASK, Mt, U, D = batchify(ch, mu, sd)
            if mc:
                Fs = np.stack([M.cum_risk(model(X, T, DT, MASK, Mt)).numpy() for _ in range(mc)])
                F, Fsd = Fs.mean(0), np.log(np.clip(Fs[..., 1], 1e-6, 1) / np.clip(1 - Fs[..., 1], 1e-6, 1)).std(0)
                lg = np.zeros(F.shape)
            else:
                lgt = model(X, T, DT, MASK, Mt)
                F, lg = M.cum_risk(lgt).numpy(), lgt.numpy()
            for b, s in enumerate(ch):
                for k in range(len(s["t"])):
                    extra = Fsd[b, k] if mc else s["evid"][k]
                    rows.append([s["pid"], k, s["t"][k], s["time"] - s["t"][k], s["event"], *F[b, k], *lg[b, k], extra])
    return np.array(rows, np.float64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("data"); ap.add_argument("sites"); ap.add_argument("name")
    ap.add_argument("--keep", type=float, default=1.0)
    ap.add_argument("--twin_dir", default="twin_full")
    ap.add_argument("--hp_override", default="")
    ap.add_argument("--mc", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("NT", "1")))
    info = json.load(open(f"{a.run}/info.json"))
    args = argparse.Namespace(**info["args"])
    prior = pickle.load(open(f"{a.twin_dir}/twin_prior.pkl", "rb"))
    hp = json.load(open(f"{a.twin_dir}/twin_hp.json"))
    if a.hp_override:
        for kv in a.hp_override.split(","):
            k, v = kv.split("=")
            hp[k] = [float(v), float(v)] if k != "c" else float(v)
    robust = "norobust" not in args.twin
    drift = "nodrift" not in args.twin
    seqs = load_cohort(a.data, list(a.sites), args.chan, args.model, prior, hp, a.keep, a.seed, robust, drift)
    nz = np.load(f"{a.run}/norm.npz")
    model = build(args, nz["mu"].shape[0])
    model.load_state_dict(torch.load(f"{a.run}/model.pt"))
    torch.manual_seed(a.seed)
    P = run(model, seqs, nz["mu"], nz["sd"], a.mc)
    np.save(f"{a.run}/pred_{a.name}.npy", P)
    print(a.run, a.name, P.shape)
