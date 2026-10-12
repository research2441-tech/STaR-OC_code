"""Stress-test inference for all seeds of the given run prefixes."""
import sys, json, pickle, argparse, os
import numpy as np, torch
import infer as IF
from train import build

CONDS = ["base", "const0.5", "const2", "inflam0", "inflam2", "sess2", "pose2", "noise2", "keep0.5"]
torch.set_num_threads(1)
prefixes = sys.argv[1:]
prior = pickle.load(open("twin_full/twin_prior.pkl", "rb"))
hp = json.load(open("twin_full/twin_hp.json"))
for cond in CONDS:
    ddir, keep = (f"robust/{cond}", 1.0) if not cond.startswith("keep") else ("robust/base", float(cond[4:]))
    cache = {}
    for pre in prefixes:
        for s in range(5):
            run = f"runs/{pre}_s{s}"
            out = f"{run}/pred_rob_{cond}.npy"
            if os.path.exists(out):
                continue
            info = json.load(open(f"{run}/info.json"))
            a = argparse.Namespace(**info["args"])
            key = (a.chan, a.model)
            if key not in cache:
                cache[key] = IF.load_cohort(ddir, "ABC", a.chan, a.model, prior, hp, keep, 0)
            nz = np.load(f"{run}/norm.npz")
            model = build(a, nz["mu"].shape[0])
            model.load_state_dict(torch.load(f"{run}/model.pt"))
            np.save(out, IF.run(model, cache[key], nz["mu"], nz["sd"]))
        print(cond, pre, flush=True)
