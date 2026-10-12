"""Parameter counts, training time per member, and per-visit inference latency (1 CPU thread)."""
import json, time, pickle, argparse, numpy as np, torch
import twin as TW
from train import build
torch.set_num_threads(1)
out = {}
prior = pickle.load(open("twin_full/twin_prior.pkl", "rb")); hp = json.load(open("twin_full/twin_hp.json"))
raw = np.load("prep/raw_A.npy").astype(np.float32)[:10]
# twin update cost per visit
t0 = time.perf_counter()
for _ in range(200):
    TW.run_patient(raw[:1], np.array([0.0]), prior["gall"], prior["tau2"], hp)
out["twin_ms_per_visit"] = (time.perf_counter() - t0) / 200 * 1000
for pre in ["cs", "siam", "tlstm", "tlstm_twin", "final", "staroc"]:
    infos = [json.load(open(f"runs/{pre}_s{s}/info.json")) for s in range(5)]
    a = argparse.Namespace(**infos[0]["args"])
    nz = np.load(f"runs/{pre}_s0/norm.npz")
    C = nz["mu"].shape[0]
    model = build(a, C); model.eval()
    K = 6
    x = torch.randn(1, K, C, 32, 32); t = torch.arange(K).float().view(1, K) * 5; m = torch.randn(1, 19)
    mask = torch.ones(1, K, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(5): model(x, t, t, mask, m)
        t0 = time.perf_counter()
        for _ in range(100): model(x, t, t, mask, m)
    ms = (time.perf_counter() - t0) / 100 * 1000
    out[pre] = dict(params=infos[0]["nparam"], train_s=float(np.mean([i["train_time"] for i in infos])),
                    train_sd=float(np.std([i["train_time"] for i in infos])), epochs=float(np.mean([i["epochs"] for i in infos])),
                    ms_seq6=ms)
    print(pre, out[pre], flush=True)
json.dump(out, open("results/efficiency.json", "w"), indent=1)
print(out["twin_ms_per_visit"])
