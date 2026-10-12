"""Stress-test cohorts: 600 new development-site patients (200 per site A-C) simulated with one
factor of the acquisition or population model changed. Common random numbers across conditions."""
import numpy as np, pickle, sys, os
import oralthermsim as S
import preprocess as PP

COND = {
    "base": {}, "const0.5": dict(const=0.5), "const2": dict(const=2.0),
    "inflam0": dict(inflam=0.0), "inflam2": dict(inflam=2.0), "sess2": dict(sess=2.0),
    "pose2": dict(pose=2.0), "noise2": dict(noise=2.0),
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(COND)
    for name in names:
        out = f"robust/{name}"
        os.makedirs(out, exist_ok=True)
        for k in S.FACT:
            S.FACT[k] = 1.0
        S.FACT.update(COND[name])
        pid0 = 100000
        for site, seed in [("A", 701), ("B", 702), ("C", 703)]:
            pats, imgs = S.simulate_cohort(200, site, seed, pid0)
            pid0 += 200
            np.save(f"{out}/imgs_{site}.npy", imgs)
            pickle.dump([p.__dict__ for p in pats], open(f"{out}/pats_{site}.pkl", "wb"))
            pats2, raw, hc = PP.process_site(site, out, "full", np.random.default_rng(ord(site) + 50))
            np.save(f"{out}/raw_{site}.npy", raw.astype(np.float16))
            np.save(f"{out}/hc_{site}.npy", hc)
            pickle.dump(pats2, open(f"{out}/pats_{site}.pkl", "wb"))
            os.remove(f"{out}/imgs_{site}.npy")
        print(name, "done", flush=True)
