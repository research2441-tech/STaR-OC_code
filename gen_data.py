"""Generate the OralThermSim-L cohorts (development sites A-C, external sites D-E)."""
import numpy as np, pickle, sys, os
import oralthermsim as S
OUT = sys.argv[1] if len(sys.argv) > 1 else "data"
os.makedirs(OUT, exist_ok=True)
plan = [("A", 900, 11), ("B", 800, 22), ("C", 800, 33), ("D", 500, 44), ("E", 500, 55)]
pid0 = 0
for site, n, seed in plan:
    pats, imgs = S.simulate_cohort(n, site, seed, pid0)
    pid0 += n
    np.save(f"{OUT}/imgs_{site}.npy", imgs)
    with open(f"{OUT}/pats_{site}.pkl", "wb") as f:
        pickle.dump([p.__dict__ for p in pats], f)
    ev = np.mean([p.event for p in pats])
    print(site, len(pats), imgs.shape, "event", round(ev, 3), flush=True)
