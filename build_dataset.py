"""Patient-level splits, metadata encoding, twin fitting (training patients only) and
twin inference for every site. Usage: python build_dataset.py prep_dir out_dir [variant]
variant: full | nodrift | norobust  (twin variants for ablation)."""
import numpy as np, pickle, json, sys, os
import twin as TW

SITES = ["A", "B", "C", "D", "E"]


def meta_vector(p):
    v = [ (p["age"] - 52) / 11, p["female"], p["smoke"], p["areca"], p["alcohol"] ]
    v += list(np.eye(5)[p["ltype"]])
    v += list(np.eye(4)[p["grade"]])
    v += list(np.eye(4)[p["lsite"]])
    v += [p["large"]]
    return np.array(v, np.float32)

# token slices for the cross-modal attention (one token per clinical variable)
META_SLICES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 10), (10, 14), (14, 18), (18, 19)]
META_NAMES = ["age", "sex", "smoking", "areca/smokeless", "alcohol", "lesion type", "dysplasia grade", "lesion site", "size>2cm"]


def make_splits(pats_by_site, seed=2026):
    rng = np.random.default_rng(seed)
    split = {}
    for s in ["A", "B", "C"]:
        ids = np.array([p["pid"] for p in pats_by_site[s]])
        ev = np.array([p["event"] for p in pats_by_site[s]])
        for e in (0, 1):
            sel = ids[ev == e]
            rng.shuffle(sel)
            n = len(sel)
            a, b, c = int(0.56 * n), int(0.66 * n), int(0.80 * n)
            for pid in sel[:a]: split[int(pid)] = "train"
            for pid in sel[a:b]: split[int(pid)] = "val"
            for pid in sel[b:c]: split[int(pid)] = "cal"
            for pid in sel[c:]: split[int(pid)] = "test"
    for s in ["D", "E"]:
        for p in pats_by_site[s]:
            split[int(p["pid"])] = "ext" + s
    return split


if __name__ == "__main__":
    prep, out = sys.argv[1], sys.argv[2]
    variant = sys.argv[3] if len(sys.argv) > 3 else "full"
    os.makedirs(out, exist_ok=True)
    pats, raw, hc = {}, {}, {}
    for s in SITES:
        pats[s] = pickle.load(open(f"{prep}/pats_{s}.pkl", "rb"))
        raw[s] = np.load(f"{prep}/raw_{s}.npy").astype(np.float32)
        hc[s] = np.load(f"{prep}/hc_{s}.npy")
    sp_path = os.path.join(os.path.dirname(out.rstrip("/")) or ".", "splits.json")
    if os.path.exists(sp_path):
        split = {int(k): v for k, v in json.load(open(sp_path)).items()}
    else:
        split = make_splits(pats)
        json.dump(split, open(sp_path, "w"))
    # fit twin on training patients (pooled over development sites)
    tr_raw, tr_pats, off = [], [], 0
    for s in ["A", "B", "C"]:
        for p in pats[s]:
            if split[p["pid"]] == "train":
                q = dict(p); q["img_idx"] = [off + i for i in range(len(p["img_idx"]))]
                tr_raw.append(raw[s][p["img_idx"]]); off += len(p["img_idx"]); tr_pats.append(q)
    tr_raw = np.concatenate(tr_raw)
    prior = TW.population_prior(tr_raw, tr_pats)
    hp_path = f"{out}/twin_hp.json"
    if os.path.exists("twin_full/twin_hp.json") and variant != "full":
        hp = json.load(open("twin_full/twin_hp.json"))
    else:
        hp = TW.fit_hyper(tr_raw, tr_pats, prior)
        hp, table = TW.fit_drift_prior(tr_raw, tr_pats, prior, hp)
        json.dump(table, open(f"{out}/drift_prior_table.json", "w"))
    json.dump(hp, open(hp_path, "w"))
    robust = variant != "norobust"
    drift = variant != "nodrift"
    for s in SITES:
        tw, ev = TW.run_all(raw[s], pats[s], prior, hp, robust=robust, drift=drift)
        np.save(f"{out}/twin_{s}.npy", tw)
        np.save(f"{out}/evid_{s}.npy", ev)
        print(s, tw.shape, flush=True)
    pickle.dump(prior, open(f"{out}/twin_prior.pkl", "wb"))
