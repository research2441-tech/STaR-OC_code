# STaR-OC: Patient-Specific Thermal Digital Twin with Conformal Clearance

Code accompanying the manuscript *"Patient-Specific Thermal Digital Twin and Censoring-Weighted Conformal Clearance for Longitudinal Prediction of Oral Malignant Transformation"*.

The repository contains the physics-based longitudinal intraoral thermography simulator (OralThermSim-L), registration and session normalisation, the robust state-space patient twin, the hazard networks and baselines, the conformal clearance rule, all evaluation and statistics, and the figure scripts. Every number in the manuscript is produced by `run_all.sh`; the `reference_*.json` files hold the values reported in the paper for comparison.

## Requirements

Python 3.11 or later and the packages in `requirements.txt` (CPU only; no GPU is needed).

```
pip install -r requirements.txt
sh run_all.sh
```

The full pipeline takes about 6 to 8 hours on two CPU cores and needs about 3 GB of disk.

## Files

| File | Purpose |
|---|---|
| `oralthermsim.py` | Simulator: effective 2-D Pennes balance, lesion-type signatures, angiogenic progression, inflammation, session and camera models, five site profiles |
| `gen_data.py` | Generates sites A-C (development) and D-E (external) |
| `preprocess.py` | Moment-based registration with NCC refinement, session normalisation, bilateral asymmetry, lesion-centred crops, handcrafted features |
| `twin.py` | Pixel-wise robust local-linear-trend twin, type-II likelihood fitting, drift-prior selection |
| `build_dataset.py` | Patient-level splits, clinical encoding, twin fitting on training patients and twin maps for all sites |
| `models.py` | STaR-OC encoder, candidate attention blocks, hazard head, T-LSTM, cross-sectional and Siamese baselines, censored discrete-time loss |
| `train.py` | Training of one model and seed; writes landmark predictions for every split |
| `jobs.txt`, `runq.sh` | The 115 training runs used in the paper (main comparison, ablations, 5-fold model selection) |
| `cox_baselines.py` | Landmark Cox-Clin, Cox-Asym and Cox-Long |
| `gen_robust.py`, `cox_robust.py`, `infer.py`, `infer_batch.py` | Stress-test cohorts and inference |
| `metrics.py`, `analysis.py` | IPCW time-dependent AUC, Uno's C, Brier score and IBS, ECE, cluster bootstrap, Holm correction, conformal threshold |
| `decision.py` | Conformal clearance, fixed-threshold comparator and site-local recalibration |
| `cv_eval.py`, `cohort_stats.py`, `efficiency.py`, `results.py` | Model selection, cohort table, computational cost, aggregate results and statistical tests |
| `ceiling.py`, `noise_budget.py` | Noise-free signal ceiling and per-visit noise budget |
| `fig_arch.py`, `fig_cases.py`, `fig_results.py` | Manuscript figures (written to `figs/`) |
| `splits.json` | Patient-level split used in the paper |

## Notes

- All data are simulated; no patient data are used or required.
- Seeds are fixed in the scripts (cohort seeds in `gen_data.py` and `gen_robust.py`, model seeds 0-4 in `jobs.txt`), so a rerun reproduces the reported results up to floating-point differences between PyTorch builds.
- Output folders (`data/`, `prep*/`, `twin_*/`, `runs/`, `runs_cv/`, `robust/`, `results/`, `figs/`, `logs/`) are created at run time.

## License

MIT
