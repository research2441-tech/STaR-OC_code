#!/bin/sh
# End-to-end reproduction of every number, table and figure in the manuscript.
# Expected run time on two CPU cores: about 6-8 hours (dominated by 115 training runs).
set -e
cd "$(dirname "$0")"
mkdir -p logs results figs

# 1. Simulate the five-site cohort (OralThermSim-L)
python3 gen_data.py data

# 2. Registration / normalisation variants
python3 preprocess.py data prep full
python3 preprocess.py data prep_none none
python3 preprocess.py data prep_moment moment
python3 preprocess.py data prep_nonorm nonorm

# 3. Patient-level splits (splits.json is shipped) and twins fitted on training patients only
python3 build_dataset.py prep twin_full full
python3 build_dataset.py prep twin_nodrift nodrift
python3 build_dataset.py prep twin_norobust norobust
python3 build_dataset.py prep_none twin_none full
python3 build_dataset.py prep_moment twin_moment full
python3 build_dataset.py prep_nonorm twin_nonorm full
python3 cohort_stats.py

# 4. Deep models: main comparison, ablations and 5-fold model selection (5 seeds each)
sh runq.sh jobs.txt
python3 cv_eval.py

# 5. Landmark Cox baselines
python3 cox_baselines.py prep runs

# 6. Stress-test cohorts and their inference
python3 gen_robust.py
python3 cox_robust.py base const0.5 const2 inflam0 inflam2 sess2 pose2 noise2 keep0.5
python3 infer_batch.py tlstm cs siam tlstm_twin final ab_notwin

# 7. Decision layer, cost, aggregate statistics, signal bounds
python3 decision.py final tlstm_twin tlstm siam cox_long cs cox_asym
python3 efficiency.py
BOOT=1000 python3 results.py
python3 ceiling.py
python3 noise_budget.py

# 8. Figures
python3 fig_arch.py
python3 fig_cases.py
python3 fig_results.py
