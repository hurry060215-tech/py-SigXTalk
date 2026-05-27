"""PRS Speed Optimization Benchmark"""
import pandas as pd, numpy as np, time
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pysigxtalk as psx
import anndata

print("=== PRS Speed Optimization Benchmark ===\n")

# Load data
adata = anndata.read_h5ad('pbmc3k_final.h5ad')
target_mask = adata.obs['cell_type'] == 'CD14+ Mono'
adata_target = adata[target_mask].copy()
X_data = adata_target.layers['scale_data'] if 'scale_data' in adata_target.layers else adata_target.X
exp_mat = pd.DataFrame(
    X_data.T.toarray() if hasattr(X_data, 'toarray') else X_data.T,
    index=adata_target.var.index.tolist(),
    columns=adata_target.obs.index.tolist()
)
exp_mat = psx.get_exp_clu(exp_mat, cutoff=0.1)

pathways = pd.read_csv('pathways.csv')
r_gini = pd.read_csv('r_prs_gini_sample.csv')
for col in ['Receptor', 'SSC', 'Target']:
    r_gini[col] = r_gini[col].str.strip()

active = pathways[pathways['pred_label'] >= 0.5]

# Use SAME TF-TG pairs as R (from r_prs_gini_sample.csv)
r_pairs = r_gini[['SSC', 'Target']].drop_duplicates()
tasks = []
for _, row in r_pairs.iterrows():
    tf, tg = row['SSC'], row['Target']
    if tf not in exp_mat.index or tg not in exp_mat.index:
        continue
    group = active[(active['TF'] == tf) & (active['TG'] == tg)]
    recs = [r for r in group['Receptor'].unique().tolist() if r in exp_mat.index]
    if len(recs) >= 2:
        tasks.append((tf, tg, recs))

def run_single(exp_clu, tf, tg, recs, n_est, mtry, seed, n_jobs_rf):
    y = (exp_clu.loc[tf].values * exp_clu.loc[tg].values).astype(np.float64)
    X_feat = exp_clu.loc[recs].values.T.astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    model = RandomForestRegressor(n_estimators=n_est, max_features=mtry, random_state=seed, n_jobs=n_jobs_rf)
    model.fit(X_scaled, y)
    results = []
    for i, rec in enumerate(recs):
        results.append({'Receptor': rec, 'SSC': tf, 'Target': tg, 'Weight': float(model.feature_importances_[i])})
    return results

def evaluate(all_res, t_elapsed, name):
    py_df = pd.DataFrame(all_res)
    for col in ['Receptor', 'SSC', 'Target']:
        py_df[col] = py_df[col].str.strip()
    merged = py_df.merge(r_gini, on=['Receptor', 'SSC', 'Target'])
    if len(merged) > 10:
        for (tf, tg), grp in merged.groupby(['SSC', 'Target']):
            py_sum = grp['Weight_x'].sum()
            r_sum = grp['Weight_y'].sum()
            if py_sum > 0 and r_sum > 0:
                merged.loc[grp.index, 'PY_norm'] = grp['Weight_x'] / py_sum
                merged.loc[grp.index, 'R_norm'] = grp['Weight_y'] / r_sum
        corr_p, _ = pearsonr(merged['PY_norm'], merged['R_norm'])
        corr_s, _ = spearmanr(merged['PY_norm'], merged['R_norm'])
    else:
        corr_p, corr_s = 0, 0
    mark = "PASS" if corr_p >= 0.95 else "FAIL"
    print(f"  [{mark}] {name:50s}: {t_elapsed:6.2f}s | Pearson={corr_p:.4f} Spearman={corr_s:.4f}")
    return {'name': name, 'time': t_elapsed, 'pearson': corr_p, 'spearman': corr_s}

# Config 1: Baseline (current)
t0 = time.time()
all_res = []
for tf, tg, recs in tasks:
    all_res.extend(run_single(exp_mat, tf, tg, recs, 500, min(3, len(recs)), 42, 1))
evaluate(all_res, time.time() - t0, "Baseline (n_est=500, rf_njobs=1, outer=1)")

# Config 2: Parallel RF cores
t0 = time.time()
all_res = []
for tf, tg, recs in tasks:
    all_res.extend(run_single(exp_mat, tf, tg, recs, 500, min(3, len(recs)), 42, -1))
evaluate(all_res, time.time() - t0, "Parallel RF (n_est=500, rf_njobs=-1, outer=1)")

# Config 3: Parallel outer
t0 = time.time()
all_res_nested = Parallel(n_jobs=-1, backend="threading")(
    delayed(run_single)(exp_mat, tf, tg, recs, 500, min(3, len(recs)), 42, 1)
    for tf, tg, recs in tasks
)
all_res = [item for sublist in all_res_nested for item in sublist]
evaluate(all_res, time.time() - t0, "Parallel outer (n_est=500, rf_njobs=1, outer=-1)")

# Config 4: Both parallel
t0 = time.time()
all_res_nested = Parallel(n_jobs=2, backend="threading")(
    delayed(run_single)(exp_mat, tf, tg, recs, 500, min(3, len(recs)), 42, 4)
    for tf, tg, recs in tasks
)
all_res = [item for sublist in all_res_nested for item in sublist]
evaluate(all_res, time.time() - t0, "Both parallel (n_est=500, rf_njobs=4, outer=2)")

# Config 5: Fewer trees + parallel
t0 = time.time()
all_res_nested = Parallel(n_jobs=-1, backend="threading")(
    delayed(run_single)(exp_mat, tf, tg, recs, 100, min(3, len(recs)), 42, 1)
    for tf, tg, recs in tasks
)
all_res = [item for sublist in all_res_nested for item in sublist]
evaluate(all_res, time.time() - t0, "Fewer trees (n_est=100, rf_njobs=1, outer=-1)")

# Config 6: Fewer trees + both parallel
t0 = time.time()
all_res_nested = Parallel(n_jobs=2, backend="threading")(
    delayed(run_single)(exp_mat, tf, tg, recs, 100, min(3, len(recs)), 42, 4)
    for tf, tg, recs in tasks
)
all_res = [item for sublist in all_res_nested for item in sublist]
evaluate(all_res, time.time() - t0, "Fewer trees + parallel (n_est=100, rf_njobs=4, outer=2)")

# Config 7: LightGBM
try:
    import lightgbm as lgb
    t0 = time.time()
    all_res = []
    for tf, tg, recs in tasks:
        y = (exp_mat.loc[tf].values * exp_mat.loc[tg].values).astype(np.float64)
        X_feat = exp_mat.loc[recs].values.T.astype(np.float64)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_feat)
        model = lgb.LGBMRegressor(n_estimators=500, max_features=min(3, len(recs)), random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X_scaled, y)
        for i, rec in enumerate(recs):
            all_res.append({'Receptor': rec, 'SSC': tf, 'Target': tg, 'Weight': float(model.feature_importances_[i])})
    evaluate(all_res, time.time() - t0, "LightGBM (n_est=500, njobs=-1)")

    # LightGBM fewer trees
    t0 = time.time()
    all_res = []
    for tf, tg, recs in tasks:
        y = (exp_mat.loc[tf].values * exp_mat.loc[tg].values).astype(np.float64)
        X_feat = exp_mat.loc[recs].values.T.astype(np.float64)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_feat)
        model = lgb.LGBMRegressor(n_estimators=100, max_features=min(3, len(recs)), random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X_scaled, y)
        for i, rec in enumerate(recs):
            all_res.append({'Receptor': rec, 'SSC': tf, 'Target': tg, 'Weight': float(model.feature_importances_[i])})
    evaluate(all_res, time.time() - t0, "LightGBM (n_est=100, njobs=-1)")
except ImportError:
    print("  LightGBM not installed, skipping")

print("\nR ranger baseline: 5.73s (50 pairs)")
