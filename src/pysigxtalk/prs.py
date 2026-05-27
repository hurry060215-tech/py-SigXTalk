import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor


def _run_single_regression(exp_clu, tf, tg, receptors, n_estimators, max_features, seed):
    """Run a single RF regression using sklearn with raw importance."""
    y = (exp_clu.loc[tf].values * exp_clu.loc[tg].values).astype(np.float64)
    rec_expr = exp_clu.loc[receptors].values.T.astype(np.float64)
    scaler = StandardScaler()
    X = scaler.fit_transform(rec_expr)

    if np.sum(y > 1e-5) < 10:
        return []

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=min(max_features, len(receptors)),
        random_state=seed,
        n_jobs=1,
    )
    model.fit(X, y)

    # Raw importance (not normalized) to match R's ranger
    importances = np.zeros(len(receptors))
    for tree in model.estimators_:
        importances += tree.tree_.compute_feature_importances(normalize=False)
    importances /= len(model.estimators_)

    results = []
    for i, rec in enumerate(receptors):
        if importances[i] > 1e-5:
            results.append({"Receptor": rec, "SSC": tf, "Target": tg, "Weight": float(importances[i])})
    return results


def compute_prs(exp_clu, pathways, engine="sklearn", n_estimators=50,
                max_features=3, cutoff=0.15, n_jobs=-1, seed=42):
    """Calculate Pathway Regulatory Strength using Random Forest."""
    active = pathways[pathways["pred_label"] >= cutoff].copy()
    tf_tg_groups = active.groupby(["TF", "TG"])
    tasks = []
    for (tf, tg), group in tf_tg_groups:
        receptors = group["Receptor"].unique().tolist()
        receptors = [r for r in receptors if r in exp_clu.index]
        if tf in exp_clu.index and tg in exp_clu.index and len(receptors) >= 1:
            tasks.append((tf, tg, receptors))

    all_results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_run_single_regression)(exp_clu, tf, tg, receptors, n_estimators, max_features, seed)
        for tf, tg, receptors in tasks
    )
    results = [item for sublist in all_results for item in sublist]
    return pd.DataFrame(results)


def filter_results(results, prs_threshold=0.05, remove_genes=True):
    """Filter PRS results by threshold and remove MT/RPL/RPS genes."""
    result = results.copy()
    if remove_genes:
        mask_mt = ~result["Receptor"].str.startswith("MT-")
        mask_mt &= ~result["SSC"].str.startswith("MT-")
        mask_mt &= ~result["Target"].str.startswith("MT-")
        mask_ribo = ~result["Receptor"].str.startswith("RPL")
        mask_ribo &= ~result["Receptor"].str.startswith("RPS")
        mask_ribo &= ~result["SSC"].str.startswith("RPL")
        mask_ribo &= ~result["SSC"].str.startswith("RPS")
        mask_ribo &= ~result["Target"].str.startswith("RPL")
        mask_ribo &= ~result["Target"].str.startswith("RPS")
        result = result[mask_mt & mask_ribo]

    max_weight = result["Weight"].max()
    if max_weight > 0:
        result = result[result["Weight"] >= prs_threshold * max_weight]

    return result.reset_index(drop=True)
