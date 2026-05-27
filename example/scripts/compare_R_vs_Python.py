"""
Notebook 1: Pipeline Parity - R vs Python Comparison
Pre-executed notebook showing numerical equivalence.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import pysigxtalk as psx
import anndata
import time

print("=" * 60)
print("Notebook 1: Pipeline Parity - R vs Python")
print("=" * 60)

# %% [markdown]
# # py-sigxtalk: R vs Python Pipeline Comparison
#
# This notebook demonstrates numerical equivalence between the Python `py-sigxtalk`
# implementation and the R `SigXTalkR` reference.

# %% Load data
print("\n[1/6] Loading data...")
adata = anndata.read_h5ad('pbmc3k_final.h5ad')
rtf_db, tftg_db = psx.load_databases(species='human')

target_type = 'CD14+ Mono'
target_mask = adata.obs['cell_type'] == target_type
adata_target = adata[target_mask].copy()

X = adata_target.X
if hasattr(X, 'toarray'):
    X = X.toarray()
exp_mat = pd.DataFrame(X.T, index=adata_target.var.index.tolist(),
                       columns=adata_target.obs.index.tolist())
exp_mat = psx.get_exp_clu(exp_mat, cutoff=0.1)
print(f"Expression matrix: {exp_mat.shape}")

# %% Prepare inputs
print("\n[2/6] Preparing inputs...")
target_genes = exp_mat.var(axis=1).nlargest(100).index.tolist()
all_genes = exp_mat.index.tolist()
receptors = [g for g in rtf_db['from'].unique() if g in all_genes][:15]

np.random.seed(42)
lr_pairs = pd.DataFrame({
    'Ligand': np.random.choice(receptors, 20, replace=True),
    'Receptor': np.random.choice(receptors, 20, replace=True),
    'Weight': np.random.rand(20) * 0.8 + 0.2,
})

inputs = psx.prepare_input(exp_mat=exp_mat, target_genes=target_genes,
                           lr_pairs=lr_pairs, rtf_db=rtf_db, tftg_db=tftg_db)

# %% Run HGNN
print("\n[3/6] Running HGNN...")
t0 = time.time()
pathways = psx.run_hgnn(inputs, epochs=10, device='cpu', seed=42,
                        hgnn_dims=[16, 8], linear_dims=[4, 2])
t_hgnn = time.time() - t0
print(f"HGNN: {t_hgnn:.2f}s, {len(pathways)} pathways")

# %% Run PRS
print("\n[4/6] Running PRS...")
t0 = time.time()
prs_py = psx.compute_prs(inputs.exp_clu, pathways, n_estimators=10, cutoff=0.75)
prs_py_filtered = psx.filter_results(prs_py, prs_threshold=0.01)
t_prs = time.time() - t0
print(f"PRS: {t_prs:.2f}s, {len(prs_py_filtered)} pathways")

# %% Load R results
print("\n[5/6] Loading R reference results...")
try:
    prs_r = pd.read_csv('prs_results_r.csv')
    print(f"R PRS: {len(prs_r)} pathways")

    # Compare
    merged = prs_py_filtered.merge(prs_r, on=['Receptor', 'SSC', 'Target'],
                                    suffixes=('_py', '_r'))

    corr_pearson, pval_p = pearsonr(merged['Weight_py'], merged['Weight_r'])
    corr_spearman, pval_s = spearmanr(merged['Weight_py'], merged['Weight_r'])

    print(f"\n{'='*60}")
    print("Parity Results:")
    print(f"  Matching pathways: {len(merged)}")
    print(f"  Pearson correlation: {corr_pearson:.4f} (threshold: 0.95)")
    print(f"  Spearman correlation: {corr_spearman:.4f} (threshold: 0.90)")
    print(f"  Status: {'PASS' if corr_pearson >= 0.95 else 'FAIL'}")
    print(f"{'='*60}")

    # %% Visualization
    print("\n[6/6] Generating comparison plots...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Scatter plot
    ax = axes[0]
    ax.scatter(merged['Weight_py'], merged['Weight_r'], alpha=0.3, s=5, c='steelblue')
    max_val = max(merged['Weight_py'].max(), merged['Weight_r'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='y=x')
    ax.set_xlabel('Python PRS')
    ax.set_ylabel('R PRS')
    ax.set_title(f'Python vs R (r={corr_pearson:.4f})')
    ax.legend()

    # Distribution comparison
    ax = axes[1]
    ax.hist(prs_py_filtered['Weight'], bins=50, alpha=0.5, label='Python', color='steelblue')
    ax.hist(prs_r['Weight'], bins=50, alpha=0.5, label='R', color='coral')
    ax.set_xlabel('PRS Weight')
    ax.set_ylabel('Frequency')
    ax.set_title('PRS Distribution')
    ax.legend()

    # Difference distribution
    diff = merged['Weight_py'] - merged['Weight_r']
    ax = axes[2]
    ax.hist(diff, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--')
    ax.set_xlabel('Python - R Difference')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Difference (mean={diff.mean():.4f})')

    plt.tight_layout()
    plt.savefig('compare_R_vs_Python.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: compare_R_vs_Python.png")

except FileNotFoundError:
    print("R results not found. Run R benchmark first.")
    print("See generate_heatmap.R for R reference script.")

# %% Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Python total time: {t_hgnn + t_prs:.2f}s")
print(f"Parity: {'PASS' if 'corr_pearson' in dir() and corr_pearson >= 0.95 else 'N/A'}")
print("=" * 60)
