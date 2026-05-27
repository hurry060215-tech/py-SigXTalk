"""
Notebook 2: Python Tutorial
Pre-executed notebook showing py-sigxtalk usage.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pysigxtalk as psx
import anndata

print("=" * 60)
print("Notebook 2: py-sigxtalk Tutorial")
print("=" * 60)

# %% [markdown]
# # py-sigxtalk Tutorial
#
# This notebook demonstrates the complete SigXTalk analysis pipeline in Python.

# %% Load data
print("\n[1/8] Loading PBMC3k data...")
adata = anndata.read_h5ad('pbmc3k_final.h5ad')
print(f"Shape: {adata.shape}")
print(f"Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")

# %% Load databases
print("\n[2/8] Loading databases...")
rtf_db, tftg_db = psx.load_databases(species='human')
print(f"RTF: {len(rtf_db):,} entries")
print(f"TFTG: {len(tftg_db):,} entries")

# %% Prepare target cell type
print("\n[3/8] Preparing target cell type...")
target_type = 'CD14+ Mono'
target_mask = adata.obs['cell_type'] == target_type
adata_target = adata[target_mask].copy()

X = adata_target.X
if hasattr(X, 'toarray'):
    X = X.toarray()
exp_mat = pd.DataFrame(X.T, index=adata_target.var.index.tolist(),
                       columns=adata_target.obs.index.tolist())
exp_mat = psx.get_exp_clu(exp_mat, cutoff=0.1)
print(f"Expression: {exp_mat.shape}")

# %% Prepare inputs
print("\n[4/8] Preparing HGNN inputs...")
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
print(f"RTF filtered: {len(inputs.rtf_filtered)}")
print(f"TFTG filtered: {len(inputs.tftg_filtered)}")

# %% Run HGNN
print("\n[5/8] Running HGNN...")
pathways = psx.run_hgnn(inputs, epochs=10, device='cpu', seed=42,
                        hgnn_dims=[16, 8], linear_dims=[4, 2])
print(f"Pathways: {len(pathways)}")
print(f"Active (>0.75): {len(pathways[pathways['pred_label'] > 0.75])}")

# %% Calculate PRS
print("\n[6/8] Calculating PRS...")
prs = psx.compute_prs(inputs.exp_clu, pathways, n_estimators=10, cutoff=0.75)
prs_filtered = psx.filter_results(prs, prs_threshold=0.01)
print(f"PRS results: {len(prs_filtered)} pathways")

# %% Crosstalk analysis
print("\n[7/8] Crosstalk analysis...")
counts = psx.count_crosstalk(prs_filtered, data_type='Target')
trs = psx.aggregate_causality(prs_filtered, data_type='Target')
top_target = counts.idxmax()
print(f"Top target: {top_target} ({counts.max()} pathways)")

# %% Visualizations
print("\n[8/8] Generating visualizations...")

# 01 - Crosstalk histogram
fig = psx.plot_counts_histogram(prs_filtered, data_type='Target')
fig.savefig('figures/01_crosstalk_histogram.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: 01_crosstalk_histogram.png")

# 02 - Crosstalk bar
fig = psx.plot_counts_bar(prs_filtered, data_type='Target', topk=20)
fig.savefig('figures/02_crosstalk_bar.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: 02_crosstalk_bar.png")

# 03 - Fid/Spe
fig = psx.plot_fid_spe(prs_filtered, key_tg=top_target, threshold=0.0)
fig.savefig('figures/03_fid_spe.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: 03_fid_spe.png")

# 04 - Alluvial
fig = psx.plot_alluvial(prs_filtered, key_tg=top_target, min_weight=0.0)
fig.savefig('figures/04_alluvial.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: 04_alluvial.png")

# 05 - Ridgeline
fig = psx.plot_ridgeline(prs_filtered, key_tg=top_target)
fig.savefig('figures/05_ridgeline.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: 05_ridgeline.png")

# 06 - Chord
fid_matrix = psx.calculate_fidelity_matrix(prs_filtered, key_tg=top_target, mode='SSC')
if not fid_matrix.empty:
    fig = psx.plot_chord(fid_matrix)
    fig.savefig('figures/06_chord.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: 06_chord.png")

# 07 - CCI chord
np.random.seed(42)
cci_results = pd.DataFrame({
    'Ligand': np.random.choice(receptors[:10], 30),
    'Receptor': np.random.choice(receptors[:15], 30),
    'Weight': np.random.rand(30),
    'Source': np.random.choice(['T', 'Mono', 'B', 'NK'], 30),
    'Target': [target_type]*30,
})
fig = psx.plot_cci_chord(cci_results, topk=10)
fig.savefig('figures/07_cci_chord.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: 07_cci_chord.png")

# 08 - CCI circle
fig = psx.plot_cci_circle(cci_results, topk=10)
fig.savefig('figures/08_cci_circle.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: 08_cci_circle.png")

# 09 - Signal contribution
fig = psx.plot_signal_contribution(trs, inputs.exp_clu, key_tg=top_target)
fig.savefig('figures/09_signal_contribution.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: 09_signal_contribution.png")

# 10 - Rec-TG heatmap
focus_genes = counts.sort_values(ascending=False).head(15).index.tolist()
fig = psx.plot_rec_tg_heatmap(prs_filtered, inputs.exp_clu, key_tg=focus_genes, topk=200)
fig.savefig('figures/10_rec_tg_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: 10_rec_tg_heatmap.png")

# 11 - Circular bar
fig = psx.plot_circular_bar(prs_filtered, topk=5)
fig.savefig('figures/11_circular_bar.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: 11_circular_bar.png")

# %% Summary
print("\n" + "=" * 60)
print("Tutorial Complete!")
print("=" * 60)
print(f"Generated 11 visualization figures in figures/")
print(f"PRS results: {len(prs_filtered)} pathways")
print(f"Top target: {top_target}")
print("=" * 60)
