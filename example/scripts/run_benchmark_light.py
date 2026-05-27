"""
Lightweight benchmark for py-sigxtalk vs R SigXTalkR.
Uses smaller model and fewer pathways to avoid memory issues.
"""
import time
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import sys
sys.path.insert(0, '../src')
import pysigxtalk as psx
import anndata

def main():
    print("=" * 60)
    print("py-sigxtalk Lightweight Benchmark")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading data...")
    adata = anndata.read_h5ad('pbmc3k_final.h5ad')
    rtf_db, tftg_db = psx.load_databases(species='human')
    print(f"  PBMC3k: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # Step 2: Prepare data (smaller subset)
    print("\n[2/5] Preparing data (reduced)...")
    target_type = 'CD14+ Mono'
    target_mask = adata.obs['cell_type'] == target_type
    adata_target = adata[target_mask].copy()

    X = adata_target.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    exp_mat = pd.DataFrame(X.T, index=adata_target.var.index.tolist(),
                           columns=adata_target.obs.index.tolist())
    exp_mat = psx.get_exp_clu(exp_mat, cutoff=0.1)

    # Use fewer target genes to reduce memory
    target_genes = exp_mat.var(axis=1).nlargest(100).index.tolist()
    all_genes = exp_mat.index.tolist()
    receptors_in_data = [g for g in rtf_db['from'].unique() if g in all_genes][:15]

    np.random.seed(42)
    lr_pairs = pd.DataFrame({
        'Ligand': np.random.choice(receptors_in_data, 20, replace=True),
        'Receptor': np.random.choice(receptors_in_data, 20, replace=True),
        'Weight': np.random.rand(20) * 0.8 + 0.2,
    })

    t0 = time.time()
    inputs = psx.prepare_input(exp_mat=exp_mat, target_genes=target_genes,
                               lr_pairs=lr_pairs, rtf_db=rtf_db, tftg_db=tftg_db)
    t_preprocess = time.time() - t0
    print(f"  Preprocessing: {t_preprocess:.2f}s")
    print(f"  RTF: {len(inputs.rtf_filtered)}, TFTG: {len(inputs.tftg_filtered)}")

    # Step 3: HGNN (smaller model)
    print("\n[3/5] Running HGNN (reduced)...")
    t0 = time.time()
    pathways = psx.run_hgnn(inputs, epochs=15, device='cpu', seed=42,
                            hgnn_dims=[32, 16], linear_dims=[8, 4])
    t_hgnn = time.time() - t0
    print(f"  HGNN: {t_hgnn:.2f}s")
    print(f"  Pathways: {len(pathways)}")

    # Save pathways for R comparison
    pathways.to_csv('pathways_for_r.csv', index=False)

    # Step 4: Python PRS
    print("\n[4/5] Running Python PRS...")
    t0 = time.time()
    prs_py = psx.compute_prs(inputs.exp_clu, pathways, n_estimators=30,
                             cutoff=0.5, n_jobs=2)
    prs_py_filtered = psx.filter_results(prs_py, prs_threshold=0.01)
    t_prs_py = time.time() - t0
    print(f"  Python PRS: {t_prs_py:.2f}s, {len(prs_py_filtered)} pathways")

    prs_py_filtered.to_csv('prs_results_python.csv', index=False)
    inputs.exp_clu.to_csv('exp_clu_for_r.csv')

    # Free memory
    del inputs, pathways
    gc.collect()

    # Step 5: R PRS comparison
    print("\n[5/5] Running R PRS...")
    import subprocess

    r_script = f"""
    .libPaths(unique(c("C:/Users/17904/Documents/R/win-library/4.5", .libPaths())))
    library(SigXTalkR)

    exp_clu <- read.csv("exp_clu_for_r.csv", row.names = 1)
    pathways <- read.csv("pathways_for_r.csv")
    active <- pathways[pathways$pred_label > 0.5, c("Receptor", "TF", "TG")]

    t0 <- proc.time()
    prs_r <- PRS_calc(exp_clu, active, cutoff = 0.1)
    t_prs_r <- (proc.time() - t0)[3]

    prs_r_filtered <- Filter_results(prs_r, PRS_thres = 0.01)
    write.csv(prs_r_filtered, "prs_results_r.csv", row.names = FALSE)
    cat(paste0(t_prs_r, ",", nrow(prs_r_filtered)))
    """

    try:
        result = subprocess.run(
            ['C:/Program Files/R/R-4.5.0/bin/Rscript.exe', '--vanilla', '-e', r_script],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            t_prs_r = float(parts[-1].split()[0])  # Get time from last output
            print(f"  R PRS completed")
        else:
            print(f"  R PRS failed: {result.stderr[-200:]}")
            t_prs_r = None
    except Exception as e:
        print(f"  R PRS error: {e}")
        t_prs_r = None

    # Step 6: Compare results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    try:
        prs_r = pd.read_csv('prs_results_r.csv')
        print(f"Python PRS: {len(prs_py_filtered)} pathways, {t_prs_py:.2f}s")
        print(f"R PRS: {len(prs_r)} pathways")

        merged = prs_py_filtered.merge(prs_r, on=['Receptor', 'SSC', 'Target'], suffixes=('_py', '_r'))
        if len(merged) > 0:
            corr_p, pval_p = pearsonr(merged['Weight_py'], merged['Weight_r'])
            corr_s, pval_s = spearmanr(merged['Weight_py'], merged['Weight_r'])
            diff = merged['Weight_py'] - merged['Weight_r']

            print(f"\nCorrelation:")
            print(f"  Matching: {len(merged)} pathways")
            print(f"  Pearson: {corr_p:.4f} (p={pval_p:.2e})")
            print(f"  Spearman: {corr_s:.4f} (p={pval_s:.2e})")
            print(f"  Mean diff: {diff.mean():.6f}")

            # Visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            ax = axes[0]
            ax.scatter(merged['Weight_py'], merged['Weight_r'], alpha=0.3, s=5)
            ax.plot([0, merged['Weight_py'].max()], [0, merged['Weight_py'].max()], 'r--')
            ax.set_xlabel('Python PRS')
            ax.set_ylabel('R PRS')
            ax.set_title(f'Python vs R (r={corr_p:.4f})')

            ax = axes[1]
            ax.hist(prs_py_filtered['Weight'], bins=30, alpha=0.5, label='Python', color='steelblue')
            ax.hist(prs_r['Weight'], bins=30, alpha=0.5, label='R', color='coral')
            ax.set_xlabel('PRS Weight')
            ax.set_ylabel('Frequency')
            ax.set_title('PRS Distribution')
            ax.legend()

            ax = axes[2]
            ax.hist(diff, bins=30, color='steelblue', edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--')
            ax.set_xlabel('Python - R Difference')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Difference (mean={diff.mean():.4f})')

            plt.tight_layout()
            plt.savefig('benchmark_correlation.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\nSaved benchmark_correlation.png")
        else:
            print("No matching pathways")
    except FileNotFoundError:
        print("R results not available")

    print("\nDone!")

if __name__ == '__main__':
    main()
