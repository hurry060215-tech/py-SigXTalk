"""Benchmark: Python pysigxtalk vs R SigXTalkR on PBMC3k data."""

import time
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pysigxtalk as psx
import anndata

RSCRIPT = "C:/Program Files/R/R-4.5.2/bin/Rscript.exe"

def run_python_pipeline(adata, rtf_db, tftg_db, target_type, n_estimators=100):
    """Run full Python pipeline and return timings + results."""
    timings = {}

    # Extract target cell type
    target_mask = adata.obs['cell_type'] == target_type
    adata_target = adata[target_mask].copy()

    # Expression matrix
    X = adata_target.layers['scale_data'] if 'scale_data' in adata_target.layers else adata_target.X
    exp_mat = pd.DataFrame(
        X.T.toarray() if hasattr(X, 'toarray') else X.T,
        index=adata_target.var.index.tolist(),
        columns=adata_target.obs.index.tolist()
    )
    t0 = time.time()
    exp_mat = psx.get_exp_clu(exp_mat, cutoff=0.1)
    timings['filter_genes'] = time.time() - t0

    # Target genes (top variable)
    target_genes = exp_mat.var(axis=1).nlargest(200).index.tolist()

    # LR pairs (synthetic)
    all_genes = exp_mat.index.tolist()
    receptors = [g for g in rtf_db['from'].unique() if g in all_genes][:20]
    np.random.seed(42)
    lr_pairs = pd.DataFrame({
        'Ligand': np.random.choice(receptors, 30, replace=True),
        'Receptor': np.random.choice(receptors, 30, replace=True),
        'Weight': np.random.rand(30) * 0.8 + 0.2,
    })

    # Preprocessing
    t0 = time.time()
    inputs = psx.prepare_input(
        exp_mat=exp_mat, target_genes=target_genes,
        lr_pairs=lr_pairs, rtf_db=rtf_db, tftg_db=tftg_db,
    )
    timings['preprocessing'] = time.time() - t0

    # HGNN
    t0 = time.time()
    pathways = psx.run_hgnn(
        inputs, epochs=30, device='cpu', seed=42,
        hgnn_dims=[64, 32], linear_dims=[16, 8]
    )
    timings['hgnn'] = time.time() - t0

    # PRS sklearn
    t0 = time.time()
    prs_sklearn = psx.compute_prs(inputs.exp_clu, pathways, engine='sklearn', n_estimators=n_estimators, cutoff=0.5)
    prs_sklearn = psx.filter_results(prs_sklearn, prs_threshold=0.01)
    timings['prs_sklearn'] = time.time() - t0

    # PRS LightGBM
    try:
        t0 = time.time()
        prs_lgbm = psx.compute_prs(inputs.exp_clu, pathways, engine='lightgbm', n_estimators=n_estimators, cutoff=0.5)
        prs_lgbm = psx.filter_results(prs_lgbm, prs_threshold=0.01)
        timings['prs_lightgbm'] = time.time() - t0
    except Exception:
        prs_lgbm = pd.DataFrame()
        timings['prs_lightgbm'] = 0

    # Crosstalk
    t0 = time.time()
    if len(prs_sklearn) > 0:
        counts = psx.count_crosstalk(prs_sklearn, verbose=False)
        trs = psx.aggregate_causality(prs_sklearn)
    else:
        counts = pd.Series(dtype=int)
        trs = pd.DataFrame()
    timings['crosstalk'] = time.time() - t0

    timings['total'] = sum(timings.values())

    return {
        'timings': timings,
        'pathways': pathways,
        'prs_sklearn': prs_sklearn,
        'prs_lgbm': prs_lgbm,
        'counts': counts,
        'trs': trs,
        'exp_clu': inputs.exp_clu,
    }


def run_r_pipeline(rscript_path, work_dir, target_type):
    """Run R SigXTalkR pipeline via Rscript and return timings."""
    r_script = f'''
library(Seurat)
library(SigXTalkR)
library(dplyr)

cat("Loading data...\\n")
t_start <- proc.time()
SeuratObj <- readRDS(file.path("{work_dir}", "pbmc3k_final.rds"))
t_load <- (proc.time() - t_start)[3]

cat("Loading databases...\\n")
t_start <- proc.time()
data("RTF_human")
data("TFT_human")
all_genes <- rownames(SeuratObj@assays$RNA$data)
RecTFDB <- RTF_human %>% distinct(from, to, .keep_all = TRUE) %>% Filter_DB(all_genes)
TFTGDB <- TFT_human %>% distinct(from, to, .keep_all = TRUE) %>% Filter_DB(all_genes)
t_db <- (proc.time() - t_start)[3]

cat("Finding target genes...\\n")
t_start <- proc.time()
TG_used <- FindMarkers(SeuratObj, "{target_type}", min.pct = 0.25, only.pos = TRUE, logfc.threshold = 0.25)
TG_used <- filter(TG_used, p_val_adj < 1e-3) %>% rownames()
t_markers <- (proc.time() - t_start)[3]

cat("Preparing inputs...\\n")
t_start <- proc.time()
input_dir <- file.path("{work_dir}", "r_inputs")
if (!dir.exists(input_dir)) dir.create(input_dir, recursive = TRUE)
Prepare_Input(SeuratObj, "{target_type}", TG_used, NULL, RecTFDB, TFTGDB, input_dir)
t_prep <- (proc.time() - t_start)[3]

cat("Running HGNN...\\n")
t_start <- proc.time()
py_script <- file.path("{work_dir}", "..", "SigXTalk-master", "vignettes", "main.py")
Run_py_script(py_script, "SigXTalk_py", c("--project", "benchmark_r", "--target_type", "{target_type}"))
t_hgnn <- (proc.time() - t_start)[3]

cat("Calculating PRS...\\n")
t_start <- proc.time()
pathways_file <- file.path("{work_dir}", "outputs", "benchmark_r", paste0("pathways_{target_type}.csv"))
if (file.exists(pathways_file)) {{
    RTFTG_results <- read.csv(pathways_file)
    RTFTG_results <- filter(RTFTG_results, pred_label > 0.75)
    Exp_clu <- Get_Exp_Clu(SeuratObj, "{target_type}")
    ress <- PRS_calc(Exp_clu, RTFTG_results, cutoff = 0.1)
    results_filtered <- Filter_results(ress, PRS_thres = 0.01)
    t_prs <- (proc.time() - t_start)[3]
    write.csv(results_filtered, file.path("{work_dir}", "r_prs_results.csv"), row.names = FALSE)
}} else {{
    cat("HGNN output not found\\n")
    t_prs <- 0
}}

cat("\\n=== R Timing Results ===\\n")
cat(sprintf("Load data: %.2fs\\n", t_load))
cat(sprintf("Databases: %.2fs\\n", t_db))
cat(sprintf("FindMarkers: %.2fs\\n", t_markers))
cat(sprintf("Preprocessing: %.2fs\\n", t_prep))
cat(sprintf("HGNN: %.2fs\\n", t_hgnn))
cat(sprintf("PRS: %.2fs\\n", t_prs))
cat(sprintf("Total: %.2fs\\n", t_load + t_db + t_markers + t_prep + t_hgnn + t_prs))
'''
    return r_script


if __name__ == "__main__":
    print("=" * 60)
    print("py-sigxtalk Benchmark: Python Pipeline")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading PBMC3k data...")
    adata = anndata.read_h5ad("pbmc3k_final.h5ad")
    print(f"  Cells: {adata.shape[0]}, Genes: {adata.shape[1]}")

    # Load databases
    print("[2/5] Loading databases...")
    rtf_db, tftg_db = psx.load_databases(species="human")
    print(f"  RTF: {len(rtf_db):,}, TFTG: {len(tftg_db):,}")

    # Run Python pipeline
    print("[3/5] Running Python pipeline (CD14+ Mono)...")
    result = run_python_pipeline(adata, rtf_db, tftg_db, "CD14+ Mono", n_estimators=100)

    # Print results
    print("\n" + "=" * 60)
    print("Python Pipeline Timing")
    print("=" * 60)
    for step, t in result['timings'].items():
        print(f"  {step:20s}: {t:8.2f}s")
    print(f"  {'TOTAL':20s}: {result['timings']['total']:8.2f}s")

    print(f"\n  Pathways:     {len(result['pathways']):>10,}")
    print(f"  PRS (sklearn):{len(result['prs_sklearn']):>10,}")
    print(f"  PRS (LightGBM):{len(result['prs_lgbm']):>10,}")
    print(f"  Crosstalk genes: {len(result['counts']):>7,}")
    print(f"  TRS pairs:    {len(result['trs']):>10,}")

    # Correlation between sklearn and LightGBM
    if len(result['prs_sklearn']) > 0 and len(result['prs_lgbm']) > 0:
        from scipy.stats import pearsonr
        merged = result['prs_sklearn'].merge(result['prs_lgbm'], on=["Receptor", "SSC", "Target"], suffixes=("_skl", "_lgb"))
        if len(merged) > 1:
            corr, pval = pearsonr(merged["Weight_skl"], merged["Weight_lgb"])
            print(f"\n  sklearn vs LightGBM correlation: {corr:.4f} (p={pval:.2e})")

    # Save results
    result['prs_sklearn'].to_csv("prs_results_python.csv", index=False)
    print(f"\n  Results saved to prs_results_python.csv")

    print("\n" + "=" * 60)
    print("Note: R comparison requires SigXTalkR package installed.")
    print("Run the R script separately if R SigXTalkR is available.")
    print("=" * 60)
