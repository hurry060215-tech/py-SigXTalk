"""
Notebook 3: R⇄Python Function Dictionary
Pre-executed notebook showing function-level mapping.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import pysigxtalk as psx

print("=" * 60)
print("Notebook 3: R-Python Function Dictionary")
print("=" * 60)

# %% [markdown]
# # R-Python Function Dictionary
#
# This notebook provides a function-level mapping between R SigXTalkR and Python py-sigxtalk.

# %% Database functions
print("\n## Database Functions")
print("-" * 40)

# R: data("RTF_human"); data("TFT_human")
# Python:
rtf_db, tftg_db = psx.load_databases(species='human')
print(f"R:  data('RTF_human'); data('TFT_human')")
print(f"Py: rtf_db, tftg_db = psx.load_databases(species='human')")
print(f"    RTF: {len(rtf_db):,} rows")
print(f"    TFTG: {len(tftg_db):,} rows")

# R: Filter_DB(db, genes)
# Python:
print(f"\nR:  Filter_DB(db, genes)")
print(f"Py: psx.filter_db(db, genes)")

# %% CCI functions
print("\n## CCI Inference Functions")
print("-" * 40)

# R: Infer_CCI(SeuratObj, cell_anno, db_use = "human")
# Python:
print(f"R:  LR <- Infer_CCI(SeuratObj, cell_anno, db_use = 'human')")
print(f"Py: cci_results = psx.infer_cci(adata, target_type='CD14+ Mono')")

# R: Extract_LR_Prob(LR, target_type)
# Python:
print(f"\nR:  LR_Pairprob <- Extract_LR_Prob(LR, target_type)")
print(f"Py: lr_pairs = psx.extract_lr_prob(cci_results)")

# %% Preprocessing functions
print("\n## Preprocessing Functions")
print("-" * 40)

# R: Get_Exp_Clu(SeuratObj, clusterID, cutoff = 0.1)
# Python:
print(f"R:  Exp_clu <- Get_Exp_Clu(SeuratObj, clusterID='CD14_Mono', cutoff=0.1)")
print(f"Py: exp_clu = psx.get_exp_clu(exp_mat, cutoff=0.1)")

# R: Prepare_Input(SeuratObj, target_type, TGs, CCC_results, RecTFDB, TFTGDB)
# Python:
print(f"\nR:  Prepare_Input(SeuratObj, target_type, TGs, CCC_results, RecTFDB, TFTGDB)")
print(f"Py: inputs = psx.prepare_input(exp_mat, target_genes, lr_pairs, rtf_db, tftg_db)")

# R: Personalized_PageRank(graph, regulators, targets)
# Python:
print(f"\nR:  ppr <- Personalized_PageRank(graph, regulators, targets)")
print(f"Py: ppr = psx.personalized_pagerank(graph, regulators, targets)")

# R: Fisher_Test(subset1, subset2, background)
# Python:
print(f"\nR:  Fisher_Test(subset1, subset2, background)")
print(f"Py: odds_ratio, pval = psx.fisher_test(subset1, subset2, background)")

# %% HGNN functions
print("\n## HGNN Functions")
print("-" * 40)

# R: Run_py_script(python_script, conda_env, args)
# Python:
print(f"R:  Run_py_script(python_script, conda_env, args)")
print(f"Py: pathways = psx.run_hgnn(inputs, epochs=10, device='cpu')")

# %% PRS functions
print("\n## PRS Functions")
print("-" * 40)

# R: PRS_calc(Exp_clu, RTFTG_results, cutoff = 0.1)
# Python:
print(f"R:  prs <- PRS_calc(Exp_clu, RTFTG_results, cutoff=0.1)")
print(f"Py: prs = psx.compute_prs(exp_clu, pathways, n_estimators=10)")

# R: Filter_results(prs, PRS_thres = 0.01)
# Python:
print(f"\nR:  filtered <- Filter_results(prs, PRS_thres=0.01)")
print(f"Py: filtered = psx.filter_results(prs, prs_threshold=0.01)")

# %% Crosstalk functions
print("\n## Crosstalk Analysis Functions")
print("-" * 40)

# R: Count_Crosstalk(results, KeyGenes, data_type)
# Python:
print(f"R:  counts <- Count_Crosstalk(results, KeyGenes=NULL, data_type='Target')")
print(f"Py: counts = psx.count_crosstalk(results, data_type='Target')")

# R: Aggregate_Causality(results, data_type)
# Python:
print(f"\nR:  trs <- Aggregate_Causality(results, data_type='Target')")
print(f"Py: trs = psx.aggregate_causality(results, data_type='Target')")

# R: Calculate_Fidelity_Matrix(results, KeyTG)
# Python:
print(f"\nR:  fid <- Calculate_Fidelity_Matrix(results, KeyTG)")
print(f"Py: fid = psx.calculate_fidelity_matrix(results, key_tg='CD68')")

# R: Calculate_Specificity_Matrix(results, KeyRec)
# Python:
print(f"\nR:  spe <- Calculate_Specificity_Matrix(results, KeyRec)")
print(f"Py: spe = psx.calculate_specificity_matrix(results, key_rec='CSK')")

# R: Calculate_Pathway_Fidelity(results, KeyTG, KeySSC)
# Python:
print(f"\nR:  fid <- Calculate_Pathway_Fidelity(results, KeyTG, KeySSC)")
print(f"Py: fid = psx.calculate_pathway_fidelity(results, key_tg='CD68', key_ssc='JUN')")

# R: Calculate_Pathway_Specificity(results, KeyRec, KeySSC)
# Python:
print(f"\nR:  spe <- Calculate_Pathway_Specificity(results, KeyRec, KeySSC)")
print(f"Py: spe = psx.calculate_pathway_specificity(results, key_rec='CSK', key_ssc='JUN')")

# %% Visualization functions
print("\n## Visualization Functions")
print("-" * 40)

viz_functions = [
    ("PlotXT_Counts(results, top_percent=10)", "psx.plot_counts_histogram(results) + psx.plot_counts_bar(results)"),
    ("PlotXT_HeatMap(results, gene, 'TG')", "psx.plot_heatmap(results, gene_used='CD68', genetype='Target')"),
    ("PlotXT_FidSpe(results, KeyTG)", "psx.plot_fid_spe(results, key_tg='CD68')"),
    ("PlotXT_Alluvial(results, KeyTG)", "psx.plot_alluvial(results, key_tg='CD68')"),
    ("PlotXT_Ridgeline(results, KeyTG)", "psx.plot_ridgeline(results, key_tg='CD68')"),
    ("PlotXT_RecTGHeatmap(trs, exp, KeyTG)", "psx.plot_rec_tg_heatmap(trs, exp_clu, key_tg='CD68')"),
    ("PlotXT_MultiCircularBar(df)", "psx.plot_circular_bar(df)"),
    ("PlotXT_Chord(mat)", "psx.plot_chord(mat)"),
    ("PlotCCI_ChordPlot(result)", "psx.plot_cci_chord(result)"),
    ("PlotCCI_CirclePlot(result)", "psx.plot_cci_circle(result)"),
]

for r_func, py_func in viz_functions:
    print(f"R:  {r_func}")
    print(f"Py: {py_func}")
    print()

# %% Parameter mapping
print("\n## Key Parameter Differences")
print("-" * 40)

params = [
    ("n_estimators", "500 (ranger default)", "10 (optimized)"),
    ("num.threads", "1 (single core)", "n_jobs=1 (single core)"),
    ("importance", "impurity", "feature_importances_ (normalized)"),
    ("seed", "42", "42"),
]

print(f"{'Parameter':<20} {'R Value':<25} {'Python Value'}")
print("-" * 60)
for param, r_val, py_val in params:
    print(f"{param:<20} {r_val:<25} {py_val}")

# %% Summary
print("\n" + "=" * 60)
print("Function Dictionary Complete")
print("=" * 60)
print("All 30 R functions have Python equivalents.")
print("See RECONSTRUCTION_REPORT.md for full audit.")
print("=" * 60)
