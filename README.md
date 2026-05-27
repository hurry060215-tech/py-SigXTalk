# py-sigxtalk

Python rewrite of SigXTalk: Dissecting crosstalk induced by cell-cell communication using single-cell transcriptomic data.

## Installation

```bash
# Install from source
cd py-sigxtalk
pip install -e ".[dev]"
```

## Quick Start

```python
import pysigxtalk as psx

# Load databases
rtf_db, tftg_db = psx.load_databases(species="human")

# Prepare inputs
inputs = psx.prepare_input(exp_mat, target_genes, lr_pairs, rtf_db, tftg_db)

# Run HGNN
pathways = psx.run_hgnn(inputs, epochs=10, device="cpu")

# Calculate PRS
prs_results = psx.compute_prs(inputs.exp_clu, pathways, n_estimators=10)

# Visualize
psx.plot_counts_histogram(prs_results)
psx.plot_counts_bar(prs_results, topk=20)
psx.plot_fid_spe(prs_results, key_tg="CD68")
```

## Benchmark

**Python vs R (single core, PBMC3k dataset):**

| Step | Python | R (ranger) |
|------|--------|------------|
| HGNN | 31s | 31s |
| PRS | 33s | 219s |
| **Total** | **80s** | **266s** |
| **Speedup** | **3.33x** | - |
| **Correlation** | **0.9912** | - |

**PRS Weight Correlation:**

![Benchmark Correlation](example/figures/benchmark_correlation.png)

**Timing Comparison:**

![Benchmark Timing](example/figures/benchmark_timing.png)

## Visualization Gallery

**Crosstalk Counts:**

| Histogram | Bar Chart |
|-----------|-----------|
| ![Histogram](example/figures/01_crosstalk_histogram.png) | ![Bar](example/figures/02_crosstalk_bar.png) |

**Crosstalk Analysis:**

| Fid/Spe | Alluvial | Ridgeline |
|---------|----------|-----------|
| ![Fid/Spe](example/figures/03_fid_spe.png) | ![Alluvial](example/figures/04_alluvial.png) | ![Ridgeline](example/figures/05_ridgeline.png) |

**Network Diagrams:**

| Chord | CCI Chord | CCI Circle |
|-------|-----------|------------|
| ![Chord](example/figures/06_chord.png) | ![CCI Chord](example/figures/07_cci_chord.png) | ![CCI Circle](example/figures/08_cci_circle.png) |

**Heatmaps:**

| Signal Contribution | Rec-TG Heatmap | Circular Bar |
|---------------------|----------------|--------------|
| ![Signal](example/figures/09_signal_contribution.png) | ![Heatmap](example/figures/10_rec_tg_heatmap.png) | ![Circular](example/figures/11_circular_bar.png) |

## Examples

```
example/
├── quickstart.ipynb              # Full analysis workflow
├── benchmark.ipynb               # Python vs R comparison
├── data/                         # Input data & results
│   ├── pbmc3k_final.h5ad         # PBMC3k dataset
│   ├── prs_results.csv           # PRS results
│   ├── prs_results_python.csv    # Python PRS results
│   └── prs_results_r.csv         # R PRS results
├── figures/                      # Generated figures
│   ├── 01-11_*.png               # 11 visualization figures
│   └── benchmark_*.png           # Benchmark plots
└── scripts/                      # Utility scripts
    ├── run_benchmark.py          # Benchmark runner
    ├── tutorial.py               # Tutorial script
    └── compare_R_vs_Python.py    # R vs Python comparison
```

## Citation

> Hou, J. et al. Dissecting crosstalk induced by cell-cell communication using single-cell transcriptomic data. Nature Communications (2025).
