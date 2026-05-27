<p align="center">
  <img src="data/logo.png" width="360px" alt="py-sigxtalk logo">
</p>

<div align="center">

| | |
|---:|:---|
| **Package** | [![PyPI](https://img.shields.io/pypi/v/py-sigxtalk?color=blue)](https://pypi.org/project/py-sigxtalk/) ![Python Versions](https://img.shields.io/pypi/pyversions/py-sigxtalk) [![Downloads](https://static.pepy.tech/badge/py-sigxtalk)](https://pepy.tech/project/py-sigxtalk) |
| **Meta** | [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Stars](https://img.shields.io/github/stars/omicverse/py-SigXTalk?style=social)](https://github.com/omicverse/py-SigXTalk) |

</div>

---

# py-sigxtalk

Python rewrite of **SigXTalk**: Dissecting crosstalk induced by cell-cell communication using single-cell transcriptomic data.

## Installation

```bash
pip install py-sigxtalk
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

![Benchmark Correlation](example/figures/benchmark_correlation.png)
![Benchmark Timing](example/figures/benchmark_timing.png)

## Visualization Gallery

| Histogram | Bar Chart | Fid/Spe |
|-----------|-----------|---------|
| ![Histogram](example/figures/01_crosstalk_histogram.png) | ![Bar](example/figures/02_crosstalk_bar.png) | ![Fid/Spe](example/figures/03_fid_spe.png) |

| Alluvial | Ridgeline | Chord |
|----------|-----------|-------|
| ![Alluvial](example/figures/04_alluvial.png) | ![Ridgeline](example/figures/05_ridgeline.png) | ![Chord](example/figures/06_chord.png) |

| CCI Chord | CCI Circle | Signal Contribution |
|-----------|------------|---------------------|
| ![CCI Chord](example/figures/07_cci_chord.png) | ![CCI Circle](example/figures/08_cci_circle.png) | ![Signal](example/figures/09_signal_contribution.png) |

| Rec-TG Heatmap | Circular Bar |
|----------------|--------------|
| ![Heatmap](example/figures/10_rec_tg_heatmap.png) | ![Circular](example/figures/11_circular_bar.png) |

## Examples

```
example/
├── quickstart.ipynb              # Full analysis workflow
├── benchmark.ipynb               # Python vs R comparison
├── data/                         # Input data & results
├── figures/                      # Generated figures
└── scripts/                      # Utility scripts
```

## Citation

> Hou, J. et al. Dissecting crosstalk induced by cell-cell communication using single-cell transcriptomic data. Nature Communications (2025).
