"""py-sigxtalk: Python rewrite of SigXTalk."""

__version__ = "0.1.0"

from .databases import load_databases, filter_db
from .cci import infer_cci, extract_lr_prob
from .preprocessing import (
    prepare_input, get_exp_clu, normalize_data,
    personalized_pagerank, fisher_test, HGNNInputs,
)
from .hgnn import run_hgnn, HGNNPredictor
from .prs import compute_prs, filter_results
from .crosstalk import (
    count_crosstalk, aggregate_causality,
    calculate_fidelity_matrix, calculate_specificity_matrix,
    calculate_pathway_fidelity, calculate_pathway_specificity,
    calculate_pairwise,
)
from .visualization import (
    plot_cci_chord, plot_cci_circle, plot_counts,
    plot_counts_histogram, plot_counts_bar,
    plot_rec_tg_heatmap, plot_alluvial, plot_fid_spe,
    plot_heatmap, plot_ridgeline, plot_circular_bar,
    plot_chord, plot_signal_contribution,
)

__all__ = [
    "__version__",
    "load_databases", "filter_db",
    "infer_cci", "extract_lr_prob",
    "prepare_input", "get_exp_clu", "normalize_data",
    "personalized_pagerank", "fisher_test", "HGNNInputs",
    "run_hgnn", "HGNNPredictor",
    "compute_prs", "filter_results",
    "count_crosstalk", "aggregate_causality",
    "calculate_fidelity_matrix", "calculate_specificity_matrix",
    "calculate_pathway_fidelity", "calculate_pathway_specificity",
    "calculate_pairwise",
    "plot_cci_chord", "plot_cci_circle", "plot_counts",
    "plot_counts_histogram", "plot_counts_bar",
    "plot_rec_tg_heatmap", "plot_alluvial", "plot_fid_spe",
    "plot_heatmap", "plot_ridgeline", "plot_circular_bar",
    "plot_chord", "plot_signal_contribution",
]
