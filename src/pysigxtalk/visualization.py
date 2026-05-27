import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# Match R's styling: Arial, large fonts, bold
FONT_FAMILY = "Arial"
FONT_SIZES = {"axis_title": 24, "axis_text": 20, "legend_title": 18, "legend_text": 14, "label": 16}
COLOR_FIDELITY = "#006400"  # green4
COLOR_SPECIFICITY = "#BA55D3"  # mediumorchid
COLOR_HIST = "#69b3a2"  # R's teal

def _setup_style():
    """Apply R-like ggplot2 styling."""
    mpl.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.size': FONT_SIZES['axis_text'],
        'axes.titlesize': FONT_SIZES['axis_title'],
        'axes.labelsize': FONT_SIZES['axis_title'],
        'xtick.labelsize': FONT_SIZES['axis_text'],
        'ytick.labelsize': FONT_SIZES['axis_text'],
        'legend.title_fontsize': FONT_SIZES['legend_title'],
        'legend.fontsize': FONT_SIZES['legend_text'],
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2,
        'grid.color': '#E5E5E5',
        'grid.linewidth': 0.8,
    })

_setup_style()


def plot_cci_chord(result, topk=10):
    """Chord diagram of LR interactions using circular layout with Bezier chords."""
    from matplotlib.patches import Wedge, PathPatch
    from matplotlib.path import Path

    top = result.nlargest(topk, "Weight").copy()
    ligands = top["Ligand"].unique().tolist()
    receptors = top["Receptor"].unique().tolist()
    all_nodes = list(dict.fromkeys(ligands + receptors))
    n = len(all_nodes)

    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16)
        return fig

    # Colors
    node_colors = plt.cm.Set3(np.linspace(0, 1, n))
    color_map = {name: node_colors[i] for i, name in enumerate(all_nodes)}

    # Calculate node weights
    node_weights = {}
    for node in all_nodes:
        w = top[top["Ligand"] == node]["Weight"].sum() + top[top["Receptor"] == node]["Weight"].sum()
        node_weights[node] = w

    total_weight = sum(node_weights.values())
    if total_weight == 0:
        total_weight = 1

    # Sector angles
    gap = 3
    available = 360 - n * gap
    sectors = {}
    cur = 0
    for name in all_nodes:
        span = max((node_weights[name] / total_weight) * available, 2)
        sectors[name] = {'start': cur, 'end': cur + span, 'mid': cur + span / 2}
        cur += span + gap

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    ax.axis('off')

    inner_r = 0.85
    outer_r = 1.0

    # Draw sectors
    for name in all_nodes:
        s = sectors[name]
        wedge = Wedge((0, 0), outer_r, s['start'], s['end'],
                      width=outer_r - inner_r,
                      facecolor=color_map[name], edgecolor='black', linewidth=1.5, alpha=0.9)
        ax.add_patch(wedge)

        mid_rad = np.radians(s['mid'])
        lr = outer_r + 0.12
        lx, ly = lr * np.cos(mid_rad), lr * np.sin(mid_rad)
        rot = s['mid']
        if 90 < rot < 270:
            rot += 180
        ax.text(lx, ly, name, ha='center', va='center',
                fontsize=10, fontweight='bold', rotation=rot, rotation_mode='anchor')

    # Draw chords
    sector_cur = {name: sectors[name]['start'] for name in all_nodes}
    max_w = top["Weight"].max()

    for _, row in top.iterrows():
        src, tgt, val = row["Ligand"], row["Receptor"], row["Weight"]

        src_span = (val / node_weights[src]) * (sectors[src]['end'] - sectors[src]['start'])
        tgt_span = (val / node_weights[tgt]) * (sectors[tgt]['end'] - sectors[tgt]['start'])

        src_a1 = sector_cur[src]
        src_a2 = src_a1 + src_span
        sector_cur[src] = src_a2

        tgt_a1 = sector_cur[tgt]
        tgt_a2 = tgt_a1 + tgt_span
        sector_cur[tgt] = tgt_a2

        def a2p(a):
            r = np.radians(a)
            return (inner_r * np.cos(r), inner_r * np.sin(r))

        p_s1, p_s2 = a2p(src_a1), a2p(src_a2)
        p_t1, p_t2 = a2p(tgt_a1), a2p(tgt_a2)
        cp = (0.0, 0.0)

        verts = [p_s1, p_s2, cp, cp, p_t1, p_t2, cp, cp, p_s1, p_s1]
        codes = [Path.MOVETO, Path.LINETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.LINETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.CLOSEPOLY]

        path = Path(verts, codes)
        alpha = 0.25 + 0.45 * (val / max_w)
        patch = PathPatch(path, facecolor=color_map[src], edgecolor='none', alpha=alpha)
        ax.add_patch(patch)

    ax.set_title("Ligand-Receptor Interactions", fontsize=20, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig


def plot_cci_circle(result, topk=10):
    """Circle diagram of LR interactions. Matches R's PlotCCI_CirclePlot.

    Nodes arranged in a circle with curved edges between them.
    Edge width and color reflect interaction weight.
    """
    from matplotlib.patches import FancyArrowPatch, Arc

    top = result.nlargest(topk, "Weight").copy()
    top["Weight_norm"] = top["Weight"] / top["Weight"].max()

    # Collect unique nodes
    ligands = top["Ligand"].unique().tolist()
    receptors = top["Receptor"].unique().tolist()
    all_nodes = list(dict.fromkeys(ligands + receptors))
    n = len(all_nodes)

    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16)
        return fig

    # Colors (rainbow like R)
    node_colors = plt.cm.rainbow(np.linspace(0, 1, n))
    color_map = {name: node_colors[i] for i, name in enumerate(all_nodes)}

    # Layout nodes in circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {name: (np.cos(a), np.sin(a)) for name, a in zip(all_nodes, angles)}

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw edges with curved arrows (like R's edge.curved = 0.2)
    max_w = top["Weight_norm"].max()
    for _, row in top.iterrows():
        src, tgt, w = row["Ligand"], row["Receptor"], row["Weight_norm"]
        if src not in pos or tgt not in pos:
            continue

        x1, y1 = pos[src]
        x2, y2 = pos[tgt]

        # Color matches source node (like R)
        color = color_map[src]
        lw = 0.5 + 4 * (w / max_w)

        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle="arc3,rad=0.2",
            arrowstyle='->,head_width=0.06,head_length=0.04',
            color=color,
            linewidth=lw,
            alpha=0.6,
            mutation_scale=12,
        )
        ax.add_patch(arrow)

    # Draw nodes (like R: vertex.size = 5, vertex.label.dist = 2)
    for name in all_nodes:
        x, y = pos[name]
        ax.scatter(x, y, s=300, c=[color_map[name]], edgecolors='black',
                   linewidths=1.5, zorder=5)
        # Label outside circle (vertex.label.dist = 2)
        label_r = 1.35
        lx, ly = x * label_r, y * label_r
        ax.text(lx, ly, name, ha='center', va='center',
                fontsize=10, fontweight='bold')

    ax.set_title("Ligand-Receptor Interactions", fontsize=20, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig


def plot_counts_histogram(cc_results, key_genes=None, data_type="Target"):
    """Histogram of crosstalk pathway counts. Matches R's PlotXT_Counts (outer plot)."""
    from .crosstalk import count_crosstalk
    counts = count_crosstalk(cc_results, key_genes=key_genes, data_type=data_type, verbose=False)
    counts_sorted = counts.sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))

    bins = range(int(counts_sorted.min()), int(counts_sorted.max()) + 2)
    ax.hist(counts_sorted.values, bins=bins,
            fill=True, color=COLOR_HIST, edgecolor="black", alpha=0.9, linewidth=0.8)
    ax.set_xlabel("Number of crosstalk pathways", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(f"Crosstalk pathways for {data_type}s", fontsize=20, fontweight="bold")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_counts_bar(cc_results, key_genes=None, data_type="Target", top_percent=10, topk=20):
    """Top genes bar chart by crosstalk count. Matches R's PlotXT_Counts (inner plot).

    Parameters
    ----------
    cc_results : pd.DataFrame
        PRS results with columns [Receptor, SSC, Target, Weight].
    key_genes : list, optional
        Specific genes to consider.
    data_type : str
        'Target' or 'SSC'.
    top_percent : int
        Percentile threshold for selecting top genes.
    topk : int
        Maximum number of genes to display (default 20).
    """
    from .crosstalk import count_crosstalk
    counts = count_crosstalk(cc_results, key_genes=key_genes, data_type=data_type, verbose=False)
    counts_sorted = counts.sort_values(ascending=False)

    # Select top genes by count
    top_genes = counts_sorted.head(topk).sort_values(ascending=True)
    n_genes = len(top_genes)

    # Match R's rainbow color scheme: colors based on unique pathway counts
    unique_counts = top_genes.unique()
    rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_counts)))
    count_to_color = {cnt: rainbow_colors[i] for i, cnt in enumerate(unique_counts)}
    bar_colors = [count_to_color[cnt] for cnt in top_genes.values]

    fig, ax = plt.subplots(figsize=(10, max(4, n_genes * 0.4)))

    ax.barh(range(n_genes), top_genes.values, color=bar_colors, alpha=0.6,
            edgecolor="black", linewidth=0.5, height=0.4)
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(top_genes.index, fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of pathways", fontweight="bold")
    ax.set_title(f"Top {topk} {data_type}s by crosstalk count", fontsize=20, fontweight="bold")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_counts(cc_results, key_genes=None, data_type="Target", top_percent=10, topk=20):
    """Histogram + bar chart of crosstalk counts (legacy combined view)."""
    from .crosstalk import count_crosstalk
    counts = count_crosstalk(cc_results, key_genes=key_genes, data_type=data_type, verbose=False)
    counts_sorted = counts.sort_values()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    bins = range(int(counts_sorted.min()), int(counts_sorted.max()) + 2)
    ax1.hist(counts_sorted.values, bins=bins,
             fill=True, color=COLOR_HIST, edgecolor="black", alpha=0.9, linewidth=0.8)
    ax1.set_xlabel("Number of crosstalk pathways", fontweight="bold")
    ax1.set_ylabel("Frequency", fontweight="bold")
    ax1.set_title(f"Crosstalk pathways for {data_type}s", fontsize=20, fontweight="bold")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    top_genes = counts.sort_values(ascending=False).head(topk).sort_values(ascending=True)
    n_genes = len(top_genes)

    # Match R's rainbow color scheme: colors based on unique pathway counts
    unique_counts = top_genes.unique()
    rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_counts)))
    count_to_color = {cnt: rainbow_colors[i] for i, cnt in enumerate(unique_counts)}
    bar_colors = [count_to_color[cnt] for cnt in top_genes.values]

    ax2.barh(range(n_genes), top_genes.values, color=bar_colors, alpha=0.6,
             edgecolor="black", linewidth=0.5, height=0.4)
    ax2.set_yticks(range(n_genes))
    ax2.set_yticklabels(top_genes.index, fontsize=14, fontweight="bold")
    ax2.set_xlabel("Number of pathways", fontweight="bold")
    ax2.set_title(f"Top {topk} {data_type}s", fontsize=20, fontweight="bold")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_rec_tg_heatmap(cc_pair_results, exp_clu, key_tg, topk=25):
    """Signal contribution stacked bar + TRS heatmap. Matches R's PlotXT_RecTGHeatmap.

    Parameters
    ----------
    cc_pair_results : pd.DataFrame
        PRS results with columns [Receptor, SSC, Target, Weight].
    exp_clu : pd.DataFrame
        Expression matrix (genes x cells).
    key_tg : str or list
        Target gene(s) to plot.
    topk : int
        Maximum number of receptor-target pairs to show.
    """
    from .crosstalk import aggregate_causality

    # Filter for target gene(s)
    if isinstance(key_tg, str):
        key_tg = [key_tg]

    # Aggregate to TRS (like R's Aggregate_Causality)
    trs = aggregate_causality(cc_pair_results, data_type="Target")
    subset = trs[trs["Target"].isin(key_tg)].copy()
    subset = subset.nlargest(topk, "Weight")

    if subset.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16)
        return fig

    key_tg_list = sorted(subset["Target"].unique())

    # Create matrix: Receptor (rows) x Target (columns)
    pivot = subset.pivot_table(index="Receptor", columns="Target", values="Weight", fill_value=0)
    # Sort by mean weight (descending)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    # Calculate expression contribution (like R)
    from .utils import calculate_average_expression
    ave_exp = {}
    for tg in key_tg_list:
        ave_exp[tg] = calculate_average_expression(tg, exp_clu, nonzero=False)

    # Normalized weights for stacked bar
    pivot_norm = pivot.div(pivot.sum(axis=0).replace(0, 1), axis=1)
    pivot_exp = pivot_norm.copy()
    for tg in key_tg_list:
        if tg in pivot_exp.columns:
            pivot_exp[tg] = pivot_exp[tg] * ave_exp.get(tg, 0)

    # Figure size
    n_targets = len(key_tg_list)
    n_rec = pivot.shape[0]
    fig_width = max(8, n_targets * 0.6 + 5)
    fig_height = max(16, n_rec * 0.6 + 5)

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create heatmap first to get its position
    ax2 = fig.add_axes([0, 0, 1, 1])  # temporary
    sns.heatmap(pivot, ax=ax2, cmap="Reds", annot=False, linewidths=0, linecolor="none",
                cbar_kws={"label": "Activity", "shrink": 0.6, "aspect": 25})

    # Get heatmap axes position for alignment
    pos = ax2.get_position()

    # Resize heatmap smaller, bar chart taller
    total_height = pos.y1 - pos.y0
    heatmap_height = total_height * 0.45
    bar_height = total_height * 0.45
    gap = 0.01

    # Move heatmap down to make room for title/legend at top
    ax2.set_position([pos.x0, pos.y0, pos.width, heatmap_height])

    # Create bar chart axes with exact same x position and width as heatmap
    ax1 = fig.add_axes([pos.x0, pos.y0 + heatmap_height + gap, pos.width, bar_height])

    # Upper: stacked bar (signal contribution to expression)
    bottom_arr = np.zeros(n_targets)
    rec_colors = plt.cm.rainbow(np.linspace(0.1, 0.9, pivot_exp.shape[0]))
    for i, rec in enumerate(pivot_exp.index):
        ax1.bar(range(n_targets), pivot_exp.loc[rec].values, bottom=bottom_arr,
                color=rec_colors[i], label=rec, edgecolor="none", width=0.8)
        bottom_arr += pivot_exp.loc[rec].values
    ax1.set_ylabel("Expression", fontweight="bold", fontsize=14)
    ax1.set_xticks(range(n_targets))
    ax1.set_xticklabels([])
    ax1.set_xlim(-0.5, n_targets - 0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Legend at the top of the figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, title="Signal", loc="upper center",
               bbox_to_anchor=(0.5, 0.98), ncol=min(5, len(handles)),
               fontsize=8, title_fontsize=10, frameon=True)


    # Lower: heatmap labels
    ax2.set_xlabel("Target gene", fontweight="bold", fontsize=14)
    ax2.set_xlabel("Target gene", fontweight="bold", fontsize=14)
    ax2.set_ylabel("Signal", fontweight="bold", fontsize=14)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, fontweight="bold", fontsize=11)
    ax2.set_yticklabels(ax2.get_yticklabels(), fontweight="bold", fontsize=11)

    return fig


def plot_alluvial(cc_results, key_tg, min_weight=0.45):
    """Sankey flow diagram using matplotlib. Matches R's PlotXT_Alluvial.

    Shows the regulatory flow: Target -> SSC -> Receptor.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    subset = cc_results[cc_results["Target"] == key_tg].copy()
    threshold = subset["Weight"].quantile(min_weight)
    subset = subset[subset["Weight"] >= threshold]

    if subset.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16)
        return fig

    # Group by pathway and aggregate weights
    pathways = subset.groupby(["Target", "SSC", "Receptor"])["Weight"].sum().reset_index()
    pathways = pathways.sort_values("Weight", ascending=False)

    # Get unique nodes at each stage
    targets = pathways["Target"].unique().tolist()
    sscs = pathways["SSC"].unique().tolist()
    receptors = pathways["Receptor"].unique().tolist()

    # Color maps
    tgt_colors = {t: plt.cm.Blues(0.4 + 0.4 * i / max(1, len(targets) - 1)) for i, t in enumerate(targets)}
    ssc_colors = {s: plt.cm.Oranges(0.3 + 0.5 * i / max(1, len(sscs) - 1)) for i, s in enumerate(sscs)}
    rec_colors = {r: plt.cm.Greens(0.3 + 0.5 * i / max(1, len(receptors) - 1)) for i, r in enumerate(receptors)}

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Layout: 3 columns
    x_tgt = 0.5
    x_ssc = 4.5
    x_rec = 8.5

    # Calculate node sizes based on total weight
    def calc_node_sizes(nodes, col_name):
        sizes = {}
        for node in nodes:
            total_w = pathways[pathways[col_name] == node]["Weight"].sum()
            sizes[node] = total_w
        return sizes

    tgt_sizes = calc_node_sizes(targets, "Target")
    ssc_sizes = calc_node_sizes(sscs, "SSC")
    rec_sizes = calc_node_sizes(receptors, "Receptor")

    # Normalize sizes to fit in plot
    max_total = max(max(tgt_sizes.values()), max(ssc_sizes.values()), max(rec_sizes.values()))
    scale = 7.0 / max_total  # Total height available

    # Calculate node positions (stacked vertically)
    def calc_positions(sizes, x_center, scale):
        positions = {}
        y_current = 0.5
        gap = 0.15
        for node, size in sizes.items():
            height = size * scale
            positions[node] = (x_center, y_current, height)
            y_current += height + gap
        return positions

    tgt_pos = calc_positions(tgt_sizes, x_tgt, scale)
    ssc_pos = calc_positions(ssc_sizes, x_ssc, scale)
    rec_pos = calc_positions(rec_sizes, x_rec, scale)

    # Track current y-offset for each node's outgoing/incoming flows
    tgt_out_offset = {t: 0 for t in targets}
    ssc_in_offset = {s: 0 for s in sscs}
    ssc_out_offset = {s: 0 for s in sscs}
    rec_in_offset = {r: 0 for r in receptors}

    # Draw flows first (behind nodes)
    max_weight = pathways["Weight"].max()

    for _, row in pathways.iterrows():
        tgt, ssc, rec, w = row["Target"], row["SSC"], row["Receptor"], row["Weight"]
        flow_height = w * scale

        # Get node positions
        tx, ty_base, th = tgt_pos[tgt]
        sx, sy_base, sh = ssc_pos[ssc]
        rx, ry_base, rh = rec_pos[rec]

        # Target -> SSC flow
        y1_start = ty_base + tgt_out_offset[tgt]
        y1_end = y1_start + flow_height
        tgt_out_offset[tgt] += flow_height

        y2_start = sy_base + ssc_in_offset[ssc]
        y2_end = y2_start + flow_height
        ssc_in_offset[ssc] += flow_height

        # Draw curved band (Target -> SSC)
        verts = [
            (tx + 0.35, y1_start),  # start left bottom
            (tx + 0.35, y1_end),    # start left top
            (sx - 0.35, y2_end),    # end right top
            (sx - 0.35, y2_start),  # end right bottom
            (tx + 0.35, y1_start),  # close
        ]

        # Create Bezier path for smooth curve
        cx1 = (tx + 0.35 + sx - 0.35) / 2
        path_data = [
            (Path.MOVETO, (tx + 0.35, y1_start)),
            (Path.CURVE4, (cx1, y1_start)),
            (Path.CURVE4, (cx1, y2_start)),
            (Path.CURVE4, (sx - 0.35, y2_start)),
            (Path.LINETO, (sx - 0.35, y2_end)),
            (Path.CURVE4, (cx1, y2_end)),
            (Path.CURVE4, (cx1, y1_end)),
            (Path.CURVE4, (tx + 0.35, y1_end)),
            (Path.CLOSEPOLY, (tx + 0.35, y1_start)),
        ]
        codes, verts_zip = zip(*path_data)
        path = Path(verts_zip, codes)
        color = tgt_colors.get(tgt, 'lightblue')
        patch = PathPatch(path, facecolor=color, edgecolor='none', alpha=0.4)
        ax.add_patch(patch)

        # SSC -> Receptor flow
        y3_start = sy_base + ssc_out_offset[ssc]
        y3_end = y3_start + flow_height
        ssc_out_offset[ssc] += flow_height

        y4_start = ry_base + rec_in_offset[rec]
        y4_end = y4_start + flow_height
        rec_in_offset[rec] += flow_height

        # Draw curved band (SSC -> Receptor)
        cx2 = (sx + 0.35 + rx - 0.35) / 2
        path_data2 = [
            (Path.MOVETO, (sx + 0.35, y3_start)),
            (Path.CURVE4, (cx2, y3_start)),
            (Path.CURVE4, (cx2, y4_start)),
            (Path.CURVE4, (rx - 0.35, y4_start)),
            (Path.LINETO, (rx - 0.35, y4_end)),
            (Path.CURVE4, (cx2, y4_end)),
            (Path.CURVE4, (cx2, y3_end)),
            (Path.CURVE4, (sx + 0.35, y3_end)),
            (Path.CLOSEPOLY, (sx + 0.35, y3_start)),
        ]
        codes2, verts_zip2 = zip(*path_data2)
        path2 = Path(verts_zip2, codes2)
        patch2 = PathPatch(path2, facecolor=color, edgecolor='none', alpha=0.4)
        ax.add_patch(patch2)

    # Draw nodes on top
    from matplotlib.patches import FancyBboxPatch

    def draw_nodes(positions, colors, ax):
        for node, (x, y, h) in positions.items():
            color = colors.get(node, 'gray')
            rect = FancyBboxPatch((x - 0.35, y), 0.7, h,
                                  boxstyle="round,pad=0.02",
                                  facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.9)
            ax.add_patch(rect)
            # Label inside if height allows, otherwise beside
            if h > 0.3:
                ax.text(x, y + h / 2, node, ha='center', va='center',
                        fontsize=8, fontweight='bold')
            else:
                ax.text(x + 0.5, y + h / 2, node, ha='left', va='center',
                        fontsize=7, fontweight='bold')

    draw_nodes(tgt_pos, tgt_colors, ax)
    draw_nodes(ssc_pos, ssc_colors, ax)
    draw_nodes(rec_pos, rec_colors, ax)

    # Column labels
    ax.text(x_tgt, 9.5, "Target", ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(x_ssc, 9.5, "SSC", ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(x_rec, 9.5, "Receptor", ha='center', va='center', fontsize=16, fontweight='bold')

    ax.set_title(f"Regulatory flow for {key_tg}", fontsize=20, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig


def plot_fid_spe(cc_results, key_tg, threshold=0.15):
    """Side-by-side fidelity and specificity heatmaps. Matches R's PlotXT_FidSpe."""
    from .crosstalk import calculate_fidelity_matrix, calculate_specificity_matrix

    subset = cc_results[cc_results["Target"] == key_tg]
    if subset.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16)
        return fig

    fid_mat = calculate_fidelity_matrix(cc_results, key_tg=key_tg, mode="all")
    if fid_mat.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16)
        return fig

    col_sums = fid_mat.sum(axis=0)
    fid_mat = fid_mat.loc[:, col_sums >= threshold * col_sums.max()]

    spe_data = []
    for rec in fid_mat.index:
        for ssc in fid_mat.columns:
            val = fid_mat.loc[rec, ssc]
            if val > 0:
                spe = calculate_specificity_matrix(cc_results, key_rec=rec, mode="all")
                if not spe.empty and ssc in spe.columns and key_tg in spe.index:
                    spe_val = spe.loc[key_tg, ssc]
                else:
                    spe_val = 0
                spe_data.append({"Receptor": rec, "SSC": ssc, "Specificity": spe_val})
    spe_df = pd.DataFrame(spe_data)
    if not spe_df.empty:
        spe_mat = spe_df.pivot_table(index="Receptor", columns="SSC", values="Specificity", fill_value=0)
        common_recs = [r for r in fid_mat.index if r in spe_mat.index]
        common_sscs = [s for s in fid_mat.columns if s in spe_mat.columns]
        fid_mat = fid_mat.loc[common_recs, common_sscs]
        spe_mat = spe_mat.loc[common_recs, common_sscs]
    else:
        spe_mat = fid_mat * 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(6, len(fid_mat) * 0.5 + 2)))

    # Left: Fidelity (green)
    sns.heatmap(fid_mat, ax=ax1, cmap=sns.color_palette("YlGn", as_cmap=True),
                annot=False, linewidths=2, linecolor="black",
                cbar_kws={"label": "Fidelity", "shrink": 0.7})
    ax1.set_xlabel("SSC", fontweight="bold", fontsize=FONT_SIZES['axis_title'])
    ax1.set_ylabel("Signal", fontweight="bold", fontsize=FONT_SIZES['axis_title'])
    ax1.set_title(f"Fidelity for {key_tg}", fontsize=20, fontweight="bold")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, fontweight="bold", fontsize=FONT_SIZES['axis_text'])
    ax1.set_yticklabels(ax1.get_yticklabels(), fontweight="bold", fontsize=FONT_SIZES['axis_text'])

    # Right: Specificity (purple)
    sns.heatmap(spe_mat, ax=ax2, cmap=sns.color_palette("RdPu", as_cmap=True),
                annot=False, linewidths=2, linecolor="black",
                cbar_kws={"label": "Specificity", "shrink": 0.7})
    ax2.set_xlabel("SSC", fontweight="bold", fontsize=FONT_SIZES['axis_title'])
    ax2.set_ylabel("")
    ax2.set_yticklabels([])
    ax2.set_title(f"Specificity for {key_tg}", fontsize=20, fontweight="bold")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, fontweight="bold", fontsize=FONT_SIZES['axis_text'])

    plt.tight_layout()
    return fig


def plot_heatmap(cc_results, gene_used, genetype, topk=25):
    """Complex heatmap with boxplot annotations. Matches R's PlotXT_HeatMap."""
    if genetype in ("Target", "TG"):
        subset = cc_results[cc_results["Target"] == gene_used]
        mat = subset.pivot_table(index="Receptor", columns="SSC", values="Weight", fill_value=0)
        legend_name = "Fidelity"
    elif genetype in ("SSC", "TF"):
        subset = cc_results[cc_results["SSC"] == gene_used]
        mat = subset.pivot_table(index="Receptor", columns="Target", values="Weight", fill_value=0)
        legend_name = "PRS"
    elif genetype in ("Receptor", "Rec"):
        subset = cc_results[cc_results["Receptor"] == gene_used]
        mat = subset.pivot_table(index="SSC", columns="Target", values="Weight", fill_value=0)
        legend_name = "Specificity"
    else:
        raise ValueError("genetype must be Target/TG, SSC/TF, or Receptor/Rec")

    if mat.empty or mat.shape[0] == 0 or mat.shape[1] == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16)
        return fig

    mat = mat / mat.sum().sum()
    mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]
    mat = mat[mat.mean(axis=0).sort_values(ascending=False).index]
    mat = mat.iloc[:topk, :topk]

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.05, wspace=0.05)
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_row = fig.add_subplot(gs[1, 1])
    ax_col = fig.add_subplot(gs[0, 0])

    sns.heatmap(mat, ax=ax_heat, cmap="Reds", annot=True, fmt=".3f",
                linewidths=1.5, linecolor="black",
                cbar_kws={"label": legend_name, "shrink": 0.8})
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=90, fontweight="bold", fontsize=13)
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), fontweight="bold", fontsize=13)

    if mat.shape[0] > 1:
        row_data = [mat.iloc[i].values for i in range(mat.shape[0])]
        ax_row.boxplot(row_data, vert=True, positions=range(mat.shape[0]),
                       patch_artist=True, boxprops=dict(facecolor="#69b3a2", alpha=0.7),
                       medianprops=dict(color="red", linewidth=2))
        ax_row.set_ylim(ax_heat.get_ylim())
        ax_row.set_yticklabels([])
        ax_row.set_xticklabels([])
        ax_row.spines['top'].set_visible(False)
        ax_row.spines['right'].set_visible(False)

    if mat.shape[1] > 1:
        col_data = [mat.iloc[:, i].values for i in range(mat.shape[1])]
        ax_col.boxplot(col_data, vert=False, positions=range(mat.shape[1]),
                       patch_artist=True, boxprops=dict(facecolor="#69b3a2", alpha=0.7),
                       medianprops=dict(color="red", linewidth=2))
        ax_col.set_xlim(ax_heat.get_xlim())
        ax_col.set_xticklabels([])
        ax_col.set_yticklabels([])
        ax_col.spines['top'].set_visible(False)
        ax_col.spines['right'].set_visible(False)

    fig.suptitle(f"{legend_name} heatmap for {gene_used}", fontsize=22, fontweight="bold", y=0.98)
    return fig


def plot_ridgeline(cc_results, key_tg):
    """Ridgeline density plot. Matches R's PlotXT_Ridgeline with ggridges.

    Each receptor gets a complete density curve, allowing overlap with adjacent rows.
    """
    from .crosstalk import calculate_fidelity_matrix
    from scipy.stats import gaussian_kde

    fid_mat = calculate_fidelity_matrix(cc_results, key_tg=key_tg, mode="all")
    if fid_mat.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16)
        return fig

    df = fid_mat.stack().reset_index()
    df.columns = ["Receptor", "SSC", "Fidelity"]
    # Keep zeros to show the peak at 0 like R
    # df = df[df["Fidelity"] > 0]

    # Order receptors by mean fidelity (ascending, bottom to top like R)
    rec_order = df.groupby("Receptor")["Fidelity"].mean().sort_values(ascending=True).index.tolist()
    n_rec = len(rec_order)

    fig, ax = plt.subplots(figsize=(10, max(4, n_rec * 0.5)))

    rec_colors = plt.cm.rainbow(np.linspace(0.1, 0.9, n_rec))
    overlap = 0.6  # How much ridges can overlap

    # Find global x range (extend slightly below 0 to show complete peak)
    all_data = df["Fidelity"].values
    data_range = all_data.max() - all_data.min()
    x_min = min(all_data.min() - data_range * 0.1, -0.02)
    x_max = max(all_data.max() * 1.3, 0.01)
    x = np.linspace(x_min, x_max, 300)

    # Draw ridges from bottom to top
    densities = []

    # First pass: compute all densities
    for i, rec in enumerate(rec_order):
        rec_data = df[df["Receptor"] == rec]["Fidelity"].values
        if len(rec_data) < 1:
            densities.append((x, np.zeros_like(x)))
            continue
        if len(rec_data) < 2:
            # For single point, create a Gaussian
            sigma = max(0.05, x_max * 0.05)
            kde_vals = np.exp(-0.5 * ((x - rec_data[0]) / sigma) ** 2)
            kde_vals = kde_vals / kde_vals.max() if kde_vals.max() > 0 else kde_vals
            densities.append((x, kde_vals))
        else:
            # Use larger bandwidth like R's default to get smooth distributions
            kde = gaussian_kde(rec_data, bw_method='scott')
            y = kde(x)
            densities.append((x, y))

    # Second pass: draw ridges with overlap
    for i, (rec, (x_vals, y_vals)) in enumerate(zip(rec_order, densities)):
        # Normalize density
        if y_vals.max() > 0:
            y_norm = y_vals / y_vals.max() * (1 - overlap)
        else:
            y_norm = y_vals

        # Base y position
        y_base = i

        # Fill the ridge
        ax.fill_between(x_vals, y_base, y_base + y_norm,
                        alpha=0.7, color=rec_colors[i], edgecolor='none')
        # Draw the outline
        ax.plot(x_vals, y_base + y_norm, color="black", linewidth=0.8)

    # Set y-axis to show receptor names
    ax.set_yticks(range(n_rec))
    ax.set_yticklabels(rec_order, fontweight="bold", fontsize=FONT_SIZES['axis_text'])
    ax.set_xlabel("Fidelity", fontweight="bold", fontsize=FONT_SIZES['axis_title'])
    ax.set_ylabel("Signal", fontweight="bold", fontsize=FONT_SIZES['axis_title'])
    ax.set_title(f"Fidelity distribution for {key_tg}", fontsize=20, fontweight="bold")
    ax.set_ylim(-0.3, n_rec)
    ax.set_xlim(x_min, x_max)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def plot_circular_bar(df, key_factors=None, topk=5, label_max=None):
    """Circular bar chart. Matches R's PlotXT_MultiCircularBar.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns [individual, group, value].
    key_factors : list, optional
        Groups to include.
    topk : int
        Top-k individuals per group.
    label_max : float, optional
        Maximum value for the radial axis.
    """
    col0 = df.columns[0]
    col1 = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    col_val = df.columns[-1]

    if key_factors is not None:
        df = df[df[col0].isin(key_factors)]

    # Get top-k per group
    top_df = df.groupby(col0, group_keys=False).apply(lambda x: x.nlargest(topk, col_val))
    top_df = top_df.reset_index(drop=True)

    groups = top_df[col0].unique().tolist()
    n_groups = len(groups)

    # Build data with empty bars between groups (like R)
    empty_bar = 3
    all_rows = []
    for g in groups:
        grp = top_df[top_df[col0] == g].copy()
        for _, row in grp.iterrows():
            all_rows.append({col0: g, col1: row[col1], col_val: row[col_val]})
        for _ in range(empty_bar):
            all_rows.append({col0: g, col1: "", col_val: 0})

    data = pd.DataFrame(all_rows)
    data["id"] = range(1, len(data) + 1)

    n_bars = len(data)
    bar_width = 2 * np.pi / n_bars * 0.8  # width of each bar

    # Angle for each bar (evenly spaced around circle)
    angles = np.linspace(0, 2 * np.pi, n_bars, endpoint=False)

    if label_max is None:
        label_max = float(data[col_val].max()) * 1.05

    # Colors per group
    group_colors = plt.cm.Set2(np.linspace(0, 1, n_groups))
    color_map = {g: group_colors[i] for i, g in enumerate(groups)}
    bar_colors = [color_map.get(g, "gray") for g in data[col0]]

    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))

    # Draw bars
    bars = ax.bar(angles, data[col_val].values, width=bar_width, color=bar_colors,
                  alpha=0.6, edgecolor="black", linewidth=0.3)

    # Labels on the outer circle (facing outward like R)
    label_radius = label_max * 1.12
    for i, row in data.iterrows():
        if row[col1]:  # not empty
            angle_rad = angles[i]
            angle_deg = np.degrees(angle_rad)
            # Rotation: text follows the circle
            rot = angle_deg
            if 90 < angle_deg < 270:
                rot = angle_deg + 180
                ha = "right"
            else:
                ha = "left"
            ax.text(angle_rad, label_radius, str(row[col1])[:12],
                    ha=ha, va="center", fontsize=8, fontweight="bold",
                    rotation=rot, rotation_mode="anchor")

    # Grid circles at 20%, 40%, 60%, 80%
    for frac in [0.2, 0.4, 0.6, 0.8]:
        r = label_max * frac
        ax.plot(np.linspace(0, 2 * np.pi, 100), [r] * 100,
                color="gray", linewidth=0.3, alpha=0.5)

    # Inner circle: baseline segments and group labels
    baseline_r = -label_max * 0.05
    label_r = -label_max * 0.25

    start = 0
    for g in groups:
        grp_data = data[data[col0] == g]
        n_real = len(grp_data[grp_data[col1] != ""])
        n_total = n_real + empty_bar

        if n_real > 0:
            # Baseline segment (arc)
            start_idx = start
            end_idx = start + n_real - 1
            if start_idx < len(angles) and end_idx < len(angles):
                arc_angles = np.linspace(angles[start_idx], angles[end_idx], 50)
                ax.plot(arc_angles, [baseline_r] * len(arc_angles),
                        color="black", linewidth=2, alpha=0.8)

            # Group label along the arc (facing center)
            mid_idx = start + n_real // 2
            if mid_idx < len(angles):
                mid_angle = angles[mid_idx]
                angle_deg = np.degrees(mid_angle)
                # Text follows the arc, rotated to be readable
                rot = angle_deg - 90
                if 90 < angle_deg < 270:
                    rot = angle_deg + 90
                ax.text(mid_angle, label_r, str(g), ha="center", va="center",
                        fontsize=11, fontweight="bold", color="black",
                        rotation=rot, rotation_mode="anchor")

        start += n_total

    ax.set_ylim(label_max * -0.7, label_max * 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)

    return fig


def plot_chord(mat, orders=None, edge_colors=None):
    """Chord diagram using matplotlib. Matches R's PlotXT_Chord with circlize.

    Parameters
    ----------
    mat : pd.DataFrame
        Matrix with row nodes as targets, column nodes as sources.
    orders : list, optional
        Order of nodes around the circle.
    edge_colors : dict, optional
        Custom colors for edges.
    """
    from matplotlib.patches import Wedge, PathPatch
    from matplotlib.path import Path

    rows = mat.index.tolist()
    cols = mat.columns.tolist()
    all_nodes = list(dict.fromkeys(cols + rows))

    if orders is not None:
        all_nodes = [n for n in orders if n in all_nodes]

    n = len(all_nodes)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16)
        return fig

    # Colors for each node
    node_colors = plt.cm.Set3(np.linspace(0, 1, n))
    color_map = {name: node_colors[i] for i, name in enumerate(all_nodes)}

    # Calculate total weight for each node
    node_weights = {}
    for name in all_nodes:
        if name in cols:
            node_weights[name] = float(mat.loc[:, name].sum())
        elif name in rows:
            node_weights[name] = float(mat.loc[name, :].sum())
        else:
            node_weights[name] = 0

    total_weight = sum(node_weights.values())
    if total_weight == 0:
        total_weight = 1

    # Calculate angles for each sector
    gap_angle = 3
    available_angle = 360 - n * gap_angle

    sectors = {}
    current_angle = 0
    for name in all_nodes:
        sector_angle = (node_weights[name] / total_weight) * available_angle
        sector_angle = max(sector_angle, 2)  # minimum size
        sectors[name] = {
            'start': current_angle,
            'end': current_angle + sector_angle,
            'mid': current_angle + sector_angle / 2,
        }
        current_angle += sector_angle + gap_angle

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    ax.axis('off')

    inner_radius = 0.85
    outer_radius = 1.0

    # Draw sectors
    for name in all_nodes:
        s = sectors[name]
        color = color_map[name]
        wedge = Wedge((0, 0), outer_radius, s['start'], s['end'],
                      width=outer_radius - inner_radius,
                      facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.9)
        ax.add_patch(wedge)

        # Label
        mid_rad = np.radians(s['mid'])
        label_r = outer_radius + 0.12
        lx = label_r * np.cos(mid_rad)
        ly = label_r * np.sin(mid_rad)
        rot = s['mid']
        if 90 < rot < 270:
            rot += 180
        ax.text(lx, ly, name, ha='center', va='center',
                fontsize=10, fontweight='bold', rotation=rot, rotation_mode='anchor')

    # Track position within each sector for stacking chords
    sector_cur = {name: sectors[name]['start'] for name in all_nodes}

    # Draw chords
    max_mat_val = mat.values.max()
    if max_mat_val == 0:
        max_mat_val = 1

    for src in cols:
        for tgt in rows:
            val = float(mat.loc[tgt, src])
            if val <= 0:
                continue

            # Source sector span
            src_total = node_weights[src]
            tgt_total = node_weights[tgt]
            if src_total == 0 or tgt_total == 0:
                continue

            src_sector = sectors[src]
            tgt_sector = sectors[tgt]

            src_span = (val / src_total) * (src_sector['end'] - src_sector['start'])
            tgt_span = (val / tgt_total) * (tgt_sector['end'] - tgt_sector['start'])

            # Current positions within sectors
            src_a1 = sector_cur[src]
            src_a2 = src_a1 + src_span
            sector_cur[src] = src_a2

            tgt_a1 = sector_cur[tgt]
            tgt_a2 = tgt_a1 + tgt_span
            sector_cur[tgt] = tgt_a2

            # Points on the inner arc
            def angle_to_point(angle_deg):
                rad = np.radians(angle_deg)
                return (inner_radius * np.cos(rad), inner_radius * np.sin(rad))

            p_src1 = angle_to_point(src_a1)
            p_src2 = angle_to_point(src_a2)
            p_tgt1 = angle_to_point(tgt_a1)
            p_tgt2 = angle_to_point(tgt_a2)

            # Control point at center
            cp = (0.0, 0.0)

            # Build path: src arc -> bezier to tgt -> tgt arc -> bezier back
            path_verts = [
                p_src1,
                p_src2,
            ]
            path_codes = [
                Path.MOVETO,
                Path.LINETO,
            ]

            # Bezier from src2 to tgt1 (through center)
            path_verts.append(cp)
            path_codes.append(Path.CURVE4)
            path_verts.append(cp)
            path_codes.append(Path.CURVE4)
            path_verts.append(p_tgt1)
            path_codes.append(Path.CURVE4)

            # Arc along tgt
            path_verts.append(p_tgt2)
            path_codes.append(Path.LINETO)

            # Bezier from tgt2 back to src1 (through center)
            path_verts.append(cp)
            path_codes.append(Path.CURVE4)
            path_verts.append(cp)
            path_codes.append(Path.CURVE4)
            path_verts.append(p_src1)
            path_codes.append(Path.CURVE4)

            # Close
            path_verts.append(p_src1)
            path_codes.append(Path.CLOSEPOLY)

            path = Path(path_verts, path_codes)
            color = color_map[src]
            alpha = 0.25 + 0.45 * (val / max_mat_val)

            patch = PathPatch(path, facecolor=color, edgecolor='none', alpha=alpha)
            ax.add_patch(patch)

    ax.set_title("Chord Diagram", fontsize=20, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig


def plot_signal_contribution(cc_pair_results, exp_clu, key_tg, topk=25):
    """Stacked bar of signal contributions."""
    from .crosstalk import aggregate_causality
    trs = aggregate_causality(cc_pair_results, data_type="Target")
    subset = trs[trs["Target"] == key_tg].nlargest(topk, "Weight")

    if subset.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16)
        return fig

    fig, ax = plt.subplots(figsize=(12, 6))

    receptors = subset["Receptor"].unique()
    colors = plt.cm.rainbow(np.linspace(0.1, 0.9, len(receptors)))
    color_map = {r: colors[i] for i, r in enumerate(receptors)}

    ax.bar(range(len(subset)), subset["Weight"].values,
           color=[color_map[r] for r in subset["Receptor"]],
           edgecolor="black", linewidth=0.5, width=0.7)
    ax.set_xticks(range(len(subset)))
    ax.set_xticklabels(subset["Receptor"], rotation=45, ha="right", fontweight="bold", fontsize=13)
    ax.set_ylabel("TRS", fontweight="bold", fontsize=FONT_SIZES['axis_title'])
    ax.set_title(f"Signal contribution to {key_tg}", fontsize=20, fontweight="bold")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig
