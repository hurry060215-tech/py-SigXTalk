"""Preprocessing module for SigXTalk.

Handles database filtering, PageRank, Fisher test, correlation calculation,
hypergraph construction, and the main prepare_input function.
"""

import itertools
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, kendalltau, spearmanr
from sklearn.preprocessing import StandardScaler


@dataclass
class HGNNInputs:
    """Container for HGNN model inputs."""

    exp_clu: pd.DataFrame
    rtf_filtered: pd.DataFrame
    tftg_filtered: pd.DataFrame
    lr_pairs: pd.DataFrame


def get_exp_clu(exp_mat: pd.DataFrame, cutoff: float = 0.05) -> pd.DataFrame:
    """Extract and filter expression matrix for a cell type.

    Parameters
    ----------
    exp_mat : pd.DataFrame
        Expression matrix (genes x cells).
    cutoff : float
        Minimum fraction of cells expressing a gene.

    Returns
    -------
    pd.DataFrame
        Filtered expression matrix.
    """
    n_cells = exp_mat.shape[1]
    min_cells = int(cutoff * n_cells)
    cell_counts = (exp_mat > 0).sum(axis=1)
    return exp_mat.loc[cell_counts >= min_cells]


def normalize_data(df: pd.DataFrame, log2: bool = False) -> pd.DataFrame:
    """Normalize expression data by median library size.

    Parameters
    ----------
    df : pd.DataFrame
        Expression matrix (cells x genes).
    log2 : bool
        Whether to apply log2(1 + x) transformation.

    Returns
    -------
    pd.DataFrame
        Normalized expression matrix.
    """
    exp = df.values
    lib_sizes = np.sum(exp, axis=1)
    median_size = np.median(lib_sizes)
    lib_sizes = np.where(lib_sizes == 0, 1, lib_sizes)
    scaling_factors = median_size / lib_sizes
    exp_nor = np.diag(scaling_factors) @ exp
    if log2:
        exp_nor = np.log2(1 + exp_nor)
    return pd.DataFrame(exp_nor, index=df.index, columns=df.columns)


def fisher_test(
    subset1: list, subset2: list, background: list
) -> tuple[float, float]:
    """Fisher's exact test for gene overlap significance.

    Parameters
    ----------
    subset1 : list
        First gene set.
    subset2 : list
        Second gene set.
    background : list
        Background gene universe.

    Returns
    -------
    tuple[float, float]
        (odds_ratio, p_value)
    """
    set1 = set(subset1)
    set2 = set(subset2)
    bg = set(background)
    overlap = len(set1 & set2 & bg)
    only1 = len((set1 & bg) - set2)
    only2 = len((set2 & bg) - set1)
    neither = len(bg) - overlap - only1 - only2
    table = [[overlap, only1], [only2, neither]]
    odds_ratio, p_value = fisher_exact(table, alternative="greater")
    return odds_ratio, p_value


def personalized_pagerank(graph, regulators, targets, damping=0.85):
    """Personalized PageRank for Receptor-TF filtering.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed graph of regulatory interactions.
    regulators : list
        Source nodes (e.g., receptors).
    targets : list
        Target nodes (e.g., transcription factors).
    damping : float
        Damping factor for PageRank.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [regulator, target, ppr].
    """
    results = []
    for reg in regulators:
        if reg not in graph.nodes():
            continue
        personalization = {node: 0.0 for node in graph.nodes()}
        reg_nodes = [n for n in graph.nodes() if reg in str(n)]
        if not reg_nodes:
            continue
        for node in reg_nodes:
            personalization[node] = 1.0
        try:
            pr = nx.pagerank(
                graph,
                alpha=damping,
                personalization=personalization,
                max_iter=100,
            )
        except Exception:
            continue
        for tf in targets:
            if tf not in graph.nodes():
                continue
            tf_nodes = [n for n in graph.nodes() if tf in str(n)]
            score = sum(pr.get(node, 0.0) for node in tf_nodes)
            if score > 0:
                results.append({"regulator": reg, "target": tf, "ppr": score})
    return pd.DataFrame(results)


def find_regulator(db, exp_genes, selected_genes, pv_threshold=1.0):
    """Filter TF-TG database using Fisher's exact test.

    Parameters
    ----------
    db : pd.DataFrame
        TF-TG database.
    exp_genes : list
        All expressed genes (background).
    selected_genes : list
        Target gene set.
    pv_threshold : float
        P-value threshold for filtering.

    Returns
    -------
    pd.DataFrame
        Filtered database.
    """
    result = db.copy()
    if pv_threshold < 1.0:
        keep_idx = []
        for i, row in result.iterrows():
            tf_targets = db[db.iloc[:, 0] == row.iloc[0]].iloc[:, 1].tolist()
            _, pval = fisher_test(tf_targets, selected_genes, exp_genes)
            if pval < pv_threshold:
                keep_idx.append(i)
        result = result.loc[keep_idx]
    return result.reset_index(drop=True)


def prepare_input(
    exp_mat,
    target_genes,
    lr_pairs,
    rtf_db,
    tftg_db,
    ccc_threshold=0.05,
    fisher_threshold=1.0,
):
    """Prepare all inputs for the HGNN module.

    Parameters
    ----------
    exp_mat : pd.DataFrame
        Expression matrix (genes x cells).
    target_genes : list
        Target gene set for Fisher filtering.
    lr_pairs : pd.DataFrame
        Ligand-receptor pairs.
    rtf_db : pd.DataFrame
        Receptor-TF database.
    tftg_db : pd.DataFrame
        TF-TG database.
    ccc_threshold : float
        CCC score threshold (fraction of max).
    fisher_threshold : float
        Fisher test p-value threshold.

    Returns
    -------
    HGNNInputs
        Container with all filtered inputs.
    """
    all_genes = exp_mat.index.tolist()
    max_weight = lr_pairs["Weight"].max()
    lr_filtered = lr_pairs[
        lr_pairs["Weight"] >= ccc_threshold * max_weight
    ].copy()

    tftg_filtered = tftg_db.copy()
    tftg_filtered.columns = ["from", "to"] + list(tftg_filtered.columns[2:])
    tftg_filtered = tftg_filtered[
        tftg_filtered["from"].isin(all_genes)
        & tftg_filtered["to"].isin(all_genes)
    ]
    tftg_filtered = tftg_filtered[tftg_filtered["from"] != tftg_filtered["to"]]

    if fisher_threshold < 1.0:
        tftg_filtered = find_regulator(
            tftg_filtered, all_genes, target_genes, fisher_threshold
        )

    rtf_filtered = rtf_db.copy()
    rtf_filtered.columns = ["from", "to"] + list(rtf_filtered.columns[2:])
    rtf_filtered = rtf_filtered[
        rtf_filtered["from"].isin(all_genes)
        & rtf_filtered["to"].isin(all_genes)
    ]
    rtf_filtered = rtf_filtered[rtf_filtered["from"] != rtf_filtered["to"]]

    G = nx.DiGraph()
    for _, row in rtf_filtered.iterrows():
        G.add_edge(row["from"], row["to"])

    receptors = (
        lr_filtered["Receptor"].unique().tolist()
        if "Receptor" in lr_filtered.columns
        else []
    )
    tfs = tftg_filtered["from"].unique().tolist()

    if receptors and tfs:
        ppr_df = personalized_pagerank(G, receptors, tfs)
        if len(ppr_df) > 0:
            valid_pairs = set(zip(ppr_df["regulator"], ppr_df["target"]))
            rtf_filtered = rtf_filtered[
                rtf_filtered.apply(
                    lambda r: (r["from"], r["to"]) in valid_pairs, axis=1
                )
            ]

    return HGNNInputs(
        exp_clu=exp_mat,
        rtf_filtered=rtf_filtered.reset_index(drop=True),
        tftg_filtered=tftg_filtered.reset_index(drop=True),
        lr_pairs=lr_filtered.reset_index(drop=True),
    )


def calculate_corr(exp_mat, db, type="Spearman", abss=True):
    """Calculate correlation between gene pairs in a database.

    Parameters
    ----------
    exp_mat : pd.DataFrame
        Expression matrix (cells x genes).
    db : pd.DataFrame
        Database with gene pairs (first two columns are 'from' and 'to').
    type : str
        Correlation type: 'Spearman', 'Pearson', or 'Kendall'.
    abss : bool
        If True, take absolute value of correlations.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [From, To, Correlation].
    """
    corr_df = db.iloc[:, :2].copy()
    corr_df.columns = ["From", "To"]

    # Pre-compute ranks for Spearman (vectorized)
    type_lower = type.lower()
    if type_lower == "spearman":
        from scipy.stats import rankdata
        ranked = exp_mat.apply(lambda x: rankdata(x), axis=0)
    else:
        ranked = exp_mat

    # Get unique genes needed
    unique_genes = set(corr_df["From"].unique()) | set(corr_df["To"].unique())
    available_genes = [g for g in unique_genes if g in ranked.columns]

    # Pre-compute correlation matrix for available genes
    if type_lower == "spearman" and len(available_genes) > 1:
        ranked_sub = ranked[available_genes].values
        corr_matrix = np.corrcoef(ranked_sub.T)
        gene_idx = {g: i for i, g in enumerate(available_genes)}
    else:
        gene_idx = {g: i for i, g in enumerate(available_genes)}
        corr_matrix = None

    # Compute correlations
    results = []
    for _, row in corr_df.iterrows():
        v1, v2 = row["From"], row["To"]
        if v1 not in gene_idx or v2 not in gene_idx:
            results.append(0.0)
            continue

        if corr_matrix is not None:
            i, j = gene_idx[v1], gene_idx[v2]
            corr = corr_matrix[i, j]
        else:
            x = ranked[v1].values
            y = ranked[v2].values
            if type_lower == "pearson":
                corr = np.corrcoef(x, y)[0, 1]
            else:
                corr = 0.0

        if abss:
            results.append(abs(corr))
        else:
            results.append(max(corr, 0))

    corr_df["Correlation"] = results
    return corr_df


def Generate_negative_3(df, seed, sample_n=0):
    """Generate negative samples by random combination.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns [Receptor, TF, TG].
    seed : int
        Random seed.
    sample_n : int
        Number of negative samples to generate. If 0, matches positives.

    Returns
    -------
    pd.DataFrame
        Negative samples.
    """
    N1 = set(df["Receptor"])
    N2 = set(df["TF"])
    N3 = set(df["TG"])
    all_combinations = list(itertools.product(N1, N2, N3))
    df_all = pd.DataFrame(all_combinations, columns=["Receptor", "TF", "TG"])
    df_pos = df[["Receptor", "TF", "TG"]]
    merged_df = df_all.merge(
        df_pos, on=["Receptor", "TF", "TG"], how="left", indicator=True
    )
    neg_samples = merged_df[merged_df["_merge"] == "left_only"].drop(
        columns="_merge"
    )
    if sample_n == 0:
        sample_n = len(df_pos)
    return neg_samples.sample(
        n=min(sample_n, len(neg_samples)), random_state=seed
    )


def Filter_RTTDB(
    df1,
    df2,
    thres1,
    thres2,
    genes,
    args,
    first="score",
    sample_scale=0.5,
    ood_frac=0.5,
):
    """Construct hypergraph and generate training samples.

    Parameters
    ----------
    df1 : pd.DataFrame
        Receptor-TF database with scores.
    df2 : pd.DataFrame
        TF-TG database with scores.
    thres1 : float
        Threshold for first filtering step.
    thres2 : float
        Threshold for second filtering step.
    genes : list
        All gene names.
    args : object
        Arguments object with seed attribute.
    first : str
        Filtering strategy: 'score', 'RecTF', or 'TFTG'.
    sample_scale : float
        Fraction of samples to use.
    ood_frac : float
        Fraction for out-of-distribution negative samples.

    Returns
    -------
    tuple
        (hypergraph, samples_df, all_df)
    """
    import dhg

    seed = args.seed if hasattr(args, "seed") else 42
    df1 = df1.copy()
    df2 = df2.copy()
    df1.columns = ["Receptor", "TF", "score1"]
    df2.columns = ["TF", "TG", "score2"]
    df1 = df1.sort_values(by="score1", ascending=False)
    df2 = df2.sort_values(by="score2", ascending=False)
    df_all = pd.merge(df1, df2, on="TF", how="inner")
    df_all = df_all[["Receptor", "TF", "TG", "score1", "score2"]]
    df_all["score"] = df_all["score1"] * df_all["score2"]
    df_all = df_all.sort_values(by="score", ascending=False)

    if first in ("RecTF", "TFTG"):
        n1 = int(len(df1) * thres1)
        n2 = int(len(df2) * thres1)
        df_pos = pd.merge(df1.head(n1), df2.head(n2), on="TF", how="inner")
        df_neg = pd.merge(df1.tail(n1), df2.tail(n2), on="TF", how="inner")
    else:
        n_selected = max(1, int(len(df_all) * thres1 * thres2))
        df_pos = df_all.head(n_selected)
        df_neg = df_all.tail(n_selected)

    df_hg = df_pos[["Receptor", "TF", "TG"]].copy()
    df_pos = df_pos[["Receptor", "TF", "TG"]].copy()
    df_neg = df_neg[["Receptor", "TF", "TG"]].copy()
    df_pos["label"] = 1
    df_neg1 = Generate_negative_3(
        df_all[["Receptor", "TF", "TG"]],
        seed,
        sample_n=round(len(df_pos) * ood_frac),
    )
    df_neg1["label"] = 0
    neg2_count = max(0, len(df_pos) - len(df_neg1))
    if neg2_count > 0 and len(df_neg) > 0:
        df_neg2 = df_neg.sample(
            n=min(neg2_count, len(df_neg)), random_state=seed
        )
        df_neg2["label"] = 0
    else:
        df_neg2 = pd.DataFrame(columns=["Receptor", "TF", "TG", "label"])
    df_pos = df_pos.sample(n=max(1, round(len(df_pos) * sample_scale)), random_state=seed)
    df_neg1 = df_neg1.sample(
        n=max(1, round(len(df_neg1) * sample_scale)), random_state=seed
    )
    if len(df_neg2) > 0:
        df_neg2 = df_neg2.sample(
            n=max(1, round(len(df_neg2) * sample_scale)), random_state=seed
        )
    df_samples = pd.concat([df_pos, df_neg1, df_neg2], axis=0)
    df_samples["label"] = df_samples["label"].astype(int)

    string_to_index = {s: i for i, s in enumerate(genes)}

    def _map(val):
        return string_to_index.get(val, val)

    # Use applymap (map for newer pandas) to convert string values to indices
    _apply = getattr(df_hg, "map", None) or df_hg.applymap
    df_hg = _apply(_map)
    _apply_all = getattr(
        df_all[["Receptor", "TF", "TG"]], "map", None
    ) or df_all[["Receptor", "TF", "TG"]].applymap
    df_all[["Receptor", "TF", "TG"]] = _apply_all(_map)
    _apply_samples = getattr(
        df_samples[["Receptor", "TF", "TG"]], "map", None
    ) or df_samples[["Receptor", "TF", "TG"]].applymap
    df_samples[["Receptor", "TF", "TG"]] = _apply_samples(_map)

    edge_list = [tuple(row) for row in df_hg.values]
    hg = dhg.Hypergraph(len(genes), edge_list)
    return hg, df_samples, df_all
