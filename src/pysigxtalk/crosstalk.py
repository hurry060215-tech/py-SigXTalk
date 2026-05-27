import numpy as np
import pandas as pd
from .utils import nonlinear_sum


def count_crosstalk(cc_results, key_genes=None, data_type="Target", verbose=True):
    """Count the number of crosstalk pathways for each gene."""
    if data_type == "Target":
        col = "Target"
    elif data_type == "SSC":
        col = "SSC"
    else:
        raise ValueError("data_type must be 'Target' or 'SSC'")
    counts = cc_results.groupby(col).size()
    if key_genes is not None:
        counts = counts.reindex(key_genes, fill_value=0)
    return counts


def aggregate_causality(cc_results, data_type="Target"):
    """Calculate TRS by aggregating PRS values."""
    result = cc_results.copy()
    max_w = result["Weight"].max()
    if max_w > 0:
        result["Weight"] = result["Weight"] / max_w * 0.999
    if data_type == "Target":
        grouped = result.groupby(["Receptor", "Target"])["Weight"].apply(
            lambda x: nonlinear_sum(x.values, type="sum")
        ).reset_index()
    elif data_type == "SSC":
        grouped = result.groupby(["Receptor", "SSC"])["Weight"].apply(
            lambda x: nonlinear_sum(x.values, type="sum")
        ).reset_index()
    else:
        raise ValueError("data_type must be 'Target' or 'SSC'")
    return grouped


def calculate_fidelity_matrix(cc_results, key_tg, mode="SSC"):
    """Compute fidelity matrix for a target gene."""
    subset = cc_results[cc_results["Target"] == key_tg].copy()
    if subset.empty:
        return pd.DataFrame()
    if mode == "SSC":
        pivot = subset.pivot_table(index="Receptor", columns="SSC", values="Weight", fill_value=0)
        row_sums = pivot.sum(axis=1)
        row_sums = row_sums.replace(0, 1)
        return pivot.div(row_sums, axis=0)
    elif mode == "Receptor":
        pivot = subset.pivot_table(index="SSC", columns="Receptor", values="Weight", fill_value=0)
        col_sums = pivot.sum(axis=0)
        col_sums = col_sums.replace(0, 1)
        return pivot.div(col_sums, axis=1)
    elif mode == "all":
        pivot = subset.pivot_table(index="Receptor", columns="SSC", values="Weight", fill_value=0)
        total = pivot.values.sum()
        if total == 0:
            return pivot
        return pivot / total
    else:
        raise ValueError("mode must be 'SSC', 'Receptor', or 'all'")


def calculate_specificity_matrix(cc_results, key_rec, mode="SSC"):
    """Compute specificity matrix for a receptor."""
    subset = cc_results[cc_results["Receptor"] == key_rec].copy()
    if subset.empty:
        return pd.DataFrame()
    if mode == "SSC":
        pivot = subset.pivot_table(index="Target", columns="SSC", values="Weight", fill_value=0)
        row_sums = pivot.sum(axis=1)
        row_sums = row_sums.replace(0, 1)
        return pivot.div(row_sums, axis=0)
    elif mode == "Target":
        pivot = subset.pivot_table(index="SSC", columns="Target", values="Weight", fill_value=0)
        col_sums = pivot.sum(axis=0)
        col_sums = col_sums.replace(0, 1)
        return pivot.div(col_sums, axis=1)
    elif mode == "all":
        pivot = subset.pivot_table(index="Target", columns="SSC", values="Weight", fill_value=0)
        total = pivot.values.sum()
        if total == 0:
            return pivot
        return pivot / total
    else:
        raise ValueError("mode must be 'SSC', 'Target', or 'all'")


def calculate_pathway_fidelity(cc_results, key_tg, key_ssc):
    """Fidelity of each receptor for a given target and SSC."""
    subset = cc_results[
        (cc_results["Target"] == key_tg) & (cc_results["SSC"] == key_ssc)
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    total = subset["Weight"].sum()
    if total == 0:
        subset["Fidelity"] = 0.0
    else:
        subset["Fidelity"] = subset["Weight"] / total
    return subset[["Receptor", "Fidelity"]].sort_values("Fidelity", ascending=False)


def calculate_pathway_specificity(cc_results, key_rec, key_ssc):
    """Specificity of each target for a given receptor and SSC."""
    subset = cc_results[
        (cc_results["Receptor"] == key_rec) & (cc_results["SSC"] == key_ssc)
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    total = subset["Weight"].sum()
    if total == 0:
        subset["Specificity"] = 0.0
    else:
        subset["Specificity"] = subset["Weight"] / total
    return subset[["Target", "Specificity"]].sort_values("Specificity", ascending=False)


def calculate_pairwise(cc_pair_results, key_gene=None, type="Fid"):
    """Calculate pairwise fidelity or specificity from TRS-aggregated results."""
    result = cc_pair_results.copy()
    if key_gene is not None:
        if "Target" in result.columns:
            result = result[result["Target"] == key_gene]
        elif "SSC" in result.columns:
            result = result[result["SSC"] == key_gene]
    if type == "Fid":
        if "Target" in result.columns:
            grouped = result.groupby("Target")
        else:
            grouped = result.groupby("SSC")
        result["Weight"] = grouped["Weight"].transform(lambda x: x / x.sum())
    elif type == "Spe":
        grouped = result.groupby("Receptor")
        result["Weight"] = grouped["Weight"].transform(lambda x: x / x.sum())
    return result
