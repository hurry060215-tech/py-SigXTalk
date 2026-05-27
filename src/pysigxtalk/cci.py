import pandas as pd
import anndata


def infer_cci(
    adata: anndata.AnnData,
    target_type: str,
    species: str = "human",
    min_cells: int = 10,
    resource_name: str = "consensus",
) -> pd.DataFrame:
    """Infer cell-cell communication using LIANA+.

    Replaces R's CellChat-based Infer_CCI() function.

    Args:
        adata: AnnData object with single-cell expression data.
        target_type: Target cell type name (receiver).
        species: 'human' or 'mouse'.
        min_cells: Minimum number of cells per cell type.
        resource_name: LIANA resource to use.

    Returns:
        DataFrame with columns: Ligand, Receptor, Weight, Source, Target.
    """
    import liana
    from liana.method import cellphonedb

    # Filter cell types with enough cells
    cell_counts = adata.obs["cell_type"].value_counts()
    valid_types = cell_counts[cell_counts >= min_cells].index.tolist()
    adata = adata[adata.obs["cell_type"].isin(valid_types)].copy()

    # Run LIANA+ CellPhoneDB method
    cellphonedb(
        adata,
        groupby="cell_type",
        resource_name=resource_name,
        expr_prop=0.1,
        verbose=False,
    )

    # Extract results targeting the receiver cell type
    liana_results = adata.uns["liana_res"]

    # Filter for interactions where target is the receiver cell type
    result = liana_results[liana_results["target"] == target_type].copy()

    # Standardize output format
    result = result.rename(columns={
        "ligand_complex": "Ligand",
        "receptor_complex": "Receptor",
        "lr_means": "Weight",
        "source": "Source",
        "target": "Target",
    })

    result = result[["Ligand", "Receptor", "Weight", "Source", "Target"]]
    result = result.sort_values("Weight", ascending=False).reset_index(drop=True)

    return result


def extract_lr_prob(
    cci_results: pd.DataFrame,
    source_type: str = None,
    target_type: str = None,
    pv_threshold: float = 0.05,
) -> pd.DataFrame:
    """Extract ligand-receptor pair probabilities from CCI results.

    Args:
        cci_results: Output from infer_cci().
        source_type: Filter by source cell type (optional).
        target_type: Filter by target cell type (optional).
        pv_threshold: P-value threshold for filtering.

    Returns:
        DataFrame with columns: Ligand, Receptor, Weight.
    """
    result = cci_results.copy()

    if source_type is not None:
        result = result[result["Source"] == source_type]
    if target_type is not None:
        result = result[result["Target"] == target_type]

    # Handle multi-subunit receptors (split weight equally)
    expanded_rows = []
    for _, row in result.iterrows():
        rec = row["Receptor"]
        if "_" in rec:
            subunits = rec.split("_")
            for subunit in subunits:
                new_row = row.copy()
                new_row["Receptor"] = subunit
                new_row["Weight"] = row["Weight"] / len(subunits)
                expanded_rows.append(new_row)
        else:
            expanded_rows.append(row)

    result = pd.DataFrame(expanded_rows)

    # Aggregate duplicates
    result = (
        result.groupby(["Ligand", "Receptor"])["Weight"]
        .sum()
        .reset_index()
    )

    return result.sort_values("Weight", ascending=False).reset_index(drop=True)
