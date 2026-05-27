from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_databases(species: str = "human") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load RTF and TFTG databases for the given species.

    Parameters
    ----------
    species : str
        Either 'human' or 'mouse'.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (rtf, tftg) DataFrames with 'from' and 'to' columns.
    """
    if species not in ("human", "mouse"):
        raise ValueError(f"species must be 'human' or 'mouse', got '{species}'")

    rtf = pd.read_csv(_DATA_DIR / f"RTF_{species}.csv")
    tftg = pd.read_csv(_DATA_DIR / f"TFT_{species}.csv")

    # Standardize column names - first two columns become 'from' and 'to'
    rtf.columns = ["from", "to"] + list(rtf.columns[2:])
    tftg.columns = ["from", "to"] + list(tftg.columns[2:])

    return rtf, tftg


def filter_db(db: pd.DataFrame, all_genes: list[str]) -> pd.DataFrame:
    """Filter database to genes present in dataset. Removes self-regulatory pairs.

    Parameters
    ----------
    db : pd.DataFrame
        Database DataFrame with 'from' and 'to' columns (first two columns).
    all_genes : list[str]
        List of gene symbols to keep.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only rows where both 'from' and 'to' are in
        all_genes, and with self-regulatory pairs removed.
    """
    result = db.copy()
    col_from = result.columns[0]
    col_to = result.columns[1]

    gene_set = set(all_genes)
    result = result[
        result[col_from].isin(gene_set) & result[col_to].isin(gene_set)
    ]

    # Remove self-regulatory pairs
    result = result[result[col_from] != result[col_to]]

    return result.reset_index(drop=True)
