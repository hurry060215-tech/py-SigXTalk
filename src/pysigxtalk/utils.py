import numpy as np
import pandas as pd


def nonlinear_sum(values: np.ndarray, type: str = "sum") -> float:
    """Nonlinear aggregation of pathway weights."""
    if len(values) == 0:
        return 0.0
    values = np.asarray(values, dtype=np.float64)
    if type == "sum":
        return float(np.sum(values))
    elif type == "prob":
        result = 1.0
        for v in values:
            result *= 1.0 - v
        return 1.0 - result
    elif type == "copula":
        csum = values[0]
        for i in range(1, len(values)):
            a, b = csum, values[i]
            csum = (a + b - 2 * a * b) / (1 - a * b)
        return float(csum)
    else:
        raise ValueError(f"Unknown type: {type}. Use 'sum', 'prob', or 'copula'.")


def trimean(values: np.ndarray, nonzero: bool = True) -> float:
    """Trimean statistic: mean of Q1, 2*Q2, Q3."""
    values = np.asarray(values, dtype=np.float64)
    if nonzero:
        values = values[values > 0]
    if len(values) == 0:
        return 0.0
    q1 = np.percentile(values, 25)
    q2 = np.percentile(values, 50)
    q3 = np.percentile(values, 75)
    return float((q1 + 2 * q2 + q3) / 4)


def calculate_average_expression(
    gene: str, exp_mat: pd.DataFrame, nonzero: bool = True
) -> float:
    """Calculate average expression for a gene using trimean."""
    if gene not in exp_mat.index:
        return 0.0
    values = exp_mat.loc[gene].values
    return trimean(values, nonzero=nonzero)
