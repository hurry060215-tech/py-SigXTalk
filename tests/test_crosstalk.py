import numpy as np
import pandas as pd
import pytest
from pysigxtalk.crosstalk import (
    count_crosstalk,
    aggregate_causality,
    calculate_fidelity_matrix,
    calculate_specificity_matrix,
    calculate_pairwise,
)


@pytest.fixture
def sample_prs():
    return pd.DataFrame({
        "Receptor": ["R1", "R1", "R2", "R2", "R1"],
        "SSC": ["T1", "T2", "T1", "T2", "T1"],
        "Target": ["G1", "G1", "G1", "G2", "G2"],
        "Weight": [0.3, 0.2, 0.4, 0.1, 0.5],
    })


class TestCountCrosstalk:
    def test_count_target(self, sample_prs):
        result = count_crosstalk(sample_prs, data_type="Target")
        assert isinstance(result, pd.Series)
        assert result["G1"] == 3
        assert result["G2"] == 2


class TestAggregateCausality:
    def test_aggregate_by_target(self, sample_prs):
        result = aggregate_causality(sample_prs, data_type="Target")
        assert isinstance(result, pd.DataFrame)
        assert "Receptor" in result.columns
        assert "Target" in result.columns
        assert "Weight" in result.columns


class TestFidelityMatrix:
    def test_fidelity_ssc(self, sample_prs):
        result = calculate_fidelity_matrix(sample_prs, key_tg="G1")
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            row_sums = result.sum(axis=1)
            np.testing.assert_array_almost_equal(row_sums.values, 1.0, decimal=5)


class TestSpecificityMatrix:
    def test_specificity_ssc(self, sample_prs):
        result = calculate_specificity_matrix(sample_prs, key_rec="R1")
        assert isinstance(result, pd.DataFrame)


class TestCalculatePairwise:
    def test_pairwise_fid(self, sample_prs):
        trs = aggregate_causality(sample_prs, data_type="Target")
        result = calculate_pairwise(trs, type="Fid")
        assert isinstance(result, pd.DataFrame)
