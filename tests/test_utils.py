import numpy as np
import pandas as pd
import pytest
from pysigxtalk.utils import nonlinear_sum, trimean, calculate_average_expression


class TestNonlinearSum:
    def test_sum_mode(self):
        values = np.array([0.1, 0.2, 0.3])
        result = nonlinear_sum(values, type="sum")
        assert np.isclose(result, 0.6)

    def test_prob_mode(self):
        values = np.array([0.1, 0.2])
        result = nonlinear_sum(values, type="prob")
        assert np.isclose(result, 0.28)

    def test_copula_mode(self):
        values = np.array([0.1, 0.2])
        result = nonlinear_sum(values, type="copula")
        assert np.isclose(result, 0.26 / 0.98)

    def test_single_value(self):
        values = np.array([0.5])
        result = nonlinear_sum(values, type="sum")
        assert np.isclose(result, 0.5)

    def test_empty_array(self):
        values = np.array([])
        result = nonlinear_sum(values, type="sum")
        assert np.isclose(result, 0.0)


class TestTrimean:
    def test_trimean_basic(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        result = trimean(values, nonzero=False)
        assert np.isclose(result, 2.5)

    def test_trimean_with_zeros(self):
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = trimean(values, nonzero=True)
        assert np.isclose(result, 2.5)

    def test_trimean_all_zeros(self):
        values = np.array([0.0, 0.0, 0.0])
        result = trimean(values, nonzero=True)
        assert np.isclose(result, 0.0)


class TestCalculateAverageExpression:
    def test_average_expression(self):
        exp_mat = pd.DataFrame(
            {"cell1": [1.0, 2.0], "cell2": [3.0, 4.0]},
            index=["geneA", "geneB"],
        )
        result = calculate_average_expression("geneA", exp_mat, nonzero=True)
        assert np.isclose(result, 2.0)
