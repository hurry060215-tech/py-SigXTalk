import numpy as np
import pandas as pd
import pytest
from pysigxtalk.prs import compute_prs, filter_results


class TestComputePrs:
    def test_basic_prs(self):
        np.random.seed(42)
        n_cells = 50
        exp = pd.DataFrame(
            np.random.rand(5, n_cells),
            index=["Rec1", "Rec2", "TF1", "TG1", "TG2"],
        )
        pathways = pd.DataFrame({
            "Receptor": ["Rec1", "Rec1", "Rec2"],
            "TF": ["TF1", "TF1", "TF1"],
            "TG": ["TG1", "TG2", "TG1"],
            "pred_label": [0.8, 0.9, 0.7],
        })
        result = compute_prs(exp, pathways, engine="sklearn", n_estimators=10)
        assert isinstance(result, pd.DataFrame)
        assert "Receptor" in result.columns
        assert "SSC" in result.columns
        assert "Target" in result.columns
        assert "Weight" in result.columns
        assert len(result) > 0

    def test_prs_lightgbm(self):
        np.random.seed(42)
        n_cells = 50
        exp = pd.DataFrame(
            np.random.rand(5, n_cells),
            index=["Rec1", "Rec2", "TF1", "TG1", "TG2"],
        )
        pathways = pd.DataFrame({
            "Receptor": ["Rec1", "Rec1", "Rec2"],
            "TF": ["TF1", "TF1", "TF1"],
            "TG": ["TG1", "TG2", "TG1"],
            "pred_label": [0.8, 0.9, 0.7],
        })
        result = compute_prs(exp, pathways, engine="lightgbm", n_estimators=10)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestFilterResults:
    def test_filter_by_threshold(self):
        results = pd.DataFrame({
            "Receptor": ["R1", "R2", "R3"],
            "SSC": ["T1", "T2", "T3"],
            "Target": ["G1", "G2", "G3"],
            "Weight": [0.1, 0.5, 0.9],
        })
        # threshold=0.5 means keep weights >= 0.5 * max(0.9) = 0.45
        # So 0.5 and 0.9 pass
        filtered = filter_results(results, prs_threshold=0.5)
        assert len(filtered) == 2

    def test_remove_mitochondrial(self):
        results = pd.DataFrame({
            "Receptor": ["R1", "MT-ND1", "RPS3"],
            "SSC": ["T1", "T2", "T3"],
            "Target": ["G1", "G2", "G3"],
            "Weight": [0.5, 0.5, 0.5],
        })
        filtered = filter_results(results, remove_genes=True)
        assert "MT-ND1" not in filtered["Receptor"].values
