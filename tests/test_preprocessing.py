import numpy as np
import pandas as pd
import pytest
from pysigxtalk.preprocessing import (
    personalized_pagerank,
    fisher_test,
    get_exp_clu,
    normalize_data,
    HGNNInputs,
    calculate_corr,
    Filter_RTTDB,
)


class TestFisherTest:
    def test_fisher_significant(self):
        background = list(range(100))
        subset1 = list(range(20))
        subset2 = list(range(15))
        _, pval = fisher_test(subset1, subset2, background)
        assert pval < 0.05

    def test_fisher_no_overlap(self):
        background = list(range(100))
        subset1 = list(range(10))
        subset2 = list(range(50, 60))
        _, pval = fisher_test(subset1, subset2, background)
        assert pval > 0.05


class TestPersonalizedPageRank:
    def test_basic_ppr(self):
        import networkx as nx

        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
        regulators = ["A"]
        targets = ["C"]
        result = personalized_pagerank(G, regulators, targets, damping=0.85)
        assert isinstance(result, pd.DataFrame)
        assert "regulator" in result.columns
        assert "target" in result.columns
        assert "ppr" in result.columns
        assert len(result) > 0


class TestGetExpClu:
    def test_filters_low_expression(self):
        data = np.zeros((5, 10))
        data[0, :] = 1.0
        data[1, 0] = 1.0
        data[2, :] = 2.0
        adata_df = pd.DataFrame(data, index=["E1", "E2", "E3", "E4", "E5"])
        result = get_exp_clu(adata_df, cutoff=0.05)
        assert "E1" in result.index
        assert "E3" in result.index


class TestNormalizeData:
    def test_normalize(self):
        df = pd.DataFrame(
            {"cell1": [10, 20], "cell2": [30, 60]},
            index=["gene1", "gene2"],
        )
        result = normalize_data(df.T, log2=False)
        assert result.shape == df.T.shape


class TestCalculateCorr:
    def test_spearman_correlation(self):
        exp = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [2, 4, 6, 8, 10],
                "C": [5, 4, 3, 2, 1],
            }
        )
        db = pd.DataFrame({"from": ["A", "A"], "to": ["B", "C"]})
        result = calculate_corr(exp, db, type="Spearman", abss=True)
        assert len(result) == 2
        assert result.iloc[0]["Correlation"] > 0.9  # A-B highly correlated


class TestFilterRTTDB:
    def test_basic_filter(self):
        rtf = pd.DataFrame(
            {
                "Receptor": ["R1", "R1", "R2"],
                "TF": ["T1", "T2", "T1"],
                "score1": [0.8, 0.6, 0.7],
            }
        )
        tftg = pd.DataFrame(
            {
                "TF": ["T1", "T1", "T2"],
                "TG": ["G1", "G2", "G1"],
                "score2": [0.9, 0.5, 0.3],
            }
        )
        genes = ["R1", "R2", "T1", "T2", "G1", "G2"]
        args = type("Args", (), {"seed": 42})()
        hg, samples, all_df = Filter_RTTDB(
            rtf, tftg, thres1=0.5, thres2=0.5, genes=genes, args=args
        )
        assert hg is not None
        assert len(samples) > 0
