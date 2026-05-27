import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pysigxtalk.visualization import (
    plot_counts,
    plot_counts_histogram,
    plot_counts_bar,
    plot_fid_spe,
    plot_heatmap,
    plot_ridgeline,
    plot_cci_chord,
    plot_cci_circle,
    plot_alluvial,
    plot_circular_bar,
    plot_chord,
    plot_signal_contribution,
    plot_rec_tg_heatmap,
)


@pytest.fixture
def sample_prs():
    np.random.seed(42)
    return pd.DataFrame({
        "Receptor": np.random.choice(["R1", "R2", "R3"], 50),
        "SSC": np.random.choice(["T1", "T2", "T3"], 50),
        "Target": np.random.choice(["G1", "G2", "G3"], 50),
        "Weight": np.random.rand(50),
    })


@pytest.fixture
def sample_cci():
    return pd.DataFrame({
        "Ligand": ["L1", "L2", "L3", "L1", "L2"],
        "Receptor": ["R1", "R2", "R3", "R2", "R1"],
        "Weight": [0.8, 0.6, 0.5, 0.4, 0.3],
        "Source": ["S1", "S2", "S1", "S2", "S1"],
        "Target": ["T1", "T1", "T2", "T2", "T1"],
    })


class TestPlotCounts:
    def test_returns_figure(self, sample_prs):
        result = plot_counts(sample_prs, data_type="Target")
        assert result is not None
        plt.close("all")


class TestPlotCountsHistogram:
    def test_returns_figure(self, sample_prs):
        result = plot_counts_histogram(sample_prs, data_type="Target")
        assert result is not None
        plt.close("all")


class TestPlotCountsBar:
    def test_returns_figure(self, sample_prs):
        result = plot_counts_bar(sample_prs, data_type="Target", top_percent=50)
        assert result is not None
        plt.close("all")


class TestPlotFidSpe:
    def test_returns_figure(self, sample_prs):
        result = plot_fid_spe(sample_prs, key_tg="G1", threshold=0.0)
        assert result is not None
        plt.close("all")


class TestPlotHeatmap:
    def test_returns_figure(self, sample_prs):
        result = plot_heatmap(sample_prs, gene_used="G1", genetype="Target")
        assert result is not None
        plt.close("all")


class TestPlotRidgeline:
    def test_returns_figure(self, sample_prs):
        result = plot_ridgeline(sample_prs, key_tg="G1")
        assert result is not None
        plt.close("all")


class TestPlotCciChord:
    def test_returns_figure(self, sample_cci):
        result = plot_cci_chord(sample_cci, topk=5)
        assert result is not None


class TestPlotCciCircle:
    def test_returns_figure(self, sample_cci):
        result = plot_cci_circle(sample_cci, topk=5)
        assert result is not None
        plt.close("all")


class TestPlotAlluvial:
    def test_returns_figure(self, sample_prs):
        result = plot_alluvial(sample_prs, key_tg="G1", min_weight=0.0)
        assert result is not None


class TestPlotCircularBar:
    def test_returns_figure(self, sample_prs):
        result = plot_circular_bar(sample_prs, topk=3)
        assert result is not None
        plt.close("all")


class TestPlotChord:
    def test_returns_figure(self, sample_prs):
        mat = pd.DataFrame(np.random.rand(3, 3), index=["A", "B", "C"], columns=["X", "Y", "Z"])
        result = plot_chord(mat)
        assert result is not None


class TestPlotSignalContribution:
    def test_returns_figure(self, sample_prs):
        trs = sample_prs.groupby(["Receptor", "Target"])["Weight"].sum().reset_index()
        exp = pd.DataFrame(np.random.rand(10, 50), index=["R1","R2","R3","T1","T2","T3","G1","G2","G3","G10"])
        result = plot_signal_contribution(trs, exp, key_tg="G1")
        assert result is not None
        plt.close("all")


class TestPlotRecTgHeatmap:
    def test_returns_figure(self, sample_prs):
        trs = sample_prs.groupby(["Receptor", "Target"])["Weight"].sum().reset_index()
        exp = pd.DataFrame(np.random.rand(10, 50), index=["R1","R2","R3","T1","T2","T3","G1","G2","G3","G10"])
        result = plot_rec_tg_heatmap(trs, exp, key_tg="G1")
        assert result is not None
        plt.close("all")
