import pandas as pd
import pytest

from pysigxtalk.databases import filter_db, load_databases


class TestFilterDb:
    def test_filters_to_present_genes(self):
        db = pd.DataFrame({"from": ["A", "B", "C"], "to": ["B", "C", "D"]})
        all_genes = ["A", "B", "C"]
        result = filter_db(db, all_genes)
        assert set(result["from"].unique()).issubset(set(all_genes))
        assert set(result["to"].unique()).issubset(set(all_genes))

    def test_removes_self_regulatory(self):
        db = pd.DataFrame({"from": ["A", "B"], "to": ["A", "C"]})
        all_genes = ["A", "B", "C"]
        result = filter_db(db, all_genes)
        assert len(result) == 1
        assert result.iloc[0]["from"] == "B"

    def test_empty_result(self):
        db = pd.DataFrame({"from": ["X"], "to": ["Y"]})
        all_genes = ["A", "B"]
        result = filter_db(db, all_genes)
        assert len(result) == 0


class TestLoadDatabases:
    def test_load_human(self):
        rtf, tftg = load_databases(species="human")
        assert isinstance(rtf, pd.DataFrame)
        assert isinstance(tftg, pd.DataFrame)
        assert "from" in rtf.columns
        assert "to" in rtf.columns

    def test_load_mouse(self):
        rtf, tftg = load_databases(species="mouse")
        assert isinstance(rtf, pd.DataFrame)
        assert isinstance(tftg, pd.DataFrame)

    def test_invalid_species(self):
        with pytest.raises(ValueError):
            load_databases(species="invalid")
