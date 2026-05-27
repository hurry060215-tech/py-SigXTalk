import numpy as np
import pandas as pd
import pytest
import torch
from pysigxtalk.hgnn import HGNNPredictor, HGNNP_layer, MLP_layer


class TestHGNNPredictor:
    def test_model_init(self):
        model = HGNNPredictor(
            in_channels=100,
            hgnn_channels=[256, 128],
            linear_channels=[64, 32],
        )
        assert isinstance(model, torch.nn.Module)

    def test_model_forward(self):
        model = HGNNPredictor(
            in_channels=10,
            hgnn_channels=[16, 8],
            linear_channels=[4, 2],
        )
        x = torch.randn(20, 10)
        import dhg
        hg = dhg.Hypergraph(20, [(0, 1, 2), (1, 2, 3)])
        training_set = torch.tensor([[0, 1, 2], [1, 2, 3]])
        output = model(x, hg, training_set)
        assert output.shape[0] == 2


class TestHGNNPLayer:
    def test_layer_forward(self):
        layer = HGNNP_layer(in_channels=10, out_channels=16)
        x = torch.randn(20, 10)
        import dhg
        hg = dhg.Hypergraph(20, [(0, 1, 2), (1, 2, 3)])
        output = layer(x, hg)
        assert output.shape == (20, 16)


class TestMLPLayer:
    def test_mlp_forward(self):
        mlp = MLP_layer(layer_sizes=[10, 16, 4])
        x = torch.randn(5, 10)
        output = mlp(x)
        assert output.shape == (5, 4)
