import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from typing import List, Optional
import dhg

from .preprocessing import HGNNInputs, Filter_RTTDB, calculate_corr


class HGNNPredictor(nn.Module):
    def __init__(self, in_channels, hgnn_channels, linear_channels, use_bn=None, drop_rate=0.5):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        hgnn_channel_list = [in_channels] + hgnn_channels
        for _idx in range(1, len(hgnn_channel_list) - 1):
            self.encoder_layers.append(HGNNP_layer(hgnn_channel_list[_idx-1], hgnn_channel_list[_idx], use_bn=use_bn, drop_rate=drop_rate))
        self.encoder_layers.append(HGNNP_layer(hgnn_channel_list[-2], hgnn_channel_list[-1], use_bn=use_bn, drop_rate=drop_rate, is_last=True))
        self.rec_MLP = MLP_layer(layer_sizes=[hgnn_channel_list[-1]]+linear_channels, use_bn=True)
        self.tf_MLP = MLP_layer(layer_sizes=[hgnn_channel_list[-1]]+linear_channels, use_bn=True)
        self.tg_MLP = MLP_layer(layer_sizes=[hgnn_channel_list[-1]]+linear_channels, use_bn=True)

    def encode(self, x, hg):
        for layer in self.encoder_layers:
            x = layer(x, hg)
        x = F.elu(x)
        return x

    def decode(self, rec_out, tf_out, tg_out):
        prob1 = torch.cosine_similarity(rec_out, tf_out, dim=1)
        prob2 = torch.cosine_similarity(tf_out, tg_out, dim=1)
        prob = torch.abs(prob1*prob2)
        return prob.view(-1,1)

    def forward(self, x, hg, training_set):
        all_embeddings = self.encode(x, hg)
        rec_embd = self.rec_MLP(all_embeddings)
        tf_embd = self.tf_MLP(all_embeddings)
        tg_embd = self.tg_MLP(all_embeddings)
        self.rec_output = rec_embd[training_set[:, 0]]
        self.tf_output = tf_embd[training_set[:, 1]]
        self.tg_output = tg_embd[training_set[:, 2]]
        return self.decode(self.rec_output, self.tf_output, self.tg_output)


class HGNNP_layer(nn.Module):
    def __init__(self, in_channels, out_channels, leaky_alpha=0.1, bias=True, use_bn=True, drop_rate=0.5, is_last=True):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act_func = nn.LeakyReLU(negative_slope=leaky_alpha)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, hg):
        x = self.theta(x)
        x = hg.v2v(x, aggr="mean")
        if not self.is_last:
            x = self.act_func(x)
            if self.bn is not None:
                x = self.bn(x)
            x = F.dropout(x, p=0.01)
            x = F.normalize(x, p=2, dim=1)
        return x


class MLP_layer(nn.Module):
    def __init__(self, layer_sizes, use_bn=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_bn = use_bn
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.norms.append(nn.BatchNorm1d(layer_sizes[i+1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.use_bn:
                x = self.norms[i](x)
            x = F.elu(x)
        x = self.layers[-1](x)
        return x


class _DataFrameDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        features = sample[["Receptor", "TF", "TG"]].values
        labels = sample["label"]
        return torch.tensor(features), torch.tensor(labels)


def _set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _train_model(args, Exp, model, hypergraph, samples, device):
    Exp_tensor = torch.tensor(Exp.values, dtype=torch.float32).to(device)
    hypergraph = hypergraph.to(device)
    model = model.to(device)
    # Check if stratified split is feasible (each class needs >= 2 members)
    label_counts = samples["label"].value_counts()
    can_stratify = (label_counts.min() >= 2)
    if can_stratify:
        training_data, val_data = train_test_split(
            samples, test_size=1-args["train_size"]-0.001,
            stratify=samples["label"], random_state=args["seed"]
        )
    else:
        training_data, val_data = train_test_split(
            samples, test_size=1-args["train_size"]-0.001,
            random_state=args["seed"]
        )
    training_dataset = _DataFrameDataset(training_data)
    training_load = DataLoader(training_dataset, batch_size=args["batch_size"], shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=args["lr"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)
    for epoch_i in range(args["epochs"]):
        running_loss = 0.0
        for train_x, train_y in training_load:
            model.train()
            optimizer.zero_grad()
            train_x = train_x.to(device)
            train_y = train_y.to(torch.float).to(device).view(-1, 1)
            pred = model(Exp_tensor, hypergraph, train_x)
            pred = F.relu(pred)
            loss = F.binary_cross_entropy(pred, train_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
    return model


def _predict(Exp, model, input_samples, hypergraph, genes, device):
    input_tensor = torch.tensor(Exp.values, dtype=torch.float32).to(device)
    hypergraph = hypergraph.to(device)
    model.eval()
    pred_results = input_samples[["Receptor", "TF", "TG"]].copy()
    pred_tensor = torch.tensor(pred_results.values).to(device)
    with torch.no_grad():
        predictions = model(input_tensor, hypergraph, pred_tensor)
        predictions = F.relu(predictions)
    pred_results["pred_label"] = predictions.cpu().numpy()
    mapping = {i: genes[i] for i in range(len(genes))}
    for col in pred_results.columns[:-1]:
        pred_results[col] = pred_results[col].map(mapping)
    return pred_results


def run_hgnn(inputs, epochs=100, batch_size=64, lr=0.01, hgnn_dims=None,
             linear_dims=None, seed=42, device="cuda", thres=None, train_size=0.75):
    """Run HGNN model and return predicted pathways."""
    if hgnn_dims is None:
        hgnn_dims = [256, 128]
    if linear_dims is None:
        linear_dims = [64, 32]
    if thres is None:
        thres = [0.1, 0.1]
    _set_seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    gene_all = inputs.exp_clu.index.tolist()
    df_rt = calculate_corr(inputs.exp_clu.T, inputs.rtf_filtered, type="Spearman", abss=False)
    df_tftg = calculate_corr(inputs.exp_clu.T, inputs.tftg_filtered, type="Spearman", abss=False)
    hg, all_samples, input_all = Filter_RTTDB(
        df_rt, df_tftg,
        thres1=thres[0], thres2=thres[1], genes=gene_all,
        args=type("Args", (), {"seed": seed})(),
        first="score", sample_scale=0.75, ood_frac=0.5,
    )
    model = HGNNPredictor(
        in_channels=inputs.exp_clu.shape[1],
        hgnn_channels=hgnn_dims,
        linear_channels=linear_dims,
    )
    args = {"epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed, "train_size": train_size}
    model = _train_model(args, inputs.exp_clu, model, hg, all_samples, dev)
    pred_results = _predict(inputs.exp_clu, model, input_all, hg, gene_all, dev)
    return pred_results
