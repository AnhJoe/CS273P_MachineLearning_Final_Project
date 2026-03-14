from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import copy
import random

import libpysal
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


"""
Utility functions and model code for county-level graph convolutional regression.

This module is designed for the SVI project notebook so that graph construction,
model definition, training, prediction, and evaluation logic stay out of the
notebook. The notebook can then focus on analysis, tables, and figures.
"""


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Get the best available PyTorch device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_queen_adjacency(
    geo_df,
    id_column: Optional[str] = None,
    use_index: bool = True,
) -> Tuple[sparse.csr_matrix, libpysal.weights.W]:
    """
    Build a Queen contiguity adjacency matrix from a GeoDataFrame.
    """
    if id_column is not None:
        weights = libpysal.weights.Queen.from_dataframe(
            geo_df,
            ids=geo_df[id_column].tolist(),
            use_index=use_index,
        )
    else:
        weights = libpysal.weights.Queen.from_dataframe(
            geo_df,
            use_index=use_index,
        )

    adjacency = weights.sparse.tocsr()
    return adjacency, weights


def add_self_loops(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Add self-loops to an adjacency matrix.
    """
    n_nodes = adjacency.shape[0]
    identity = sparse.eye(n_nodes, format="csr")
    return (adjacency + identity).tocsr()


def symmetric_normalize_adjacency(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Apply symmetric GCN normalization to a sparse adjacency matrix.
    """
    degrees = np.asarray(adjacency.sum(axis=1)).flatten()

    degrees_inv_sqrt = np.zeros_like(degrees, dtype=np.float64)
    mask = degrees > 0
    degrees_inv_sqrt[mask] = np.power(degrees[mask], -0.5)

    d_inv_sqrt = sparse.diags(degrees_inv_sqrt)
    adjacency_norm = d_inv_sqrt @ adjacency @ d_inv_sqrt
    return adjacency_norm.tocsr()


def scipy_sparse_to_torch_sparse(matrix: sparse.csr_matrix) -> torch.Tensor:
    """
    Convert a SciPy sparse matrix into a PyTorch sparse COO tensor.
    """
    matrix = matrix.tocoo()
    indices = np.vstack((matrix.row, matrix.col))
    indices_t = torch.tensor(indices, dtype=torch.long)
    values_t = torch.tensor(matrix.data, dtype=torch.float32)
    shape = torch.Size(matrix.shape)
    return torch.sparse_coo_tensor(indices_t, values_t, shape).coalesce()


def build_gcn_support(
    geo_df,
    id_column: Optional[str] = None,
    add_loops: bool = True,
    use_index: bool = True,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, torch.Tensor, libpysal.weights.W]:
    """
    Build graph support objects needed for the GCN.
    """
    adjacency_raw, weights = build_queen_adjacency(
        geo_df=geo_df,
        id_column=id_column,
        use_index=use_index,
    )

    adjacency_work = add_self_loops(adjacency_raw) if add_loops else adjacency_raw
    adjacency_norm = symmetric_normalize_adjacency(adjacency_work)
    adjacency_torch = scipy_sparse_to_torch_sparse(adjacency_norm)

    return adjacency_raw, adjacency_norm, adjacency_torch, weights


def summarize_graph(adjacency: sparse.csr_matrix) -> pd.DataFrame:
    """
    Compute a compact summary of graph structure.
    """
    degrees = np.asarray(adjacency.sum(axis=1)).flatten()
    n_nodes = adjacency.shape[0]
    n_edges = int(adjacency.nnz / 2)

    return pd.DataFrame(
        {
            "n_nodes": [n_nodes],
            "n_edges_undirected": [n_edges],
            "degree_min": [degrees.min()],
            "degree_mean": [degrees.mean()],
            "degree_median": [np.median(degrees)],
            "degree_max": [degrees.max()],
            "n_isolated_nodes": [int((degrees == 0).sum())],
        }
    )


def make_masks(
    n_nodes: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create boolean node masks for train, validation, and test sets.
    """
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def stack_split_arrays(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate split arrays into full node-level matrices and return indices.
    """
    X_all = np.vstack([X_train, X_val, X_test]).astype(np.float32)
    y_all = np.vstack([y_train, y_val, y_test]).astype(np.float32)

    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)

    node_order = np.arange(X_all.shape[0])
    split_ids = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    split_sizes = np.array([n_train, n_val, n_test])

    return X_all, y_all, train_idx, val_idx, test_idx, node_order, split_ids, split_sizes


def numpy_to_torch_features_targets(
    X: np.ndarray,
    y: np.ndarray,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert NumPy feature and target arrays to PyTorch float tensors.
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    if device is not None:
        X_t = X_t.to(device)
        y_t = y_t.to(device)

    return X_t, y_t


class GraphConvolution(nn.Module):
    """
    Simple graph convolution layer using a precomputed normalized adjacency.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.sparse.mm(adjacency, x)
        return x


class GCNRegressor(nn.Module):
    """
    Graph Convolutional Network for regression.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = nn.ReLU()

        layers: List[GraphConvolution] = []

        if num_layers == 1:
            layers.append(GraphConvolution(in_features, out_features))
        else:
            layers.append(GraphConvolution(in_features, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(GraphConvolution(hidden_dim, hidden_dim))
            layers.append(GraphConvolution(hidden_dim, out_features))

        self.layers = nn.ModuleList(layers)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, adjacency)

            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout_layer(x)

        return x


@dataclass
class TrainHistory:
    """
    Container for training history.
    """

    train_loss: List[float]
    val_loss: List[float]
    best_epoch: int
    best_val_loss: float


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """
    Build an Adam optimizer for GCN training.
    """
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_gcn(
    model: nn.Module,
    adjacency: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    n_epochs: int = 300,
    patience: int = 30,
    verbose: bool = False,
) -> Tuple[nn.Module, TrainHistory]:
    """
    Train a GCN for regression with early stopping.
    """
    if criterion is None:
        criterion = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = -1
    wait = 0

    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X, adjacency)
        train_loss = criterion(y_pred[train_mask], y[train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X, adjacency)
            val_loss = criterion(y_val_pred[val_mask], y[val_mask])

        train_loss_value = float(train_loss.item())
        val_loss_value = float(val_loss.item())
        train_losses.append(train_loss_value)
        val_losses.append(val_loss_value)

        if verbose and (epoch % 25 == 0 or epoch == n_epochs - 1):
            print(
                f"Epoch {epoch:04d} | "
                f"train_loss={train_loss_value:.6f} | "
                f"val_loss={val_loss_value:.6f}"
            )

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch:04d}.")
            break

    model.load_state_dict(best_state)
    history = TrainHistory(
        train_loss=train_losses,
        val_loss=val_losses,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
    )
    return model, history


def predict_gcn(
    model: nn.Module,
    adjacency: torch.Tensor,
    X: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Generate predictions from a trained GCN.
    """
    model.eval()
    with torch.no_grad():
        preds = model(X, adjacency)
        if mask is not None:
            preds = preds[mask]
    return preds.detach().cpu().numpy()


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Compute regression metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return pd.DataFrame(
        {
            "RMSE": [rmse],
            "MAE": [mae],
            "R2": [r2],
        }
    )


def evaluate_regression_per_target(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute regression metrics separately for each target.
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_targets = y_true.shape[1]

    if target_names is None:
        target_names = [f"target_{i}" for i in range(n_targets)]

    rows: List[Dict[str, Any]] = []
    for i in range(n_targets):
        rows.append(
            {
                "Target": target_names[i],
                "RMSE": np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
                "MAE": mean_absolute_error(y_true[:, i], y_pred[:, i]),
                "R2": r2_score(y_true[:, i], y_pred[:, i]),
            }
        )

    return pd.DataFrame(rows)


def history_to_dataframe(history: TrainHistory) -> pd.DataFrame:
    """
    Convert training history into a tidy DataFrame for plotting.
    """
    return pd.DataFrame(
        {
            "epoch": np.arange(len(history.train_loss)),
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
        }
    )


def run_gcn_experiment(
    adjacency: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: Optional[torch.Tensor] = None,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    n_epochs: int = 300,
    patience: int = 30,
    seed: int = 42,
    target_names: Optional[List[str]] = None,
    scaler_y=None,
    verbose: bool = False,
    evaluate_test: bool = False,
) -> Dict[str, Any]:
    """
    Run one complete GCN experiment from initialization through evaluation.
    """
    set_seed(seed)
    device = X.device

    model = GCNRegressor(
        in_features=X.shape[1],
        hidden_dim=hidden_dim,
        out_features=y.shape[1],
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)

    model, history = train_gcn(
        model=model,
        adjacency=adjacency,
        X=X,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        optimizer=optimizer,
        n_epochs=n_epochs,
        patience=patience,
        verbose=verbose,
    )

    val_pred = predict_gcn(model, adjacency, X, mask=val_mask)
    y_val = y[val_mask].detach().cpu().numpy()

    val_metrics_scaled = evaluate_regression(y_val, val_pred)
    val_metrics_per_target_scaled = evaluate_regression_per_target(
        y_val, val_pred, target_names
    )

    y_val_original = None
    val_pred_original = None

    if scaler_y is not None:
        y_val_original = scaler_y.inverse_transform(y_val)
        val_pred_original = scaler_y.inverse_transform(val_pred)

        val_metrics = evaluate_regression(y_val_original, val_pred_original)
        val_metrics_per_target = evaluate_regression_per_target(
            y_val_original, val_pred_original, target_names
        )
    else:
        val_metrics = val_metrics_scaled
        val_metrics_per_target = val_metrics_per_target_scaled

    y_test = None
    test_pred = None
    test_metrics_scaled = None
    test_metrics_per_target_scaled = None
    y_test_original = None
    test_pred_original = None
    test_metrics = None
    test_metrics_per_target = None

    if evaluate_test:
        if test_mask is None:
            raise ValueError("test_mask must be provided when evaluate_test=True.")

        test_pred = predict_gcn(model, adjacency, X, mask=test_mask)
        y_test = y[test_mask].detach().cpu().numpy()

        test_metrics_scaled = evaluate_regression(y_test, test_pred)
        test_metrics_per_target_scaled = evaluate_regression_per_target(
            y_test, test_pred, target_names
        )

        if scaler_y is not None:
            y_test_original = scaler_y.inverse_transform(y_test)
            test_pred_original = scaler_y.inverse_transform(test_pred)

            test_metrics = evaluate_regression(y_test_original, test_pred_original)
            test_metrics_per_target = evaluate_regression_per_target(
                y_test_original, test_pred_original, target_names
            )
        else:
            test_metrics = test_metrics_scaled
            test_metrics_per_target = test_metrics_per_target_scaled

    return {
        "model": model,
        "history": history,
        "history_df": history_to_dataframe(history),
        "config": {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "n_epochs": n_epochs,
            "patience": patience,
            "seed": seed,
            "n_parameters": count_parameters(model),
        },
        "y_val": y_val,
        "val_pred": val_pred,
        "val_metrics_scaled": val_metrics_scaled,
        "val_metrics_per_target_scaled": val_metrics_per_target_scaled,
        "y_val_original": y_val_original,
        "val_pred_original": val_pred_original,
        "val_metrics": val_metrics,
        "val_metrics_per_target": val_metrics_per_target,
        "y_test": y_test,
        "test_pred": test_pred,
        "test_metrics_scaled": test_metrics_scaled,
        "test_metrics_per_target_scaled": test_metrics_per_target_scaled,
        "y_test_original": y_test_original,
        "test_pred_original": test_pred_original,
        "test_metrics": test_metrics,
        "test_metrics_per_target": test_metrics_per_target,
    }


def results_dict_to_frame(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of experiment result dictionaries into a summary DataFrame.
    """
    rows: List[Dict[str, Any]] = []

    for result in results_list:
        row = dict(result["config"])
        row["val_RMSE"] = result["val_metrics"].iloc[0]["RMSE"]
        row["val_MAE"] = result["val_metrics"].iloc[0]["MAE"]
        row["val_R2"] = result["val_metrics"].iloc[0]["R2"]

        if result["test_metrics"] is not None:
            row["test_RMSE"] = result["test_metrics"].iloc[0]["RMSE"]
            row["test_MAE"] = result["test_metrics"].iloc[0]["MAE"]
            row["test_R2"] = result["test_metrics"].iloc[0]["R2"]
        else:
            row["test_RMSE"] = np.nan
            row["test_MAE"] = np.nan
            row["test_R2"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)