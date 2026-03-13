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

Main responsibilities
---------------------
1. Build a county adjacency graph from a GeoDataFrame using Queen contiguity.
2. Convert adjacency matrices into normalized sparse tensors for GCN training.
3. Define a flexible multi-output GCN regressor.
4. Train and evaluate the GCN using fixed train/validation/test masks.
5. Summarize metrics overall and per target.
"""


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int, default=42
        Random seed used across Python, NumPy, and PyTorch.

    Returns
    -------
    None
        This function updates global random states in-place.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def get_device() -> torch.device:
    """
    Get the best available PyTorch device.

    Returns
    -------
    torch.device
        CUDA device if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def build_queen_adjacency(
    geo_df,
    id_column: Optional[str] = None,
    use_index: bool = True,
) -> Tuple[sparse.csr_matrix, libpysal.weights.W]:
    """
    Build a Queen contiguity adjacency matrix from a GeoDataFrame.

    Queen contiguity connects two polygons if they share either a border edge
    or a corner vertex. This is a common spatial graph construction strategy
    for county-level analysis.

    Parameters
    ----------
    geo_df : geopandas.GeoDataFrame
        GeoDataFrame containing one geometry per county.
    id_column : str or None, default=None
        Optional column to use as IDs in the PySAL weights object. If None,
        the GeoDataFrame index is used.
    use_index : bool, default=True
        Passed to libpysal so the caller can explicitly control index usage.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix of shape (n_nodes, n_nodes)
        Binary sparse adjacency matrix with one row/column per county.
    weights : libpysal.weights.W
        PySAL spatial weights object for additional diagnostics if needed.
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

    Self-loops allow each node to preserve its own features during graph
    convolution. In standard GCN formulations, the adjacency matrix is often
    augmented as A_tilde = A + I.

    Parameters
    ----------
    adjacency : scipy.sparse.csr_matrix of shape (n_nodes, n_nodes)
        Sparse adjacency matrix without or with self-loops.

    Returns
    -------
    scipy.sparse.csr_matrix of shape (n_nodes, n_nodes)
        Sparse adjacency matrix with self-loops added.
    """
    n_nodes = adjacency.shape[0]
    identity = sparse.eye(n_nodes, format="csr")
    return (adjacency + identity).tocsr()



def symmetric_normalize_adjacency(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Apply symmetric GCN normalization to a sparse adjacency matrix.

    The normalized adjacency is defined as
        D^(-1/2) A D^(-1/2)
    where D is the degree matrix.

    This is the standard normalization used by the original GCN formulation.

    Parameters
    ----------
    adjacency : scipy.sparse.csr_matrix of shape (n_nodes, n_nodes)
        Sparse adjacency matrix, typically with self-loops already added.

    Returns
    -------
    scipy.sparse.csr_matrix of shape (n_nodes, n_nodes)
        Symmetrically normalized sparse adjacency matrix.
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

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix or compatible sparse matrix
        Sparse matrix to convert.

    Returns
    -------
    torch.Tensor
        Sparse COO tensor containing the same nonzero structure and values.
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
    Build all graph support objects needed for the GCN.

    This helper creates the raw Queen adjacency, optionally adds self-loops,
    applies symmetric normalization, and converts the result to a PyTorch sparse
    tensor for model training.

    Parameters
    ----------
    geo_df : geopandas.GeoDataFrame
        GeoDataFrame containing county geometries.
    id_column : str or None, default=None
        Optional ID column used by PySAL.
    add_loops : bool, default=True
        Whether to add self-loops before normalization.
    use_index : bool, default=True
        Passed to PySAL Queen contiguity builder.

    Returns
    -------
    adjacency_raw : scipy.sparse.csr_matrix
        Raw binary adjacency matrix.
    adjacency_norm : scipy.sparse.csr_matrix
        Symmetrically normalized adjacency matrix used by the GCN.
    adjacency_torch : torch.Tensor
        PyTorch sparse adjacency tensor for model training.
    weights : libpysal.weights.W
        PySAL spatial weights object for diagnostics and inspection.
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

    Parameters
    ----------
    adjacency : scipy.sparse.csr_matrix of shape (n_nodes, n_nodes)
        Sparse adjacency matrix.

    Returns
    -------
    pandas.DataFrame
        One-row DataFrame with node count, edge count, degree statistics, and
        isolated node count.
    """
    degrees = np.asarray(adjacency.sum(axis=1)).flatten()
    n_nodes = adjacency.shape[0]
    n_edges = int(adjacency.nnz / 2)

    summary = pd.DataFrame(
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
    return summary



def make_masks(
    n_nodes: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create boolean node masks for train, validation, and test sets.

    Parameters
    ----------
    n_nodes : int
        Total number of graph nodes.
    train_idx : np.ndarray
        Integer indices for training nodes.
    val_idx : np.ndarray
        Integer indices for validation nodes.
    test_idx : np.ndarray
        Integer indices for test nodes.

    Returns
    -------
    train_mask : torch.Tensor of shape (n_nodes,)
        Boolean mask marking training nodes.
    val_mask : torch.Tensor of shape (n_nodes,)
        Boolean mask marking validation nodes.
    test_mask : torch.Tensor of shape (n_nodes,)
        Boolean mask marking test nodes.
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

    This helper assumes the graph node ordering will match the stacked order:
    first all training nodes, then validation nodes, then test nodes.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
        Split feature matrices.
    y_train, y_val, y_test : np.ndarray
        Split target matrices.

    Returns
    -------
    X_all : np.ndarray
        Concatenated feature matrix of shape (n_nodes, n_features).
    y_all : np.ndarray
        Concatenated target matrix of shape (n_nodes, n_targets).
    train_idx : np.ndarray
        Integer indices for training nodes.
    val_idx : np.ndarray
        Integer indices for validation nodes.
    test_idx : np.ndarray
        Integer indices for test nodes.
    node_order : np.ndarray
        Integer node order from 0 to n_nodes - 1.
    split_ids : np.ndarray
        String array identifying each row as train, val, or test.
    split_sizes : np.ndarray
        Array with the sizes of the three splits.
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

    Parameters
    ----------
    X : np.ndarray of shape (n_nodes, n_features)
        Node feature matrix.
    y : np.ndarray of shape (n_nodes, n_targets)
        Node target matrix.
    device : torch.device or None, default=None
        Optional device to move tensors onto.

    Returns
    -------
    X_t : torch.Tensor
        Float tensor version of X.
    y_t : torch.Tensor
        Float tensor version of y.
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

    This layer performs
        H_next = A_norm @ H @ W + b
    where A_norm is a normalized sparse adjacency matrix.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    bias : bool, default=True
        Whether to include an additive bias term.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Apply one graph convolution step.

        Parameters
        ----------
        x : torch.Tensor of shape (n_nodes, in_features)
            Input node feature matrix.
        adjacency : torch.Tensor
            Sparse normalized adjacency tensor of shape (n_nodes, n_nodes).

        Returns
        -------
        torch.Tensor of shape (n_nodes, out_features)
            Updated node representations.
        """
        x = self.linear(x)
        x = torch.sparse.mm(adjacency, x)
        return x


class GCNRegressor(nn.Module):
    """
    Multi-output Graph Convolutional Network for regression.

    The model uses a configurable number of graph convolution layers. Hidden
    layers apply ReLU activation and optional dropout. The final layer outputs
    one regression value per target.

    Parameters
    ----------
    in_features : int
        Number of input node features.
    hidden_dim : int
        Width of hidden graph convolution layers.
    out_features : int
        Number of regression targets.
    num_layers : int, default=2
        Total number of graph convolution layers.
    dropout : float, default=0.0
        Dropout probability applied after hidden activations.
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
        """
        Run a forward pass through the GCN.

        Parameters
        ----------
        x : torch.Tensor of shape (n_nodes, n_features)
            Input node feature matrix.
        adjacency : torch.Tensor
            Sparse normalized adjacency tensor.

        Returns
        -------
        torch.Tensor of shape (n_nodes, n_targets)
            Predicted target values for all nodes.
        """
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

    Attributes
    ----------
    train_loss : list of float
        Training loss values by epoch.
    val_loss : list of float
        Validation loss values by epoch.
    best_epoch : int
        Epoch index with the best validation loss.
    best_val_loss : float
        Best validation loss achieved during training.
    """

    train_loss: List[float]
    val_loss: List[float]
    best_epoch: int
    best_val_loss: float



def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.

    Returns
    -------
    int
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """
    Build an Adam optimizer for GCN training.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters will be optimized.
    lr : float, default=1e-3
        Learning rate.
    weight_decay : float, default=0.0
        L2 regularization strength.

    Returns
    -------
    torch.optim.Optimizer
        Adam optimizer.
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
    Train a GCN for multi-output regression with early stopping.

    Parameters
    ----------
    model : torch.nn.Module
        GCN model to train.
    adjacency : torch.Tensor
        Sparse normalized adjacency tensor.
    X : torch.Tensor
        Full node feature matrix.
    y : torch.Tensor
        Full node target matrix.
    train_mask : torch.Tensor
        Boolean mask indicating training nodes.
    val_mask : torch.Tensor
        Boolean mask indicating validation nodes.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    criterion : torch.nn.Module or None, default=None
        Loss function. If None, MSELoss is used.
    n_epochs : int, default=300
        Maximum number of epochs.
    patience : int, default=30
        Early stopping patience based on validation loss.
    verbose : bool, default=False
        Whether to print training progress.

    Returns
    -------
    model : torch.nn.Module
        Model restored to the best validation checkpoint.
    history : TrainHistory
        Training and validation loss history.
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

    Parameters
    ----------
    model : torch.nn.Module
        Trained GCN model.
    adjacency : torch.Tensor
        Sparse normalized adjacency tensor.
    X : torch.Tensor
        Full node feature matrix.
    mask : torch.Tensor or None, default=None
        Optional boolean mask selecting a subset of nodes.

    Returns
    -------
    np.ndarray
        Predicted values for all nodes or the masked subset.
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
    Compute overall multi-output regression metrics.

    Metrics are aggregated across all targets and samples.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples, n_targets)
        Ground-truth target values.
    y_pred : np.ndarray of shape (n_samples, n_targets)
        Predicted target values.

    Returns
    -------
    pandas.DataFrame
        One-row DataFrame containing RMSE, MAE, and R2.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")

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

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples, n_targets)
        Ground-truth target values.
    y_pred : np.ndarray of shape (n_samples, n_targets)
        Predicted target values.
    target_names : list of str or None, default=None
        Optional target names. If None, generic names are created.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per target and columns for RMSE, MAE, and R2.
    """
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

    Parameters
    ----------
    history : TrainHistory
        Training history object returned by train_gcn.

    Returns
    -------
    pandas.DataFrame
        DataFrame with epoch, training loss, and validation loss columns.
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

    Parameters
    ----------
    adjacency : torch.Tensor
        Sparse normalized adjacency tensor.
    X : torch.Tensor
        Full node feature matrix.
    y : torch.Tensor
        Full node target matrix.
    train_mask : torch.Tensor
        Boolean mask for training nodes.
    val_mask : torch.Tensor
        Boolean mask for validation nodes.
    test_mask : torch.Tensor or None, default=None
        Boolean mask for test nodes. Only required when evaluate_test=True.
    hidden_dim : int, default=64
        Hidden layer width.
    num_layers : int, default=2
        Number of graph convolution layers.
    dropout : float, default=0.0
        Dropout probability.
    lr : float, default=1e-3
        Learning rate.
    weight_decay : float, default=0.0
        Weight decay strength.
    n_epochs : int, default=300
        Maximum number of training epochs.
    patience : int, default=30
        Early stopping patience.
    seed : int, default=42
        Random seed.
    target_names : list of str or None, default=None
        Optional target names for per-target metrics.
    scaler_y : fitted scaler or None, default=None
        Target scaler used during preprocessing. If provided, metrics are
        computed on both scaled and inverse-transformed original target scales.
    verbose : bool, default=False
        Whether to print training progress.
    evaluate_test : bool, default=False
        Whether to evaluate on the test split. Keep False during baseline and
        ablation experiments, and set to True only for the final locked model.

    Returns
    -------
    dict
        Dictionary containing the trained model, history, predictions, config,
        and evaluation tables. Test-related outputs are None unless
        evaluate_test=True.
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

    # ----------------------------
    # Validation predictions/metrics
    # ----------------------------
    val_pred = predict_gcn(model, adjacency, X, mask=val_mask)
    y_val = y[val_mask].detach().cpu().numpy()

    val_metrics_scaled = evaluate_regression(y_val, val_pred)
    val_metrics_per_target_scaled = evaluate_regression_per_target(
        y_val, val_pred, target_names
    )

    # Default outputs for original-scale validation values
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

    # ----------------------------
    # Test predictions/metrics
    # ----------------------------
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

    results = {
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
        # Validation outputs
        "y_val": y_val,
        "val_pred": val_pred,
        "val_metrics_scaled": val_metrics_scaled,
        "val_metrics_per_target_scaled": val_metrics_per_target_scaled,
        "y_val_original": y_val_original,
        "val_pred_original": val_pred_original,
        "val_metrics": val_metrics,
        "val_metrics_per_target": val_metrics_per_target,
        # Test outputs (None unless evaluate_test=True)
        "y_test": y_test,
        "test_pred": test_pred,
        "test_metrics_scaled": test_metrics_scaled,
        "test_metrics_per_target_scaled": test_metrics_per_target_scaled,
        "y_test_original": y_test_original,
        "test_pred_original": test_pred_original,
        "test_metrics": test_metrics,
        "test_metrics_per_target": test_metrics_per_target,
    }

    return results



def results_dict_to_frame(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of experiment result dictionaries into a summary DataFrame.

    Parameters
    ----------
    results_list : list of dict
        Output dictionaries returned by run_gcn_experiment.

    Returns
    -------
    pandas.DataFrame
        Tidy DataFrame summarizing configuration and headline metrics.
    """
    rows: List[Dict[str, Any]] = []

    for result in results_list:
        row = dict(result["config"])
        row["val_RMSE"] = result["val_metrics"].iloc[0]["RMSE"]
        row["val_MAE"] = result["val_metrics"].iloc[0]["MAE"]
        row["val_R2"] = result["val_metrics"].iloc[0]["R2"]
        row["test_RMSE"] = result["test_metrics"].iloc[0]["RMSE"]
        row["test_MAE"] = result["test_metrics"].iloc[0]["MAE"]
        row["test_R2"] = result["test_metrics"].iloc[0]["R2"]
        rows.append(row)

    return pd.DataFrame(rows)
