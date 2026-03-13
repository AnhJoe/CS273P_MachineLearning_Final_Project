import copy
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Inputs
    ------
    seed : int
        Random seed value.

    Outputs
    -------
    None
        This function updates global random states in-place.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Select the best available PyTorch device.

    Inputs
    ------
    None

    Outputs
    -------
    torch.device
        CUDA device if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_graph_from_weights(
    w,
    X: np.ndarray,
    y: np.ndarray,
    node_indices: Optional[np.ndarray] = None,
) -> Data:
    """
    Build a PyTorch Geometric Data object from spatial weights and feature/target arrays.

    Inputs
    ------
    w : libpysal.weights.W
        Spatial weights object defining county adjacency.
    X : np.ndarray
        Node feature matrix of shape (n_nodes, n_features).
    y : np.ndarray
        Target matrix of shape (n_nodes, n_targets).
    node_indices : np.ndarray or None, default=None
        Optional subset of node indices to include in the graph.
        If None, all nodes are included.

    Outputs
    -------
    Data
        PyTorch Geometric Data object containing node features, edge indices, and targets.
    """
    if node_indices is None:
        node_indices = np.arange(len(X))

    X_subset = X[node_indices]
    y_subset = y[node_indices]

    edge_list = []
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_indices)}

    for i, node_id in enumerate(node_indices):
        if node_id in w.neighbors:
            for neighbor_id in w.neighbors[node_id]:
                if neighbor_id in id_to_idx:
                    j = id_to_idx[neighbor_id]
                    edge_list.append([i, j])

    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    data = Data(
        x=torch.tensor(X_subset, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y_subset, dtype=torch.float32),
    )

    return data


class GAT(nn.Module):
    """
    Graph Attention Network for multi-output regression on spatial data.

    This architecture uses multi-head attention to learn heterogeneous spatial influence
    weights between neighboring counties when aggregating information across the graph.

    Inputs
    ------
    input_dim : int
        Number of input node features.
    hidden_dim : int
        Hidden layer dimensionality.
    output_dim : int
        Number of output targets per node.
    num_heads : int, default=4
        Number of attention heads in the first GAT layer.
    num_heads_out : int, default=1
        Number of attention heads in the second GAT layer.
    dropout : float, default=0.2
        Dropout probability applied to attention coefficients and features.
    negative_slope : float, default=0.2
        Negative slope for LeakyReLU activation in attention mechanism.

    Outputs
    -------
    nn.Module
        A PyTorch GAT model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4,
        num_heads_out: int = 1,
        dropout: float = 0.2,
        negative_slope: float = 0.2,
    ):
        super().__init__()

        self.dropout = dropout

        self.conv1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            negative_slope=negative_slope,
            concat=True,
        )

        self.conv2 = GATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=num_heads_out,
            dropout=dropout,
            negative_slope=negative_slope,
            concat=False,
        )

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ):
        """
        Forward pass through the GAT network.

        Inputs
        ------
        x : torch.Tensor
            Node feature matrix of shape (n_nodes, input_dim).
        edge_index : torch.Tensor
            Edge connectivity in COO format of shape (2, n_edges).
        return_attention_weights : bool, default=False
            If True, returns attention weights along with predictions.

        Outputs
        -------
        out : torch.Tensor
            Predicted node outputs of shape (n_nodes, output_dim).
        attention_weights : tuple or None
            If return_attention_weights=True, returns (edge_index, attention_weights).
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, (edge_index_1, alpha_1) = self.conv1(
            x, edge_index, return_attention_weights=True
        )
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x, (edge_index_2, alpha_2) = self.conv2(
            x, edge_index, return_attention_weights=True
        )
        x = F.elu(x)

        out = self.fc_out(x)

        if return_attention_weights:
            attention_weights = {
                "layer1": (edge_index_1, alpha_1),
                "layer2": (edge_index_2, alpha_2),
            }
            return out, attention_weights
        else:
            return out


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Inputs
    ------
    model : nn.Module
        PyTorch model.

    Outputs
    -------
    int
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """
    Build an Adam optimizer for the provided model.

    Inputs
    ------
    model : nn.Module
        PyTorch model whose parameters will be optimized.
    learning_rate : float, default=1e-3
        Optimizer learning rate.
    weight_decay : float, default=1e-4
        L2 regularization strength.

    Outputs
    -------
    torch.optim.Optimizer
        Configured Adam optimizer.
    """
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def run_epoch_gat(
    model: nn.Module,
    data: Data,
    criterion,
    mask: Optional[torch.Tensor] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> float:
    """
    Run one full epoch for training or evaluation on graph data.

    Inputs
    ------
    model : nn.Module
        GAT model.
    data : Data
        PyTorch Geometric Data object containing the full graph.
    criterion : loss function
        PyTorch loss function such as nn.MSELoss().
    mask : torch.Tensor or None, default=None
        Boolean mask indicating which nodes to include in loss computation.
        If None, all nodes are used.
    optimizer : torch.optim.Optimizer or None, default=None
        Optimizer used for training. If None, evaluation mode is used.
    device : torch.device or None, default=None
        Device on which computation is performed.

    Outputs
    -------
    float
        Average loss over the masked nodes.
    """
    if device is None:
        device = get_device()

    data = data.to(device)

    if optimizer is not None:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(optimizer is not None):
        preds = model(data.x, data.edge_index)

        if mask is not None:
            preds = preds[mask]
            targets = data.y[mask]
        else:
            targets = data.y

        loss = criterion(preds, targets)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss.item()


def train_gat(
    model: nn.Module,
    train_data: Data,
    val_data: Data,
    criterion,
    optimizer: torch.optim.Optimizer,
    max_epochs: int = 500,
    patience: int = 50,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, List], int, float]:
    """
    Train a GAT model with early stopping based on validation loss.

    Inputs
    ------
    model : nn.Module
        GAT model to train.
    train_data : Data
        Training graph data.
    val_data : Data
        Validation graph data.
    criterion : loss function
        PyTorch loss function such as nn.MSELoss().
    optimizer : torch.optim.Optimizer
        Optimizer used for training.
    max_epochs : int, default=500
        Maximum number of epochs.
    patience : int, default=50
        Number of epochs to wait without validation improvement before stopping.
    device : torch.device or None, default=None
        Device on which computation is performed.

    Outputs
    -------
    model : nn.Module
        Best model based on validation loss.
    history : dict
        Dictionary containing epoch numbers, training losses, and validation losses.
    best_epoch : int
        Epoch with the best validation performance.
    best_val_loss : float
        Best validation loss observed.
    """
    if device is None:
        device = get_device()

    model = model.to(device)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch_gat(
            model=model,
            data=train_data,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss = run_epoch_gat(
            model=model,
            data=val_data,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)

    return model, history, best_epoch, best_val_loss


def predict_gat(
    model: nn.Module,
    data: Data,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Generate predictions from a trained GAT model.

    Inputs
    ------
    model : nn.Module
        Trained GAT model.
    data : Data
        PyTorch Geometric Data object.
    device : torch.device or None, default=None
        Device on which computation is performed.

    Outputs
    -------
    np.ndarray
        Predicted outputs of shape (n_nodes, output_dim).
    """
    if device is None:
        device = get_device()

    model.eval()
    model = model.to(device)
    data = data.to(device)

    with torch.no_grad():
        preds = model(data.x, data.edge_index)

    return preds.cpu().numpy()


def extract_attention_weights(
    model: nn.Module,
    data: Data,
    layer: str = "layer1",
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract attention weights from a trained GAT model for analysis.

    Inputs
    ------
    model : nn.Module
        Trained GAT model.
    data : Data
        PyTorch Geometric Data object.
    layer : str, default="layer1"
        Which attention layer to extract. Options: "layer1", "layer2".
    device : torch.device or None, default=None
        Device on which computation is performed.

    Outputs
    -------
    edge_index : np.ndarray
        Edge connectivity of shape (2, n_edges).
    attention_weights : np.ndarray
        Attention coefficients of shape (n_edges, n_heads) or (n_edges,).
    """
    if device is None:
        device = get_device()

    model.eval()
    model = model.to(device)
    data = data.to(device)

    with torch.no_grad():
        _, attention_dict = model(
            data.x, data.edge_index, return_attention_weights=True
        )

    edge_index, alpha = attention_dict[layer]

    edge_index = edge_index.cpu().numpy()
    alpha = alpha.cpu().numpy()

    return edge_index, alpha


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics across all targets.

    Inputs
    ------
    y_true : np.ndarray
        Ground truth targets of shape (n_samples, n_targets).
    y_pred : np.ndarray
        Predicted targets of shape (n_samples, n_targets).

    Outputs
    -------
    dict
        Dictionary containing RMSE and R² scores.
    """
    from sklearn.metrics import mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": rmse,
        "r2": r2,
    }


def evaluate_regression_per_target(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute per-target regression metrics.

    Inputs
    ------
    y_true : np.ndarray
        Ground truth targets of shape (n_samples, n_targets).
    y_pred : np.ndarray
        Predicted targets of shape (n_samples, n_targets).
    target_names : list[str] or None, default=None
        Names of the target variables. If None, uses generic names.

    Outputs
    -------
    pd.DataFrame
        DataFrame with per-target RMSE and R² scores.
    """
    import pandas as pd
    from sklearn.metrics import mean_squared_error, r2_score

    n_targets = y_true.shape[1]

    if target_names is None:
        target_names = [f"target_{i}" for i in range(n_targets)]

    results = []
    for i, name in enumerate(target_names):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true[:, i], y_pred[:, i])

        results.append({
            "target": name,
            "rmse": rmse,
            "r2": r2,
        })

    return pd.DataFrame(results)
