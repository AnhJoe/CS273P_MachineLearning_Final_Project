import copy
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


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


class FlexibleMLP(nn.Module):
    """
    Configurable multilayer perceptron for multi-output regression.

    Inputs
    ------
    input_dim : int
        Number of input features.
    hidden_dims : list[int]
        List of hidden layer widths. Example: [64, 32].
    output_dim : int
        Number of output targets.
    dropout : float, default=0.1
        Dropout probability applied after each hidden layer.
    activation : str, default="relu"
        Activation function name. Currently supports: "relu".

    Outputs
    -------
    nn.Module
        A PyTorch MLP model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        if activation.lower() != "relu":
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the network.

        Inputs
        ------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Outputs
        -------
        torch.Tensor
            Predicted outputs of shape (batch_size, output_dim).
        """
        return self.model(x)


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


def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> float:
    """
    Run one full epoch over a dataloader.

    If an optimizer is provided, the model is trained.
    If optimizer is None, the model is evaluated without gradient updates.

    Inputs
    ------
    model : nn.Module
        PyTorch model.
    loader : DataLoader
        Dataloader returning (X_batch, y_batch).
    criterion : loss function
        PyTorch loss function such as nn.MSELoss().
    optimizer : torch.optim.Optimizer or None, default=None
        Optimizer used for training. If None, evaluation mode is used.
    device : torch.device or None, default=None
        Device on which computation is performed.

    Outputs
    -------
    float
        Average loss over all samples in the dataloader.
    """
    if device is None:
        device = get_device()

    if optimizer is not None:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_samples = 0

    with torch.set_grad_enabled(optimizer is not None):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

    return total_loss / n_samples


def train_mlp(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer: torch.optim.Optimizer,
    max_epochs: int = 300,
    patience: int = 25,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, List[float]], int, float]:
    """
    Train an MLP with early stopping based on validation loss.

    Inputs
    ------
    model : nn.Module
        PyTorch model to train.
    train_loader : DataLoader
        Training dataloader.
    val_loader : DataLoader
        Validation dataloader.
    criterion : loss function
        PyTorch loss function such as nn.MSELoss().
    optimizer : torch.optim.Optimizer
        Optimizer used for training.
    max_epochs : int, default=300
        Maximum number of epochs.
    patience : int, default=25
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
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
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


def predict_mlp(
    model: nn.Module,
    loader,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Generate predictions from a trained model over a dataloader.

    Inputs
    ------
    model : nn.Module
        Trained PyTorch model.
    loader : DataLoader
        Dataloader returning (X_batch, y_batch) or compatible tuples.
    device : torch.device or None, default=None
        Device on which computation is performed.

    Outputs
    -------
    np.ndarray
        Model predictions stacked across all batches with shape (n_samples, n_outputs).
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    preds_all = []

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            preds_all.append(preds.cpu().numpy())

    return np.vstack(preds_all)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute aggregate regression metrics across all outputs.

    Inputs
    ------
    y_true : np.ndarray
        Ground-truth targets of shape (n_samples, n_outputs).
    y_pred : np.ndarray
        Predicted targets of shape (n_samples, n_outputs).

    Outputs
    -------
    dict
        Dictionary containing overall MSE, RMSE, MAE, and R2.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def evaluate_regression_per_target(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute regression metrics separately for each target.

    Inputs
    ------
    y_true : np.ndarray
        Ground-truth targets of shape (n_samples, n_outputs).
    y_pred : np.ndarray
        Predicted targets of shape (n_samples, n_outputs).
    target_names : list[str]
        Names of target variables in output order.

    Outputs
    -------
    dict
        Nested dictionary mapping each target name to its MSE, RMSE, MAE, and R2.
    """
    results = {}

    for i, name in enumerate(target_names):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]

        mse = np.mean((y_true_i - y_pred_i) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true_i - y_pred_i))

        ss_res = np.sum((y_true_i - y_pred_i) ** 2)
        ss_tot = np.sum((y_true_i - np.mean(y_true_i)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot)

        results[name] = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
        }

    return results