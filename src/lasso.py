import random
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import Lasso


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python and NumPy.

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


def run_lasso_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    alpha: float,
    seed: int = 42,
    max_iter: int = 20000,
    scaler_y: Optional[object] = None,
) -> Dict:
    """
    Fit a Lasso regression model at a given alpha and evaluate it on a held-out split.

    Inputs
    ------
    X_train : np.ndarray
        Standardized training feature matrix.
    y_train : np.ndarray
        Standardized training targets, shape (n_samples,) or (n_samples, 1).
    X_eval : np.ndarray
        Standardized evaluation feature matrix (validation or test).
    y_eval : np.ndarray
        Standardized evaluation targets, shape (n_samples,) or (n_samples, 1).
    alpha : float
        L1 regularization strength.
    seed : int, default=42
        Random seed for reproducibility.
    max_iter : int, default=20000
        Maximum number of coordinate-descent iterations.
    scaler_y : fitted sklearn scaler or None, default=None
        If provided, predictions and targets are inverse-transformed back to
        the original log1p(FEMA IHP) scale before computing evaluation
        metrics, matching the convention used by the MLP/GCN pipelines. If
        None, metrics are reported on the standardized scale.

    Outputs
    -------
    dict
        Dictionary containing the fitted model, alpha, number of non-zero
        coefficients, and evaluation metrics (mse, rmse, mae, r2).
    """
    set_seed(seed)

    model = Lasso(alpha=alpha, max_iter=max_iter, random_state=seed)
    model.fit(X_train, np.ravel(y_train))

    y_pred = model.predict(X_eval).reshape(-1, 1)
    y_true = np.asarray(y_eval).reshape(-1, 1)

    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred)
        y_true = scaler_y.inverse_transform(y_true)

    metrics = evaluate_regression(y_true, y_pred)

    return {
        "model": model,
        "alpha": alpha,
        "n_nonzero_coef": count_nonzero_coefficients(model),
        **metrics,
    }


def count_nonzero_coefficients(model: Lasso) -> int:
    """
    Count the number of non-zero coefficients (selected features) in a fitted Lasso model.

    Inputs
    ------
    model : sklearn.linear_model.Lasso
        Fitted Lasso model.

    Outputs
    -------
    int
        Number of features with non-zero coefficients.
    """
    return int(np.sum(model.coef_ != 0))


def get_coefficient_table(model: Lasso, feature_names: List[str]):
    """
    Build a coefficient table for a fitted Lasso model, sorted by absolute magnitude.

    Inputs
    ------
    model : sklearn.linear_model.Lasso
        Fitted Lasso model.
    feature_names : list[str]
        Names of the input features, in the same order used for training.

    Outputs
    -------
    pandas.DataFrame
        DataFrame with columns "Feature", "Coefficient", and "Abs. Coefficient",
        sorted by descending absolute coefficient magnitude.
    """
    import pandas as pd

    coef_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": model.coef_,
        }
    )
    coef_df["Abs. Coefficient"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs. Coefficient", ascending=False).reset_index(drop=True)

    return coef_df
