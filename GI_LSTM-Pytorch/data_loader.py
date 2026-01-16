import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, List, Optional, Union


class TimeSeriesScaler:
    """
    Handles Z-score normalization and denormalization.
    Stores statistics (mean/std) for validation/test data
    Scaled using only training statistics.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.device = None

    def fit(self, data: np.ndarray, axis: Tuple[int, ...] = (0, 1)):
        """Compute mean and std from training data."""
        self.mean = data.mean(axis=axis, keepdims=True)
        self.std = data.std(axis=axis, keepdims=True) + 1e-8
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (data - self.mean) / self.std

    def denormalize(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        Reverse  normalization (for generic use or plotting).
        Moves statistics to the same device as the input tensor on the fly.
        """
        if self.device != x_tensor.device:
            self.mean_t = torch.tensor(self.mean, dtype=torch.float32, device=x_tensor.device)
            self.std_t = torch.tensor(self.std, dtype=torch.float32, device=x_tensor.device)
            self.device = x_tensor.device

        return x_tensor * self.std_t + self.mean_t


def make_windows(
        X_seg: np.ndarray,
        y_seg: np.ndarray,
        L: int,
        H: int,
        window_step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sliding windows from segmented data (data leakage prevented) .
    """
    T_seg = X_seg.shape[0]
    # Handle single vs multi-channel Y shape
    Cout = y_seg.shape[1] if y_seg.ndim > 1 else 1

    # Calculate number of samples
    N = T_seg - L - H + 1
    assert N > 0, f"Segment length {T_seg} too short for L={L}, H={H}"

    N_samples = int(np.floor(N / window_step))

    # Pre-allocate arrays
    X_out = np.zeros((N_samples, L, X_seg.shape[1]), dtype=np.float32)

    if Cout > 1:
        Y_out = np.zeros((N_samples, H, Cout), dtype=np.float32)
    else:
        Y_out = np.zeros((N_samples, H), dtype=np.float32)

    # Fill arrays
    for n in range(N_samples):
        t_initial = n * window_step
        t_end = t_initial + L

        X_out[n] = X_seg[t_initial:t_end, :]

        if Cout > 1:
            Y_out[n] = y_seg[t_end:t_end + H, :]
        else:
            # Flatten last dim if single channel, matching original logic
            Y_out[n] = y_seg[t_end:t_end + H, 0]

    return X_out, Y_out


def get_dataloaders(
        file_path: str,
        lookback: int,
        horizon: int,
        batch_size: int,
        output_channels: Union[int, List] = 1,
        window_step: int = 1,
        split_counts: Optional[Tuple[int, int, int]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, TimeSeriesScaler, int, int, List]:
    """
    Args:
        file_path: Path to CSV.
        lookback (L): Input window length.
        horizon (H): Forecast horizon.
        batch_size: Batch size for loaders.
        output_channels: 1, specific int, or 'all'.
        window_step: Step size for sliding window.
        split_counts: (n_train, n_val, n_test). If None, defaults to standard ETTh2 split.

    Returns:
        train_loader, val_loader, test_loader, target_scaler, input_dim (D), targets_dim (Cout)
    """

    # Load Dataset
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)

    # Drop timestamp (assumed col 0), keep numeric
    values = df.iloc[:, 1:].astype(np.float32).to_numpy()
    T_total, D = values.shape
    print(f"Total length {T_total}, num features {D}")

    # Define Output Channels
    if output_channels == D or output_channels == 'all':
        target_cols = list(range(D))
        target_full = values[:, target_cols].copy()
    else:
        # Subset of channels
        target_cols = output_channels
        target_full = values[:, target_cols].copy()

    Cout = len(target_cols)
    print(f"Output channels: {Cout}, indices: {target_cols}")

    # Split Indices (Chronological)
    # Default to the fixed split from original code if not provided
    if split_counts is None:
        n_train = 24 * 30 * 12
        n_val = 24 * 30 * 4
        n_test = 24 * 30 * 4
    else:
        n_train, n_val, n_test = split_counts

    print(f"Splitting: Train={n_train}, Val={n_val}, Test={n_test}")

    idx_tr = slice(0, n_train)
    idx_va = slice(n_train, n_train + n_val)
    idx_te = slice(n_train + n_val, n_train + n_val + n_test)

    X_tr_raw, y_tr_raw = values[idx_tr], target_full[idx_tr]
    X_va_raw, y_va_raw = values[idx_va], target_full[idx_va]
    X_te_raw, y_te_raw = values[idx_te], target_full[idx_te]

    # 4. Make Sliding Windows
    Xtr_win, Ytr_win = make_windows(X_tr_raw, y_tr_raw, lookback, horizon, window_step)
    Xva_win, Yva_win = make_windows(X_va_raw, y_va_raw, lookback, horizon, window_step)
    Xte_win, Yte_win = make_windows(X_te_raw, y_te_raw, lookback, horizon, window_step)

    print("Windowed shapes:")
    print("  Train shapes X:", Xtr_win.shape, " Y:", Ytr_win.shape)
    print("  Val shapes  X:", Xva_win.shape, " Y:", Yva_win.shape)
    print("  Test shapes X:", Xte_win.shape, " Y:", Yte_win.shape)

    # 5. Normalization (Fit on Train, Apply to All)
    # Input Scaler (per-feature)
    x_scaler = TimeSeriesScaler().fit(Xtr_win, axis=(0, 1))
    Xtr_n = x_scaler.transform(Xtr_win)
    Xva_n = x_scaler.transform(Xva_win)
    Xte_n = x_scaler.transform(Xte_win)

    # Target Scaler
    if Cout > 1:
        # Per-channel z-score for multi-channel target: (N, H, Cout) -> fit on (0, 1)
        y_scaler = TimeSeriesScaler().fit(Ytr_win, axis=(0, 1))
    else:
        # Single-channel z-score: (N, H) -> fit global scalar (axis=None in original logic used mean() over all)
        # Original code: Ytr_win.mean() -> Global mean over (N, H)
        y_scaler = TimeSeriesScaler().fit(Ytr_win, axis=None)

    Ytr_n = y_scaler.transform(Ytr_win)
    Yva_n = y_scaler.transform(Yva_win)
    Yte_n = y_scaler.transform(Yte_win)

    # 6. Convert to Tensor
    train_ds = TensorDataset(torch.from_numpy(Xtr_n), torch.from_numpy(Ytr_n))
    val_ds = TensorDataset(torch.from_numpy(Xva_n), torch.from_numpy(Yva_n))
    test_ds = TensorDataset(torch.from_numpy(Xte_n), torch.from_numpy(Yte_n))

    # 7. Create Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, y_scaler, D, Cout, target_cols