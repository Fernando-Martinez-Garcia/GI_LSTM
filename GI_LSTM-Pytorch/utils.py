import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

def set_seed(seed: int = 12345):
    """Sets the seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Info] Random seed set to {seed}")

# --- Evaluation Functions  ---
@torch.no_grad()
def evaluate_metrics(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        scaler=None,
        flag_denormalize: bool = False
) -> Tuple[float, float]:
    """
    Iterates through a loader to calculate MSE and MAE metrics.
    """
    model.eval()
    se_sum = 0.0
    ae_sum = 0.0
    n_elem = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        # Forward
        pred_n = model(xb)

        #  Denormalization
        if flag_denormalize and scaler is not None:
            pred = scaler.denormalize(pred_n).cpu().numpy()
            true = scaler.denormalize(yb).cpu().numpy()
        else:
            pred = pred_n.cpu().numpy()
            true = yb.cpu().numpy()

        diff = pred - true
        se_sum += float(np.sum(diff ** 2))
        ae_sum += float(np.sum(np.abs(diff)))
        n_elem += diff.size

    mse = se_sum / n_elem
    mae = ae_sum / n_elem
    return mse, mae


@torch.no_grad()
def compute_validation_loss(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device
) -> float:
    """
    Calculates the average loss over the validation set.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred_n = model(xb)
        loss = criterion(pred_n, yb)

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)

class TrainingVisualizer:
    """
    Manages real-time visualization of forecasts and internal Theta parameters
    during training. Handles figure creation, updating, and cleanup.
    """
    def __init__(self, save_dir: str = "plots"):
        self.save_dir = save_dir
        self.fig_forecast = None
        self.fig_theta_list = []
        self.x_ref = None
        self.y_ref = None

        plt.style.use('ggplot')

    def set_reference_sample(self, loader, device, sample_idx: int = 0):
        """
        Extracts a specific sample by index from the underlying dataset.
        This allows for using the exact same curve (e.g., index 1500) every time.
        """
        dataset = loader.dataset

        # Safety Check
        if sample_idx >= len(dataset):
            print(f"[Visualizer] Warning: Index {sample_idx} is out of bounds (Size: {len(dataset)}). Defaulting to 0.")
            sample_idx = 0

        # Get the single item (Returns tensors: (L, D), (H, C))
        x_item, y_item = dataset[sample_idx]

        # Verifyin they are Tensors
        if not torch.is_tensor(x_item):
            x_item = torch.from_numpy(x_item)
        if not torch.is_tensor(y_item):
            y_item = torch.from_numpy(y_item)

        # The model expects a batch dimension (1, L, D), so we unsqueeze at dim 0
        self.x_ref = x_item.unsqueeze(0).to(device)
        self.y_ref = y_item.unsqueeze(0).to(device)

        print(f"[Visualizer] Reference sample set to Index {sample_idx}.")

    def update_forecast(
            self,
            model: nn.Module,
            epoch: int,
            device: torch.device,
            scaler=None
    ):

        if self.x_ref is None or self.y_ref is None:
            print("[Visualizer] Warning: No reference sample set. Call set_reference_sample() first.")
            return

        """
        Runs inference on a single sample and plots True vs Predicted.
        """
        # Close previous figure safely
        if self.fig_forecast is not None:
            try:
                plt.close(self.fig_forecast)
            except Exception:
                pass

        was_training = model.training
        model.eval()

        with torch.no_grad():
            # Forward pass
            pred_n = model(self.x_ref)

            # Denormalize if scaler provided
            if scaler:
                pred = scaler.denormalize(pred_n).cpu().numpy().squeeze(0)
                true = scaler.denormalize(self.y_ref).cpu().numpy().squeeze(0)
            else:
                pred = pred_n.cpu().numpy().squeeze(0)
                true = self.y_ref.cpu().numpy().squeeze(0)

        # Plotting
        self.fig_forecast, ax = plt.subplots(figsize=(9, 4))

        # Handle Horizon Axis
        h_axis = np.arange(len(true) if true.ndim == 1 else len(true[:, 0]))

        # Handle multi-channel vs single channel shapes for plotting
        if pred.ndim > 1:
            # Plot first channel
            ax.plot(h_axis, true[:, 0], label="True (Ch0)", linewidth=2, color='black', alpha=0.7)
            ax.plot(h_axis, pred[:, 0], label="Pred (Ch0)", linewidth=2, color='C1')
        else:
            ax.plot(h_axis, true, label="True", linewidth=2, color='black', alpha=0.7)
            ax.plot(h_axis, pred, label="Pred", linewidth=2, color='C1')

        mse = float(np.mean((pred - true) ** 2))
        mae = float(np.mean(np.abs(pred - true)))

        ax.set_title(f"Validation Forecast (Epoch {epoch}) | MSE={mse:.3f} MAE={mae:.3f}")
        ax.set_xlabel("Horizon Step")
        ax.set_ylabel("Value")
        ax.legend(loc="best")
        self.fig_forecast.tight_layout()

        plt.show(block=False)
        plt.pause(0.2)

        if was_training:
            model.train()

    def plot_theta(
            self,
            gilstm_list: List[nn.Module],
            epoch: int,
            layer_indices: Optional[List[int]] = None,
            use_weighted: bool = False
    ):
        """
        Plot mean absolute values of Theta parameters.
        Includes support for Forget Gate visualization (red divider line).
        Replaces 'plot_theta_parameters'.
        """
        # Close previous theta figures
        for fig in self.fig_theta_list:
            try:
                plt.close(fig)
            except Exception:
                pass
        self.fig_theta_list = []

        for model_idx, gilstm_model in enumerate(gilstm_list):
            if gilstm_model is None: continue

            # Basic attribute checks
            if not hasattr(gilstm_model, 'Thetas') or not hasattr(gilstm_model, 'qs'):
                continue

            num_layers = gilstm_model.num_layers
            qs = gilstm_model.qs
            S = getattr(gilstm_model, 'S', 1)  # Default to S=1 if attribute missing

            # Determine layers to plot
            if layer_indices is None:
                layers_to_plot = list(range(num_layers))
            elif isinstance(layer_indices, int):
                layers_to_plot = [layer_indices]
            else:
                layers_to_plot = list(layer_indices)

            layers_to_plot = [l for l in layers_to_plot if 0 <= l < num_layers]

            # 1. Handle Weighted Thetas
            weighted_thetas = None
            if use_weighted:
                if hasattr(gilstm_model, 'get_weighted_thetas'):
                    weighted_thetas = gilstm_model.get_weighted_thetas()
                    if weighted_thetas is None:
                        print(f"[Warn] Weighted thetas unavailable for model {model_idx}. Using raw.")
                else:
                    print(f"[Warn] Model {model_idx} has no get_weighted_thetas method.")

            # 2. Handle Avg Forget Gates
            avg_forget_gates = None
            if hasattr(gilstm_model, 'avg_forget_gates'):
                avg_forget_gates = gilstm_model.avg_forget_gates

            with torch.no_grad():
                for layer_idx in layers_to_plot:
                    # Select source theta list
                    if weighted_thetas is not None:
                        theta_plist = weighted_thetas[layer_idx]
                        plot_suffix = " (Weighted)"
                    else:
                        theta_plist = gilstm_model.Thetas[layer_idx]
                        plot_suffix = ""

                    # Get Forget Gates for this layer
                    layer_avg_fg = None
                    if avg_forget_gates is not None and layer_idx < len(avg_forget_gates):
                        layer_avg_fg = avg_forget_gates[layer_idx]

                    # Loop over Memory Groups
                    for s, (theta, q) in enumerate(zip(theta_plist, qs)):
                        # Extract tensor data
                        theta_data = theta.data if hasattr(theta, 'data') else theta
                        theta_abs = theta_data.abs()

                        # Calculate Mean per Encoder
                        if theta_abs.dim() == 2:  # (H, q) -> (1, q)
                            theta_mean = theta_abs.mean(dim=0, keepdim=True)
                        elif theta_abs.dim() == 3:  # (G, H, q) -> (G, q)
                            theta_mean = theta_abs.mean(dim=1)
                        else:
                            continue

                        G_enc = theta_mean.size(0)
                        q_val = theta_mean.size(1)

                        # Logic for Lower-Order Forget Gates (The "Left Side" of the plot)
                        lower_fg_avgs = None
                        num_lower_fg = 0

                        if s >= 1 and layer_avg_fg is not None:
                            fg_list = []
                            for j in range(s):
                                # layer_avg_fg[j] is (G, H) -> mean over H -> (G,)
                                fg_list.append(layer_avg_fg[j].mean(dim=-1))
                            lower_fg_avgs = torch.stack(fg_list, dim=1)  # (G, s)
                            num_lower_fg = s

                        # Setup Plot
                        total_len = num_lower_fg + q_val
                        x_pos = np.arange(total_len)

                        fig, ax = plt.subplots(figsize=(10, 6))
                        self.fig_theta_list.append(fig)

                        for g in range(G_enc):
                            # Data Prep
                            theta_vals = theta_mean[g].flip(dims=[-1]).cpu().numpy()

                            if lower_fg_avgs is not None:
                                fg_vals = lower_fg_avgs[g].cpu().numpy()
                                y_g = np.concatenate([fg_vals, theta_vals])
                            else:
                                y_g = theta_vals

                            ax.plot(x_pos, y_g, alpha=0.7, label=f'Enc {g}')

                        # Visual Polish: Red Divider Line for Forget Gates
                        if num_lower_fg > 0:
                            ax.axvline(x=num_lower_fg - 0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                            # Annotation Box
                            ymax = ax.get_ylim()[1]
                            ax.text(num_lower_fg - 0.5, ymax * 0.95, ' FG | θ ',
                                    ha='center', va='top', fontsize=9, color='red',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='red', alpha=0.8))

                            # X-axis annotations
                            for j in range(num_lower_fg):
                                ax.annotate(f'f{j + 1}', xy=(j, 0), xytext=(j, -0.05 * ymax),
                                            ha='center', va='top', fontsize=8, color='blue', annotation_clip=False)

                            fg_range = f"FG₁→FG_{num_lower_fg}" if num_lower_fg > 1 else "FG₁"
                            ax.set_xlabel(
                                f'Index ({fg_range}: 0→{num_lower_fg - 1}, θ: {num_lower_fg}→{total_len - 1})')
                        else:
                            ax.set_xlabel(f'Parameter Index (q={q_val})')

                        ax.set_ylabel('Mean Absolute Value')
                        ax.set_title(f"L{layer_idx} Grp{s + 1}/{S} (Order {s + 1}) - Ep{epoch}{plot_suffix}")
                        ax.grid(True, alpha=0.3, axis='y')

                        if G_enc <= 12:
                            ax.legend(fontsize='small', loc='upper right')

                        plt.tight_layout()
                        plt.show(block=False)
                        plt.pause(0.2)