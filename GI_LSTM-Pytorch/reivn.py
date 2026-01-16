import torch
import torch.nn as nn
from typing import Sequence

class LockedDrop1D(nn.Module):
    """Same dropout mask across the time axis (B, L, D)."""
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = float(p)
    def forward(self, x):               # x: (B, L, D)
        if not self.training or self.p <= 0:
            return x
        B, L, D = x.shape
        mask = x.new_empty(B, 1, D).bernoulli_(1 - self.p) / (1 - self.p)
        return x * mask


class RevIN(nn.Module):
    """
    Reversible Instance Norm for time-series.
    - Per-sample, per-channel normalization across time.
    - Optional learned affine (gamma, beta) per channel.
    Shapes:
      x: (B, T, C)
    """
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            # Broadcastable to (B, T, C)
            self.gamma = nn.Parameter(torch.ones(1, 1, num_channels))
            self.beta  = nn.Parameter(torch.zeros(1, 1, num_channels))

    def _stats(self, x: torch.Tensor):
        # Instance-wise (per sample), per-channel, over time
        # mean/std: (B, 1, C)
        mu  = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        return mu, std

    def normalize(self, x: torch.Tensor):
        """
        Returns:
          x_norm: (B, T, C)
          mu, std: (B, 1, C) – to be reused for de-normalization
        """
        mu, std = self._stats(x)
        x_norm = (x - mu) / (std + self.eps)
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm, mu, std

    def denormalize_univariate(self, y: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, target_idx: int):
        """
        y: (B, H)
        mu, std: (B, 1, C) from the *input* sample
        target_idx: which input channel corresponds to the predicted target
        """
        mu_c  = mu[:, :, target_idx]   # (B, 1)
        std_c = std[:, :, target_idx]  # (B, 1)
        if self.affine:
            beta_c  = self.beta[:, :, target_idx]   # (1, 1)
            gamma_c = self.gamma[:, :, target_idx]  # (1, 1)
            y = (y - beta_c) / (gamma_c.clamp_min(1e-6))
        y = y * (std_c + self.eps) + mu_c
        return y

    def denormalize_multivariate(self, y: torch.Tensor, mu: torch.Tensor, std: torch.Tensor):
        """
        y: (B, H, Cout). Assumes Cout == C (same channels as input).
        mu, std: (B, 1, C)
        """
        if self.affine:
            y = (y - self.beta) / (self.gamma.clamp_min(1e-6))
        y = y * (std + self.eps) + mu
        return y

    def denormalize_subset(
            self,
            y: torch.Tensor,
            mu: torch.Tensor,
            std: torch.Tensor,
            target_indices
    ):
        """
        Denormalize a subset of channels.

        Args
        ----
        y: (B, H, Cout)
            Normalized predictions for a subset of channels.
        mu, std: (B, 1, C)
            Stats from the *input* sample (all channels).
        target_indices: 1D list/tuple/LongTensor of length Cout
            For each output channel j in [0, Cout-1], tells which
            input channel index it corresponds to.

        Returns
        -------
        y_denorm: (B, H, Cout) on the original scale.
        """
        if not torch.is_tensor(target_indices):
            target_indices = torch.as_tensor(target_indices, device=y.device)

        # Select the corresponding mu/std for the subset: (B, 1, Cout)
        mu_sel = mu[:, :, target_indices]
        std_sel = std[:, :, target_indices]

        if self.affine:
            beta_sel = self.beta[:, :, target_indices]  # (1, 1, Cout)
            gamma_sel = self.gamma[:, :, target_indices]  # (1, 1, Cout)
            y = (y - beta_sel) / gamma_sel.clamp_min(1e-6)

        y = y * (std_sel + self.eps) + mu_sel  # broadcasts to (B, H, Cout)
        return y


class ModelWithRevIN(nn.Module):
    """
    Wraps a core forecaster f(x_norm) with RevIN pre/post.
    - Input x: (B, T, C)
    - Univariate output: (B, H)  -> set target_idx
    - Multivariate output: (B, H, C) with Cout == C -> target_idx=None
    """
    def __init__(self, core_model: nn.Module, num_channels: int, use_revin: bool = True,
                 affine: bool = True, target_idx: int | None = 0, eps: float = 1e-5, input_drop_p: float = 0.2,
                 output_indices: Sequence[int] | None = None,):
        super().__init__()
        self.core = core_model
        self.use_revin = use_revin
        self.target_idx = target_idx  # set None for multivariate Cout==C
        self.revin = RevIN(num_channels, eps=eps, affine=affine) if use_revin else None
        self.input_dropout = LockedDrop1D(input_drop_p) if input_drop_p > 0 else nn.Identity()
        self.output_indices = list(output_indices) if output_indices is not None else None



    def forward(self, x: torch.Tensor):
        """
        x: (B, T, C)
        returns y_hat on the original scale
        """
        if self.use_revin:
            x_norm, mu, std = self.revin.normalize(x)   # (B,T,C), (B,1,C), (B,1,C)
            x_norm = self.input_dropout(x_norm)
            y_norm = self.core(x_norm)                  # (B,H) or (B,H,Cout)

            if y_norm.dim() == 2:
                # Univariate output
                assert self.target_idx is not None, "target_idx must be set for univariate output"
                y = self.revin.denormalize_univariate(y_norm, mu, std, self.target_idx)

            else:
                # Multivariate output: (B,H,Cout)
                B, H, Cout = y_norm.shape

                if self.output_indices is None:
                    # Old behavior: assume Cout == C (all channels)
                    y = self.revin.denormalize_multivariate(y_norm, mu, std)
                else:
                    # Subset: Cout must match number of selected indices
                    assert Cout == len(self.output_indices), \
                        f"Cout={Cout} but len(output_indices)={len(self.output_indices)}"
                    y = self.revin.denormalize_subset(y_norm, mu, std, self.output_indices)

            return y

        else:
            x = self.input_dropout(x)
            # Bypass RevIN entirely
            return self.core(x)



