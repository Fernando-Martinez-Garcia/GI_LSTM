import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    Flexible embedding layer with three modes:
    - 'none': No transformation (identity), requires input_dim == output_dim
    - 'linear': Standard linear layer
    - 'per_step': Sequence of linear layers (one per time-step)

    All modes support dropout regularization.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            mode: str = 'linear',
            num_steps: int = None,
            dropout_p: float = 0.0,
            device=None,
            dtype=None,
    ):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            mode: Embedding mode - 'none', 'linear', or 'per_step'
            num_steps: Number of time steps (required for 'per_step' mode)
            dropout_p: Dropout probability applied after transformation
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()

        valid_modes = ['none', 'linear', 'per_step']
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.mode = mode
        self.num_steps = int(num_steps) if num_steps is not None else None
        self.dropout_p = float(dropout_p)

        factory = {}
        if device is not None:
            factory['device'] = device
        if dtype is not None:
            factory['dtype'] = dtype

        # Validation
        if mode == 'none' and input_dim != output_dim:
            raise ValueError(
                f"For 'none' mode, input_dim must equal output_dim. "
                f"Got input_dim={input_dim}, output_dim={output_dim}"
            )

        if mode == 'per_step' and num_steps is None:
            raise ValueError("num_steps is required for 'per_step' mode")

        # Initialize layers based on mode
        if mode == 'linear':
            self.linear = nn.Linear(self.input_dim, self.output_dim, **factory)

        elif mode == 'per_step':
            # W[t, :, :] is the weight matrix for step t
            self.W = nn.Parameter(
                torch.empty(self.num_steps, self.input_dim, self.output_dim, **factory)
            )
            self.b = nn.Parameter(
                torch.zeros(self.num_steps, self.output_dim, **factory)
            )
            nn.init.xavier_uniform_(self.W.view(self.num_steps, -1))

        # Dropout (applied after transformation)
        self.dropout = nn.Dropout(p=self.dropout_p) if self.dropout_p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
               - For 'none' and 'linear': (B, D) or (B, T, D)
               - For 'per_step': (B, T, D) where T == num_steps

        Returns:
            Transformed tensor with output_dim features in last dimension
        """
        if self.mode == 'none':
            out = x

        elif self.mode == 'linear':
            out = self.linear(x)

        elif self.mode == 'per_step':
            if x.dim() != 3:
                raise RuntimeError(
                    f"'per_step' mode expects 3D input (B, T, D), got shape {tuple(x.shape)}"
                )
            if x.size(1) != self.num_steps:
                raise RuntimeError(
                    f"Expected T={self.num_steps}, got T={x.size(1)}"
                )
            if x.size(2) != self.input_dim:
                raise RuntimeError(
                    f"Expected D={self.input_dim}, got D={x.size(2)}"
                )

            # out[b, t, o] = sum_i W[t, i, o] * x[b, t, i] + b[t, o]
            out = torch.einsum('bti,tio->bto', x, self.W) + self.b

        out = self.dropout(out)
        return out

    @property
    def effective_output_dim(self) -> int:
        """Returns the actual output dimension (useful when mode='none')."""
        return self.output_dim

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"mode='{self.mode}', num_steps={self.num_steps}, dropout_p={self.dropout_p}"
        )