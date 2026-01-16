import torch
import torch.nn as nn

from gilstm import MultiDecoderMemoryGroupLSTM

class EncoderGILSTM(nn.Module):
    """
    N independent GI-LSTM encoders (vectorized) + softmax mixture.

    - Input x: (B, T, D_emb)  (after your input embedding layer)
    - Internally uses MultiDecoderMemoryGroupLSTM with num_decoders = N.
    - Output: h_enc_mix: (B, H_enc), where H_enc = hidden_size of each small encoder.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        qs,
        *,
        num_layers: int = 1,
        num_encoders: int = 4,
        batch_first: bool = True,
        device=None,
        dtype=None,
        compute_forget_avg: bool = True
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_encoders = int(num_encoders)

        self.multi = MultiDecoderMemoryGroupLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            qs=qs,
            num_decoders=num_encoders,
            batch_first=batch_first,
            num_layers=num_layers,
            return_sequence=False,  # last hidden only
            device=device,
            dtype=dtype,
            compute_forget_avg=compute_forget_avg
        )

        factory = {} if device is None and dtype is None else {"device": device, "dtype": dtype}

        # Global (sample-independent) softmax over encoders
        # logits: (N,)
        self.gating_logits = nn.Parameter(torch.zeros(self.num_encoders, **factory))

    @torch.no_grad()
    def project_thetas_(self, eps: float = 1e-12):
        # forward to the underlying multi-encoder
        self.multi.project_thetas_(eps=eps)

    def get_weighted_thetas(self):
        # forward to the underlying multi-encoder
        return self.multi.get_weighted_thetas()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D_emb)
        returns: h_enc_mix: (B, H_enc)
        """
        # multi(x): (B, N*H_enc)
        h_vec = self.multi(x)  # (B, N * H)
        B, Dtot = h_vec.shape
        assert Dtot == self.num_encoders * self.hidden_size, \
            f"Got Dtot={Dtot}, expected {self.num_encoders * self.hidden_size}"

        # (B, N, H_enc)
        h_stack = h_vec.view(B, self.num_encoders, self.hidden_size)

        return h_stack

