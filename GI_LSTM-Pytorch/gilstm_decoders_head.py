import torch
import torch.nn as nn
import torch.nn.functional as F

from gilstm import MultiDecoderMemoryGroupLSTM
from embedding_layer import EmbeddingLayer

class HorizonLinearHeadFromSeq(nn.Module):
    """
    Input:  h_seq (B, H_out, H_dec)
    Output: y (B, H_out, Cout)
    """

    def __init__(self, H_dec: int, H_out: int, Cout: int = 1, *, device=None, dtype=None):
        super().__init__()
        self.H_dec = H_dec
        self.H_out = H_out
        self.Cout = Cout
        factory = {} if device is None and dtype is None else {"device": device, "dtype": dtype}

        # W[t, :, :] is the weight matrix for horizon t
        self.W = nn.Parameter(torch.empty(H_out, H_dec, Cout, **factory))
        self.b = nn.Parameter(torch.zeros(H_out, Cout, **factory))

        nn.init.xavier_uniform_(self.W.view(H_out, -1))

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        # h_seq: (B, H_out, H_dec)
        if h_seq.dim() != 3 or h_seq.size(1) != self.H_out or h_seq.size(2) != self.H_dec:
            raise RuntimeError(
                f"Expected h_seq (B,{self.H_out},{self.H_dec}), got {tuple(h_seq.shape)}"
            )

        # y[b, t, c] = sum_d W[t, d, c] * h_seq[b, t, d] + b[t, c]
        y = torch.einsum('btd,tdc->btc', h_seq, self.W) + self.b  # (B, H_out, Cout)

        if self.Cout == 1:
            y = y.squeeze(-1)  # (B, H_out)

        return y


class DecoderGILSTMHead(nn.Module):
    """
    Recurrent forecasting head using GI-LSTM as a decoder.
    Supports both single-channel and multi-channel output,
    and an ensemble of N parallel decoders whose outputs are
    combined per-output-variable via a softmax over decoders.
    """

    def __init__(
            self,
            H_enc: int,
            H_out: int,
            Cout: int = 1,
            qs=None,
            *,
            num_layers: int = 1,
            decoder_hidden_size: int = None,
            num_decoders: int = 1,          # number of parallel decoders
            num_encoders: int = 1,
            # Embedding configuration
            d_emb: int = 64,
            emb_mode: str = 'linear',       # 'none', 'linear', 'per_step'
            ctx_drop_p: float = 0.0,
            input_drop_p: float = 0.0,
            dec_feat_drop_p: float = 0.0,
            aug_mode:bool = False,
            compute_forget_avg: bool = True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.H_enc = int(H_enc)
        self.H_out = int(H_out)
        self.Cout = int(Cout)
        self.num_layers = int(num_layers)
        self.num_decoders = int(num_decoders)
        self.num_encoders = int(num_encoders)
        self.aug_mode = aug_mode
        self.compute_forget_avg=compute_forget_avg

        if decoder_hidden_size is None:
            decoder_hidden_size = H_enc
        self.decoder_hidden_size = int(decoder_hidden_size)

        # total hidden size if you ever need concatenation
        self.total_hidden_size = self.decoder_hidden_size * self.num_decoders

        self.ctx_drop_p = float(ctx_drop_p)
        self.input_drop_p = float(input_drop_p)
        self.dec_feat_drop_p = float(dec_feat_drop_p)

        # Embedding config
        self.emb_mode = emb_mode
        if self.emb_mode == 'none':
            self.d_emb = self.H_enc
            emb_output_dim = self.H_enc
        else:
            self.d_emb = int(d_emb)
            emb_output_dim = self.d_emb

        if self.aug_mode == True:
            emb_output_dim_aug = emb_output_dim + self.H_enc
        else:
            emb_output_dim_aug = emb_output_dim

        self.ctx_dropout = (
            nn.Dropout(p=self.ctx_drop_p) if self.ctx_drop_p > 0.0 else nn.Identity()
        )

        factory = {} if device is None and dtype is None else {"device": device, "dtype": dtype}

        # Embedding layer
        self.embedding = EmbeddingLayer(
            input_dim=self.H_enc,
            output_dim=emb_output_dim,
            mode=emb_mode,
            num_steps=self.H_out,
            dropout_p=0.0,   # keep dropout external
            device=device,
            dtype=dtype,
        )

        # Decoders: N parallel GI-LSTM decoders with identical config

        self.decoders = MultiDecoderMemoryGroupLSTM(
                input_size=emb_output_dim_aug,
                hidden_size=self.decoder_hidden_size,
                qs=qs,
                num_decoders=self.num_decoders,
                batch_first=True,
                num_layers=self.num_layers,
                return_sequence=True,
                compute_forget_avg=self.compute_forget_avg,
                device=device,
                dtype=dtype,
            )


        # Dropout on decoder features (hidden states)
        self._dummy_drop = nn.Dropout(p=dec_feat_drop_p)

        # Per-decoder softmax over N_enc encoders: logits[j, n]
        self.encoder_gating_logits = nn.Parameter(
            torch.zeros(self.num_decoders, self.num_encoders, **factory)
        )

        # --- per-output softmax over decoders ---
        if self.num_decoders > 1:
            # gating_logits[c, n] -> softmax over n gives mixture weights for output variate c
            self.gating_logits = nn.Parameter(
                torch.zeros(self.Cout, self.num_decoders, **factory)
            )
        else:
            self.gating_logits = None  # only one decoder

        # --- NEW: per-output variate, time-dependent linear heads ---
        # Each head: (B, H_out, H_dec) -> (B, H_out)
        self.var_heads = nn.ModuleList([
            HorizonLinearHeadFromSeq(
                H_dec=self.decoder_hidden_size,
                H_out=self.H_out,
                Cout=1,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.Cout)
        ])

        # Optional simple readout (no longer used in forward, but kept for compatibility)
        self.readout = nn.Linear(
            self.decoder_hidden_size,
            self.Cout,
            device=device,
            dtype=dtype,
        )

        # Kept for compatibility, but not used in forward anymore
        self.Horizon_based_readout = HorizonLinearHeadFromSeq(
            H_dec=self.total_hidden_size,
            H_out=self.H_out,
            Cout=self.Cout,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def project_thetas_(self, eps: float = 1e-12):
        """
            Hard row-wise L1 normalization for all GI-LSTM decoders.

            - If self.decoder is a plain MemoryGroupLSTM (num_decoders=1),
              this normalizes its Thetas.
            - If self.decoder is a MultiDecoderMemoryGroupLSTM (num_decoders>1),
              that class' project_thetas_ handles all groups G internally.
            """
        self.decoders.project_thetas_(eps=eps)

    def get_weighted_thetas(self):

        return self.decoders.get_weighted_thetas()

    def forward(self, h_enc_all: torch.Tensor) -> torch.Tensor:
        """
        h_enc_all: (B, N_enc, H_enc)
        """
        if h_enc_all.dim() != 3:
            raise RuntimeError(f"Expected (B, N_enc, H_enc), got {tuple(h_enc_all.shape)}")

        B, N_enc, H_enc = h_enc_all.shape
        assert N_enc == self.num_encoders and H_enc == self.H_enc

        # Optional dropout on encoder reps
        if self.ctx_drop_p > 0:
            h_enc_all = F.dropout(h_enc_all, p=self.ctx_drop_p, training=self.training)

        # --- per-decoder softmax over encoders ---
        # encoder_gating_logits: (J, N_enc)
        enc_weights = F.softmax(self.encoder_gating_logits, dim=-1)  # (J, N_enc)

        # h_mix[j, b, h] = sum_n w[j,n] * h_enc_all[b,n,h]
        # -> (B, J, H_enc)
        h_mix = torch.einsum('jn,bnh->bjh', enc_weights, h_enc_all)

        # --- embedding for all decoders at once ---
        if self.emb_mode == 'per_step':
            # (B,J,H_enc) -> (B,J,H_out,H_enc)
            h_mix_rep = h_mix.unsqueeze(2).expand(B, self.num_decoders, self.H_out, self.H_enc)
            # flatten (B*J, H_out, H_enc) for embedding
            h_mix_rep_flat = h_mix_rep.reshape(B * self.num_decoders, self.H_out, self.H_enc)
            x_dec_flat = self.embedding(h_mix_rep_flat)              # (B*J, H_out, d_emb)
            x_dec = x_dec_flat.view(B, self.num_decoders, self.H_out, self.d_emb)
            x_dec = x_dec.permute(0, 2, 1, 3)                         # (B, H_out, J, d_emb)
        else:
            # 'linear' / 'none': (B,J,H_enc) -> (B,J,d_emb)
            h_mix_flat = h_mix.reshape(B * self.num_decoders, H_enc)  # (B*J, H_enc)
            h_emb_flat = self.embedding(h_mix_flat)                   # (B*J, d_emb)
            h_emb = h_emb_flat.view(B, self.num_decoders, self.d_emb) # (B,J,d_emb)
            # broadcast over horizon: (B,H_out,J,d_emb)
            x_dec = h_emb.unsqueeze(1).expand(B, self.H_out, self.num_decoders, self.d_emb)


        # Dropout on decoder input (time-locked, shared over decoders if you wish)
        if self.training and self.input_drop_p > 0.0:
            # e.g., mask per sample & feature, shared over time & decoders
            mask = x_dec.new_ones(B, 1, 1, self.d_emb)
            mask = F.dropout(mask, p=self.input_drop_p, training=True)
            x_dec = x_dec * mask  # (B,H_out,J,d_emb)

        # ---- vectorized decoders (MultiDecoderMemoryGroupLSTM with 4D input) ----
        if self.aug_mode == True:
            # x_dec: (B, H_out, J, d_emb+H_enc)
            x_dec_aug= torch.cat([x_dec,h_mix.unsqueeze(1).expand(B, self.H_out, self.num_decoders, self.H_enc)],dim=-1)
        else:
            # x_dec: (B, H_out, J, d_emb)
            x_dec_aug = x_dec

        h_vec = self.decoders(x_dec_aug)                  # (B, H_out, J * H_dec)
        B, T, Dtot = h_vec.shape
        h_stack = h_vec.view(B, T, self.num_decoders, self.decoder_hidden_size)

        # Feature dropout on decoder outputs (shared across time),
        # applied before mixing. Mask is (B, 1, N * H_dec) -> reshape.
        if self.training and self.dec_feat_drop_p > 0.0:
            mask_flat = h_stack.new_ones(B, 1, Dtot)
            mask_flat = self._dummy_drop(mask_flat)             # (B, 1, N * H_dec)
            mask = mask_flat.view(B, 1, self.num_decoders, self.decoder_hidden_size) # (B, 1, N, H_dec)
            h_stack = h_stack * mask                            # (B, T, N, H_dec)

        # ---- Per-output softmax over decoders ----
        if self.num_decoders > 1:
            # gating_weights: (Cout, N), softmax over decoders
            gating_weights = F.softmax(self.gating_logits, dim=-1)  # (Cout, N)

            # reshape for broadcasting: (1, 1, Cout, N, 1)
            g = gating_weights.view(1, 1, self.Cout, self.num_decoders, 1)
            # expand h_stack: (B, T, 1, N, H_dec)
            h_exp = h_stack.unsqueeze(2)

            # mixture: sum over decoders dimension
            # result: (B, T, Cout, H_dec)
            h_comb = (g * h_exp).sum(dim=3)
        else:
            # N = 1, just replicate the single decoder's state across Cout
            # h_stack: (B, T, 1, H_dec)
            h_comb = h_stack.expand(-1, -1, self.Cout, -1)  # (B, T, Cout, H_dec)

        # ---- Variate-dependent, time-dependent linear heads ----
        # For each output variate c, apply its own HorizonLinearHeadFromSeq
        ys = []
        for c in range(self.Cout):
            # h_c: (B, T, H_dec)
            h_c = h_comb[:, :, c, :]
            # var_heads[c]: (B, T, H_dec) -> (B, T)
            y_c = self.var_heads[c](h_c)  # (B, T)
            ys.append(y_c.unsqueeze(-1))  # (B, T, 1)

        # Stack all output variates: (B, T, Cout)
        y = torch.cat(ys, dim=-1)

        if self.Cout == 1:
            # (B, H_out) instead of (B, H_out, 1)
            y = y.squeeze(-1)

        return y

