import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDecoderMemoryGroupLSTM(nn.Module):
    """
    Multi-layer GI-LSTM with *G independent decoders* (groups) vectorized.

    - num_decoders = G
    - Each decoder has its own W_cat, b_cat, Thetas.
    - Input x is shared across decoders; states are (G,B,H,...).

    Shapes:
      x: (B,T,D) if batch_first else (T,B,D)
      output (sequence): (B,T,G*H) if batch_first else (T,B,G*H)
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            *,
            qs,  # list[int]
            num_decoders: int,
            batch_first: bool = True,
            bias: bool = True,
            device=None, dtype=None,
            eps: float = 1e-12,
            num_layers: int = 1,
            return_sequence: bool = False,
            compute_forget_avg: bool = False  # flag to compute forget gate averages
    ):
        super().__init__()
        factory = {"device": device, "dtype": dtype}

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.qs = [int(q) for q in qs]
        self.S = len(self.qs)
        self.batch_first = batch_first
        self.eps = eps
        self.num_layers = int(num_layers)
        self.return_sequence = return_sequence
        self.num_decoders = int(num_decoders)  # G

        # Flag to compute forget gate averages in eval mode
        self.compute_forget_avg = compute_forget_avg
        self.avg_forget_gates = None  # storage for averaged forget gate values

        G = self.num_decoders
        H = self.hidden_size
        D0 = self.input_size
        S = self.S
        gate_out = (3 + S) * H

        # ---- Per-layer gate projections W_cat[ℓ,g], b_cat[ℓ,g]
        self.W_cat = nn.ParameterList()
        self.b_cat = nn.ParameterList() if bias else None

        in_sizes = [D0] + [H] * (self.num_layers - 1)  # input_size_ℓ
        for Dl in in_sizes:
            # W_cat[ℓ]: (G, gate_out, Dl + H)
            W = nn.Parameter(torch.empty(G, gate_out, Dl + H, **factory))
            # Xavier per group
            for g in range(G):
                nn.init.xavier_uniform_(W[g])
            self.W_cat.append(W)
            if bias:
                # b_cat[ℓ]: (G, gate_out)
                b = nn.Parameter(torch.zeros(G, gate_out, **factory))
                self.b_cat.append(b)

        # ---- Per-layer Θ_s kernels: Theta[ℓ][s] ∈ (G,H,q_s)
        self.Thetas = nn.ModuleList()
        for _ in range(self.num_layers):
            plist = nn.ParameterList()
            for q in self.qs:
                th = nn.Parameter(torch.empty(G, H, q, **factory))
                # Xavier over (H,q) for each group
                for g in range(G):
                    nn.init.xavier_uniform_(th[g])
                plist.append(th)
            self.Thetas.append(plist)

        # ---- P list: P[0]=1, P[s]=q1*...*q_s   (length S)
        P = [1]
        for q in self.qs[:-1]:
            P.append(P[-1] * q)
        self.P = P  # P[s] = product_{i=1..s} q_i

        # ---- History window sizes for a single layer (shared scheme)
        # M1 history length (for c_hist)
        self.Lc = self.qs[0] if S >= 1 else 0

        # For s >= 2 (memory group index), we want M_{s-1} history length:
        #   cap_s = q_s * P_{s-1}
        # where P_{s-1} = self.P[s-1].
        self.m_caps = []
        if self.S >= 2:
            for j in range(self.S - 1):
                # group index s = j+2 => d_s = P[s-1] = self.P[j+1-1] = self.P[j]
                # but the history we allocate here is for M_{s-1}, used by group s.
                # In the computation loop, for index j (1..S-1) we use self.P[j] and qs[j].
                d = self.P[j + 1]      # P_{s-1} in original code convention
                q_s = self.qs[j + 1]   # q_s
                cap = q_s * d          # full q_s taps at scale d
                self.m_caps.append(cap)

        self.register_buffer("one", torch.tensor(1.0, **factory), persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() <= 1:
                    p.uniform_(-1.0, 1.0)
                # W_cat, Theta already xavier-init

    def _theta_as_conv_kernels(self, Theta_plist):
        """
        For a given layer, take list of Theta_s ∈ (G,H,q_s)
        and return list of kernels_s ∈ (G,H,1,q_s), row-normalized (L1) per (G,H).
        """
        kernels = []
        for Theta in Theta_plist:
            # Theta: (G,H,q)
            denom = Theta.abs().sum(dim=-1, keepdim=True).clamp_min(self.eps)  # (G,H,1)
            W = (Theta / denom).unsqueeze(2)  # (G,H,1,q)
            kernels.append(W)
        return kernels

    @torch.no_grad()
    def project_thetas_(self, eps: float = 1e-12):
        """Hard row-wise L1 normalization for all layers & decoders."""
        for Theta_plist in self.Thetas:
            for Theta in Theta_plist:
                # Theta: (G,H,q)
                s = Theta.abs().sum(dim=-1, keepdim=True).clamp_min(eps)  # (G,H,1)
                Theta.div_(s)

    def get_weighted_thetas(self):
        """
        Returns Theta parameters weighted by the averaged forget gate values.

        The i-th element of the averaged forget gate vector (for memory group s)
        weights the i-th row of the corresponding Theta matrix.

        Returns:
            weighted_thetas: list[list[Tensor]] where weighted_thetas[layer][s]
                             has shape (G, H, q) for the s-th memory group.
            Returns None if avg_forget_gates is not computed.
        """
        if self.avg_forget_gates is None:
            return None

        weighted_thetas = []
        for l in range(self.num_layers):
            layer_thetas = []
            avg_f = self.avg_forget_gates[l]  # (S, G, H) or None

            if avg_f is None:
                for theta in self.Thetas[l]:
                    layer_thetas.append(theta.detach().clone())
            else:
                for s, theta in enumerate(self.Thetas[l]):
                    # theta: (G, H, q)
                    # avg_f[s]: (G, H)
                    weight = avg_f[s].unsqueeze(-1)  # (G, H, 1)
                    weighted_theta = theta.detach() * weight  # (G, H, q)
                    layer_thetas.append(weighted_theta)

            weighted_thetas.append(layer_thetas)

        return weighted_thetas

    def _layer_step_grouped(self, layer_idx, x_in, h, c, c_hist, m_hists, kernels):
        """
        Optimized and corrected version combining best of both implementations.
        """
        G = self.num_decoders
        H = self.hidden_size
        S = self.S

        # Gate computations (unchanged - already optimized)
        cat = torch.cat([x_in, h], dim=-1)
        W = self.W_cat[layer_idx]
        b = self.b_cat[layer_idx] if self.b_cat is not None else None

        gates = torch.bmm(cat, W.transpose(1, 2))
        if b is not None:
            gates = gates + b.unsqueeze(1)

        gates = gates.view(G, -1, (3 + S), H)
        i = torch.sigmoid(gates[:, :, 0, :])
        o = torch.sigmoid(gates[:, :, 1, :])
        a = gates[:, :, 2, :]
        f_chunks = [gates[:, :, 3 + s, :] for s in range(S)]

        # Memory group computation
        m_vals = []

        if S >= 1:
            # ---- M1: Optimized with conv1d
            B, Lc = c_hist.shape[1], c_hist.shape[-1]
            c_hist_ = c_hist.permute(1, 0, 2, 3).reshape(B, G * H, Lc)
            k0 = kernels[0]
            GH = G * H
            q1 = k0.shape[-1]
            k0_flat = k0.reshape(GH, 1, q1)
            m1_flat = F.conv1d(c_hist_, k0_flat, groups=GH)
            m1 = m1_flat.view(B, G, H, -1).squeeze(-1).permute(1, 0, 2)
            m_vals.append(m1)

            # ---- Higher-order memories
            new_m_hists = []
            for j in range(1, S):
                hist = m_hists[j - 1]  # (G,B,H,cap_j)
                G_, B_, H_, cap_j = hist.shape

                d_s = self.P[j]  # Scale: P_{s-1}
                q_s = self.qs[j]  # Number of taps

                # Create indices for lags d_s, 2*d_s, ..., q_s*d_s
                # hist buffer stores [t-cap_j, ..., t-1]
                # Position for lag L is: cap_j - L
                lag_multipliers = torch.arange(1, q_s + 1, device=hist.device, dtype=torch.long)
                lags = lag_multipliers * d_s  # [d_s, 2*d_s, ..., q_s*d_s]
                indices = cap_j - lags  # Positions in hist buffer

                # Expand indices for gather operation
                k = indices.view(1, 1, 1, q_s).expand(G, B, H, q_s)

                # Check if we have enough history
                if indices[0] >= 0:  # First (smallest) lag is available
                    selected = hist.gather(dim=3, index=k)

                    # Apply normalized weights
                    w = kernels[j].squeeze(2)  # (G,H,q_s)
                    w = w.unsqueeze(1).expand(G, B, H, q_s)

                    ms = (selected * w).sum(dim=3)
                else:
                    # Not enough history yet
                    ms = torch.zeros_like(h)

                m_vals.append(ms)

        # Cell update
        f_stack = None
        if S:
            f_stack = torch.stack([torch.sigmoid(fc) for fc in f_chunks], dim=0)
            denom = f_stack.sum(dim=0, keepdim=True).clamp_min(self.eps)
            w_f = f_stack / denom
            f_hat = f_stack * w_f

            M = torch.stack(m_vals, dim=0)
            forget_term = (f_hat * M).sum(dim=0)

            c = torch.tanh(a) * i + forget_term
        else:
            c = torch.tanh(a) * i

        h = o * torch.tanh(c)

        # Update histories
        if S >= 1:
            c_hist = torch.cat([c_hist[:, :, :, 1:], c.unsqueeze(-1)], dim=-1)

            new_m_hists = []
            for j in range(S - 1):
                m_j = m_vals[j]
                hist_j = m_hists[j]
                new_m = torch.cat([hist_j[:, :, :, 1:], m_j.unsqueeze(-1)], dim=-1)
                new_m_hists.append(new_m)
            m_hists = new_m_hists

        return h, c, c_hist, m_hists, m_vals, f_stack

    def forward(self, x):
        """
        x:
          - (B,T,D)  : same input for all decoders (old behaviour)
          - (B,T,G,D): per-decoder input, G = num_decoders

        RETURNS:
          if return_sequence: (B,T,G*H)
          else: (B,G*H)
        """
        if self.batch_first:
            if x.dim() == 3:
                B, T, D = x.shape
                x_has_groups = False
                x = x.transpose(0, 1)  # (T,B,D)
            elif x.dim() == 4:
                B, T, Gx, D = x.shape
                assert Gx == self.num_decoders, \
                    f"Expected G={self.num_decoders}, got {Gx}"
                x_has_groups = True
                x = x.permute(1, 2, 0, 3)  # (T,G,B,D)
            else:
                raise RuntimeError(f"Expected x dim 3 or 4, got {x.dim()}")
        else:
            raise NotImplementedError("Use batch_first=True here.")

        G = self.num_decoders
        H = self.hidden_size
        S = self.S

        # ---- Init per-layer states and histories
        hs = [x.new_zeros(G, B, H) for _ in range(self.num_layers)]
        cs = [x.new_zeros(G, B, H) for _ in range(self.num_layers)]
        if S >= 1:
            c_hists = [x.new_zeros(G, B, H, self.Lc) for _ in range(self.num_layers)]
            m_hists_all = [
                [x.new_zeros(G, B, H, cap) for cap in self.m_caps]
                for _ in range(self.num_layers)
            ]
        else:
            c_hists = [None] * self.num_layers
            m_hists_all = [None] * self.num_layers

        # ---- Precompute per-layer kernels
        kernels_per_layer = [
            self._theta_as_conv_kernels(self.Thetas[l]) for l in range(self.num_layers)
        ]

        outs_h_last = []  # collect last-layer h_t over time, shape (G,B,H) per step

        # forget gate averaging (eval mode only)
        compute_fg_avg = self.compute_forget_avg and not self.training
        if compute_fg_avg:
            forget_gate_sum = [None for _ in range(self.num_layers)]  # list of (S,G,B,H)
            forget_gate_count = 0

        # ---- Time loop
        for t in range(T):
            if x_has_groups:
                x_in = x[t]  # (G,B,D)
            else:
                x_t = x[t]  # (B,D)
                x_in = x_t.unsqueeze(0).expand(G, B, -1)  # (G,B,D)

            for l in range(self.num_layers):
                kers = kernels_per_layer[l]
                h_l, c_l = hs[l], cs[l]
                ch_l, mh_l = c_hists[l], m_hists_all[l]

                h_new, c_new, ch_new, mh_new, _, f_stack = self._layer_step_grouped(
                    l, x_in, h_l, c_l, ch_l, mh_l, kers
                )

                if compute_fg_avg and f_stack is not None:
                    if forget_gate_sum[l] is None:
                        forget_gate_sum[l] = f_stack.clone()
                    else:
                        forget_gate_sum[l] = forget_gate_sum[l] + f_stack

                hs[l], cs[l] = h_new, c_new
                if S >= 1:
                    c_hists[l], m_hists_all[l] = ch_new, mh_new

                x_in = h_new  # (G,B,H) for next layer

            if compute_fg_avg:
                forget_gate_count += 1

            outs_h_last.append(hs[-1])  # (G,B,H)

        # compute averaged forget gates
        if compute_fg_avg:
            self.avg_forget_gates = []
            for l in range(self.num_layers):
                if forget_gate_sum[l] is not None:
                    avg_over_time = forget_gate_sum[l] / forget_gate_count  # (S,G,B,H)
                    avg_over_time_batch = avg_over_time.mean(dim=2)         # (S,G,H)
                    self.avg_forget_gates.append(avg_over_time_batch)
                else:
                    self.avg_forget_gates.append(None)

        # ---- Stack over time and reshape
        y_last = torch.stack(outs_h_last, dim=0)      # (T,G,B,H)
        y_last = y_last.permute(2, 0, 1, 3)           # (B,T,G,H)
        y_last_flat = y_last.reshape(B, T, G * H)     # (B,T,G*H)

        if not self.return_sequence:
            y_last_flat = y_last_flat[:, -1, :]       # (B,G*H)

        if not self.batch_first:
            if self.return_sequence:
                y_last_flat = y_last_flat.transpose(0, 1)  # (T,B,G*H)
            else:
                y_last_flat = y_last_flat.unsqueeze(0)     # (1,B,G*H)

        return y_last_flat
