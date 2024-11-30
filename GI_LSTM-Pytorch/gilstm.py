import torch.nn as nn
import torch

class GI_LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, qs,device=torch.device("cpu")):
        """
        Args:
            input_size (int): Size of input features.
            hidden_size (int): Size of hidden state.
            qs (list): List of lags for each memory group, e.g., [q1, q2, q3, ...].
        """
        super(GI_LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.qs = qs  # List of lags for each memory group
        self.S = len(qs)  # Number of memory groups
        self.epsilon = 1e-8  # Small constant to prevent division by zero

        self.prev_cell_state = []
        self.prev_mem_grps = [[] for _ in range(self.S)]

        # Gates weights and biases
        self.W_a = nn.Linear(input_size + hidden_size, hidden_size, bias=True).to(device)
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size, bias=True).to(device)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size, bias=True).to(device)   
        self.W_fs = nn.ModuleList([
            nn.Linear(input_size + hidden_size, hidden_size, bias=True) for _ in range(self.S)
        ]).to(device)

        # Memory group weights (Theta parameters before normalization)
        self.Thetas = nn.ParameterList([
            nn.Parameter(torch.Tensor(hidden_size, q)) for q in self.qs
        ]).to(device)
        # Initialize Thetas
        for Theta in self.Thetas:
            nn.init.uniform_(Theta, -0.1, 0.1)

    def reset_states(self):
        self.prev_cell_state = []
        self.prev_mem_grps = [[] for _ in range(self.S)]

    def forward(self, x_k, h_prev):
        batch_size = x_k.size(0)

        combined = torch.cat((x_k, h_prev), dim=1).to(x_k.device)

        a_k = torch.tanh(self.W_a(combined))
        i_k = torch.sigmoid(self.W_i(combined))
        o_k = torch.sigmoid(self.W_o(combined))

        f_ks = [torch.sigmoid(W_f(combined)) for W_f in self.W_fs]
        sum_f = sum(f_ks) + self.epsilon
        w_fs = [f_k / sum_f for f_k in f_ks]
        hat_f_ks = [f_k * w_f for f_k, w_f in zip(f_ks, w_fs)]

        M_ks = []

        lag_products = [1]
        for s in range(1, self.S):
            lag_products.append(lag_products[-1] * self.qs[s - 1])

        for s in range(self.S):
            q_s = self.qs[s]
            Theta_s = self.Thetas[s]
            abs_sum_theta_s = Theta_s.abs().sum(dim=1, keepdim=True) + self.epsilon
            W_s = Theta_s / abs_sum_theta_s

            if s == 0:
                source_list = self.prev_cell_state
            else:
                source_list = self.prev_mem_grps[s - 1]

            lag_product = lag_products[s]

            required_lags = [j * lag_product for j in range(1, q_s + 1)]
            required_length = required_lags[-1]

            if len(source_list) < required_length:
                pad_size = required_length - len(source_list)
                zero_padding = [torch.zeros(batch_size, self.hidden_size, device=x_k.device) for _ in range(pad_size)]
                source_list = zero_padding + source_list
            else:
                source_list = source_list[-required_length:]

            indices = [len(source_list) - lag for lag in required_lags]

            states_needed = [source_list[idx] for idx in indices]

            states_stack = torch.stack(states_needed, dim=2).to(x_k.device)

            W_s_expanded = W_s.unsqueeze(0)
            M_k_s = (states_stack * W_s_expanded).sum(dim=2)
            M_ks.append(M_k_s)
            self.prev_mem_grps[s].append(M_k_s)

        c_k = i_k * a_k
        for s in range(self.S):
            c_k += hat_f_ks[s] * M_ks[s]

        h_k = o_k * torch.tanh(c_k)

        self.prev_cell_state.append(c_k)
        max_length_c_prev = max(self.qs)
        if len(self.prev_cell_state) > max_length_c_prev:
            self.prev_cell_state.pop(0)

        for s in range(self.S):
            lag_product = lag_products[s] * self.qs[s]
            max_length_m_prev = lag_product
            if len(self.prev_mem_grps[s]) > max_length_m_prev:
                self.prev_mem_grps[s].pop(0)

        return h_k, c_k

    
class GI_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, qs=[5,1], device=torch.device("cpu")):
        super(GI_LSTM, self).__init__()

        self.qs=qs

        self.hidden_size = hidden_size
        self.cell = GI_LSTMCell(input_size, hidden_size, qs=self.qs,device=device).to(device)
        self.output_layer = nn.Linear(hidden_size, output_size).to(device)
 
    def forward(self, input_seq):
        batch_size, seq_len, _ = input_seq.size()
        device = input_seq.device

        h_t = torch.zeros(batch_size, self.hidden_size, device=device)
        self.cell.reset_states()

        outputs = []
        for t in range(seq_len):
            x_k = input_seq[:, t, :]

            h_t, c_k = self.cell(x_k, h_t)

            y_t = self.output_layer(h_t)
            outputs.append(y_t.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs