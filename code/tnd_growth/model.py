"""TND 的最小复刻（arXiv:2606.21295 的神经元图模型）。

与上游 `tnd_pong/model/tnd.py` 等价：每个神经元带自己的动力学，神经元之间按一张
初始化后固定的随机图（adj_mask）通信。本文件只做两件事：
  1. 保持前向/初始化与上游逐位一致（换了本地 RNG，不污染全局随机状态）；
  2. 允许外部显式载入 checkpoint（实验用）或从 seed 初始化（冒烟/重训用）。

拓扑 = 邻接掩码 adj_mask（0/1，缓冲区）。"生长/剪枝"就是改这个掩码 + 给新边赋权重。
"""
from __future__ import annotations

import random

import torch
from torch import nn


class TND(nn.Module):
    def __init__(self, num_neurons: int = 600, num_inputs: int = 24, num_outputs: int = 3,
                 dt: float = 0.8, connection_prob: float = 0.4, seed: int = 0):
        super().__init__()
        self.num_neurons = int(num_neurons)
        self.num_inputs = int(num_inputs)
        self.num_outputs = int(num_outputs)
        self.dt = float(dt)

        self.input_indices = list(range(self.num_inputs))
        self.output_indices = list(range(self.num_inputs, self.num_inputs + self.num_outputs))

        py_rng = random.Random(int(seed))
        gen = torch.Generator().manual_seed(int(seed))
        adj_mask = self._build_vectorized_connections(connection_prob, py_rng, gen)
        self.register_buffer("adj_mask", adj_mask)

        self.weights = nn.Parameter(torch.empty(self.num_neurons, self.num_neurons))
        nn.init.xavier_uniform_(self.weights, gain=1.0, generator=gen)
        self._apply_spectral_normalization(target_radius=0.95)

        self.bias = nn.Parameter(torch.zeros(self.num_neurons))
        self.alpha = nn.Parameter(torch.full((self.num_neurons,), 0.5))
        self.input_weight = nn.Parameter(torch.ones(self.num_inputs))

    # ---------------------------------------------------------------- 初始化
    def _build_vectorized_connections(self, prob: float, py_rng: random.Random,
                                      gen: torch.Generator) -> torch.Tensor:
        """与上游同构的随机图（含最小入/出度保护）。"""
        mask = torch.zeros(self.num_neurons, self.num_neurons)
        hidden_start = self.num_inputs + self.num_outputs

        for i in self.output_indices:
            row_mask = (torch.rand(self.num_neurons - hidden_start, generator=gen) < prob).float()
            mask[i, hidden_start:] = row_mask

        for i in range(hidden_start, self.num_neurons):
            possible_srcs = [j for j in range(self.num_neurons) if j not in self.output_indices]
            for j in possible_srcs:
                if i != j and float(torch.rand(1, generator=gen)) < prob:
                    mask[i, j] = 1

        target_nodes = list(range(self.num_inputs, self.num_neurons))
        for i in target_nodes:
            if mask[i, :].sum() == 0:
                possible_sources = [j for j in range(self.num_neurons)
                                    if j != i and j not in self.output_indices]
                if possible_sources:
                    mask[i, py_rng.choice(possible_sources)] = 1

        hidden_indices = list(range(hidden_start, self.num_neurons))
        for j in hidden_indices:
            if mask[:, j].sum() == 0:
                possible_targets = [i for i in range(self.num_inputs, self.num_neurons) if i != j]
                if possible_targets:
                    mask[py_rng.choice(possible_targets), j] = 1
        return mask

    def _apply_spectral_normalization(self, target_radius: float = 0.95):
        """初始化稳定化：max|eig(W⊙A)| <= target_radius。"""
        with torch.no_grad():
            effective_w = self.weights * self.adj_mask
            eigs = torch.linalg.eigvals(effective_w.cpu())
            max_eig = torch.max(torch.abs(eigs))
            if float(max_eig) > 0:
                self.weights.data = (self.weights.data / max_eig) * target_radius

    # ---------------------------------------------------------------- 前向
    def forward(self, input_seq, states=None, last_neuron_outputs=None):
        batch_size, seq_len, _ = input_seq.shape
        device = input_seq.device

        if states is None:
            states = torch.zeros(batch_size, self.num_neurons, device=device)
        if last_neuron_outputs is None:
            last_neuron_outputs = torch.zeros(batch_size, self.num_neurons, device=device)

        effective_weights = self.weights * self.adj_mask
        full_input_stream = torch.zeros(batch_size, seq_len, self.num_neurons, device=device)
        full_input_stream[:, :, self.input_indices] = input_seq * self.input_weight
        w_t = effective_weights.t()

        for t in range(seq_len):
            weighted_sum = torch.matmul(last_neuron_outputs, w_t)
            total_input = full_input_stream[:, t, :] + weighted_sum + self.bias + self.alpha * states
            derivative = torch.tanh(total_input) - states
            states = states + self.dt * derivative
            last_neuron_outputs = torch.tanh(total_input)

        outputs = last_neuron_outputs[:, self.output_indices]
        return outputs, states, last_neuron_outputs

    # 单步推理（评测用，语义与 forward 完全相同，保留独立入口以对齐上游）
    def pred(self, input_seq, states=None, last_neuron_outputs=None):
        return self.forward(input_seq, states, last_neuron_outputs)

    def reset_state(self):
        pass

    # ---------------------------------------------------------------- 工具
    def edge_count(self) -> int:
        return int(self.adj_mask.sum().item())

    def spectral_radius(self) -> float:
        with torch.no_grad():
            eigs = torch.linalg.eigvals((self.weights * self.adj_mask).cpu())
            return float(torch.max(torch.abs(eigs)).item())

    def load_checkpoint(self, path: str, map_location="cpu"):
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)
        return self