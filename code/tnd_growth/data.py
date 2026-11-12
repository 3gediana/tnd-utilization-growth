"""Pong 行为克隆数据的加载与切批（口径与上游 tnd_pong/dt/c2.txt + train_tnd.py 一致）。

数据格式：每行 `24 个 0/1 输入 \\t 3 个 0/1 输出`，共 19999 行。用长度 L=40 的滑窗
构造序列样本，再按上游的 reshape 顺序切成 `num_batches` 个 batch。

返回的 x 形状 (num_batches, 1000, 40, 24)，y 形状 (num_batches, 1000, 3)。
训练时每个 batch 只取前 `train_seqs` 条（上游口径 200）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

SEQ_LEN = 40
BATCH_SIZE = 25


def default_data_path() -> Path:
    """repo/data/c2.txt（相对本包定位，不含绝对路径）。"""
    return Path(__file__).resolve().parents[2] / "data" / "c2.txt"


def load_raw(path: str | Path | None = None):
    path = Path(path) if path is not None else default_data_path()
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            x_txt, y_txt = line.strip().split("\t")
            xs.append([int(v) for v in x_txt.split(",")])
            ys.append([int(v) for v in y_txt.split(",")])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def prepare_data(path: str | Path | None = None, seq_len: int = SEQ_LEN,
                 batch_size: int = BATCH_SIZE):
    """返回 (x, y, num_batches)，与上游 prepare_data 逐位等价。"""
    x, y = load_raw(path)
    x_new, y_new = [], []
    for j in range(0, len(x) - seq_len, 1):
        x_new.append(x[j:j + seq_len])
        y_new.append(y[j + seq_len])
    x_new = np.asarray(x_new)
    y_new = np.asarray(y_new)

    cnt = int(len(x_new) / (batch_size * seq_len))
    x_new = x_new[:cnt * (batch_size * seq_len)]
    y_new = y_new[:cnt * (batch_size * seq_len)]

    x_tensor = torch.tensor(x_new, dtype=torch.float32)
    y_tensor = torch.tensor(y_new, dtype=torch.float32)

    x_tensor = x_tensor.reshape(-1, seq_len, *x_tensor.shape[1:]).transpose(0, 1)
    y_tensor = y_tensor.reshape(-1, seq_len, *y_tensor.shape[1:]).transpose(0, 1)
    x_tensor = x_tensor.reshape(seq_len * batch_size, -1, *x_tensor.shape[2:])
    y_tensor = y_tensor.reshape(seq_len * batch_size, -1, *y_tensor.shape[2:])
    x_tensor = x_tensor.permute(1, 0, 2, 3)
    y_tensor = y_tensor.permute(1, 0, 2)

    return x_tensor, y_tensor, x_tensor.shape[0]


def batch_order(num_batches: int, seed: int | None = None) -> list[int]:
    """D11 seed 语义：batch 序 = RandomState(seed) 的固定置换，逐 epoch 同序。

    seed=None → 原序 0..n-1（上游 S3 口径）。
    """
    if seed is None:
        return list(range(num_batches))
    rng = np.random.RandomState(int(seed))
    return [int(i) for i in rng.permutation(num_batches)]