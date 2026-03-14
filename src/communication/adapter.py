"""
Receiver adapter module.

Trains a lightweight adapter that helps the receiver (e.g., Judger)
better process compressed communication inputs, addressing the
distribution shift caused by communication compression.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ReceiverAdapter:
    """
    Lightweight adapter that maps compressed communication embeddings
    closer to the full communication distribution.

    When torch is available, uses a small MLP.
    Falls back to a linear projection with numpy otherwise.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256,
                 lr: float = 1e-4, epochs: int = 5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.trained = False

        if HAS_TORCH:
            self._init_torch()
        else:
            self._init_numpy()

    def _init_torch(self):
        self.model = _AdapterMLP(self.input_dim, self.hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def _init_numpy(self):
        self.W = np.eye(self.input_dim, dtype=np.float32)
        self.b = np.zeros(self.input_dim, dtype=np.float32)

    def train(self, compressed_embeddings: np.ndarray,
              full_embeddings: np.ndarray):
        """
        Train the adapter to map compressed → full embeddings.
        Both inputs: (N, dim)
        """
        assert compressed_embeddings.shape == full_embeddings.shape

        if HAS_TORCH:
            self._train_torch(compressed_embeddings, full_embeddings)
        else:
            self._train_numpy(compressed_embeddings, full_embeddings)
        self.trained = True

    def _train_torch(self, comp: np.ndarray, full: np.ndarray):
        X = torch.tensor(comp, dtype=torch.float32)
        Y = torch.tensor(full, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                             shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def _train_numpy(self, comp: np.ndarray, full: np.ndarray):
        # Least-squares linear projection: full ≈ comp @ W + b
        n = comp.shape[0]
        ones = np.ones((n, 1), dtype=np.float32)
        A = np.hstack([comp, ones])
        solution, _, _, _ = np.linalg.lstsq(A, full, rcond=None)
        self.W = solution[:-1]
        self.b = solution[-1]

    def adapt(self, compressed_embedding: np.ndarray) -> np.ndarray:
        """Transform a compressed embedding to approximate full distribution."""
        if not self.trained:
            return compressed_embedding

        if HAS_TORCH:
            self.model.eval()
            with torch.no_grad():
                x = torch.tensor(compressed_embedding, dtype=torch.float32)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                out = self.model(x).numpy()
            return out.squeeze()
        else:
            return compressed_embedding @ self.W + self.b

    def compute_shift(self, comp_emb: np.ndarray,
                      full_emb: np.ndarray) -> dict:
        """Measure distribution shift between compressed and full embeddings."""
        comp_mean = comp_emb.mean(axis=0)
        full_mean = full_emb.mean(axis=0)

        cos_dist = 1.0 - (
            np.dot(comp_mean, full_mean)
            / (np.linalg.norm(comp_mean) * np.linalg.norm(full_mean) + 1e-8)
        )
        l2_dist = np.linalg.norm(comp_mean - full_mean)

        return {
            "cosine_distance": float(cos_dist),
            "l2_distance": float(l2_dist),
        }


if HAS_TORCH:
    class _AdapterMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )
            self.skip = nn.Identity()

        def forward(self, x):
            return x + self.net(x)  # residual connection
