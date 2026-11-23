"""Stage 3: Bidirectional Transformer denoiser for 16th-note Masked Diffusion.

Architecture (MelodyTransformerDenoiser):
    - Token embedding (V = 40, including REST=0, HOLD=1, PITCH=2..38, [MASK]=39)
    - Global sequence position embedding (0..63)
    - Intra-bar metrical position embedding (pos % 8 for 2/4 16th-note grid)
    - Sinusoidal timestep embedding + MLP projection for noise step t
    - Pre-LN bidirectional TransformerEncoder (no causal mask)
    - Final LayerNorm + linear vocabulary projection -> logits (B, L, V)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from melody_diffusion.codec import SEQ_LEN, STEPS_PER_BAR, VOCAB_SIZE


class SinusoidalTimestepEmbedding(nn.Module):
    """Standard sinusoidal embedding mapping integer diffusion steps t -> (B, dim)."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Timestep embedding dim must be even, got {dim}")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / max(1, half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class MelodyTransformerDenoiser(nn.Module):
    """Bidirectional Transformer Encoder predicting clean token logits from (x_t, t)."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        seq_len: int = SEQ_LEN,
        steps_per_bar: int = STEPS_PER_BAR,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 384,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.steps_per_bar = steps_per_bar
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.bar_pos_emb = nn.Embedding(steps_per_bar, d_model)

        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.in_norm = nn.LayerNorm(d_model)
        self.in_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.bar_pos_emb.weight, mean=0.0, std=0.02)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict clean melody logits (B, L, V) given noisy sequence x_t (B, L) and step t (B,)."""
        if x_t.ndim != 2:
            raise ValueError(f"Expected x_t of shape (B, L), got {tuple(x_t.shape)}")
        B, L = x_t.shape
        if L > self.seq_len:
            raise ValueError(f"Sequence length {L} exceeds configured max seq_len {self.seq_len}")
        if t.shape != (B,):
            raise ValueError(f"Expected t of shape ({B},), got {tuple(t.shape)}")

        pos = torch.arange(L, device=x_t.device, dtype=torch.long)
        bar_pos = pos % self.steps_per_bar

        h = (
            self.tok_emb(x_t)
            + self.pos_emb(pos).unsqueeze(0)
            + self.bar_pos_emb(bar_pos).unsqueeze(0)
            + self.time_mlp(t).unsqueeze(1)
        )
        h = self.in_drop(self.in_norm(h))
        # Bidirectional attention (no causal mask)
        h = self.encoder(h)
        h = self.out_norm(h)
        return self.head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
