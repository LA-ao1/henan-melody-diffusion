"""Stage 2: Masked Diffusion (absorbing-state discrete diffusion) for 16th-note melody sequences.

Implements:
    - HenanMelodyDataset: PyTorch Dataset wrapper for Stage 1 .npy arrays
    - MaskedDiffusion: Forward absorbing masking process q(x_t | x_0),
      masked cross-entropy loss, and exact Markov posterior reverse sampling
      q(x_{t-1} | x_t, x_0_hat) from any noise step t_start down to 0.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from melody_diffusion.codec import MASK_ID, SEQ_LEN, VOCAB_SIZE


class HenanMelodyDataset(Dataset):
    """PyTorch Dataset loading preprocessed (N, 64) integer melody gene sequences."""

    def __init__(self, npy_path: str | Path):
        self.npy_path = Path(npy_path)
        self.data = np.load(self.npy_path).astype(np.int64)
        if self.data.ndim != 2 or self.data.shape[1] != SEQ_LEN:
            raise ValueError(f"Expected shape (N, {SEQ_LEN}), got {self.data.shape}")

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx])


class MaskedDiffusion:
    """Discrete absorbing-state Masked Diffusion over melody token sequences.

    Forward marginal distribution:
        q(x_t^(i) | x_0^(i)) = alpha_bar_t * delta(x_t^(i), x_0^(i))
                             + (1 - alpha_bar_t) * delta(x_t^(i), [MASK])

    For the linear schedule (default):
        gamma_t = 1 - alpha_bar_t = t / T
        Exact reverse unmasking probability at step t -> t - 1:
            p_unmask(t) = (alpha_bar_{t-1} - alpha_bar_t) / (1 - alpha_bar_t) = 1 / t
    """

    def __init__(
        self,
        num_steps: int = 50,
        mask_id: int = MASK_ID,
        vocab_size: int = VOCAB_SIZE,
        schedule: str = "linear",
    ):
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if schedule not in {"linear", "cosine"}:
            raise ValueError(f"Unsupported schedule '{schedule}'; expected 'linear' or 'cosine'")

        self.T = num_steps
        self.mask_id = mask_id
        self.vocab_size = vocab_size
        self.schedule = schedule

        t_grid = torch.arange(0, self.T + 1, dtype=torch.float32)
        if schedule == "linear":
            # gamma_t = t / T  =>  alpha_bar_t = 1 - t / T
            self.alpha_bar = 1.0 - (t_grid / float(self.T))
        else:
            # Cosine survival probability from 1.0 (t=0) down to 0.0 (t=T)
            self.alpha_bar = torch.cos((t_grid / float(self.T)) * (math.pi / 2.0)).pow(2)
        self.alpha_bar[0] = 1.0
        self.alpha_bar[-1] = 0.0
        self.gamma = 1.0 - self.alpha_bar  # mask probability at each step t in 0..T

    def mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """Return mask probability gamma_t in [0, 1] for integer step tensor t."""
        alpha_bar = self.alpha_bar.to(t.device)
        return 1.0 - alpha_bar[t]

    def sample_timesteps(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Sample uniform training timesteps t ~ Uniform{1, ..., T}."""
        return torch.randint(1, self.T + 1, (batch_size,), device=device, dtype=torch.long)

    def forward_mask(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor | int,
        ensure_non_empty: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Corrupt clean sequences x_0 (B, L) at timestep(s) t -> (x_t, is_masked)."""
        if x_0.ndim != 2:
            raise ValueError(f"Expected x_0 of shape (B, L), got {tuple(x_0.shape)}")
        B, L = x_0.shape
        device = x_0.device

        if isinstance(t, int):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        else:
            t_tensor = t.to(device=device, dtype=torch.long)
            if t_tensor.shape != (B,):
                raise ValueError(f"Expected t shape ({B},), got {tuple(t_tensor.shape)}")

        probs = self.mask_prob(t_tensor).unsqueeze(1)  # (B, 1)
        rand = torch.rand((B, L), device=device, dtype=torch.float32, generator=generator)
        is_masked = rand < probs

        if ensure_non_empty:
            needs_one = (t_tensor > 0) & (is_masked.sum(dim=1) == 0)
            if needs_one.any():
                rand_pos = torch.randint(0, L, (B,), device=device, generator=generator)
                row_idx = torch.nonzero(needs_one, as_tuple=False).squeeze(1)
                is_masked[row_idx, rand_pos[row_idx]] = True

        x_t = torch.where(is_masked, torch.full_like(x_0, self.mask_id), x_0)
        return x_t, is_masked

    def compute_masked_loss(
        self,
        logits: torch.Tensor,
        x_0: torch.Tensor,
        is_masked: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Cross-Entropy loss strictly on masked positions."""
        if not is_masked.any():
            raise ValueError("is_masked contains zero True entries; cannot compute masked loss")
        return F.cross_entropy(logits[is_masked], x_0[is_masked])

    def posterior_unmask_prob(self, t: int) -> float:
        """Exact Markov posterior probability of unmasking a masked position at step t -> t-1."""
        if t < 1 or t > self.T:
            raise ValueError(f"t must be in [1, {self.T}], got {t}")
        if t == 1:
            return 1.0
        a_prev = float(self.alpha_bar[t - 1].item())
        a_curr = float(self.alpha_bar[t].item())
        denom = max(1e-8, 1.0 - a_curr)
        return min(1.0, max(0.0, (a_prev - a_curr) / denom))

    @torch.no_grad()
    def reverse_denoise_step(
        self,
        predict_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_t: torch.Tensor,
        t: int,
        temperature: float = 0.9,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Perform one reverse Markov transition x_t -> x_{t-1}."""
        B, L = x_t.shape
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        logits = predict_fn(x_t, t_tensor).clone()
        logits[..., self.mask_id] = -1e9  # Never emit [MASK] as a music token

        if temperature <= 0.0:
            pred_tokens = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            flat_probs = probs.view(-1, probs.size(-1))
            pred_tokens = torch.multinomial(flat_probs, num_samples=1, generator=generator).view(B, L)

        p_unmask = self.posterior_unmask_prob(t)
        still_masked = x_t == self.mask_id
        rand = torch.rand((B, L), device=x_t.device, dtype=torch.float32, generator=generator)
        unmask_now = still_masked & (rand < p_unmask)
        return torch.where(unmask_now, pred_tokens, x_t)

    @torch.no_grad()
    def reverse_sample(
        self,
        predict_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_init: torch.Tensor,
        t_start: Optional[int] = None,
        temperature: float = 0.9,
        generator: Optional[torch.Generator] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, list[torch.Tensor]]:
        """Iteratively denoise from x_init at step t_start down to t = 0.

        - Unconditional generation: pass x_init full of MASK_ID and t_start = T
        - Partial reconstruction: pass forward-masked x_t and t_start = t
        """
        if t_start is None:
            t_start = self.T
        if t_start < 0 or t_start > self.T:
            raise ValueError(f"t_start must be in [0, {self.T}], got {t_start}")

        x = x_init.clone()
        traj = [x.clone()] if return_trajectory else []
        for t in range(t_start, 0, -1):
            x = self.reverse_denoise_step(
                predict_fn=predict_fn,
                x_t=x,
                t=t,
                temperature=temperature,
                generator=generator,
            )
            if return_trajectory:
                traj.append(x.clone())

        if return_trajectory:
            return x, traj
        return x
