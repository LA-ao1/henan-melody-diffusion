"""Stage 3 verification script: architecture, bidirectional context, gradient flow, and CPU speed benchmark."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from melody_diffusion.codec import MASK_ID, SEQ_LEN, VOCAB_SIZE
from melody_diffusion.diffusion import HenanMelodyDataset, MaskedDiffusion
from melody_diffusion.model import MelodyTransformerDenoiser


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    train_ds = HenanMelodyDataset(root / "data" / "processed" / "train.npy")
    diffusion = MaskedDiffusion(num_steps=50, schedule="linear")

    model = MelodyTransformerDenoiser(
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=384,
        dropout=0.15,
    )
    n_params = model.count_parameters()

    # 1. Shape & bidirectional context verification
    model.eval()
    x_a = torch.full((2, SEQ_LEN), MASK_ID, dtype=torch.long)
    x_b = x_a.clone()
    x_b[:, -1] = 14  # change only the very last token (step 63)
    t_test = torch.tensor([25, 25], dtype=torch.long)

    with torch.no_grad():
        logits_a = model(x_a, t_test)
        logits_b = model(x_b, t_test)

    assert logits_a.shape == (2, SEQ_LEN, VOCAB_SIZE)
    assert torch.isfinite(logits_a).all()
    step0_diff = float((logits_a[:, 0, :] - logits_b[:, 0, :]).abs().max().item())
    assert step0_diff > 1e-6, "Step 0 logits did not change when step 63 changed (attention not bidirectional!)"

    # 2. Training step & gradient flow check on real Stage 1 batch
    model.train()
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    x_0 = next(iter(loader))
    t = diffusion.sample_timesteps(x_0.size(0))
    x_t, is_masked = diffusion.forward_mask(x_0, t)

    logits = model(x_t, t)
    loss = diffusion.compute_masked_loss(logits, x_0, is_masked)
    loss.backward()

    missing_grads = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing_grads, f"Parameters missing gradients: {missing_grads}"

    # 3. Full reverse sampling smoke test (T=50 steps on 4 sequences)
    model.eval()
    x_init = torch.full((4, SEQ_LEN), MASK_ID, dtype=torch.long)
    t_samp_start = time.perf_counter()
    samples = diffusion.reverse_sample(model, x_init=x_init, t_start=diffusion.T, temperature=1.0)
    samp_sec = time.perf_counter() - t_samp_start
    assert samples.shape == (4, SEQ_LEN)
    assert int((samples == MASK_ID).sum().item()) == 0

    # 4. Benchmark 1 full epoch on CPU (2390 samples, batch_size=64 -> 38 batches)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    t_ep_start = time.perf_counter()
    for batch_x0 in loader:
        opt.zero_grad()
        bt = diffusion.sample_timesteps(batch_x0.size(0))
        bxt, bmask = diffusion.forward_mask(batch_x0, bt)
        out = model(bxt, bt)
        bloss = diffusion.compute_masked_loss(out, batch_x0, bmask)
        bloss.backward()
        opt.step()
    epoch_sec = time.perf_counter() - t_ep_start

    report = {
        "model_class": "MelodyTransformerDenoiser",
        "config": {
            "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN,
            "steps_per_bar": 8,
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 384,
            "dropout": 0.15,
        },
        "trainable_parameters": n_params,
        "bidirectional_step63_to_step0_max_diff": round(step0_diff, 6),
        "initial_batch_loss": round(float(loss.item()), 4),
        "all_params_have_gradients": True,
        "reverse_sample_4x64_sec": round(samp_sec, 3),
        "cpu_1_epoch_sec": round(epoch_sec, 2),
        "est_100_epochs_min": round(epoch_sec * 100 / 60.0, 1),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
