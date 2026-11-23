"""Stage 4: Training, fixed-seed validation, and 4-dimensional checkup for Masked Diffusion.

Runs smoothly on CPU, Apple Silicon (MPS), or CUDA GPU (local / Google Colab).

Usage:
    python3 -m melody_diffusion.train --epochs 100 --batch-size 64
    python3 -m melody_diffusion.train --epochs 120 --resume checkpoints/last_model.pt
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from melody_diffusion.codec import (
    MASK_ID,
    SEQ_LEN,
    STEPS_PER_BAR,
    VOCAB_SIZE,
    compute_sequence_metrics,
    decode_to_midi,
    decode_to_wav,
)
from melody_diffusion.diffusion import HenanMelodyDataset, MaskedDiffusion
from melody_diffusion.model import MelodyTransformerDenoiser
from melody_diffusion.prepare_henan import seq_to_readable_jianpu


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_fixed_val_pairs(
    val_ds: HenanMelodyDataset,
    diffusion: MaskedDiffusion,
    repeats: int = 5,
    seed: int = 12345,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Pre-generate deterministic (x_0, t, x_t, is_masked) batches for jitter-free validation loss."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x0_all = torch.from_numpy(val_ds.data)  # (N_val, L)
    pairs = []
    for _ in range(repeats):
        t = torch.randint(1, diffusion.T + 1, (x0_all.size(0),), generator=gen, dtype=torch.long)
        x_t, is_masked = diffusion.forward_mask(x0_all, t, ensure_non_empty=True, generator=gen)
        pairs.append((x0_all, t, x_t, is_masked))
    return pairs


@torch.no_grad()
def evaluate_val_loss(
    model: MelodyTransformerDenoiser,
    diffusion: MaskedDiffusion,
    val_pairs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for x_0, t, x_t, is_masked in val_pairs:
        logits = model(x_t.to(device), t.to(device))
        loss = diffusion.compute_masked_loss(logits, x_0.to(device), is_masked.to(device))
        losses.append(float(loss.item()))
    return float(sum(losses) / len(losses))


def max_contiguous_overlap(gen_seq: np.ndarray, train_seqs: np.ndarray) -> int:
    """Exact longest contiguous subarray match length between gen_seq (L,) and train_seqs (N, L)."""
    L = gen_seq.shape[0]
    best = 0
    for shift in range(-(L - 1), L):
        g0 = max(0, -shift)
        t0 = max(0, shift)
        length = L - abs(shift)
        if length <= best:
            continue
        eq = train_seqs[:, t0 : t0 + length] == gen_seq[g0 : g0 + length]
        cur = np.zeros(eq.shape[0], dtype=np.int32)
        for col in range(length):
            cur = np.where(eq[:, col], cur + 1, 0)
            m = int(cur.max())
            if m > best:
                best = m
    return best


@torch.no_grad()
def run_four_dim_checkup(
    model: MelodyTransformerDenoiser,
    diffusion: MaskedDiffusion,
    train_seqs_np: np.ndarray,
    val_loss: float,
    epoch: int,
    save_dir: Path,
    device: torch.device,
    num_samples: int = 20,
    temperature: float = 0.9,
    seed: int = 2026,
) -> dict:
    """Run the 4-dimensional health checkup:
    1) Validation loss
    2) Syntax validity (orphan HOLD ratio)
    3) Pentatonic mode purity (Henan 2/4 baseline ~97%)
    4) Novelty check against training corpus (longest contiguous match)
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed + epoch)
    x_init = torch.full((num_samples, SEQ_LEN), MASK_ID, dtype=torch.long, device=device)
    samples = diffusion.reverse_sample(
        model,
        x_init=x_init,
        t_start=diffusion.T,
        temperature=temperature,
        generator=gen,
    )
    samples_np = samples.cpu().numpy()

    seq_metrics = compute_sequence_metrics(samples_np.tolist())
    overlaps = [max_contiguous_overlap(s, train_seqs_np) for s in samples_np]
    exact_copies = sum(1 for ov in overlaps if ov == SEQ_LEN)

    # Export first 2 generated samples for inspection/listening
    sample_out = save_dir / "checkup_samples"
    sample_out.mkdir(parents=True, exist_ok=True)
    preview_lines = [f"=== Epoch {epoch:03d} | val_loss={val_loss:.4f} | pentatonic={seq_metrics['pentatonic_ratio']:.1%} | orphan_hold={seq_metrics['orphan_hold_ratio']:.2%} ==="]
    for i in range(min(2, num_samples)):
        stem = f"epoch_{epoch:03d}_sample_{i+1:02d}"
        decode_to_midi(samples_np[i].tolist(), sample_out / f"{stem}.mid")
        decode_to_wav(samples_np[i].tolist(), sample_out / f"{stem}.wav")
        preview_lines.append(f"[Sample {i+1} | max_train_overlap={overlaps[i]}/{SEQ_LEN}]")
        preview_lines.append(f"  jianpu: {seq_to_readable_jianpu(samples_np[i])}")
        preview_lines.append(f"  ints  : {samples_np[i].tolist()}")
    (sample_out / f"epoch_{epoch:03d}_preview.txt").write_text("\n".join(preview_lines) + "\n", encoding="utf-8")

    return {
        "epoch": epoch,
        "val_loss": round(val_loss, 4),
        "orphan_hold_count": seq_metrics["orphan_hold_count"],
        "orphan_hold_ratio": round(seq_metrics["orphan_hold_ratio"], 4),
        "pentatonic_ratio": round(seq_metrics["pentatonic_ratio"], 4),
        "onset_ratio": round(seq_metrics["onset_ratio"], 4),
        "hold_ratio": round(seq_metrics["hold_ratio"], 4),
        "rest_ratio": round(seq_metrics["rest_ratio"], 4),
        "avg_max_train_overlap": round(float(np.mean(overlaps)), 2),
        "max_train_overlap": int(max(overlaps)),
        "exact_copy_count": exact_copies,
    }


def save_training_plot(history: Dict[str, list], out_path: Path) -> bool:
    """Save loss & checkup metrics plot if matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    epochs = [row["epoch"] for row in history["epochs"]]
    train_losses = [row["train_loss"] for row in history["epochs"]]
    val_losses = [row["val_loss"] for row in history["epochs"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, train_losses, label="Train Loss", linewidth=1.8)
    axes[0].plot(epochs, val_losses, label="Val Loss", linewidth=1.8)
    axes[0].set_title("Masked Diffusion Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if history["checkups"]:
        ck_ep = [c["epoch"] for c in history["checkups"]]
        penta = [c["pentatonic_ratio"] * 100.0 for c in history["checkups"]]
        orphan = [c["orphan_hold_ratio"] * 100.0 for c in history["checkups"]]
        axes[1].plot(ck_ep, penta, marker="o", label="Pentatonic Purity (%)")
        axes[1].plot(ck_ep, orphan, marker="s", label="Orphan HOLD (%)")
        axes[1].axhline(97.2, linestyle="--", alpha=0.5, label="Henan Train Baseline (97.2%)")
        axes[1].set_title("Generated Sample Checkup")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Percentage (%)")
        axes[1].set_ylim(-2, 105)
        axes[1].grid(alpha=0.3)
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 4 Masked Diffusion on Henan 2/4 melodies")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--save-dir", type=Path, default=None, help="Defaults to <project-root>/checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    root = args.project_root
    save_dir = args.save_dir if args.save_dir is not None else (root / "checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_ds = HenanMelodyDataset(root / "data" / "processed" / "train.npy")
    val_ds = HenanMelodyDataset(root / "data" / "processed" / "val.npy")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    diffusion = MaskedDiffusion(num_steps=args.T, schedule=args.schedule)
    val_pairs = build_fixed_val_pairs(val_ds, diffusion, repeats=5, seed=args.seed + 999)

    model_cfg = {
        "vocab_size": VOCAB_SIZE,
        "seq_len": SEQ_LEN,
        "steps_per_bar": STEPS_PER_BAR,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }
    diffusion_cfg = {"num_steps": args.T, "schedule": args.schedule}

    model = MelodyTransformerDenoiser(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    start_epoch = 1
    best_val_loss = float("inf")
    history: Dict[str, list] = {"epochs": [], "checkups": []}

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        hist_path = save_dir / "training_history.json"
        if hist_path.exists():
            history = json.loads(hist_path.read_text(encoding="utf-8"))
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best_val_loss={best_val_loss:.4f})")

    print(
        f"Training on device={device} | params={model.count_parameters():,} | "
        f"train={len(train_ds)} | val={len(val_ds)} | T={args.T} ({args.schedule})"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        for x_0 in train_loader:
            x_0 = x_0.to(device)
            t = diffusion.sample_timesteps(x_0.size(0), device=device)
            x_t, is_masked = diffusion.forward_mask(x_0, t, ensure_non_empty=True)

            logits = model(x_t, t)
            loss = diffusion.compute_masked_loss(logits, x_0, is_masked)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.item())

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        val_loss = evaluate_val_loss(model, diffusion, val_pairs, device=device)
        elapsed = time.perf_counter() - t0
        cur_lr = float(optimizer.param_groups[0]["lr"])

        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "lr": round(cur_lr, 6),
                "sec": round(elapsed, 2),
            }
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        ckpt_payload = {
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "model_config": model_cfg,
            "diffusion_config": diffusion_cfg,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(ckpt_payload, save_dir / "last_model.pt")
        if is_best:
            torch.save(ckpt_payload, save_dir / "best_model.pt")

        msg = (
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"lr={cur_lr:.2e} | {elapsed:.1f}s"
        )
        if is_best:
            msg += " [BEST]"
        print(msg, flush=True)

        if (epoch % args.eval_every == 0) or (epoch == args.epochs):
            ck = run_four_dim_checkup(
                model=model,
                diffusion=diffusion,
                train_seqs_np=train_ds.data,
                val_loss=val_loss,
                epoch=epoch,
                save_dir=save_dir,
                device=device,
                num_samples=20,
                temperature=args.temperature,
                seed=args.seed + 2000,
            )
            history["checkups"].append(ck)
            print(
                f"  [Checkup @ {epoch:03d}] pentatonic={ck['pentatonic_ratio']:.1%} | "
                f"orphan_hold={ck['orphan_hold_ratio']:.2%} | "
                f"avg_overlap={ck['avg_max_train_overlap']}/{SEQ_LEN} (max={ck['max_train_overlap']}, copies={ck['exact_copy_count']})",
                flush=True,
            )

        with (save_dir / "training_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    plotted = save_training_plot(history, save_dir / "training_curves.png")
    print(f"Training complete. Best val_loss={best_val_loss:.4f}. Saved to {save_dir} (plot={plotted}).")


if __name__ == "__main__":
    main()
