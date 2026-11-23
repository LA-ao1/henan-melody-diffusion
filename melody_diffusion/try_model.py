"""Interactive & CLI trial script for teammates to test the trained Henan Masked Diffusion model.

Usage:
    # Interactive menu (guided prompt):
    python3 -m melody_diffusion.try_model

    # Generate 3 brand-new Henan-style melodies from scratch:
    python3 -m melody_diffusion.try_model --mode generate --num 3 --temperature 0.9

    # Corrupt a real test song by 50% masking and let the model reconstruct it:
    python3 -m melody_diffusion.try_model --mode reconstruct --song-idx 1 --mask-pct 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from melody_diffusion.codec import (
    MASK_ID,
    SEQ_LEN,
    compute_sequence_metrics,
    decode_to_midi,
    decode_to_musicxml,
    decode_to_wav,
)
from melody_diffusion.diffusion import MaskedDiffusion
from melody_diffusion.model import MelodyTransformerDenoiser
from melody_diffusion.prepare_henan import seq_to_readable_jianpu


def load_trained_model(ckpt_path: Path, device: torch.device) -> tuple[MelodyTransformerDenoiser, MaskedDiffusion, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt["model_config"]
    diff_cfg = ckpt.get("diffusion_config", {"num_steps": 50, "schedule": "linear"})

    model = MelodyTransformerDenoiser(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    diffusion = MaskedDiffusion(**diff_cfg)
    return model, diffusion, ckpt


def run_generate(
    model: MelodyTransformerDenoiser,
    diffusion: MaskedDiffusion,
    out_dir: Path,
    num: int = 3,
    temperature: float = 0.9,
    seed: int = 2026,
    device: torch.device = torch.device("cpu"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device=device).manual_seed(seed)
    x_init = torch.full((num, SEQ_LEN), MASK_ID, dtype=torch.long, device=device)
    samples = diffusion.reverse_sample(
        model,
        x_init=x_init,
        t_start=diffusion.T,
        temperature=temperature,
        generator=gen,
    ).cpu().numpy()

    print(f"\n=== 生成 {num} 首新旋律 (temperature={temperature}, seed={seed}) ===")
    for i in range(num):
        seq = samples[i]
        m = compute_sequence_metrics([seq.tolist()])
        stem = f"gen_{i+1:02d}_seed{seed}"
        decode_to_midi(seq.tolist(), out_dir / f"{stem}.mid")
        decode_to_musicxml(seq.tolist(), out_dir / f"{stem}.musicxml", title=f"Generated #{i+1}")
        decode_to_wav(seq.tolist(), out_dir / f"{stem}.wav")
        jianpu = seq_to_readable_jianpu(seq)
        (out_dir / f"{stem}.txt").write_text(
            f"mode: unconditional generation\nseed: {seed}\ntemperature: {temperature}\n"
            f"pentatonic_ratio: {m['pentatonic_ratio']:.2%}\norphan_hold_ratio: {m['orphan_hold_ratio']:.2%}\n"
            f"jianpu: {jianpu}\nints: {seq.tolist()}\n",
            encoding="utf-8",
        )
        print(f"\n[生成 #{i+1}] 五声纯度={m['pentatonic_ratio']:.1%} | 孤立HOLD={m['orphan_hold_ratio']:.1%}")
        print(f"  简谱: {jianpu}")
        print(f"  已导出: {out_dir / (stem + '.wav')} / .mid / .musicxml")


def run_reconstruct(
    root: Path,
    model: MelodyTransformerDenoiser,
    diffusion: MaskedDiffusion,
    out_dir: Path,
    song_idx: int = 0,
    mask_pct: float = 50.0,
    temperature: float = 0.9,
    seed: int = 2026,
    device: torch.device = torch.device("cpu"),
) -> None:
    test_arr = np.load(root / "data" / "processed" / "test.npy")
    with (root / "data" / "processed" / "splits_metadata.json").open("r", encoding="utf-8") as f:
        test_meta = json.load(f)["test"]

    if song_idx < 0 or song_idx >= len(test_arr):
        raise ValueError(f"song_idx must be between 0 and {len(test_arr) - 1}, got {song_idx}")

    t_start = max(1, min(diffusion.T, int(round((mask_pct / 100.0) * diffusion.T))))
    x_0 = torch.from_numpy(test_arr[song_idx : song_idx + 1]).long().to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    x_masked, is_masked = diffusion.forward_mask(x_0, t=t_start, ensure_non_empty=True, generator=gen)
    x_recon = diffusion.reverse_sample(
        model,
        x_init=x_masked,
        t_start=t_start,
        temperature=temperature,
        generator=gen,
    )

    seq_orig = x_0[0].cpu().numpy()
    seq_mask = x_masked[0].cpu().numpy()
    seq_reco = x_recon[0].cpu().numpy()

    info = test_meta[song_idx]
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = info["song_id"].replace(":", "_")
    prefix = f"recon_{safe_id}_idx{song_idx}_t{t_start:02d}"

    for label, seq in [("orig", seq_orig), ("masked", seq_mask), ("recon", seq_reco)]:
        decode_to_midi(seq.tolist(), out_dir / f"{prefix}_{label}.mid")
        decode_to_musicxml(seq.tolist(), out_dir / f"{prefix}_{label}.musicxml", title=f"{info['title']} ({label})")
        decode_to_wav(seq.tolist(), out_dir / f"{prefix}_{label}.wav")

    m_orig = compute_sequence_metrics([seq_orig.tolist()])
    m_reco = compute_sequence_metrics([seq_reco.tolist()])
    masked_n = int(is_masked.sum().item())
    exact_match = float((seq_orig == seq_reco).mean())
    masked_acc = float((seq_orig[is_masked[0].cpu().numpy()] == seq_reco[is_masked[0].cpu().numpy()]).mean())

    summary_txt = (
        f"Test song #{song_idx}: {info['song_id']} {info['title']} ({info['category']})\n"
        f"Noise step t={t_start}/{diffusion.T} (~{t_start/diffusion.T:.0%}, masked {masked_n}/64 tokens)\n"
        f"Overall match with original: {exact_match:.1%} | Masked-position recovery acc: {masked_acc:.1%}\n"
        f"Pentatonic purity: orig={m_orig['pentatonic_ratio']:.1%} -> recon={m_reco['pentatonic_ratio']:.1%}\n"
        f"Orphan HOLD ratio: recon={m_reco['orphan_hold_ratio']:.1%}\n\n"
        f"[原曲]   {seq_to_readable_jianpu(seq_orig)}\n"
        f"[掩码后] {seq_to_readable_jianpu(seq_mask)}\n"
        f"[重建后] {seq_to_readable_jianpu(seq_reco)}\n"
    )
    (out_dir / f"{prefix}_compare.txt").write_text(summary_txt, encoding="utf-8")
    print("\n" + summary_txt)
    print(f"三态对比文件 (.wav / .mid / .musicxml / .txt) 已保存至: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Try out trained Henan Masked Diffusion model")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mode", type=str, default=None, choices=["generate", "reconstruct"])
    parser.add_argument("--num", type=int, default=3, help="Number of melodies to generate")
    parser.add_argument("--song-idx", type=int, default=0, help="Test song index (0..115)")
    parser.add_argument("--mask-pct", type=float, default=50.0, help="Mask percentage (0..100)")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.project_root
    ckpt_path = args.checkpoint if args.checkpoint is not None else (root / "checkpoints" / "best_model.pt")
    out_dir = args.out_dir if args.out_dir is not None else (root / "outputs" / "try")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, diffusion, ckpt = load_trained_model(ckpt_path, device)
    print(f"已加载模型: {ckpt_path} (epoch={ckpt.get('epoch')}, best_val_loss={ckpt.get('best_val_loss', 0):.4f}, device={device})")

    mode: Optional[str] = args.mode
    if mode is None:
        print("\n请选择试用模式:")
        print("  1. 无条件从零创作新河南民歌")
        print("  2. 选取一首测试集真实民歌 -> 掩码损坏 -> 模型重建对比")
        choice = input("请输入 1 或 2 (默认 1): ").strip() or "1"
        if choice == "2":
            mode = "reconstruct"
            idx_in = input("请输入测试曲目编号 0~115 (默认 1，即《天上黄河几道湾》): ").strip()
            if idx_in:
                args.song_idx = int(idx_in)
            pct_in = input("请输入遮挡破坏百分比 10~100 (默认 50): ").strip()
            if pct_in:
                args.mask_pct = float(pct_in)
        else:
            mode = "generate"
            num_in = input("请输入要生成几首新旋律 (默认 3): ").strip()
            if num_in:
                args.num = int(num_in)

    if mode == "generate":
        run_generate(model, diffusion, out_dir, num=args.num, temperature=args.temperature, seed=args.seed, device=device)
    else:
        run_reconstruct(
            root, model, diffusion, out_dir,
            song_idx=args.song_idx, mask_pct=args.mask_pct,
            temperature=args.temperature, seed=args.seed, device=device,
        )


if __name__ == "__main__":
    main()
