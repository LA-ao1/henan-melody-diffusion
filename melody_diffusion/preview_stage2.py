"""Stage 2 verification & forward-masking preview script.

Runs self-checks on MaskedDiffusion (schedule alignment, loss computation,
and reverse sampling termination) and exports a progressive masking preview
on a real Henan 2/4 test song under data/samples_stage2/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from melody_diffusion.codec import (
    MASK_ID,
    SEQ_LEN,
    VOCAB_SIZE,
    compute_sequence_metrics,
    decode_to_midi,
    decode_to_wav,
)
from melody_diffusion.diffusion import HenanMelodyDataset, MaskedDiffusion
from melody_diffusion.prepare_henan import seq_to_readable_jianpu


def run_unit_checks(diffusion: MaskedDiffusion, train_ds: HenanMelodyDataset) -> dict:
    """Verify forward schedule accuracy, loss computation, and reverse unmasking termination."""
    gen = torch.Generator().manual_seed(42)

    # 1. Check empirical mask rate across 1,000 samples at selected steps
    sample_batch = torch.stack([train_ds[i % len(train_ds)] for i in range(1000)], dim=0)
    schedule_check = {}
    for t_val in [1, 5, 15, 25, 35, 45, 50]:
        x_t, is_masked = diffusion.forward_mask(sample_batch, t=t_val, generator=gen)
        assert x_t.shape == (1000, SEQ_LEN)
        assert is_masked.dtype == torch.bool
        assert (x_t[is_masked] == MASK_ID).all()
        assert (x_t[~is_masked] == sample_batch[~is_masked]).all()
        if t_val > 0:
            assert (is_masked.sum(dim=1) >= 1).all()
        empirical = float(is_masked.float().mean().item())
        theoretical = float(diffusion.gamma[t_val].item())
        schedule_check[f"t={t_val}"] = {
            "theoretical_mask_rate": round(theoretical, 4),
            "empirical_mask_rate": round(empirical, 4),
            "abs_diff": round(abs(empirical - theoretical), 4),
        }
        assert abs(empirical - theoretical) < 0.025, f"Mask rate mismatch at t={t_val}"

    # 2. Check masked loss with random logits
    b_x0 = sample_batch[:16]
    b_t = diffusion.sample_timesteps(16)
    b_xt, b_mask = diffusion.forward_mask(b_x0, b_t, generator=gen)
    dummy_logits = torch.zeros((16, SEQ_LEN, VOCAB_SIZE))
    loss = diffusion.compute_masked_loss(dummy_logits, b_x0, b_mask)
    expected_uniform_ce = float(np.log(VOCAB_SIZE))
    assert abs(float(loss.item()) - expected_uniform_ce) < 1e-4

    # 3. Check reverse Markov posterior terminates with ZERO [MASK] tokens
    x_init = torch.full((8, SEQ_LEN), MASK_ID, dtype=torch.long)

    def dummy_uniform_model(x_curr: torch.Tensor, t_curr: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x_curr.size(0), SEQ_LEN, VOCAB_SIZE), device=x_curr.device)

    x_out, traj = diffusion.reverse_sample(
        dummy_uniform_model,
        x_init=x_init,
        t_start=diffusion.T,
        temperature=1.0,
        generator=gen,
        return_trajectory=True,
    )
    assert len(traj) == diffusion.T + 1
    assert int((x_out == MASK_ID).sum().item()) == 0, "Reverse sampling left residual [MASK] tokens!"

    return {
        "T": diffusion.T,
        "schedule": diffusion.schedule,
        "schedule_check": schedule_check,
        "uniform_dummy_loss": round(float(loss.item()), 4),
        "reverse_final_mask_count": int((x_out == MASK_ID).sum().item()),
    }


def export_forward_mask_preview(
    root: Path,
    diffusion: MaskedDiffusion,
    test_ds: HenanMelodyDataset,
) -> dict:
    """Export progressive forward-masking examples on a real Henan test song."""
    meta_path = root / "data" / "processed" / "splits_metadata.json"
    with meta_path.open("r", encoding="utf-8") as f:
        splits_meta = json.load(f)

    # Pick sample #1 (same as sample_02 or sample_01 from test set)
    idx = 0
    info = splits_meta["test"][idx]
    x_0 = test_ds[idx].unsqueeze(0)  # (1, 64)

    out_dir = root / "data" / "samples_stage2"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator().manual_seed(2026)
    steps = [0, 5, 15, 25, 35, 45, 50]
    report_rows = []
    txt_lines = [
        f"Stage 2 Forward Masking Preview (T={diffusion.T}, schedule={diffusion.schedule})",
        f"Test sample #{idx}: {info['song_id']} {info['title']} ({info['category']}), start_bar={info['start_bar']}",
        "",
    ]

    for t_val in steps:
        if t_val == 0:
            x_t = x_0.clone()
            masked_cnt = 0
        else:
            x_t, is_masked = diffusion.forward_mask(x_0, t=t_val, generator=gen)
            masked_cnt = int(is_masked.sum().item())

        seq_list = x_t[0].tolist()
        metrics = compute_sequence_metrics([seq_list])
        tag = f"t{t_val:02d}_pct{int(round(100 * t_val / diffusion.T)):03d}"
        decode_to_midi(seq_list, out_dir / f"preview_{tag}.mid")
        decode_to_wav(seq_list, out_dir / f"preview_{tag}.wav")

        jianpu_str = seq_to_readable_jianpu(x_t[0].numpy())
        txt_lines.append(
            f"[t={t_val:02d}/{diffusion.T} | mask={masked_cnt:2d}/64 ({masked_cnt/SEQ_LEN:.1%}) | "
            f"pentatonic={metrics['pentatonic_ratio']:.1%}]\n"
            f"  ints  : {seq_list}\n"
            f"  jianpu: {jianpu_str}\n"
        )
        report_rows.append(
            {
                "t": t_val,
                "noise_pct": round(100.0 * t_val / diffusion.T, 1),
                "masked_tokens": masked_cnt,
                "actual_mask_rate": round(masked_cnt / SEQ_LEN, 4),
                "pentatonic_ratio": round(metrics["pentatonic_ratio"], 4),
                "onset_ratio": round(metrics["onset_ratio"], 4),
            }
        )

    (out_dir / "forward_mask_preview.txt").write_text("\n".join(txt_lines), encoding="utf-8")
    with (out_dir / "stage2_verification.json").open("w", encoding="utf-8") as f:
        json.dump({"sample_info": info, "progression": report_rows}, f, ensure_ascii=False, indent=2)

    return {"sample_info": info, "progression": report_rows}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    train_ds = HenanMelodyDataset(root / "data" / "processed" / "train.npy")
    test_ds = HenanMelodyDataset(root / "data" / "processed" / "test.npy")

    diffusion = MaskedDiffusion(num_steps=50, schedule="linear")
    checks = run_unit_checks(diffusion, train_ds)
    preview = export_forward_mask_preview(root, diffusion, test_ds)

    summary = {"unit_checks": checks, "preview": preview}
    with (root / "data" / "samples_stage2" / "stage2_verification.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
