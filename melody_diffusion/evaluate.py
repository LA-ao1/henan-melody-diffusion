"""Stage 5: Comprehensive evaluation across all 116 test sequences & 4 musicological phenomena.

Evaluates trained checkpoints/best_model.pt at noise levels t in [0, 5, 15, 25, 35, 45, 50]:
    1) Pentatonic Mode Attractor (pentatonic ratio, 12-tone JS divergence vs original)
    2) Melodic Contour (pitch contour correlation with original, stepwise <=2st interval ratio)
    3) Rhythmic Density & Stability (onset density/bar, downbeat ratio, mean duration, orphan HOLD %)
    4) Motif Repetition & Self-Similarity (4-step motif repeat rate, bar-level self-similarity)

Exports report-ready figures, JSON metrics, and 3 case-study packages (.mid/.musicxml/.wav/.txt)
under results/stage5/.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from melody_diffusion.codec import (
    HOLD_ID,
    MASK_ID,
    PENTATONIC_MOD12,
    PITCH_MIN,
    PITCH_MAX,
    REST_ID,
    SEQ_LEN,
    STEPS_PER_BAR,
    compute_sequence_metrics,
    decode_to_midi,
    decode_to_musicxml,
    decode_to_wav,
    token_to_midi,
)
from melody_diffusion.diffusion import MaskedDiffusion
from melody_diffusion.model import MelodyTransformerDenoiser
from melody_diffusion.prepare_henan import seq_to_readable_jianpu

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

PC_LABELS = ["C(1)", "#C", "D(2)", "bE", "E(3)", "F(4)", "#F", "G(5)", "#G", "A(6)", "bB", "B(7)"]


def extract_pitch_contour_track(seq: Sequence[int]) -> np.ndarray:
    """Return 64-step continuous pitch track (interpolating/holding across HOLD and REST) for contour corr."""
    track = np.full(len(seq), np.nan, dtype=np.float64)
    active = np.nan
    for i, tok in enumerate(seq):
        tok = int(tok)
        if 2 <= tok < MASK_ID:
            active = float(token_to_midi(tok))
            track[i] = active
        elif tok == HOLD_ID and not np.isnan(active):
            track[i] = active
    valid = ~np.isnan(track)
    if not valid.any():
        return np.zeros(len(seq), dtype=np.float64)
    idx = np.arange(len(seq))
    return np.interp(idx, idx[valid], track[valid])


def contour_correlation(seq_a: Sequence[int], seq_b: Sequence[int]) -> float:
    """Pearson correlation of mean-centered pitch contours between two 64-step sequences."""
    ta = extract_pitch_contour_track(seq_a)
    tb = extract_pitch_contour_track(seq_b)
    da = ta - ta.mean()
    db = tb - tb.mean()
    denom = float(np.linalg.norm(da) * np.linalg.norm(db))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(da, db) / denom)


def pitch_class_histogram(seqs: np.ndarray) -> np.ndarray:
    """Normalized 12-semitone pitch-class distribution across sounding steps."""
    counts = np.zeros(12, dtype=np.float64)
    for seq in seqs:
        active_midi = None
        for tok in seq.tolist():
            if 2 <= tok < MASK_ID:
                active_midi = token_to_midi(tok)
                counts[active_midi % 12] += 1.0
            elif tok == HOLD_ID and active_midi is not None:
                counts[active_midi % 12] += 1.0
            else:
                active_midi = None
    total = counts.sum()
    return counts / total if total > 0 else np.ones(12) / 12.0


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence (base 2, bounded in [0, 1])."""
    p_s = np.clip(p, eps, 1.0)
    q_s = np.clip(q, eps, 1.0)
    p_s /= p_s.sum()
    q_s /= q_s.sum()
    m = 0.5 * (p_s + q_s)
    kl_pm = float(np.sum(p_s * np.log2(p_s / m)))
    kl_qm = float(np.sum(q_s * np.log2(q_s / m)))
    return 0.5 * (kl_pm + kl_qm)


def compute_phenomena_metrics(
    orig_arr: np.ndarray,
    cand_arr: np.ndarray,
    orig_pc_dist: np.ndarray,
) -> dict:
    """Compute quantitative metrics for all 4 musical phenomena on cand_arr vs orig_arr."""
    base = compute_sequence_metrics(cand_arr.tolist())
    cand_pc = pitch_class_histogram(cand_arr)
    js_div = js_divergence(cand_pc, orig_pc_dist)

    contour_corrs = []
    stepwise_hits = 0
    total_intervals = 0
    abs_intervals = []
    downbeat_onsets = 0
    total_onsets = 0
    note_durations = []
    motif_4gram_rep_rates = []
    bar_self_sims = []

    for i in range(cand_arr.shape[0]):
        seq = cand_arr[i].tolist()
        contour_corrs.append(contour_correlation(orig_arr[i].tolist(), seq))

        # Intervals, durations, downbeat onsets
        onset_midis = []
        idx = 0
        while idx < len(seq):
            tok = seq[idx]
            if 2 <= tok < MASK_ID:
                midi = token_to_midi(tok)
                onset_midis.append(midi)
                total_onsets += 1
                if idx % 4 == 0:  # quarter-note beat head in 16th grid
                    downbeat_onsets += 1
                dur = 1
                while idx + dur < len(seq) and seq[idx + dur] == HOLD_ID:
                    dur += 1
                note_durations.append(dur)
                idx += dur
            else:
                idx += 1

        for j in range(1, len(onset_midis)):
            diff = abs(onset_midis[j] - onset_midis[j - 1])
            abs_intervals.append(diff)
            total_intervals += 1
            if diff <= 2:
                stepwise_hits += 1

        # 4-step motif repetition rate (non-trivial 4-grams with >=1 onset occurring >=2 times in 64 steps)
        grams = []
        for pos in range(0, SEQ_LEN - 3, 2):  # stride 2 (eighth-note aligned)
            g = tuple(seq[pos : pos + 4])
            if any(2 <= t < MASK_ID for t in g):
                grams.append(g)
        if grams:
            counts: Dict[tuple, int] = {}
            for g in grams:
                counts[g] = counts.get(g, 0) + 1
            rep = sum(1 for g in grams if counts[g] >= 2)
            motif_4gram_rep_rates.append(rep / len(grams))

        # Bar-level self-similarity (8 bars of 8 steps, exact token overlap between distinct bars)
        bars = [seq[b * 8 : (b + 1) * 8] for b in range(8)]
        sims = []
        for b1 in range(8):
            for b2 in range(b1 + 1, 8):
                sims.append(sum(1 for a, b in zip(bars[b1], bars[b2]) if a == b) / 8.0)
        bar_self_sims.append(float(np.mean(sims)))

    token_match_rate = float((orig_arr == cand_arr).mean())
    onsets_per_bar = (total_onsets / cand_arr.shape[0]) / 8.0

    return {
        "token_match_rate": round(token_match_rate, 4),
        # 1. Mode attractor
        "pentatonic_ratio": round(base["pentatonic_ratio"], 4),
        "pitch_class_js_div": round(js_div, 5),
        "pitch_class_dist": [round(float(x), 4) for x in cand_pc],
        # 2. Melodic contour
        "contour_corr": round(float(np.mean(contour_corrs)), 4),
        "stepwise_ratio": round(stepwise_hits / total_intervals if total_intervals else 0.0, 4),
        "mean_abs_interval_semitones": round(float(np.mean(abs_intervals)) if abs_intervals else 0.0, 3),
        # 3. Rhythmic density & stability
        "onsets_per_bar": round(onsets_per_bar, 3),
        "mean_note_duration_16ths": round(float(np.mean(note_durations)) if note_durations else 0.0, 3),
        "downbeat_onset_ratio": round(downbeat_onsets / total_onsets if total_onsets else 0.0, 4),
        "orphan_hold_ratio": round(base["orphan_hold_ratio"], 4),
        "rest_ratio": round(base["rest_ratio"], 4),
        # 4. Motif repetition & self-similarity
        "motif_4gram_rep_rate": round(float(np.mean(motif_4gram_rep_rates)) if motif_4gram_rep_rates else 0.0, 4),
        "bar_self_similarity": round(float(np.mean(bar_self_sims)), 4),
    }


def make_random_fill_baseline(x_masked: np.ndarray, seed: int = 999) -> np.ndarray:
    """Randomly replace MASK_ID tokens uniformly from non-mask vocab [0..38] for comparison."""
    rng = np.random.default_rng(seed)
    out = x_masked.copy()
    mask = out == MASK_ID
    out[mask] = rng.integers(0, MASK_ID, size=int(mask.sum()))
    return out


def plot_metrics_curves(rows: List[dict], rand_rows: List[dict], out_path: Path) -> None:
    pcts = [r["noise_pct"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    # Panel A: Mode Attractor
    ax = axes[0, 0]
    ax.plot(pcts, [r["pentatonic_ratio"] * 100 for r in rows], "o-", linewidth=2, label="掩码扩散重建 (Pentatonic %)")
    ax.plot(pcts, [r["pentatonic_ratio"] * 100 for r in rand_rows], "s--", alpha=0.75, label="均匀随机填充基线")
    ax.axhline(rows[0]["pentatonic_ratio"] * 100, linestyle=":", color="gray", label=f"原曲基线 ({rows[0]['pentatonic_ratio']*100:.1f}%)")
    ax.set_title("1. 五声调式吸引子 (Pentatonic Mode Attractor)")
    ax.set_xlabel("掩码噪声比例 (%)")
    ax.set_ylabel("五声骨干音占比 (%)")
    ax.set_ylim(35, 102)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Panel B: Melodic Contour
    ax = axes[0, 1]
    ax.plot(pcts, [r["contour_corr"] for r in rows], "o-", linewidth=2, label="轮廓相关系数 (vs 原曲)")
    ax.plot(pcts, [r["stepwise_ratio"] for r in rows], "^-", linewidth=2, label="级进运动占比 (≤2半音)")
    ax.plot(pcts, [r["contour_corr"] for r in rand_rows], "s--", alpha=0.65, label="随机填充轮廓相关性")
    ax.set_title("2. 旋律轮廓与级进规律 (Melodic Contour)")
    ax.set_xlabel("掩码噪声比例 (%)")
    ax.set_ylabel("相关度 / 比例")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Panel C: Rhythmic Density & Syntax
    ax = axes[1, 0]
    ax.plot(pcts, [r["onsets_per_bar"] for r in rows], "o-", linewidth=2, label="每小节起音数 (notes/bar)")
    ax.plot(pcts, [r["mean_note_duration_16ths"] for r in rows], "^-", linewidth=2, label="平均音符时值 (16分步数)")
    ax2 = ax.twinx()
    ax2.plot(pcts, [r["orphan_hold_ratio"] * 100 for r in rows], "x--", color="#d97706", linewidth=1.8, label="孤立 HOLD 错误率 (%)")
    ax.set_title("3. 节奏密度与长短音稳定性 (Rhythm)")
    ax.set_xlabel("掩码噪声比例 (%)")
    ax.set_ylabel("起音数 / 时值步长")
    ax2.set_ylabel("孤立 HOLD (%)")
    ax2.set_ylim(-1, 25)
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    # Panel D: Motif Repetition & Recovery
    ax = axes[1, 1]
    ax.plot(pcts, [r["motif_4gram_rep_rate"] * 100 for r in rows], "o-", linewidth=2, label="重建 4-step 动机重复率 (%)")
    ax.plot(pcts, [r["motif_4gram_rep_rate"] * 100 for r in rand_rows], "s--", alpha=0.65, label="随机填充动机重复率 (%)")
    ax.plot(pcts, [r["token_match_rate"] * 100 for r in rows], "^:", linewidth=1.8, label="与原曲精确吻合率 (%)")
    ax.set_title("4. 重复动机保持与结构重建 (Motif & Recovery)")
    ax.set_xlabel("掩码噪声比例 (%)")
    ax.set_ylabel("百分比 (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("河南 2/4 民歌测试集 (N=116) 不同掩码噪声下重建特性演化分析", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pitch_histogram(rows: List[dict], rand_last: dict, out_path: Path) -> None:
    r0 = rows[0]["pitch_class_dist"]
    r25 = next(r for r in rows if r["t"] == 25)["pitch_class_dist"]
    r50 = next(r for r in rows if r["t"] == 50)["pitch_class_dist"]
    rr = rand_last["pitch_class_dist"]

    x = np.arange(12)
    w = 0.2
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.bar(x - 1.5 * w, [v * 100 for v in r0], width=w, label="原曲测试集 (t=0)")
    ax.bar(x - 0.5 * w, [v * 100 for v in r25], width=w, label="50% 掩码重建 (t=25)")
    ax.bar(x + 0.5 * w, [v * 100 for v in r50], width=w, label="100% 无条件生成 (t=50)")
    ax.bar(x + 1.5 * w, [v * 100 for v in rr], width=w, alpha=0.65, label="均匀随机填充基线")

    for xi in [0, 2, 4, 7, 9]:
        ax.axvspan(xi - 0.45, xi + 0.45, color="#3b82f6", alpha=0.07)

    ax.set_xticks(x)
    ax.set_xticklabels(PC_LABELS)
    ax.set_ylabel("音级出现比例 (%)")
    ax.set_title("十二半音音级分布直方图（浅蓝背景柱为五声正音：宫1 商2 角3 徵5 羽6）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_case_study_piano_rolls(
    cases_data: List[dict],
    out_path: Path,
) -> None:
    """Plot 3-row piano roll figure for case study #1 across t=0, t=15 (30%), t=25 (50%), t=50 (100%)."""
    case = cases_data[0]
    stages = [
        ("原曲 Original (t=0)", case["sequences"]["0"]["recon"]),
        ("30% 掩码态 Masked (t=15)", case["sequences"]["15"]["masked"]),
        ("30% 重建后 Recon (t=15)", case["sequences"]["15"]["recon"]),
        ("50% 掩码态 Masked (t=25)", case["sequences"]["25"]["masked"]),
        ("50% 重建后 Recon (t=25)", case["sequences"]["25"]["recon"]),
        ("100% 无条件重生成 (t=50)", case["sequences"]["50"]["recon"]),
    ]

    fig, axes = plt.subplots(len(stages), 1, figsize=(12, 9.2), sharex=True)
    for ax, (label, seq) in zip(axes, stages):
        active = None
        for i, tok in enumerate(seq):
            if tok == MASK_ID:
                ax.axvspan(i, i + 1, color="#ef4444", alpha=0.22)
                active = None
            elif tok == REST_ID:
                active = None
            elif tok == HOLD_ID:
                if active is not None:
                    ax.barh(active + 46, 1.0, left=i, height=0.8, color="#10b981", alpha=0.85)
            else:
                active = tok
                midi = tok + 46
                col = "#2563eb" if (midi % 12) in PENTATONIC_MOD12 else "#f59e0b"
                ax.barh(midi, 0.9, left=i, height=0.85, color=col)
        for b in range(0, 65, 8):
            ax.axvline(b, color="#64748b", linewidth=0.8 if b % 16 else 1.2, alpha=0.5)
        ax.set_ylim(53, 79)
        ax.set_ylabel("MIDI", fontsize=8)
        ax.set_title(label, loc="left", fontsize=9.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)

    axes[-1].set_xlim(0, 64)
    axes[-1].set_xticks(range(0, 65, 8))
    axes[-1].set_xticklabels([f"Bar {b+1}" for b in range(9)])
    fig.suptitle(f"典型测试曲梯度重建钢琴卷帘对比：{case['title']} ({case['song_id']})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 evaluation on full test set")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    root = args.project_root
    ckpt_path = args.checkpoint if args.checkpoint is not None else (root / "checkpoints" / "best_model.pt")
    out_dir = args.out_dir if args.out_dir is not None else (root / "results" / "stage5")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MelodyTransformerDenoiser(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    diffusion = MaskedDiffusion(**ckpt.get("diffusion_config", {"num_steps": 50, "schedule": "linear"}))

    test_arr = np.load(root / "data" / "processed" / "test.npy").astype(np.int64)
    with (root / "data" / "processed" / "splits_metadata.json").open("r", encoding="utf-8") as f:
        test_meta = json.load(f)["test"]

    orig_pc_dist = pitch_class_histogram(test_arr)
    x0_tensor = torch.from_numpy(test_arr).long().to(device)
    steps = [0, 5, 15, 25, 35, 45, 50]

    recon_rows: List[dict] = []
    rand_rows: List[dict] = []
    all_recons_by_t: Dict[int, np.ndarray] = {}
    all_masked_by_t: Dict[int, np.ndarray] = {}

    for t_val in steps:
        gen = torch.Generator(device=device).manual_seed(args.seed + t_val * 13)
        if t_val == 0:
            xm_np = test_arr.copy()
            xr_np = test_arr.copy()
        else:
            xm, _ = diffusion.forward_mask(x0_tensor, t=t_val, ensure_non_empty=True, generator=gen)
            xr = diffusion.reverse_sample(model, xm, t_start=t_val, temperature=args.temperature, generator=gen)
            xm_np = xm.cpu().numpy()
            xr_np = xr.cpu().numpy()

        all_masked_by_t[t_val] = xm_np
        all_recons_by_t[t_val] = xr_np
        rand_np = make_random_fill_baseline(xm_np, seed=args.seed + t_val)

        m_rec = compute_phenomena_metrics(test_arr, xr_np, orig_pc_dist)
        m_rnd = compute_phenomena_metrics(test_arr, rand_np, orig_pc_dist)
        pct = round(100.0 * t_val / diffusion.T, 1)

        recon_rows.append({"t": t_val, "noise_pct": pct, **m_rec})
        rand_rows.append({"t": t_val, "noise_pct": pct, **m_rnd})

    # Export 3 representative case studies (henan:97, henan:884, henan:842)
    target_ids = ["henan:97", "henan:884", "henan:842"]
    case_indices = []
    for tid in target_ids:
        found = next((i for i, m in enumerate(test_meta) if m["song_id"] == tid), None)
        if found is not None:
            case_indices.append(found)
    if not case_indices:
        case_indices = [0, 1, 2]

    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    cases_data = []

    for c_idx in case_indices:
        info = test_meta[c_idx]
        safe_id = info["song_id"].replace(":", "_")
        c_dir = cases_dir / f"{safe_id}_idx{c_idx}"
        c_dir.mkdir(parents=True, exist_ok=True)

        seq_dict = {}
        txt_lines = [
            f"Case Study: {info['song_id']} {info['title']} ({info['category']})",
            f"Original Jianpu: {seq_to_readable_jianpu(test_arr[c_idx])}",
            "",
        ]
        for t_val in [0, 15, 25, 35, 50]:
            s_mask = all_masked_by_t[t_val][c_idx]
            s_reco = all_recons_by_t[t_val][c_idx]
            seq_dict[str(t_val)] = {"masked": s_mask.tolist(), "recon": s_reco.tolist()}

            tag = f"t{t_val:02d}_pct{int(round(100*t_val/diffusion.T)):03d}"
            if t_val == 0:
                decode_to_midi(s_reco.tolist(), c_dir / f"orig_{tag}.mid")
                decode_to_musicxml(s_reco.tolist(), c_dir / f"orig_{tag}.musicxml", title=f"{info['title']} (Original)")
                decode_to_wav(s_reco.tolist(), c_dir / f"orig_{tag}.wav")
            else:
                decode_to_midi(s_mask.tolist(), c_dir / f"masked_{tag}.mid")
                decode_to_wav(s_mask.tolist(), c_dir / f"masked_{tag}.wav")
                decode_to_midi(s_reco.tolist(), c_dir / f"recon_{tag}.mid")
                decode_to_musicxml(s_reco.tolist(), c_dir / f"recon_{tag}.musicxml", title=f"{info['title']} (Recon {tag})")
                decode_to_wav(s_reco.tolist(), c_dir / f"recon_{tag}.wav")

                match = float((test_arr[c_idx] == s_reco).mean())
                m_r = compute_sequence_metrics([s_reco.tolist()])
                txt_lines.append(
                    f"[t={t_val:02d}/50 ({t_val*2}%) | match={match:.1%} | penta={m_r['pentatonic_ratio']:.1%} | orphan_hold={m_r['orphan_hold_ratio']:.1%}]\n"
                    f"  masked: {seq_to_readable_jianpu(s_mask)}\n"
                    f"  recon : {seq_to_readable_jianpu(s_reco)}\n"
                )
        (c_dir / "comparison_jianpu.txt").write_text("\n".join(txt_lines), encoding="utf-8")
        cases_data.append({"song_id": info["song_id"], "title": info["title"], "category": info["category"], "sequences": seq_dict})

    plot_metrics_curves(recon_rows, rand_rows, out_dir / "stage5_metrics_curves.png")
    plot_pitch_histogram(recon_rows, rand_rows[-1], out_dir / "stage5_pitch_histogram.png")
    # Put henan:884 (2nd in target_ids) first in piano roll plot for visual continuity if present
    ordered_cases = sorted(cases_data, key=lambda x: 0 if x["song_id"] == "henan:884" else 1)
    plot_case_study_piano_rolls(ordered_cases, out_dir / "stage5_case_studies.png")

    summary = {
        "checkpoint": str(ckpt_path.name),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_best_val_loss": round(float(ckpt.get("best_val_loss", 0.0)), 4),
        "num_test_sequences": int(test_arr.shape[0]),
        "temperature": args.temperature,
        "diffusion_reconstruction": recon_rows,
        "random_fill_baseline": rand_rows,
        "case_studies": [{"song_id": c["song_id"], "title": c["title"], "category": c["category"]} for c in cases_data],
    }
    (out_dir / "stage5_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
