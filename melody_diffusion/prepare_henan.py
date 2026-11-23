"""Stage 1 pipeline: filter, split by song_id, encode Henan 2/4 folk songs into 16th-note integer sequences.

Outputs under data/processed/:
    - train.npy, val.npy, test.npy      (int64 arrays of shape [N, 64])
    - splits_metadata.json              (song_id, title, category, start_bar per window)
    - dataset_stats.json                (dataset counts, musicological stats, round-trip check)
Outputs under data/samples_stage1/:
    - 5 test-set sample reconstructions (.mid, .musicxml, .wav, .txt) for listening/inspection
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np

from melody_diffusion.codec import (
    BARS_PER_SEQ,
    HOLD_ID,
    MASK_ID,
    REST_ID,
    SEQ_LEN,
    VOCAB_SIZE,
    compute_sequence_metrics,
    decode_to_midi,
    decode_to_musicxml,
    decode_to_wav,
    extract_clean_windows,
    parse_musicxml_bars,
    token_to_midi,
)

NOTE_NAMES_CN = {0: "1", 1: "#1", 2: "2", 3: "b3", 4: "3", 5: "4", 6: "#4", 7: "5", 8: "#5", 9: "6", 10: "b7", 11: "7"}


def seq_to_readable_jianpu(seq: np.ndarray) -> str:
    parts = []
    for i, tok in enumerate(seq.tolist()):
        if i > 0 and i % 8 == 0:
            parts.append("|")
        if tok == REST_ID:
            parts.append("0")
        elif tok == HOLD_ID:
            parts.append("-")
        elif tok == MASK_ID:
            parts.append("?")
        else:
            midi = token_to_midi(tok)
            octave_shift = (midi // 12) - 5  # C4 (60) -> 0
            label = NOTE_NAMES_CN[midi % 12]
            if octave_shift > 0:
                label += "'" * octave_shift
            elif octave_shift < 0:
                label += "," * (-octave_shift)
            parts.append(label)
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Henan 2/4 folk song 16th-note dataset")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-stride-bars", type=int, default=2)
    parser.add_argument("--eval-stride-bars", type=int, default=8)
    args = parser.parse_args()

    root = args.project_root
    meta_path = root / "data" / "raw" / "henan.jsonl"
    zip_path = root / "data" / "raw" / "ACFS-v261103-scores-henan.zip"
    out_dir = root / "data" / "processed"
    sample_dir = root / "data" / "samples_stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    with meta_path.open("r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f]

    records_24 = [
        r
        for r in all_records
        if not r.get("empty_score", False)
        and r.get("fields", {}).get("meter", {}).get("value") == "2/4"
        and r.get("assets", {}).get("musicxml")
    ]

    # First check which 2/4 songs yield >=1 clean 8-bar window (regardless of stride)
    song_xml_cache: Dict[str, bytes] = {}
    valid_records: List[dict] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for r in records_24:
            xml_bytes = zf.read(r["assets"]["musicxml"])
            wins_probe = extract_clean_windows(xml_bytes, bars_per_seq=BARS_PER_SEQ, stride_bars=1)
            if wins_probe:
                song_xml_cache[r["id"]] = xml_bytes
                valid_records.append(r)

    # Split strictly by song_id BEFORE windowing to prevent data leakage
    valid_records.sort(key=lambda r: r["id"])
    rng = random.Random(args.seed)
    shuffled = valid_records.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(round(0.80 * n_total))
    n_val = int(round(0.10 * n_total))
    split_records = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    split_arrays: Dict[str, np.ndarray] = {}
    splits_meta: Dict[str, List[dict]] = {}

    for split_name, recs in split_records.items():
        stride = args.train_stride_bars if split_name == "train" else args.eval_stride_bars
        seqs: List[List[int]] = []
        meta_rows: List[dict] = []
        for r in recs:
            wins = extract_clean_windows(
                song_xml_cache[r["id"]],
                bars_per_seq=BARS_PER_SEQ,
                stride_bars=stride,
            )
            for win_idx, (start_bar, seq) in enumerate(wins):
                seqs.append(seq)
                meta_rows.append(
                    {
                        "split": split_name,
                        "index": len(seqs) - 1,
                        "song_id": r["id"],
                        "title": r.get("title", ""),
                        "category": r.get("fields", {}).get("category", {}).get("value"),
                        "header_tonic": r.get("fields", {}).get("tonic", {}).get("value"),
                        "start_bar": start_bar,
                        "window_in_song": win_idx,
                    }
                )
        arr = np.asarray(seqs, dtype=np.int64)
        split_arrays[split_name] = arr
        splits_meta[split_name] = meta_rows
        np.save(out_dir / f"{split_name}.npy", arr)

    with (out_dir / "splits_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(splits_meta, f, ensure_ascii=False, indent=2)

    # Verify lossless MusicXML round-trip across ALL test windows
    test_arr = split_arrays["test"]
    roundtrip_exact = 0
    for idx, seq in enumerate(test_arr):
        tmp_xml = sample_dir / "_tmp_roundtrip.musicxml"
        decode_to_musicxml(seq.tolist(), tmp_xml, title="roundtrip_check")
        re_wins = extract_clean_windows(tmp_xml.read_bytes(), bars_per_seq=BARS_PER_SEQ, stride_bars=8)
        if len(re_wins) == 1 and re_wins[0][1] == seq.tolist():
            roundtrip_exact += 1
        if tmp_xml.exists():
            tmp_xml.unlink()

    # Export 5 diverse test samples (MIDI + MusicXML + WAV + readable gene text)
    sample_indices = np.linspace(0, len(test_arr) - 1, num=min(5, len(test_arr)), dtype=int).tolist()
    exported_samples = []
    for s_num, idx in enumerate(sample_indices, start=1):
        seq = test_arr[idx]
        info = splits_meta["test"][idx]
        safe_id = info["song_id"].replace(":", "_")
        stem = f"sample_{s_num:02d}_{safe_id}_bar{info['start_bar']}"
        mid_p = decode_to_midi(seq.tolist(), sample_dir / f"{stem}.mid")
        xml_p = decode_to_musicxml(seq.tolist(), sample_dir / f"{stem}.musicxml", title=f"{info['title']} ({info['song_id']})")
        wav_p = decode_to_wav(seq.tolist(), sample_dir / f"{stem}.wav")
        txt_p = sample_dir / f"{stem}.txt"
        txt_p.write_text(
            f"song_id: {info['song_id']}\n"
            f"title: {info['title']}\n"
            f"category: {info['category']}\n"
            f"header_tonic (XML already 1=C): {info['header_tonic']}\n"
            f"start_bar (0-indexed): {info['start_bar']}\n"
            f"gene_int_seq (len={len(seq)}): {seq.tolist()}\n"
            f"jianpu_preview (1=C): {seq_to_readable_jianpu(seq)}\n",
            encoding="utf-8",
        )
        exported_samples.append({"stem": stem, "song_id": info["song_id"], "title": info["title"]})

    stats = {
        "province": "henan",
        "meter": "2/4",
        "grid_resolution": "16th_note (8 steps/bar)",
        "seq_len": SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "vocab_spec": {
            "0": "REST",
            "1": "HOLD",
            "2..38": "MIDI 48..84 (token = midi - 46)",
            "39": "[MASK]",
        },
        "seed": args.seed,
        "train_stride_bars": args.train_stride_bars,
        "eval_stride_bars": args.eval_stride_bars,
        "song_counts": {
            "henan_total": len(all_records),
            "henan_2_4": len(records_24),
            "songs_with_clean_8bar_windows": len(valid_records),
            "train_songs": len(split_records["train"]),
            "val_songs": len(split_records["val"]),
            "test_songs": len(split_records["test"]),
        },
        "window_counts": {k: int(v.shape[0]) for k, v in split_arrays.items()},
        "split_metrics": {k: compute_sequence_metrics(v) for k, v in split_arrays.items()},
        "test_musicxml_roundtrip_exact": f"{roundtrip_exact}/{len(test_arr)}",
        "exported_samples": exported_samples,
    }

    with (out_dir / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
