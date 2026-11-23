"""Bidirectional 16th-note melody gene codec for Henan 2/4 folk songs.

Token vocabulary (V = 40):
    0       : REST (silence)
    1       : HOLD (sustain previous pitch)
    2..38   : MIDI pitch onset 48 (C3) .. 84 (C6)  [token = midi - 46]
    39      : [MASK] (absorbing mask token for Masked Diffusion)
"""

from __future__ import annotations

import math
import wave
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import mido
import numpy as np

REST_ID = 0
HOLD_ID = 1
PITCH_MIN = 48
PITCH_MAX = 84
PITCH_OFFSET = 46  # token = midi - 46 => 48 -> 2, 84 -> 38
MASK_ID = 39
VOCAB_SIZE = 40

STEPS_PER_BAR = 8   # 2/4 meter at 16th-note grid (8 sixteenth notes per bar)
BARS_PER_SEQ = 8    # 8 bars per window
SEQ_LEN = STEPS_PER_BAR * BARS_PER_SEQ  # 64

STEP_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PENTATONIC_MOD12 = {0, 2, 4, 7, 9}  # C, D, E, G, A (Movable-Do 1, 2, 3, 5, 6)

NOTE_SPELLING = [
    ("C", 0), ("C", 1), ("D", 0), ("E", -1), ("E", 0), ("F", 0),
    ("F", 1), ("G", 0), ("A", -1), ("A", 0), ("B", -1), ("B", 0),
]

DUR_TO_NOTATION = {
    1: [(1, "16th", False)],
    2: [(2, "eighth", False)],
    3: [(3, "eighth", True)],
    4: [(4, "quarter", False)],
    5: [(4, "quarter", False), (1, "16th", False)],
    6: [(6, "quarter", True)],
    7: [(6, "quarter", True), (1, "16th", False)],
    8: [(8, "half", False)],
}


def midi_to_token(midi: int) -> int:
    if midi < PITCH_MIN or midi > PITCH_MAX:
        raise ValueError(f"MIDI pitch {midi} out of range [{PITCH_MIN}, {PITCH_MAX}]")
    return midi - PITCH_OFFSET


def token_to_midi(token: int) -> int:
    if token < 2 or token >= MASK_ID:
        raise ValueError(f"Token {token} is not a pitch token")
    return token + PITCH_OFFSET


def parse_musicxml_bars(xml_bytes: bytes) -> List[Optional[Tuple[List[int], List[Optional[int]]]]]:
    """Parse a MusicXML score into 8-step 2/4 bars.

    Returns a list with one entry per XML measure:
        - None if the measure fails quality checks (divisions != 4, zero-duration
          truncated 32nd notes, total duration != 8 sixteenths, or pitch out of range).
        - (tokens, underlying_pitch_tokens) for valid 8-step 2/4 bars.
    """
    root = ET.fromstring(xml_bytes)
    div = 4
    bars: List[Optional[Tuple[List[int], List[Optional[int]]]]] = []
    tie_midi: Optional[int] = None

    for measure in root.findall(".//measure"):
        div_el = measure.find("./attributes/divisions")
        if div_el is not None and div_el.text:
            div = int(div_el.text)
        if div != 4:
            bars.append(None)
            tie_midi = None
            continue

        bar_tokens: List[int] = []
        bar_pitches: List[Optional[int]] = []
        bad_bar = False

        for note in measure.findall("./note"):
            if note.find("grace") is not None:
                continue
            if note.find("chord") is not None:
                bad_bar = True
                break
            dur_text = note.findtext("duration")
            dur = int(dur_text) if dur_text is not None else 0
            if dur <= 0:
                # Truncated 32nd notes or malformed durations
                bad_bar = True
                break

            tie_types = {t.attrib.get("type") for t in note.findall("tie")}
            if note.find("rest") is not None:
                tie_midi = None
                bar_tokens.extend([REST_ID] * dur)
                bar_pitches.extend([None] * dur)
            else:
                pitch_el = note.find("pitch")
                if pitch_el is None:
                    bad_bar = True
                    break
                step = pitch_el.findtext("step")
                octave = int(pitch_el.findtext("octave", "4"))
                alter = int(pitch_el.findtext("alter", "0"))
                if step not in STEP_TO_SEMITONE:
                    bad_bar = True
                    break
                midi = (octave + 1) * 12 + STEP_TO_SEMITONE[step] + alter
                if midi < PITCH_MIN or midi > PITCH_MAX:
                    bad_bar = True
                    break

                tok_pitch = midi_to_token(midi)
                if "stop" in tie_types and tie_midi == midi:
                    first_tok = HOLD_ID
                else:
                    first_tok = tok_pitch

                bar_tokens.append(first_tok)
                bar_tokens.extend([HOLD_ID] * (dur - 1))
                bar_pitches.extend([tok_pitch] * dur)
                tie_midi = midi if "start" in tie_types else None

        if bad_bar or len(bar_tokens) != STEPS_PER_BAR:
            bars.append(None)
            tie_midi = None
        else:
            bars.append((bar_tokens, bar_pitches))

    return bars


def extract_clean_windows(
    xml_bytes: bytes,
    bars_per_seq: int = BARS_PER_SEQ,
    stride_bars: int = 2,
) -> List[Tuple[int, List[int]]]:
    """Extract contiguous clean windows of `bars_per_seq` 2/4 bars.

    Returns list of (start_bar_index, token_sequence).
    """
    bars = parse_musicxml_bars(xml_bytes)
    windows: List[Tuple[int, List[int]]] = []
    n_bars = len(bars)
    i = 0
    while i <= n_bars - bars_per_seq:
        chunk = bars[i : i + bars_per_seq]
        if all(b is not None for b in chunk):
            seq: List[int] = []
            for bar_tokens, _ in chunk:  # type: ignore[misc]
                seq.extend(bar_tokens)
            # If window starts on a tied continuation from the previous bar, turn it into an onset
            if seq[0] == HOLD_ID:
                lead_pitch = chunk[0][1][0]  # type: ignore[index]
                if lead_pitch is None:
                    i += 1
                    continue
                seq[0] = lead_pitch
            if not all(tok == REST_ID for tok in seq):
                windows.append((i, seq))
            i += stride_bars
        else:
            i += 1
    return windows


def _sequence_to_spans(seq: Sequence[int]) -> List[Tuple[str, Optional[int], int, int]]:
    """Convert token sequence into contiguous (kind, midi_pitch, start_step, end_step) spans."""
    spans: List[Tuple[str, Optional[int], int, int]] = []
    cur_kind: Optional[str] = None
    cur_midi: Optional[int] = None
    cur_start = 0

    for idx, tok in enumerate(seq):
        tok = int(tok)
        if tok == REST_ID or tok == MASK_ID:
            if cur_kind == "REST":
                continue
            if cur_kind is not None:
                spans.append((cur_kind, cur_midi, cur_start, idx))
            cur_kind, cur_midi, cur_start = "REST", None, idx
        elif tok == HOLD_ID:
            if cur_kind == "PITCH":
                continue
            # Orphan hold (only possible in noisy/generated outputs) -> treat as silence
            if cur_kind == "REST":
                continue
            cur_kind, cur_midi, cur_start = "REST", None, idx
        else:
            midi = token_to_midi(tok)
            if cur_kind is not None:
                spans.append((cur_kind, cur_midi, cur_start, idx))
            cur_kind, cur_midi, cur_start = "PITCH", midi, idx

    if cur_kind is not None:
        spans.append((cur_kind, cur_midi, cur_start, len(seq)))
    return spans


def decode_to_midi(seq: Sequence[int], out_path: str | Path, tempo_bpm: int = 100) -> Path:
    """Decode an integer gene sequence into a standard monophonic MIDI file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("time_signature", numerator=2, denominator=4, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    track.append(mido.Message("program_change", program=0, time=0))

    ticks_per_step = 120  # 1 quarter beat = 480 ticks => 1 sixteenth = 120 ticks
    pending_rest_ticks = 0

    for kind, midi, start, end in _sequence_to_spans(seq):
        dur_ticks = (end - start) * ticks_per_step
        if kind == "REST" or midi is None:
            pending_rest_ticks += dur_ticks
        else:
            track.append(mido.Message("note_on", note=midi, velocity=80, time=pending_rest_ticks))
            track.append(mido.Message("note_off", note=midi, velocity=0, time=dur_ticks))
            pending_rest_ticks = 0

    track.append(mido.MetaMessage("end_of_track", time=pending_rest_ticks))
    mid.save(str(out_path))
    return out_path


def decode_to_musicxml(
    seq: Sequence[int],
    out_path: str | Path,
    title: str = "Henan Folk Song Sample",
) -> Path:
    """Decode a sequence (multiple of 8 steps) into a valid 2/4 MusicXML file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(seq) % STEPS_PER_BAR != 0:
        raise ValueError(f"Sequence length {len(seq)} must be a multiple of {STEPS_PER_BAR}")

    n_bars = len(seq) // STEPS_PER_BAR
    bar_items: List[List[Tuple[str, Optional[int], int, bool, bool]]] = [[] for _ in range(n_bars)]

    for kind, midi, start, end in _sequence_to_spans(seq):
        pos = start
        while pos < end:
            bar_idx = pos // STEPS_PER_BAR
            bar_end = (bar_idx + 1) * STEPS_PER_BAR
            seg_end = min(end, bar_end)
            dur = seg_end - pos
            tie_stop = (kind == "PITCH") and (pos > start)
            tie_start = (kind == "PITCH") and (seg_end < end)

            sub_parts = DUR_TO_NOTATION[dur]
            for s_idx, (s_dur, _, _) in enumerate(sub_parts):
                sub_tie_stop = tie_stop or (kind == "PITCH" and s_idx > 0)
                sub_tie_start = tie_start or (kind == "PITCH" and s_idx < len(sub_parts) - 1)
                bar_items[bar_idx].append((kind, midi, s_dur, sub_tie_stop, sub_tie_start))
            pos = seg_end

    score = ET.Element("score-partwise", version="3.1")
    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = title
    part_list = ET.SubElement(score, "part-list")
    sp = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(sp, "part-name").text = "Melody"
    part = ET.SubElement(score, "part", id="P1")

    dur_lookup = {}
    for d, parts in DUR_TO_NOTATION.items():
        if len(parts) == 1:
            dur_lookup[d] = (parts[0][1], parts[0][2])

    for b_idx, items in enumerate(bar_items, start=1):
        m_el = ET.SubElement(part, "measure", number=str(b_idx))
        if b_idx == 1:
            attr = ET.SubElement(m_el, "attributes")
            ET.SubElement(attr, "divisions").text = "4"
            key = ET.SubElement(attr, "key")
            ET.SubElement(key, "fifths").text = "0"
            time_el = ET.SubElement(attr, "time")
            ET.SubElement(time_el, "beats").text = "2"
            ET.SubElement(time_el, "beat-type").text = "4"
            clef = ET.SubElement(attr, "clef")
            ET.SubElement(clef, "sign").text = "G"
            ET.SubElement(clef, "line").text = "2"

        for kind, midi, s_dur, t_stop, t_start in items:
            n_el = ET.SubElement(m_el, "note")
            if kind == "REST" or midi is None:
                ET.SubElement(n_el, "rest")
            else:
                step, alter = NOTE_SPELLING[midi % 12]
                octave = (midi // 12) - 1
                p_el = ET.SubElement(n_el, "pitch")
                ET.SubElement(p_el, "step").text = step
                if alter != 0:
                    ET.SubElement(p_el, "alter").text = str(alter)
                ET.SubElement(p_el, "octave").text = str(octave)

            ET.SubElement(n_el, "duration").text = str(s_dur)
            if t_stop:
                ET.SubElement(n_el, "tie", type="stop")
            if t_start:
                ET.SubElement(n_el, "tie", type="start")
            ntype, is_dot = dur_lookup[s_dur]
            ET.SubElement(n_el, "type").text = ntype
            if is_dot:
                ET.SubElement(n_el, "dot")

    ET.indent(score, space="  ")
    tree = ET.ElementTree(score)
    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
    return out_path


def decode_to_wav(
    seq: Sequence[int],
    out_path: str | Path,
    tempo_bpm: int = 100,
    sample_rate: int = 22050,
) -> Path:
    """Synthesize a clean audio preview (.wav) from an integer gene sequence."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    step_sec = 60.0 / tempo_bpm / 4.0  # 1 sixteenth note duration in seconds
    total_samples = int(round(len(seq) * step_sec * sample_rate))
    buf = np.zeros(total_samples, dtype=np.float32)

    for kind, midi, start, end in _sequence_to_spans(seq):
        if kind != "PITCH" or midi is None:
            continue
        i0 = int(round(start * step_sec * sample_rate))
        i1 = int(round(end * step_sec * sample_rate))
        n = max(1, i1 - i0)
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        t = np.arange(n, dtype=np.float32) / sample_rate

        # Soft flute-like harmonic blend with click-free attack/release envelope
        wave_sig = (
            0.70 * np.sin(2.0 * math.pi * freq * t)
            + 0.20 * np.sin(2.0 * math.pi * 2.0 * freq * t)
            + 0.10 * np.sin(2.0 * math.pi * 3.0 * freq * t)
        )
        atk = min(int(0.015 * sample_rate), n // 3)
        rel = min(int(0.030 * sample_rate), n // 3)
        env = np.ones(n, dtype=np.float32)
        if atk > 0:
            env[:atk] = np.linspace(0.0, 1.0, atk, dtype=np.float32)
        if rel > 0:
            env[-rel:] = np.linspace(1.0, 0.0, rel, dtype=np.float32)
        buf[i0 : i0 + n] += 0.35 * env * wave_sig

    pcm = np.clip(buf * 32767.0, -32767, 32767).astype(np.int16)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return out_path


def compute_sequence_metrics(seqs: Iterable[Sequence[int]]) -> dict:
    """Compute musicological and validity statistics across sequences."""
    total_seqs = 0
    total_steps = 0
    sound_steps = 0
    pentatonic_steps = 0
    onset_steps = 0
    rest_steps = 0
    hold_steps = 0
    orphan_holds = 0
    mask_steps = 0

    for seq in seqs:
        total_seqs += 1
        active_midi: Optional[int] = None
        for idx, tok in enumerate(seq):
            tok = int(tok)
            total_steps += 1
            if tok == MASK_ID:
                mask_steps += 1
                active_midi = None
            elif tok == REST_ID:
                rest_steps += 1
                active_midi = None
            elif tok == HOLD_ID:
                hold_steps += 1
                if idx == 0 or active_midi is None:
                    orphan_holds += 1
                else:
                    sound_steps += 1
                    if (active_midi % 12) in PENTATONIC_MOD12:
                        pentatonic_steps += 1
            else:
                onset_steps += 1
                midi = token_to_midi(tok)
                active_midi = midi
                sound_steps += 1
                if (midi % 12) in PENTATONIC_MOD12:
                    pentatonic_steps += 1

    return {
        "num_sequences": total_seqs,
        "total_steps": total_steps,
        "pentatonic_ratio": pentatonic_steps / sound_steps if sound_steps else 0.0,
        "onset_ratio": onset_steps / total_steps if total_steps else 0.0,
        "hold_ratio": hold_steps / total_steps if total_steps else 0.0,
        "rest_ratio": rest_steps / total_steps if total_steps else 0.0,
        "mask_ratio": mask_steps / total_steps if total_steps else 0.0,
        "orphan_hold_count": orphan_holds,
        "orphan_hold_ratio": orphan_holds / hold_steps if hold_steps else 0.0,
    }
