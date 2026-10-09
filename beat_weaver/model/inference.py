"""Autoregressive generation with grammar-constrained decoding."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.tokenizer import (
    BAR_BASE,
    BAR_COUNT,
    COLS,
    DIRS,
    ROWS,
    SUBDIVISIONS_PER_BAR,
    _decode_note_token,
    bar_token,
    is_bar_token,
    DIFF_EASY,
    DIFF_EXPERT_PLUS,
    END,
    LEFT_BASE,
    LEFT_COUNT,
    LEFT_EMPTY,
    POS_BASE,
    POS_COUNT,
    RIGHT_BASE,
    RIGHT_COUNT,
    RIGHT_EMPTY,
    START,
    VOCAB_SIZE,
    decode_tokens,
    difficulty_to_token,
)
from beat_weaver.model.transformer import BeatWeaverModel
from beat_weaver.schemas.normalized import Note


def _audio_bars(real_frames: int) -> int:
    """Number of bars (64 frames each) needed to cover ``real_frames`` of audio."""
    return max(1, min(BAR_COUNT, -(-real_frames // SUBDIVISIONS_PER_BAR)))


# Saber travel direction for each cut direction (0=up 1=down 2=left 3=right
# 4=up-left 5=up-right 6=down-left 7=down-right 8=any). Two consecutive cuts on
# one hand "flow" when the second roughly reverses the first (a down-swing is
# followed by an up-swing). The exception is 8 (any direction).
_CUT_VEC = torch.tensor(
    [[0, 1], [0, -1], [-1, 0], [1, 0], [-1, 1], [1, 1], [-1, -1], [1, -1], [0, 0]],
    dtype=torch.float32,
)
_CUT_UNIT = torch.nn.functional.normalize(_CUT_VEC, dim=1)


def _note_tokens_on_cell(base: int, x: int, y: int) -> slice:
    """The 9 compound tokens (all cut directions) for one grid cell of one hand."""
    start = base + x * (ROWS * DIRS) + y * DIRS  # same layout as _encode_note_token
    return slice(start, start + DIRS)


def _same_direction_penalty(base: int, prev_direction: int) -> torch.Tensor:
    """Per-token penalty (0..1) for cutting the same way as the hand's last cut.

    Returns a (VOCAB_SIZE,) tensor that is non-zero only over this hand's 108
    note tokens. 1.0 = exact same direction, 0.0 = reversed or perpendicular.
    """
    pen = torch.zeros(VOCAB_SIZE)
    if prev_direction < 0 or prev_direction == 8:
        return pen
    # dot product of each of the 9 directions with the previous one, clipped at 0
    dots = (_CUT_UNIT @ _CUT_UNIT[prev_direction]).clamp(min=0.0)  # (9,)
    dots[8] = 0.0
    pen[base: base + COLS * ROWS * DIRS] = dots.repeat(COLS * ROWS)
    return pen


# Cut directions pointing away from the body for a hand that has crossed over:
# left hand in columns 2-3 may not cut Right/UpRight/DownRight, right hand in
# columns 0-1 may not cut Left/UpLeft/DownLeft. Pattern seen in generated maps:
# a crossover block facing outward is nearly unhittable; facing inward it is fine.
_OUTWARD = {LEFT_BASE: ((2, 3), (3, 5, 7)), RIGHT_BASE: ((0, 1), (2, 4, 6))}


def _crossover_outward_tokens(base: int) -> list[int]:
    cols, dirs = _OUTWARD[base]
    return [base + x * 27 + y * 9 + d for x in cols for y in range(3) for d in dirs]


def _build_grammar_mask(
    last_token: int, last_pos_in_bar: int = -1, current_bar: int = -1,
    audio_bars: int | None = None, left_cell: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Build a boolean mask over the vocabulary for valid next tokens.

    Returns a tensor of shape (VOCAB_SIZE,) where True = allowed.

    Args:
        last_token: The most recently generated token.
        last_pos_in_bar: The last POS offset used in the current bar (-1 if none).
            Used to enforce strictly increasing positions within a bar,
            preventing multiple notes at the same beat.
        current_bar: Index of the bar currently open (-1 before the first bar).
            The only bar token allowed next is current_bar + 1, so bars can
            neither repeat nor skip and the sequence cannot outrun the audio.
        audio_bars: Bars of real audio in the window. When given, END is not
            allowed until the last of those bars is open, and no bar may follow
            it. Without this the model stops early: END is legal after every
            bar, so a small chance per bar compounds over 64 bars (measured on
            a trained model: windows ending after 14 and 17 of 64 bars).
        left_cell: (x, y) of the LEFT note just placed at this position, if any.
            The RIGHT note may not occupy the same cell: two blocks cannot share
            a grid square at one instant. Human maps never do this; the model
            did it 1.5-2.5% of the time.

    Grammar rules:
        START      → DIFF_*
        DIFF_*     → BAR_0
        BAR_k      → POS_* | BAR_k+1 | END
        POS_*      → LEFT_* | LEFT_EMPTY
        LEFT_*     → RIGHT_* | RIGHT_EMPTY
        RIGHT_*    → POS_* (strictly >) | BAR_k+1 | END
    """
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    next_bar = current_bar + 1
    next_bar_ok = next_bar < BAR_COUNT

    if last_token == START:
        # After START → only difficulty tokens
        mask[DIFF_EASY: DIFF_EXPERT_PLUS + 1] = True

    elif DIFF_EASY <= last_token <= DIFF_EXPERT_PLUS:
        # After DIFF → the first bar
        mask[bar_token(0)] = True

    elif is_bar_token(last_token):
        # After BAR_k → POS, BAR_k+1 (if any audio left), or END
        mask[POS_BASE: POS_BASE + POS_COUNT] = True
        if next_bar_ok:
            mask[bar_token(next_bar)] = True
        mask[END] = True

    elif POS_BASE <= last_token < POS_BASE + POS_COUNT:
        # After POS → LEFT note or LEFT_EMPTY
        mask[LEFT_EMPTY] = True
        mask[LEFT_BASE: LEFT_BASE + LEFT_COUNT] = True
        mask[_crossover_outward_tokens(LEFT_BASE)] = False

    elif last_token == LEFT_EMPTY or (LEFT_BASE <= last_token < LEFT_BASE + LEFT_COUNT):
        # After LEFT → RIGHT note or RIGHT_EMPTY (never on the LEFT note's cell)
        mask[RIGHT_EMPTY] = True
        mask[RIGHT_BASE: RIGHT_BASE + RIGHT_COUNT] = True
        mask[_crossover_outward_tokens(RIGHT_BASE)] = False
        if left_cell is not None:
            mask[_note_tokens_on_cell(RIGHT_BASE, *left_cell)] = False

    elif last_token == RIGHT_EMPTY or (RIGHT_BASE <= last_token < RIGHT_BASE + RIGHT_COUNT):
        # After RIGHT → POS (strictly increasing), BAR, or END
        # Only allow POS tokens with offset > last_pos_in_bar
        min_next = last_pos_in_bar + 1
        if min_next < POS_COUNT:
            mask[POS_BASE + min_next: POS_BASE + POS_COUNT] = True
        if next_bar_ok:
            mask[bar_token(next_bar)] = True
        mask[END] = True

    else:
        # Unknown state — allow everything except PAD/START
        mask[2:] = True

    if audio_bars is not None:
        audio_bars = min(audio_bars, BAR_COUNT)
        if current_bar < audio_bars - 1:
            mask[END] = False  # audio remains: the sequence must reach the last bar
        else:
            mask[BAR_BASE: BAR_BASE + BAR_COUNT] = False  # last bar is open: only notes or END

    return mask


def _onset_schedule(
    model: BeatWeaverModel, memory: torch.Tensor, n_frames: int,
    notes_per_beat: float | None = None, min_gap: int = 4,
) -> list[int]:
    """Stage 1 of two-stage decoding: the frames that get a note.

    Read straight off the encoder's onset head (trained per frame on where the
    map's notes fall, top-k precision 0.56 vs 0.08 chance), with no decoder
    involved. Each bar gets the head's own expected count there (sum of its
    probabilities), or ``notes_per_beat * 4`` when overridden, spread across
    the bar's beats with ``min_gap`` frames between notes.
    """
    p = torch.sigmoid(model.onset_head(memory)[0, :n_frames, 0].float())
    beat = SUBDIVISIONS_PER_BAR // 4
    chosen: list[int] = []
    for b0 in range(0, n_frames, SUBDIVISIONS_PER_BAR):
        pb = p[b0: b0 + SUBDIVISIONS_PER_BAR]
        k = round(pb.sum().item()) if notes_per_beat is None else round(notes_per_beat * len(pb) / 16)
        # Round-robin over the bar's beats, strongest frame first within each, so
        # the budget is spread through the bar instead of bunching on its loudest
        # moment (which played as a burst then a wait).
        queues = [
            [b0 + q0 + i for i in torch.argsort(pb[q0: q0 + beat], descending=True).tolist()]
            for q0 in range(0, len(pb), beat)
        ]
        picked = 0
        while picked < k and any(queues):
            for q in queues:
                while q:
                    f = q.pop(0)
                    if all(abs(f - c) >= min_gap for c in chosen):
                        chosen.append(f)
                        picked += 1
                        break
                if picked >= k:
                    break
    return sorted(chosen)


def _sample_with_filter(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Sample a token from logits with temperature, top-k, and top-p filtering."""
    if temperature <= 0:
        return logits.argmax().item()

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_val = values[-1]
        logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative prob above threshold
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float("-inf")
        # Scatter back
        logits = torch.zeros_like(logits).scatter(0, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


@torch.no_grad()
def generate(
    model: BeatWeaverModel,
    mel_spectrogram: torch.Tensor,
    difficulty: str,
    config: ModelConfig,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = None,
    mel_mask: torch.Tensor | None = None,
    audio_bars: int | None = None,
    flow_penalty: float = 2.0,
    onset_guidance: float = 0.0,
    two_stage: bool = False,
    notes_per_beat: float | None = None,
    min_gap: int = 4,
) -> list[int]:
    """Generate a token sequence autoregressively.

    Args:
        model: Trained BeatWeaverModel.
        mel_spectrogram: (n_mels, T_audio) — single spectrogram (no batch dim).
        difficulty: Difficulty name (e.g., "Expert").
        config: Model configuration.
        temperature: Sampling temperature (0 = greedy).
        top_k: Top-k filtering (0 = disabled).
        top_p: Top-p / nucleus filtering (1.0 = disabled).
        seed: Random seed for reproducibility.
        mel_mask: (T_audio,) — True for valid positions.
        flow_penalty: Logit penalty applied to a note that cuts in the same
            direction as that hand's previous cut (scaled by how parallel the
            two cuts are). Encourages alternating swings so the sabers flow from
            one note to the next. 0 disables it. Human Expert maps repeat a
            direction ~4-5% of the time, the model ~7-12%, so this is a nudge
            the model can still override, not a ban.
        onset_guidance: Weight on the encoder's per-frame onset logit, added to
            the logit of each POS token (POS p in bar b is audio frame
            64*b + p). The decoder alone does not reliably use the audio (on
            held-out charts its loss was the same with the right song's audio
            and a wrong one's), but the encoder's onset head does locate where
            notes fall (top-k precision 0.56 vs 0.08 chance). This ties note
            timing to the song directly. The logits are centred on the window
            mean so the overall density is unchanged: it moves notes to the
            onsets, it does not add or remove them. 0 disables it.
        two_stage: Rhythm from the audio, placement from the decoder. The
            decoder's cross-attention never attended to the audio at any point
            in training (entropy at the uniform ceiling; forcing it onto the
            right frame changed the loss by nothing), so every POS/BAR/END
            token is taken from the onset head (see ``_onset_schedule``) and
            the decoder only chooses what goes at each given position. A
            scheduled position must hold a note, so LEFT_EMPTY + RIGHT_EMPTY
            is forbidden there.
        notes_per_beat: Two-stage note density override (default: the onset
            head's own expected count).

    Returns:
        List of token IDs including START and END.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    device = next(model.parameters()).device

    # Prepare mel: add batch dimension
    mel = mel_spectrogram.unsqueeze(0).to(device)  # (1, n_mels, T_audio)
    if mel_mask is not None:
        mel_mask = mel_mask.unsqueeze(0).to(device)  # (1, T_audio)

    # Encode audio once
    memory = model.encode_audio(mel, mel_mask)

    # Per-frame onset guidance, centred on the window's real audio and padded to
    # the full BAR_COUNT * 64 frames so every bar can slice 64 entries.
    guidance = None
    if onset_guidance > 0:
        onset = model.onset_head(memory)[0, :, 0].float()
        n_real = min(onset.numel(), (audio_bars or BAR_COUNT) * SUBDIVISIONS_PER_BAR)
        guidance = torch.zeros(BAR_COUNT * SUBDIVISIONS_PER_BAR, device=device)
        m = min(onset.numel(), guidance.numel())
        guidance[:m] = onset[:m] - onset[:n_real].mean()

    n_bars = min(audio_bars or BAR_COUNT, BAR_COUNT)
    schedule = None
    if two_stage:
        schedule = _onset_schedule(model, memory, n_bars * SUBDIVISIONS_PER_BAR, notes_per_beat, min_gap)
    sched_i = 0

    # Start with [START, DIFF_x]
    diff_token = difficulty_to_token(difficulty)
    tokens = [START, diff_token]
    last_pos_in_bar = -1  # Track last POS offset in current bar
    current_bar = -1  # Index of the open bar; grammar only permits current_bar + 1 next
    left_cell: tuple[int, int] | None = None  # cell of the LEFT note at the open position
    last_dir = {LEFT_BASE: -1, RIGHT_BASE: -1}  # previous cut direction per hand

    for _ in range(config.max_seq_len - 2):
        # Two-stage: timing tokens come from the schedule, not the decoder
        if schedule is not None and (
            is_bar_token(tokens[-1]) or tokens[-1] == RIGHT_EMPTY
            or RIGHT_BASE <= tokens[-1] < RIGHT_BASE + RIGHT_COUNT
        ):
            if sched_i < len(schedule) and schedule[sched_i] // SUBDIVISIONS_PER_BAR == current_bar:
                next_token = POS_BASE + schedule[sched_i] % SUBDIVISIONS_PER_BAR
                sched_i += 1
            elif current_bar + 1 < n_bars:
                next_token = bar_token(current_bar + 1)
            else:
                next_token = END
            tokens.append(next_token)
            if is_bar_token(next_token):
                last_pos_in_bar, current_bar = -1, next_token - BAR_BASE
            elif next_token != END:
                last_pos_in_bar, left_cell = next_token - POS_BASE, None
            if next_token == END:
                break
            continue

        # Prepare decoder input
        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        token_mask = torch.ones(1, len(tokens), dtype=torch.bool, device=device)

        logits = model.decoder(token_tensor, memory, token_mask, mel_mask)
        # logits: (1, seq_len, vocab_size) — take last position
        next_logits = logits[0, -1]  # (vocab_size,)

        # Apply grammar mask (with position tracking for one-note-per-color-per-beat)
        grammar_mask = _build_grammar_mask(
            tokens[-1], last_pos_in_bar, current_bar, audio_bars, left_cell,
        ).to(device)
        if schedule is not None and tokens[-1] == LEFT_EMPTY:
            grammar_mask[RIGHT_EMPTY] = False  # a scheduled position holds a note
        next_logits[~grammar_mask] = float("-inf")

        # Onset guidance: POS p of the open bar is frame 64*bar + p
        if guidance is not None and current_bar >= 0:
            f0 = current_bar * SUBDIVISIONS_PER_BAR
            next_logits[POS_BASE: POS_BASE + POS_COUNT] += (
                onset_guidance * guidance[f0: f0 + POS_COUNT]
            )

        # Flow: discourage repeating a hand's last cut direction
        if flow_penalty > 0:
            if POS_BASE <= tokens[-1] < POS_BASE + POS_COUNT:
                next_logits -= flow_penalty * _same_direction_penalty(LEFT_BASE, last_dir[LEFT_BASE]).to(device)
            elif tokens[-1] == LEFT_EMPTY or LEFT_BASE <= tokens[-1] < LEFT_BASE + LEFT_COUNT:
                next_logits -= flow_penalty * _same_direction_penalty(RIGHT_BASE, last_dir[RIGHT_BASE]).to(device)

        # Sample
        next_token = _sample_with_filter(next_logits, temperature, top_k, top_p)
        tokens.append(next_token)

        # Update position tracking
        if is_bar_token(next_token):
            last_pos_in_bar = -1  # Reset on new bar
            current_bar = next_token - BAR_BASE
        elif POS_BASE <= next_token < POS_BASE + POS_COUNT:
            last_pos_in_bar = next_token - POS_BASE
            left_cell = None
        elif LEFT_BASE <= next_token < LEFT_BASE + LEFT_COUNT:
            x, y, d = _decode_note_token(next_token, LEFT_BASE)
            left_cell = (x, y)
            last_dir[LEFT_BASE] = d
        elif RIGHT_BASE <= next_token < RIGHT_BASE + RIGHT_COUNT:
            last_dir[RIGHT_BASE] = _decode_note_token(next_token, RIGHT_BASE)[2]

        if next_token == END:
            break

    return tokens


def generate_full_song(
    model: BeatWeaverModel,
    mel_spectrogram: torch.Tensor,
    difficulty: str,
    config: ModelConfig,
    bpm: float,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = None,
    onset_guidance: float = 0.0,
    two_stage: bool = False,
    notes_per_beat: float | None = None,
    min_gap_seconds: float = 0.2,
) -> list[Note]:
    """Generate a complete Beat Saber map by processing audio in overlapping windows.

    For short audio that fits in a single window, this is equivalent to calling
    generate() + decode_tokens(). For longer audio, the mel is split into
    overlapping windows, each generating a token sequence that is decoded to
    notes and merged using midpoint ownership in the overlap zones.

    Args:
        model: Trained BeatWeaverModel.
        mel_spectrogram: (n_mels, T_audio) — full beat-aligned spectrogram.
        difficulty: Difficulty name (e.g., "Expert").
        config: Model configuration.
        bpm: Song BPM (needed for beat offset calculation and token decoding).
        temperature: Sampling temperature.
        top_k: Top-k filtering (0 = disabled).
        top_p: Top-p / nucleus filtering (1.0 = disabled).
        seed: Random seed for reproducibility.

    Returns:
        List of Note objects spanning the full song, sorted by beat.
    """
    # ponytail: min spacing between notes in frames (1/16 beats); ~an 8th at 140 BPM, a 1/4 at 80
    min_gap = max(2, round(min_gap_seconds * bpm / 60 * SUBDIVISIONS_PER_BAR / 4))

    total_frames = mel_spectrogram.shape[1]
    max_len = config.max_audio_len
    # The model sometimes keeps writing notes into the zero padding after the
    # audio ends (the last window is real audio + zeros). Nothing past the real
    # audio can be a valid note, so drop it. Frames are 1/16 beat.
    audio_end_beat = total_frames / 16.0

    # Single window — generate directly
    if total_frames <= max_len:
        tokens = generate(
            model, mel_spectrogram, difficulty, config,
            temperature=temperature, top_k=top_k, top_p=top_p, seed=seed,
            audio_bars=_audio_bars(total_frames), onset_guidance=onset_guidance,
            two_stage=two_stage, notes_per_beat=notes_per_beat, min_gap=min_gap,
        )
        return [n for n in decode_tokens(tokens, bpm) if n.beat < audio_end_beat]

    # Multi-window generation
    overlap = min(max_len // 4, 1024)
    stride = max_len - overlap

    # Compute window start positions
    starts: list[int] = []
    pos = 0
    while pos < total_frames:
        starts.append(pos)
        if pos + max_len >= total_frames:
            break
        pos += stride

    all_window_notes: list[tuple[int, list[Note]]] = []

    for i, start in enumerate(starts):
        end = start + max_len
        window_mel = mel_spectrogram[:, start:end]

        # Zero-pad last window if needed
        if window_mel.shape[1] < max_len:
            pad_size = max_len - window_mel.shape[1]
            window_mel = torch.nn.functional.pad(window_mel, (0, pad_size))

        # Use different seed per window for variety (if seed provided)
        window_seed = seed + i if seed is not None else None

        tokens = generate(
            model, window_mel, difficulty, config,
            temperature=temperature, top_k=top_k, top_p=top_p, seed=window_seed,
            audio_bars=_audio_bars(min(max_len, total_frames - start)),
            onset_guidance=onset_guidance, two_stage=two_stage, notes_per_beat=notes_per_beat, min_gap=min_gap,
        )

        # Decode tokens — notes have beats relative to window start (bar 0)
        window_notes = decode_tokens(tokens, bpm)

        # Offset all beats by the window's start position in frames
        # Each frame = 1/16th note subdivision, beat = frame / 16
        beat_offset = start / 16.0
        for note in window_notes:
            note.beat += beat_offset
            note.time_seconds = note.beat * 60.0 / bpm

        all_window_notes.append((start, window_notes))

    # Merge with midpoint ownership — each window owns notes up to the
    # midpoint of its overlap with the next window, and from the midpoint
    # of its overlap with the previous window.
    result: list[Note] = []
    for i, (start, notes) in enumerate(all_window_notes):
        min_beat = 0.0
        max_beat = audio_end_beat

        if i > 0:
            prev_start = all_window_notes[i - 1][0]
            # Overlap zone between prev and current: [start, prev_start + max_len)
            overlap_mid_frame = (start + prev_start + max_len) / 2.0
            min_beat = overlap_mid_frame / 16.0

        if i < len(all_window_notes) - 1:
            next_start = all_window_notes[i + 1][0]
            # Overlap zone between current and next: [next_start, start + max_len)
            overlap_mid_frame = (next_start + start + max_len) / 2.0
            max_beat = overlap_mid_frame / 16.0

        result.extend(n for n in notes if min_beat <= n.beat < max_beat)

    result.sort(key=lambda n: (n.beat, n.color))
    return result
