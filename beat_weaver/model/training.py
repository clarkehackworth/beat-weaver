"""Training loop with mixed-precision, checkpointing, and TensorBoard logging."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.dataset import BeatSaberDataset, build_weighted_sampler, collate_fn
from beat_weaver.model.tokenizer import (
    LEFT_EMPTY, RIGHT_EMPTY,
    BAR_BASE,
    BOMB_BASE,
    BAR_COUNT,
    LEFT_BASE,
    LEFT_COUNT,
    PAD,
    POS_BASE,
    POS_COUNT,
    RIGHT_BASE,
    RIGHT_COUNT,
    SUBDIVISIONS_PER_BAR,
)
from beat_weaver.model.transformer import BeatWeaverModel

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: ModelConfig, steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine LR schedule with linear warmup.

    steps_per_epoch should be the number of *optimizer steps* per epoch
    (i.e. len(train_loader) // gradient_accumulation_steps), not the raw
    batch count.
    """
    total_steps = config.max_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def note_frame_targets(tokens: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Per-audio-frame note targets (batch, n_frames) from a token sequence.

    A note sits at frame ``bar * 64 + pos`` (16 subdivisions per beat, 4 beats
    per bar), read straight off the BAR_k and POS_p tokens, so the target comes
    from the map and not from the audio. Frames past ``n_frames`` are dropped.
    """
    batch, seq = tokens.shape
    targets = torch.zeros(batch, n_frames, device=tokens.device)
    is_bar = (tokens >= BAR_BASE) & (tokens < BAR_BASE + BAR_COUNT)
    is_pos = (tokens >= POS_BASE) & (tokens < POS_BASE + POS_COUNT)
    # bar index in force at each position: the most recent BAR token
    bar_val = torch.where(is_bar, tokens - BAR_BASE, torch.full_like(tokens, -1))
    bar_cur = torch.cummax(bar_val, dim=1).values  # BAR tokens only increase
    frame = bar_cur * SUBDIVISIONS_PER_BAR + (tokens - POS_BASE)
    valid = is_pos & (bar_cur >= 0) & (frame < n_frames)
    b_idx = torch.arange(batch, device=tokens.device).unsqueeze(1).expand(batch, seq)
    targets[b_idx[valid], frame[valid]] = 1.0
    return targets


def _is_extras_target(flat: torch.Tensor, placement_only: bool) -> torch.Tensor:
    """BOMB/WALL targets, plus (in placement mode) the POS that opens an extras-only position."""
    extra = flat >= BOMB_BASE
    if placement_only:
        extra = extra | ((flat >= POS_BASE) & (flat < POS_BASE + POS_COUNT))
    return extra


def token_loss(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float,
               extras_weight: float = 1.0, placement_only: bool = False) -> torch.Tensor:
    """Mean cross-entropy over non-PAD targets, with extras targets weighted by ``extras_weight``.

    At 1.0 this is nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing). Below 1.0 the bomb/wall targets pull
    less on the shared decoder: the extras4 fine-tune (weight 1) learned bombs but drifted the notes (doubles
    26-52% -> 5-13%, big fast reversals 11% -> 19% at equal decode flags)."""
    flat = targets.reshape(-1)
    per = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), flat, ignore_index=PAD, reduction="none",
                          label_smoothing=label_smoothing)
    w = (flat != PAD).float()
    if extras_weight != 1.0:
        w = torch.where(_is_extras_target(flat, placement_only), w * extras_weight, w)
    return (per * w).sum() / w.sum().clamp(min=1)


def placement_targets(targets: torch.Tensor) -> torch.Tensor:
    """Targets with everything but LEFT/RIGHT (and BOMB/WALL) tokens set to PAD (ignored).

    Also kept: a POS that opens a bomb/wall-only position (followed by LEFT_EMPTY, RIGHT_EMPTY). Notes get their
    timing from the onset head, but nothing schedules these, so the decoder has to learn where they go."""
    pos = (targets >= POS_BASE) & (targets < POS_BASE + POS_COUNT)
    nxt = torch.zeros_like(pos)
    nxt[:, :-2] = (targets[:, 1:-1] == LEFT_EMPTY) & (targets[:, 2:] == RIGHT_EMPTY)
    keep = (pos & nxt) | (targets == LEFT_EMPTY) | (targets == RIGHT_EMPTY) | (
        (targets >= LEFT_BASE) & (targets < LEFT_BASE + LEFT_COUNT)
    ) | ((targets >= RIGHT_BASE) & (targets < RIGHT_BASE + RIGHT_COUNT)) | (targets >= BOMB_BASE)
    return torch.where(keep, targets, torch.full_like(targets, PAD))


def apply_token_dropout(
    tokens: torch.Tensor, p: float, protect: int = 2,
) -> torch.Tensor:
    """Replace a random ``p`` fraction of input tokens with PAD.

    The first ``protect`` tokens (START, DIFF) are kept so the sequence still
    says which difficulty it is. Targets are never touched; only the decoder's
    view of the past is corrupted, which is what makes the audio worth using.
    """
    if p <= 0:
        return tokens
    drop = torch.rand(tokens.shape, device=tokens.device) < p
    drop[:, :protect] = False
    return tokens.masked_fill(drop, PAD)


def _color_balance_loss(logits: torch.Tensor) -> torch.Tensor:
    """Penalize deviation from 50/50 LEFT/RIGHT token probability.

    Only considers positions where a note token (LEFT or RIGHT) is likely.
    """
    probs = torch.softmax(logits, dim=-1)  # (batch, seq, vocab)
    left_prob = probs[:, :, LEFT_BASE : LEFT_BASE + LEFT_COUNT].sum(dim=-1)
    right_prob = probs[:, :, RIGHT_BASE : RIGHT_BASE + RIGHT_COUNT].sum(dim=-1)
    total = left_prob + right_prob + 1e-8
    # Only count positions where note tokens have meaningful probability
    note_mask = total > 0.1
    if not note_mask.any():
        return torch.tensor(0.0, device=logits.device)
    left_ratio = left_prob[note_mask] / total[note_mask]
    return ((left_ratio - 0.5) ** 2).mean()


class Trainer:
    """Wraps training state and provides train/validate methods."""

    def __init__(
        self,
        model: BeatWeaverModel,
        config: ModelConfig,
        output_dir: Path,
        device: torch.device | None = None,
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device or _get_device()

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=PAD,
            label_smoothing=config.label_smoothing,
        )
        self.scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        accum_steps = self.config.gradient_accumulation_steps

        self.optimizer.zero_grad()

        for batch_idx, (mel, mel_mask, tokens, token_mask) in enumerate(dataloader):
            mel = mel.to(self.device)
            mel_mask = mel_mask.to(self.device)
            tokens = tokens.to(self.device)
            token_mask = token_mask.to(self.device)

            # Teacher forcing: input is tokens[:-1], target is tokens[1:]
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            input_mask = token_mask[:, :-1]

            # Corrupt the decoder's view of the past (targets stay intact) and
            # build per-frame note targets from the clean tokens.
            decoder_input = apply_token_dropout(input_tokens, self.config.token_dropout)
            onset_targets = note_frame_targets(tokens, mel.size(2))
            if self.config.placement_only:
                target_tokens = placement_targets(target_tokens)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                logits, onset_logits = self.model.forward_with_onset(
                    mel, decoder_input, mel_mask, input_mask,
                )
                # logits: (batch, seq_len-1, vocab_size)
                # target: (batch, seq_len-1)
                loss = token_loss(logits, target_tokens, self.config.label_smoothing,
                                  self.config.extras_loss_weight, self.config.placement_only)
                # Onset-alignment auxiliary loss on the encoder output
                if self.config.onset_loss_weight > 0:
                    onset_loss = F.binary_cross_entropy_with_logits(
                        onset_logits.float(), onset_targets, reduction="none",
                    )
                    onset_loss = (onset_loss * mel_mask).sum() / mel_mask.sum().clamp(min=1)
                    loss = loss + self.config.onset_loss_weight * onset_loss
                # Color balance auxiliary loss
                if self.config.color_balance_weight > 0:
                    loss = loss + self.config.color_balance_weight * _color_balance_loss(logits)
                loss = loss / accum_steps  # Scale for accumulation

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if hasattr(self, "scheduler"):
                    self.scheduler.step()

            total_loss += loss.item() * accum_steps  # Unscale for logging
            n_batches += 1
            self.global_step += 1

            # Log every 50 steps
            if self.global_step % 50 == 0:
                self.writer.add_scalar("train/loss_step", loss.item() * accum_steps, self.global_step)
                if self.config.onset_loss_weight > 0:
                    self.writer.add_scalar("train/onset_loss", onset_loss.item(), self.global_step)
                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/lr", lr, self.global_step)

        avg_loss = total_loss / max(1, n_batches)
        return avg_loss

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Run validation. Returns dict of metrics."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        n_batches = 0
        note_sum = extra_sum = 0.0  # unweighted CE split: notes (LEFT/RIGHT incl. empties) vs extras, to see drift
        note_n = extra_n = 0

        for mel, mel_mask, tokens, token_mask in dataloader:
            mel = mel.to(self.device)
            mel_mask = mel_mask.to(self.device)
            tokens = tokens.to(self.device)
            token_mask = token_mask.to(self.device)

            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            input_mask = token_mask[:, :-1]
            target_mask = token_mask[:, 1:]
            if self.config.placement_only:
                target_tokens = placement_targets(target_tokens)

            logits = self.model(mel, input_tokens, mel_mask, input_mask)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1),
            )
            total_loss += loss.item()
            n_batches += 1
            flat_t = target_tokens.reshape(-1)
            per = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), flat_t, ignore_index=PAD,
                                  reduction="none", label_smoothing=self.config.label_smoothing)
            is_note = (flat_t >= LEFT_EMPTY) & (flat_t < RIGHT_BASE + RIGHT_COUNT)
            is_extra = _is_extras_target(flat_t, self.config.placement_only)
            note_sum += per[is_note].sum().item(); note_n += int(is_note.sum())
            extra_sum += per[is_extra].sum().item(); extra_n += int(is_extra.sum())

            # Token accuracy (ignoring padding)
            preds = logits.argmax(dim=-1)
            mask = target_mask & (target_tokens != PAD)
            total_correct += (preds == target_tokens)[mask].sum().item()
            total_tokens += mask.sum().item()

        avg_loss = total_loss / max(1, n_batches)
        accuracy = total_correct / max(1, total_tokens)
        return {"val_loss": avg_loss, "val_token_accuracy": accuracy,
                "val_note_loss": note_sum / max(1, note_n), "val_extras_loss": extra_sum / max(1, extra_n)}

    def save_checkpoint(self, name: str) -> Path:
        """Save model + optimizer + scheduler + training state."""
        ckpt_dir = self.output_dir / "checkpoints" / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), ckpt_dir / "model.pt")
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        if hasattr(self, "scheduler"):
            torch.save(self.scheduler.state_dict(), ckpt_dir / "scheduler.pt")
        torch.save(self.scaler.state_dict(), ckpt_dir / "scaler.pt")
        self.config.save(ckpt_dir / "config.json")

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        (ckpt_dir / "training_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        return ckpt_dir

    def load_checkpoint(self, ckpt_dir: Path) -> None:
        """Resume from a checkpoint."""
        ckpt_dir = Path(ckpt_dir)
        self._resume_dir = ckpt_dir
        self.model.load_state_dict(
            torch.load(ckpt_dir / "model.pt", map_location=self.device, weights_only=True),
        )
        self.optimizer.load_state_dict(
            torch.load(ckpt_dir / "optimizer.pt", map_location=self.device, weights_only=True),
        )
        scaler_path = ckpt_dir / "scaler.pt"
        if scaler_path.exists():
            self.scaler.load_state_dict(
                torch.load(scaler_path, map_location=self.device, weights_only=True),
            )
            logger.info("Restored GradScaler state from %s", scaler_path)
        elif self.device.type == "cuda":
            # No scaler.pt — use conservative scale=1.0 to avoid overflow
            self.scaler = torch.amp.GradScaler(init_scale=1.0, growth_interval=1000)
            logger.info("No scaler.pt found; using conservative init_scale=1.0")
        state = json.loads((ckpt_dir / "training_state.json").read_text(encoding="utf-8"))
        self.epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_val_loss = state["best_val_loss"]

    def restore_scheduler(self) -> None:
        """Restore scheduler state if resuming and scheduler.pt exists.

        If no scheduler.pt is found, fast-forward the scheduler to the
        current global_step so the LR matches where training left off.
        """
        resume_dir = getattr(self, "_resume_dir", None)
        if resume_dir is None:
            return
        if not hasattr(self, "scheduler"):
            return
        scheduler_path = resume_dir / "scheduler.pt"
        if scheduler_path.exists():
            self.scheduler.load_state_dict(
                torch.load(scheduler_path, map_location=self.device, weights_only=True),
            )
            logger.info("Restored LR scheduler state from %s", scheduler_path)
        elif self.global_step > 0:
            # No scheduler.pt — fast-forward to match resumed global_step
            for _ in range(self.global_step):
                self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Fast-forwarded LR scheduler to step %d (lr=%.2e)", self.global_step, lr,
            )


def train(
    config: ModelConfig,
    train_dataset: BeatSaberDataset,
    val_dataset: BeatSaberDataset,
    output_dir: Path,
    resume_from: Path | None = None,
    init_from: Path | None = None,
    probe_audio: Path | None = None,
) -> Path:
    """Main training entry point.

    ``probe_audio`` generates that song after every epoch with ``probe.PROBE_FLAGS`` and logs the doubles,
    reversal, left-hand and nps shares (``probe.drift_stats``): the drift that validation loss cannot see.

    ``init_from`` loads only the weights of a checkpoint (fresh optimizer,
    schedule and best-loss), for fine-tuning under a different objective.

    Returns path to the best checkpoint directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = BeatWeaverModel(config)
    logger.info("Model parameters: %s", f"{model.count_parameters():,}")

    trainer = Trainer(model, config, output_dir)

    if resume_from:
        trainer.load_checkpoint(resume_from)
        logger.info("Resumed from %s (epoch %d)", resume_from, trainer.epoch)
    elif init_from:
        state = torch.load(Path(init_from) / "model.pt", map_location=trainer.device, weights_only=True)
        # A notes-only checkpoint (vocab 355) warm-starts the 495 vocab: copy the old rows, new rows stay random
        own = model.state_dict()
        for k in ("decoder.embedding.weight", "decoder.output_proj.weight", "decoder.output_proj.bias"):
            if k in state and state[k].shape != own[k].shape:
                grown = own[k].clone()
                grown[: state[k].shape[0]] = state[k]
                logger.info("Grew %s from %s to %s", k, tuple(state[k].shape), tuple(own[k].shape))
                state[k] = grown
        model.load_state_dict(state)
        logger.info("Initialised weights from %s", init_from)

    sampler = build_weighted_sampler(
        train_dataset, config.official_ratio, config.source_ratios,
    )
    use_cuda = trainer.device.type == "cuda"
    # Windows spawn-based multiprocessing causes DataLoader worker deadlocks
    # between epochs with persistent_workers. Use num_workers=0 on Windows.
    num_workers = 0 if sys.platform == "win32" else (2 if use_cuda else 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,  # shuffle only when no weighted sampler
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )

    # Build scheduler after knowing steps_per_epoch.
    # The scheduler steps once per optimizer step, which happens every
    # gradient_accumulation_steps batches (not every batch).
    optimizer_steps_per_epoch = max(1, len(train_loader) // config.gradient_accumulation_steps)
    trainer.scheduler = _build_lr_scheduler(
        trainer.optimizer, config, optimizer_steps_per_epoch,
    )
    trainer.restore_scheduler()

    config.save(output_dir / "config.json")

    training_start = time.time()
    epoch_times: list[float] = []

    logger.info(
        "Training: %d train samples, %d val samples, %d batches/epoch, device=%s",
        len(train_dataset), len(val_dataset), len(train_loader), trainer.device,
    )

    probe_mel = probe_bpm = None
    if probe_audio is not None:
        from beat_weaver.model.probe import load_probe_audio
        probe_mel, probe_bpm = load_probe_audio(probe_audio, config)
        logger.info("Drift probe: %s at %.1f BPM after every epoch", probe_audio, probe_bpm)

    for epoch in range(trainer.epoch, config.max_epochs):
        trainer.epoch = epoch
        t0 = time.time()

        train_loss = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        total_elapsed = time.time() - training_start

        logger.info(
            "Epoch %d/%d (%.1fs, total %.0fs): train_loss=%.4f val_loss=%.4f val_acc=%.4f "
            "val_note_loss=%.4f val_extras_loss=%.4f",
            epoch + 1, config.max_epochs, elapsed, total_elapsed,
            train_loss, val_metrics["val_loss"], val_metrics["val_token_accuracy"],
            val_metrics["val_note_loss"], val_metrics["val_extras_loss"],
        )

        # TensorBoard
        trainer.writer.add_scalar("train/loss_epoch", train_loss, epoch)
        trainer.writer.add_scalar("val/loss", val_metrics["val_loss"], epoch)
        trainer.writer.add_scalar("val/token_accuracy", val_metrics["val_token_accuracy"], epoch)
        trainer.writer.add_scalar("val/note_loss", val_metrics["val_note_loss"], epoch)
        trainer.writer.add_scalar("val/extras_loss", val_metrics["val_extras_loss"], epoch)
        trainer.writer.add_scalar("timing/epoch_seconds", elapsed, epoch)
        trainer.writer.add_scalar("timing/total_seconds", total_elapsed, epoch)

        if probe_mel is not None:
            from beat_weaver.model.probe import probe
            stats = probe(model, config, probe_mel, probe_bpm)
            if stats:
                logger.info("Probe epoch %d: nps=%.2f doubles=%.3f reversal=%.3f left=%.3f",
                            epoch + 1, stats["nps"], stats["doubles"], stats["reversal"], stats["left"])
                for k, v in stats.items():
                    trainer.writer.add_scalar(f"probe/{k}", v, epoch)

        # Checkpoint every epoch
        trainer.save_checkpoint(f"epoch_{epoch + 1:03d}")

        # Best model
        if val_metrics["val_loss"] < trainer.best_val_loss:
            trainer.best_val_loss = val_metrics["val_loss"]
            trainer.patience_counter = 0
            trainer.save_checkpoint("best")
            logger.info("New best model (val_loss=%.4f)", trainer.best_val_loss)
        else:
            trainer.patience_counter += 1
            if trainer.patience_counter >= config.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    trainer.writer.close()

    # Write training summary
    total_time = time.time() - training_start
    epochs_completed = len(epoch_times)
    avg_epoch = sum(epoch_times) / max(1, epochs_completed)
    summary = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "batches_per_epoch": len(train_loader),
        "batch_size": config.batch_size,
        "epochs_completed": epochs_completed,
        "total_time_seconds": round(total_time, 1),
        "avg_epoch_seconds": round(avg_epoch, 1),
        "samples_per_second": round(len(train_dataset) / avg_epoch, 1),
        "best_val_loss": round(trainer.best_val_loss, 4),
        "device": str(trainer.device),
        "model_parameters": model.count_parameters(),
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Training complete: %d epochs in %.0fs (avg %.1fs/epoch, %.1f samples/s)",
        epochs_completed, total_time, avg_epoch,
        len(train_dataset) / avg_epoch,
    )

    return output_dir / "checkpoints" / "best"
