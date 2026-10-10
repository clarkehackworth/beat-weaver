"""Score checkpoints on the placement-only validation loss: the loss on LEFT/RIGHT targets only.

    python scripts/score_placement.py output/training/checkpoints/epoch_014 \\
                                      output/placement/checkpoints/epoch_003

Validation loss from different training runs is only comparable when it covers the same tokens.
The main run's loss covers every token; the placement run's covers only the two hands. This scores
every checkpoint you give it the placement way, so an epoch of the main run and an epoch of the
placement run can be compared. Lower is better. Uses the placement config's validation split.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.dataset import BeatSaberDataset, collate_fn
from beat_weaver.model.training import Trainer
from beat_weaver.model.transformer import BeatWeaverModel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", type=Path, help="checkpoint folders (each has model.pt and config.json)")
    ap.add_argument("--config", type=Path, default=Path("configs/medium_conformer_placement.json"),
                    help="config that defines the placement-only loss and the data filters")
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--audio-manifest", type=Path, default=Path("data/audio_manifest.json"))
    args = ap.parse_args()

    cfg = ModelConfig.load(args.config)
    assert cfg.placement_only, f"{args.config} is not a placement_only config, so the loss would cover every token"
    val = BeatSaberDataset(args.data, args.audio_manifest, cfg, split="val")
    loader = DataLoader(val, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    for ck in args.checkpoints:
        model = BeatWeaverModel(ModelConfig.load(ck / "config.json"))
        model.load_state_dict(torch.load(ck / "model.pt", map_location="cpu", weights_only=True))
        with tempfile.TemporaryDirectory() as tmp:
            r = Trainer(model, cfg, Path(tmp)).validate(loader)
        print(f"{ck}: placement val_loss={r['val_loss']:.4f} acc={r['val_token_accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
