"""Cold-start and off-policy SFT (PLAN 5.2, baseline in 8.2).

Full fine-tune, AdamW, BF16, loss on response tokens only, gradient checkpointing,
lr 1e-5, 1 epoch (configs/train/sft.yaml). Memory logged every step.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qopd.config import SFTCfg
    from qopd.utils.run import RunDir


def build_sft_dataset(traces: Path, tokenizer: object, max_len: int) -> object:
    """Tokenized (prompt, response) pairs with response-only labels."""
    raise NotImplementedError("step 5.2")


def train_sft(cfg: SFTCfg, run: RunDir) -> Path:
    """Train and return the final checkpoint directory (resumable from ``state.json``)."""
    raise NotImplementedError("step 5.2")
