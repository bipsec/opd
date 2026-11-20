"""Hard check that teacher and student tokenizers are identical (PLAN 1.1).

OPD compares per-token log-probs, so teacher and student must share the vocabulary,
merges, special tokens and chat template exactly. Any mismatch raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class TokenizerMismatchError(RuntimeError):
    """Teacher and student tokenizers differ."""


@dataclass
class TokenizerDiff:
    vocab: list[str] = field(default_factory=list)
    merges: list[str] = field(default_factory=list)
    special_tokens: list[str] = field(default_factory=list)
    chat_template: bool = False

    @property
    def identical(self) -> bool:
        return not (self.vocab or self.merges or self.special_tokens or self.chat_template)


def compare_tokenizers(a: Any, b: Any) -> TokenizerDiff:
    """Compare two HF tokenizers field by field."""
    raise NotImplementedError("step 1.1")


def assert_same_tokenizer(teacher: Any, student: Any) -> None:
    """Raise :class:`TokenizerMismatchError` with a readable diff on any mismatch."""
    raise NotImplementedError("step 1.1")
