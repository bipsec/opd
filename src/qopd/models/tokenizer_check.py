"""Hard check that teacher and student tokenizers are identical (PLAN 1.1).

OPD compares per-token log-probs, so teacher and student must share the vocabulary,
merges, special/added tokens, chat template and the rest of the tokenization pipeline
(normalizer, pre-tokenizer, decoder, post-processor). Any mismatch raises.

Works on HF "fast" tokenizers (``PreTrainedTokenizerFast``); Qwen3 ships only fast ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

PIPELINE_PARTS = ("normalizer", "pre_tokenizer", "post_processor", "decoder")
_MAX_EXAMPLES = 10


class TokenizerMismatchError(RuntimeError):
    """Teacher and student tokenizers differ."""


@dataclass
class TokenizerDiff:
    vocab: list[str] = field(default_factory=list)
    merges: list[str] = field(default_factory=list)
    special_tokens: list[str] = field(default_factory=list)
    chat_template: bool = False
    pipeline: list[str] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        return not (
            self.vocab or self.merges or self.special_tokens or self.chat_template or self.pipeline
        )

    def summary(self) -> str:
        if self.identical:
            return "tokenizers identical"
        parts = []
        for name in ("vocab", "merges", "special_tokens", "pipeline"):
            items = getattr(self, name)
            if items:
                shown = "; ".join(items[:_MAX_EXAMPLES])
                more = f" (+{len(items) - _MAX_EXAMPLES} more)" if len(items) > _MAX_EXAMPLES else ""
                parts.append(f"{name}: {shown}{more}")
        if self.chat_template:
            parts.append("chat_template: differs")
        return " | ".join(parts)


def _backend_json(tok: Any) -> dict[str, Any]:
    backend = getattr(tok, "backend_tokenizer", None)
    if backend is None:
        raise TypeError(f"{type(tok).__name__} is not a fast tokenizer (no backend_tokenizer)")
    return json.loads(backend.to_str())


def _diff_dict(a: dict[Any, Any], b: dict[Any, Any], label: str) -> list[str]:
    out = []
    for k in sorted(set(a) | set(b), key=str):
        if k not in b:
            out.append(f"{label} {k!r} only in first")
        elif k not in a:
            out.append(f"{label} {k!r} only in second")
        elif a[k] != b[k]:
            out.append(f"{label} {k!r}: {a[k]!r} != {b[k]!r}")
    return out


def _added_tokens(tok: Any) -> dict[int, tuple[str, bool]]:
    return {i: (t.content, bool(t.special)) for i, t in tok.added_tokens_decoder.items()}


def compare_tokenizers(a: Any, b: Any) -> TokenizerDiff:
    """Compare two HF fast tokenizers field by field."""
    diff = TokenizerDiff()
    diff.vocab = _diff_dict(a.get_vocab(), b.get_vocab(), "token")

    ja, jb = _backend_json(a), _backend_json(b)
    ma, mb = ja.get("model", {}), jb.get("model", {})
    merges_a, merges_b = ma.get("merges", []), mb.get("merges", [])
    if merges_a != merges_b:
        n = min(len(merges_a), len(merges_b))
        first = next((i for i in range(n) if merges_a[i] != merges_b[i]), n)
        diff.merges.append(
            f"{len(merges_a)} vs {len(merges_b)} merges, first difference at index {first}"
        )
    for key in sorted((set(ma) | set(mb)) - {"vocab", "merges"}):
        if ma.get(key) != mb.get(key):
            diff.pipeline.append(f"model.{key}: {ma.get(key)!r} != {mb.get(key)!r}")
    for part in PIPELINE_PARTS:
        if ja.get(part) != jb.get(part):
            diff.pipeline.append(f"{part} differs")

    diff.special_tokens = _diff_dict(_added_tokens(a), _added_tokens(b), "added token id")
    diff.special_tokens += _diff_dict(
        dict(a.special_tokens_map), dict(b.special_tokens_map), "special token role"
    )
    diff.chat_template = (a.chat_template or None) != (b.chat_template or None)
    return diff


def assert_same_tokenizer(teacher: Any, student: Any) -> TokenizerDiff:
    """Raise :class:`TokenizerMismatchError` with a readable diff on any mismatch."""
    diff = compare_tokenizers(teacher, student)
    if not diff.identical:
        raise TokenizerMismatchError(diff.summary())
    return diff
