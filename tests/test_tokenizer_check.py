"""Tokenizer identity check (PLAN 1.1): identical pairs pass, mismatched pairs fail."""

import copy
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from qopd.config import CONFIG_ROOT
from qopd.models.loading import load_tokenizer
from qopd.models.registry import (
    ModelNotFoundError,
    load_registry,
    pin_revision,
    resolve,
    verify_exists,
)
from qopd.models.tokenizer_check import (
    TokenizerMismatchError,
    assert_same_tokenizer,
    compare_tokenizers,
)

CORPUS = ["the student samples and the teacher scores every token"] * 20 + [
    "on policy distillation with quantized teachers",
    "how many bits does a teacher need",
]
TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
)


def make_tok(
    corpus=CORPUS, template=TEMPLATE, extra_special=(), pre=None
) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre or pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=120, special_tokens=["<unk>", "<|im_start|>", "<|im_end|>"]
    )
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<|im_end|>")
    fast.chat_template = template
    if extra_special:
        fast.add_special_tokens({"additional_special_tokens": list(extra_special)})
    return fast


def test_identical_pair_passes():
    a, b = make_tok(), make_tok()
    diff = assert_same_tokenizer(a, b)
    assert diff.identical


def test_extra_special_token_fails():
    with pytest.raises(TokenizerMismatchError, match="special_tokens"):
        assert_same_tokenizer(make_tok(), make_tok(extra_special=["<tool>"]))


def test_chat_template_fails():
    diff = compare_tokenizers(make_tok(), make_tok(template=TEMPLATE + "{{ eos_token }}"))
    assert diff.chat_template and not diff.vocab


def test_different_merges_fail():
    diff = compare_tokenizers(make_tok(), make_tok(corpus=[*CORPUS, "zebra zebra zebra quartz"]))
    assert diff.merges or diff.vocab
    assert not diff.identical


def test_pre_tokenizer_fails():
    diff = compare_tokenizers(make_tok(), make_tok(pre=pre_tokenizers.ByteLevel()))
    assert diff.pipeline


def test_local_rung_not_built():
    with pytest.raises(ModelNotFoundError, match="not built yet"):
        verify_exists(resolve("T4-int"))


def test_pin_revision_keeps_comments(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    src = (CONFIG_ROOT / "models" / "S2.yaml").read_text()
    (models_dir / "S2.yaml").write_text(src)
    pin_revision("S2", "abc123", models_dir=models_dir)
    text = (models_dir / "S2.yaml").read_text()
    assert "revision: abc123" in text
    assert text.splitlines()[0] == src.splitlines()[0]  # header comment kept
    load_registry.cache_clear()


def _cached(model_id: str) -> PreTrainedTokenizerFast | None:
    try:
        return load_tokenizer(resolve(model_id), local_files_only=True)
    except OSError:
        return None


def test_real_students_identical_and_mutation_fails():
    s1, s2 = _cached("S1"), _cached("S2")
    if s1 is None or s2 is None:
        pytest.skip("S1/S2 tokenizers not in the HF cache (qopd models download S1 S2)")
    assert assert_same_tokenizer(s1, s2).identical
    mutated = copy.deepcopy(s2)
    mutated.add_special_tokens({"additional_special_tokens": ["<|qopd_mismatch|>"]})
    with pytest.raises(TokenizerMismatchError):
        assert_same_tokenizer(s1, mutated)
    templ = copy.deepcopy(s2)
    templ.chat_template = (s2.chat_template or "") + " "
    assert compare_tokenizers(s1, templ).chat_template
