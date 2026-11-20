"""HF transformers teacher backend: plain forward pass, the slow reference (PLAN 1.3).

Logits at position j-1 give log p(input_ids[j] | input_ids[:j]); log-softmax is taken in
float32. Used to validate the vLLM backend (mean |Δ| < 0.02 nats/token on 20 sequences).
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from typing import Any

from qopd.teacher.base import TeacherScorer, check_scores
from qopd.types import Rollout, TeacherScores
from qopd.utils.logging import get_logger
from qopd.utils.memory import track_memory

log = get_logger("teacher.hf")


class HFTeacher(TeacherScorer):
    backend = "hf"

    def __init__(
        self,
        model_path: str,
        revision: str | None = None,
        batch_size: int = 1,
        dtype: str = "bfloat16",
        device: str = "cuda",
        pad_token_id: int = 0,
    ) -> None:
        super().__init__(model_path)
        self.revision = revision
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device
        self.pad_token_id = pad_token_id
        self.model: Any = None

    def start(self) -> None:
        if self.model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM  # noqa: PLC0415

        with track_memory(f"load HF teacher {self.model_path}", log):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, revision=self.revision, dtype=getattr(torch, self.dtype)
            ).to(self.device)
            self.model.eval()

    def stop(self) -> None:
        if self.model is None:
            return
        import torch  # noqa: PLC0415

        with track_memory(f"unload HF teacher {self.model_path}", log):
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()

    def score(self, rollouts: Sequence[Rollout], top_k: int) -> list[TeacherScores]:
        if self.model is None:
            raise RuntimeError("call start() first")
        import torch  # noqa: PLC0415

        scores: list[TeacherScores] = []
        for start in range(0, len(rollouts), self.batch_size):
            batch = rollouts[start : start + self.batch_size]
            seqs = [r.input_ids for r in batch]
            width = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), width), self.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for i, s in enumerate(seqs):  # right padding keeps positions 0..len-1
                ids[i, : len(s)] = torch.tensor(s)
                mask[i, : len(s)] = 1
            with torch.inference_mode():
                logits = self.model(
                    input_ids=ids.to(self.device), attention_mask=mask.to(self.device)
                ).logits
            for i, r in enumerate(batch):
                n_p, n_r = len(r.prompt_ids), len(r.response_ids)
                # logits[t] predicts token t+1, so response token j uses logits[j-1]
                lp = torch.log_softmax(logits[i, n_p - 1 : n_p + n_r - 1].float(), dim=-1)
                target = torch.tensor(r.response_ids, device=lp.device)
                token_lp = lp.gather(1, target[:, None]).squeeze(1)
                if top_k:
                    top_v, top_i = lp.topk(top_k, dim=-1)
                    topk_ids, topk_lp = top_i.tolist(), top_v.tolist()
                else:
                    topk_ids, topk_lp = [], []
                scores.append(TeacherScores(token_lp.tolist(), topk_ids, topk_lp))
            del logits
        check_scores(rollouts, scores, top_k)
        return scores
