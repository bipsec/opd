"""One module per quantization format; all implement :class:`Recipe`.

Every recipe uses the 512-prompt calibration split and writes
``models/quant/<rung>/recipe.json`` with tool versions and the calibration hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class RecipeResult:
    rung: str
    out_dir: Path
    format: str
    tool: str
    tool_version: str
    calibration_sha256: str
    seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


class Recipe(Protocol):
    """A quantization recipe for one format."""

    format: str
    tool: str

    def build(self, src: str, calibration: Path, out_dir: Path, **kwargs: Any) -> RecipeResult:
        """Quantize ``src`` (HF repo/path) with the calibration split into ``out_dir``."""
        ...
