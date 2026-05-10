"""Per-model configuration shared by the unified pipeline.

Each reservoir model (SPE9, Norne, Volve) registers a `DeckConfig`
instance in `scripts/decks/{name}.py`. The orchestrator, runner, and
extractor all consume this config and have no model-specific branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional


@dataclass(frozen=True)
class DeckConfig:
    name: str
    deck_dir: Path
    main_deck_filename: str  # e.g. "SPE9.DATA"

    # LHS sampling
    lever_ranges: dict[str, tuple[float, float]]
    sample_validator: Optional[Callable[[dict], bool]] = None

    # Runtime templating: render_deck(params) returns
    # {relative_path_inside_deck_dir: rendered_text}. The runner
    # overwrites those files in the per-sim copy.
    render_deck: Callable[[dict], dict[str, str]] = field(
        default=lambda params: {}
    )

    # Static features (volume-weighted baselines, before per-sim levers)
    static_features: dict[str, float] = field(default_factory=dict)

    # Unit system. METRIC -> FIELD conversion happens inside the extractor.
    unit_system: Literal["FIELD", "METRIC"] = "FIELD"

    # Pre-shift bubble point in the deck unit (psia for FIELD, bar for
    # METRIC). The extractor converts to psi for the schema column.
    baseline_pb: float = 0.0

    flow_timeout_s: int = 1800

    @property
    def summary_basename(self) -> str:
        """Name of the .UNSMRY file (without extension)."""
        return self.main_deck_filename.removesuffix(".DATA")
