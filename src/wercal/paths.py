"""Where every pipeline artifact lives on disk, so stages agree on file locations."""

from __future__ import annotations

import os
from pathlib import Path

# Override with WERCAL_DATA_DIR on Delta to point at scratch storage.
DATA_DIR = Path(os.environ.get("WERCAL_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))


def model_slug(model: str) -> str:
    """Turn a Hub model id like 'openai/whisper-large-v3' into a filesystem-safe name."""
    return model.replace("/", "__")


def manifest_path(corpus: str, split: str) -> Path:
    """Fixed utterance list for one corpus split."""
    return DATA_DIR / "manifests" / f"{corpus}_{split}.parquet"


def hypotheses_path(model: str, corpus: str, split: str) -> Path:
    """Raw model transcripts, one JSONL record {"utt_id", "hyp"} per utterance."""
    return DATA_DIR / "hyps" / model_slug(model) / f"{corpus}_{split}.jsonl"


def responses_path(model: str, corpus: str, split: str) -> Path:
    """Scored response-matrix rows for one model on one corpus split."""
    return DATA_DIR / "responses" / f"{corpus}_{split}" / f"{model_slug(model)}.parquet"
