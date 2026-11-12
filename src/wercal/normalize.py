"""The one text normalizer every reference and hypothesis goes through before scoring."""

from __future__ import annotations

from functools import lru_cache

from whisper_normalizer.english import EnglishTextNormalizer

# Markers that appear in corpus references but are never spoken words.
_CORPUS_MARKERS = ("<unk>",)


@lru_cache(maxsize=1)
def _whisper_normalizer() -> EnglishTextNormalizer:
    return EnglishTextNormalizer()


def normalize(text: str) -> str:
    """Normalize a reference or hypothesis into lowercase, space-separated words.

    Uses Whisper's English normalizer, which lowercases, strips punctuation,
    expands contractions, standardizes numbers and spellings, and drops fillers
    such as "um", "uh" and "mm-hmm". The same function must be applied to both
    sides, or normalization differences show up as fake dataset difficulty.
    """
    for marker in _CORPUS_MARKERS:
        text = text.replace(marker, " ")
    return " ".join(_whisper_normalizer()(text).split())
