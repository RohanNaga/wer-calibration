"""Registry of evaluation corpora and how to stream them from the Hugging Face Hub."""

from __future__ import annotations

from dataclasses import dataclass

from datasets import Audio, IterableDataset, load_dataset


@dataclass(frozen=True)
class Corpus:
    """One evaluation corpus and the Hub coordinates needed to load it.

    `splits` maps our split names (used in file paths and the response matrix)
    to the split names on the Hub.
    """

    name: str
    hf_path: str
    hf_config: str | None
    splits: dict[str, str]
    id_col: str
    text_col: str


CORPORA: dict[str, Corpus] = {
    "librispeech": Corpus(
        name="librispeech",
        hf_path="openslr/librispeech_asr",
        hf_config="all",
        splits={
            "dev-clean": "validation.clean",
            "dev-other": "validation.other",
            "test-clean": "test.clean",
            "test-other": "test.other",
        },
        id_col="id",
        text_col="text",
    ),
    # The official LIUM/tedlium repo is gated and script-based, so we use a Parquet copy.
    # TED-LIUM releases 1-3 share the same dev and test talks, so these are the TED-LIUM 2 sets.
    "tedlium2": Corpus(
        name="tedlium2",
        hf_path="distil-whisper/tedlium-dev-test",
        hf_config=None,
        splits={"dev": "validation", "test": "test"},
        id_col="id",
        text_col="text",
    ),
    "ami_ihm": Corpus(
        name="ami_ihm",
        hf_path="edinburghcstr/ami",
        hf_config="ihm",
        splits={"dev": "validation", "test": "test"},
        id_col="audio_id",
        text_col="text",
    ),
    "ami_sdm": Corpus(
        name="ami_sdm",
        hf_path="edinburghcstr/ami",
        hf_config="sdm",
        splits={"dev": "validation", "test": "test"},
        id_col="audio_id",
        text_col="text",
    ),
}


def get_corpus(name: str) -> Corpus:
    """Look up a corpus by name, with a helpful error for typos."""
    try:
        return CORPORA[name]
    except KeyError:
        raise KeyError(f"unknown corpus {name!r}; choose from {sorted(CORPORA)}") from None


def stream_split(corpus_name: str, split: str, *, with_audio: bool = False) -> IterableDataset:
    """Stream one split of a corpus without downloading the whole thing.

    With `with_audio=False` the audio column is left undecoded, so building
    manifests needs no audio backend and stays fast.
    """
    corpus = get_corpus(corpus_name)
    if split not in corpus.splits:
        raise KeyError(f"{corpus_name} has no split {split!r}; choose from {sorted(corpus.splits)}")
    ds = load_dataset(
        corpus.hf_path,
        corpus.hf_config,
        split=corpus.splits[split],
        streaming=True,
    )
    if not with_audio:
        ds = ds.cast_column("audio", Audio(decode=False))
    return ds
