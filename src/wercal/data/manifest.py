"""Build fixed, seeded utterance lists so every model is scored on exactly the same clips."""

from __future__ import annotations

import argparse
from itertools import islice

import pandas as pd

from wercal.data.corpora import get_corpus, stream_split
from wercal.normalize import normalize
from wercal.paths import manifest_path

# TED-LIUM marks stretches of non-speech (music, applause) with this text; they are never scored.
_TEDLIUM_IGNORE = "ignore_time_segment_in_scoring"


def keep_utterance(raw_text: str, ref: str, corpus: str) -> bool:
    """Decide whether an utterance enters the manifest.

    Rules that are not judgment calls live here. The length policy, which is a
    real modeling choice, lives in `passes_length_policy`.
    """
    if _TEDLIUM_IGNORE in raw_text:
        return False
    n_words = len(ref.split())
    if n_words == 0:  # e.g. AMI "MM-HMM", which the normalizer removes as a filler
        return False
    return passes_length_policy(n_words, corpus)


def passes_length_policy(n_words: int, corpus: str) -> bool:
    """Decide whether an utterance is long enough to keep, after normalization.

    TODO(Rohan): choose the policy. Currently every non-empty utterance is kept.
    """
    return True


def build_manifest(
    corpus_name: str,
    split: str,
    n: int,
    seed: int = 0,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Stream a split, filter it, and draw a fixed-seed sample of `n` utterances.

    Rows are sorted by utterance id before sampling, so the same seed always
    yields the same manifest regardless of the order the Hub streams rows in.
    `max_rows` caps how many rows are read, for quick smoke tests.
    """
    corpus = get_corpus(corpus_name)
    rows = []
    for ex in islice(stream_split(corpus_name, split), max_rows):
        raw_text = ex[corpus.text_col]
        ref = normalize(raw_text)
        if keep_utterance(raw_text, ref, corpus_name):
            rows.append(
                {
                    "dataset": corpus_name,
                    "split": split,
                    "utt_id": str(ex[corpus.id_col]),
                    "ref_raw": raw_text,
                    "ref": ref,
                    "n_ref": len(ref.split()),
                }
            )

    pool = pd.DataFrame(rows).sort_values("utt_id", ignore_index=True)
    if pool["utt_id"].duplicated().any():
        raise ValueError(f"{corpus_name}/{split}: duplicate utterance ids in source data")
    sample = pool.sample(n=min(n, len(pool)), random_state=seed)
    return sample.sort_values("utt_id", ignore_index=True)


def load_manifest(corpus: str, split: str) -> pd.DataFrame:
    """Read a manifest written by `build_manifest`."""
    return pd.read_parquet(manifest_path(corpus, split))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a seeded utterance manifest for one corpus split.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--n", type=int, default=500, help="utterances to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None, help="read at most this many rows (smoke tests)")
    args = parser.parse_args()

    manifest = build_manifest(args.corpus, args.split, args.n, args.seed, args.max_rows)
    out = manifest_path(args.corpus, args.split)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out, index=False)
    print(f"{args.corpus}/{args.split}: {len(manifest)} utterances, {manifest['n_ref'].sum()} words -> {out}")


if __name__ == "__main__":
    main()
