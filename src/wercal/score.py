"""Turn model hypotheses into rows of the response matrix (error counts per utterance)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import jiwer
import pandas as pd

from wercal.data.manifest import load_manifest
from wercal.normalize import normalize
from wercal.paths import hypotheses_path, model_slug, responses_path


@dataclass(frozen=True)
class ErrorCounts:
    """Word-level alignment counts for one utterance."""

    n_ref: int
    sub: int
    dele: int
    ins: int


def count_errors(ref: str, hyp: str) -> ErrorCounts:
    """Align two normalized strings and count substitutions, deletions and insertions."""
    n_ref = len(ref.split())
    if not hyp.strip():
        return ErrorCounts(n_ref=n_ref, sub=0, dele=n_ref, ins=0)
    out = jiwer.process_words(ref, hyp)
    return ErrorCounts(n_ref=n_ref, sub=out.substitutions, dele=out.deletions, ins=out.insertions)


def load_hypotheses(path: Path) -> dict[str, str]:
    """Read a JSONL file of {"utt_id": ..., "hyp": ...} records into a dict."""
    with path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {r["utt_id"]: r["hyp"] for r in records}


def score_model(manifest: pd.DataFrame, raw_hyps: dict[str, str], model: str) -> pd.DataFrame:
    """Score one model on one manifest and return response-matrix rows.

    Every manifest utterance must have a hypothesis: a silent gap would make the
    model look better on exactly the utterances it failed to produce output for.
    """
    missing = set(manifest["utt_id"]) - set(raw_hyps)
    if missing:
        raise ValueError(f"{model}: {len(missing)} manifest utterances have no hypothesis, e.g. {sorted(missing)[:3]}")

    rows = []
    for utt in manifest.itertuples(index=False):
        hyp = normalize(raw_hyps[utt.utt_id])
        counts = count_errors(utt.ref, hyp)
        rows.append(
            {
                "model": model,
                "dataset": utt.dataset,
                "split": utt.split,
                "utt_id": utt.utt_id,
                "ref": utt.ref,
                "hyp": hyp,
                "n_ref": counts.n_ref,
                "sub": counts.sub,
                "del": counts.dele,
                "ins": counts.ins,
            }
        )
    return pd.DataFrame(rows)


def corpus_wer(responses: pd.DataFrame) -> float:
    """Corpus-level WER: total errors over total reference words (not a mean of per-utterance WERs)."""
    errors = responses[["sub", "del", "ins"]].to_numpy().sum()
    return float(errors / responses["n_ref"].sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one model's hypotheses against a manifest.")
    parser.add_argument("--model", required=True, help="model id, e.g. openai/whisper-large-v3")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.corpus, args.split)
    raw_hyps = load_hypotheses(hypotheses_path(args.model, args.corpus, args.split))
    responses = score_model(manifest, raw_hyps, args.model)

    out = responses_path(args.model, args.corpus, args.split)
    out.parent.mkdir(parents=True, exist_ok=True)
    responses.to_parquet(out, index=False)
    print(f"{model_slug(args.model)} {args.corpus}/{args.split}: WER {corpus_wer(responses):.2%} on {len(responses)} utterances -> {out}")


if __name__ == "__main__":
    main()
