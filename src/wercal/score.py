import argparse
import json

import jiwer

from wercal.data.manifest import load_manifest
from wercal.normalize import normalize
from wercal.paths import hyps_path, responses_path


def count_errors(ref, hyp):
    if not hyp:
        return 0, len(ref.split()), 0
    out=jiwer.process_words(ref, hyp)
    return out.substitutions, out.deletions, out.insertions


def load_hyps(path):
    with open(path) as f:
        return {r["utt_id"]: r["hyp"] for r in map(json.loads, f)}


def score_model(manifest, hyps, model):
    missing=set(manifest.utt_id) - set(hyps)
    if missing:
        raise ValueError(f"{model}: {len(missing)} utterances have no hypothesis")
    df=manifest[["dataset", "split", "utt_id", "ref", "n_ref"]].copy()
    df.insert(0, "model", model)
    df["hyp"]=[normalize(hyps[u]) for u in df.utt_id]
    df["sub"], df["del"], df["ins"]=zip(*map(count_errors, df.ref, df.hyp))
    return df[["model", "dataset", "split", "utt_id", "ref", "hyp", "n_ref", "sub", "del", "ins"]]


def corpus_wer(df):
    return df[["sub", "del", "ins"]].values.sum() / df.n_ref.sum()


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--split", required=True)
    a=p.parse_args()
    df=score_model(load_manifest(a.corpus, a.split), load_hyps(hyps_path(a.model, a.corpus, a.split)), a.model)
    out=responses_path(a.model, a.corpus, a.split)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{a.model} {a.corpus}/{a.split}: WER {corpus_wer(df):.2%} on {len(df)} utterances")
