import argparse
from itertools import islice

import pandas as pd

from wercal.data.corpora import CORPORA, stream_split
from wercal.normalize import normalize
from wercal.paths import manifest_path


def build_manifest(corpus, split, n, seed=0, min_words=1, max_rows=None):
    id_col, rows=CORPORA[corpus][2], []
    for ex in islice(stream_split(corpus, split), max_rows):
        ref=normalize(ex["text"])
        if "ignore_time_segment_in_scoring" not in ex["text"] and len(ref.split()) >= min_words:
            rows.append({"dataset": corpus, "split": split, "utt_id": str(ex[id_col]), "ref_raw": ex["text"], "ref": ref, "n_ref": len(ref.split())})
    pool=pd.DataFrame(rows).sort_values("utt_id", ignore_index=True)
    return pool.sample(n=min(n, len(pool)), random_state=seed).sort_values("utt_id", ignore_index=True)


def load_manifest(corpus, split):
    return pd.read_parquet(manifest_path(corpus, split))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-words", type=int, default=1)
    p.add_argument("--max-rows", type=int)
    a=p.parse_args()
    df=build_manifest(a.corpus, a.split, a.n, a.seed, a.min_words, a.max_rows)
    out=manifest_path(a.corpus, a.split)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{a.corpus}/{a.split}: {len(df)} utterances, {df.n_ref.sum()} words")
