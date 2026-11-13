import os
from pathlib import Path

DATA_DIR=Path(os.environ.get("WERCAL_DATA_DIR", Path(__file__).parents[2] / "data"))


def manifest_path(corpus, split):
    return DATA_DIR / "manifests" / f"{corpus}_{split}.parquet"


def hyps_path(model, corpus, split):
    return DATA_DIR / "hyps" / model.replace("/", "__") / f"{corpus}_{split}.jsonl"


def responses_path(model, corpus, split):
    return DATA_DIR / "responses" / f"{corpus}_{split}" / f"{model.replace('/', '__')}.parquet"
