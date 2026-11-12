"""End-to-end tests: stream real corpus rows, build manifests, score fake hypotheses.

These hit the Hugging Face Hub, so they need network access.
"""

from __future__ import annotations

import json

import pytest

from wercal.data.manifest import build_manifest
from wercal.paths import hypotheses_path, manifest_path, responses_path


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point every pipeline path at a temp directory."""
    import wercal.paths as paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    return tmp_path


def test_librispeech_manifest_to_response_matrix(data_dir, monkeypatch):
    """Manifest -> hypotheses file -> score CLI -> response matrix with known error counts."""
    manifest = build_manifest("librispeech", "dev-clean", n=10, seed=0, max_rows=40)
    assert len(manifest) == 10
    assert manifest.equals(build_manifest("librispeech", "dev-clean", n=10, seed=0, max_rows=40)), "sampling must be deterministic"

    path = manifest_path("librispeech", "dev-clean")
    path.parent.mkdir(parents=True)
    manifest.to_parquet(path, index=False)

    # Fake model: perfect on every utterance except the first, where it drops the first word
    # and swaps the last one. Raw hypotheses use normal casing and punctuation, like real ASR.
    model = "fake/perfect-but-one"
    first = manifest.iloc[0]["utt_id"]
    hyp_file = hypotheses_path(model, "librispeech", "dev-clean")
    hyp_file.parent.mkdir(parents=True)
    with hyp_file.open("w") as f:
        for utt in manifest.itertuples():
            words = utt.ref.split()
            if utt.utt_id == first:
                words = words[1:-1] + ["zebra"]
            hyp = " ".join(words).capitalize() + "."
            f.write(json.dumps({"utt_id": utt.utt_id, "hyp": hyp}) + "\n")

    from wercal import score

    monkeypatch.setattr("sys.argv", ["score", "--model", model, "--corpus", "librispeech", "--split", "dev-clean"])
    score.main()

    import pandas as pd

    responses = pd.read_parquet(responses_path(model, "librispeech", "dev-clean"))
    assert list(responses.columns) == ["model", "dataset", "split", "utt_id", "ref", "hyp", "n_ref", "sub", "del", "ins"]
    bad = responses.set_index("utt_id").loc[first]
    assert (bad["sub"], bad["del"], bad["ins"]) == (1, 1, 0)
    assert responses.drop(responses.index[responses.utt_id == first])[["sub", "del", "ins"]].to_numpy().sum() == 0


@pytest.mark.parametrize(
    ("corpus", "split", "bad_marker"),
    [("tedlium2", "test", "ignore_time_segment_in_scoring"), ("ami_ihm", "dev", None)],
)
def test_manifests_drop_unscorable_utterances(corpus, split, bad_marker):
    """TED-LIUM non-speech segments and AMI filler-only utterances never reach the manifest."""
    manifest = build_manifest(corpus, split, n=1000, max_rows=300)
    assert len(manifest) > 0
    assert (manifest["n_ref"] > 0).all()
    if bad_marker:
        assert not manifest["ref_raw"].str.contains(bad_marker).any()
