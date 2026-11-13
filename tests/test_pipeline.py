import pytest

from wercal.data.manifest import build_manifest
from wercal.score import corpus_wer, score_model


def test_librispeech_end_to_end():
    m=build_manifest("librispeech", "dev-clean", n=10, max_rows=40)
    assert len(m) == 10
    assert m.equals(build_manifest("librispeech", "dev-clean", n=10, max_rows=40))
    first=m.utt_id[0]
    hyps={u: (" ".join(r.split()[1:-1] + ["zebra"]) if u == first else r).capitalize() + "." for u, r in zip(m.utt_id, m.ref)}
    df=score_model(m, hyps, "fake")
    assert tuple(df.loc[0, ["sub", "del", "ins"]]) == (1, 1, 0)
    assert corpus_wer(df) == 2 / df.n_ref.sum()


@pytest.mark.parametrize("corpus,split", [("tedlium2", "test"), ("ami_ihm", "dev")])
def test_unscorable_utterances_dropped(corpus, split):
    m=build_manifest(corpus, split, n=1000, max_rows=300)
    assert len(m) and (m.n_ref > 0).all()
    assert not m.ref_raw.str.contains("ignore_time_segment_in_scoring").any()
