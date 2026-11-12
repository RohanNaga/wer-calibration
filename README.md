# Calibrating Word Error Rate Across Speech Datasets

Course project for 11-751 / 18-781 (Speech Recognition and Understanding), CMU.

Team: Peiyi Wang, Furi Xiang, Arav Tewari, Rohan Nagabhirava. Mentor: Masao Someki.

## Goal

Raw WER depends on dataset difficulty, so 20% on AMI and 5% on LibriSpeech are not directly comparable.
We use a pool of fixed, pretrained ASR models as anchors to estimate utterance and dataset difficulty,
then map WER onto a common scale. We compare simple normalization against 1D and multidimensional
item response theory (IRT).

## Data

| Dataset | Calibration split | Evaluation split |
|---|---|---|
| LibriSpeech | dev-clean, dev-other | test-clean, test-other |
| TED-LIUM 2 | dev | test |
| AMI (IHM and/or SDM) | dev | test |

Each split is subsampled with a fixed seed. No ASR model is trained; all models stay frozen.

## Pipeline

1. **Transcribe**: run every anchor model on the same utterances and cache hypotheses.
2. **Normalize and score**: apply one shared text normalizer, then count errors per utterance.
3. **Response matrix**: one row per (model, utterance), schema below.
4. **Calibrate**: baselines, 1D binomial IRT, multidimensional IRT.
5. **Evaluate**: leave one model out, estimate its ability on dataset A, predict its WER on dataset B.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # add ",asr" for model inference deps
pytest                          # end-to-end tests; needs network for the Hugging Face Hub
```

Set `WERCAL_DATA_DIR` to put data somewhere other than `./data` (for example scratch storage on Delta).

## Data prep usage

```bash
# 1. Build a fixed-seed utterance list (manifest) for a corpus split
python -m wercal.data.manifest --corpus librispeech --split dev-clean --n 500 --seed 0

# 2. Run a model and write data/hyps/{model}/{corpus}_{split}.jsonl,
#    one line per utterance: {"utt_id": "...", "hyp": "raw model output"}

# 3. Normalize and score into response-matrix rows
python -m wercal.score --model openai/whisper-large-v3 --corpus librispeech --split dev-clean
```

Corpora: `librispeech` (dev-clean, dev-other, test-clean, test-other), `tedlium2` (dev, test),
`ami_ihm` and `ami_sdm` (dev, test). All stream from the Hugging Face Hub, so nothing needs
downloading by hand. TED-LIUM comes from `distil-whisper/tedlium-dev-test`, because the official
repo is gated. Releases 1 to 3 share the same dev and test talks.

Manifests drop TED-LIUM `ignore_time_segment_in_scoring` segments and utterances that are empty
after normalization, such as AMI "mm-hmm" backchannels.

## Response matrix schema

Stored as Parquet at `data/responses/{corpus}_{split}/{model}.parquet`, one file per model so runs
on different machines never overwrite each other.

| Column | Type | Meaning |
|---|---|---|
| `model` | str | Anchor model id, e.g. `openai/whisper-large-v3` |
| `dataset` | str | `librispeech`, `tedlium2`, `ami_ihm`, `ami_sdm` |
| `split` | str | e.g. `dev-clean`, `test` |
| `utt_id` | str | Utterance id from the source corpus |
| `ref` | str | Normalized reference text |
| `hyp` | str | Normalized hypothesis text |
| `n_ref` | int | Reference word count |
| `sub` | int | Substitutions |
| `del` | int | Deletions |
| `ins` | int | Insertions |

Store counts, not per-utterance WER. Per-utterance WER is unbounded and noisy on short utterances,
and the IRT likelihood needs counts.

## Layout

```
src/wercal/
  data/corpora.py   corpus registry and Hub streaming
  data/manifest.py  filtering and fixed-seed subsampling
  normalize.py      shared text normalizer (Whisper's English normalizer)
  score.py          error counting into the response matrix
  paths.py          where every artifact lives
  asr/              one runner per toolkit (HF, ESPnet, NeMo)
  irt/              baselines, 1D IRT, MIRT
  eval/             leave-one-model-out harness
tests/              end-to-end tests
data/               manifests, hypotheses, response matrices (git-ignored)
```

## Work split

| Area | Owner |
|---|---|
| Data loading, subsampling, text normalization, scoring | Rohan |
| ASR inference runners | TBD |
| IRT models | TBD |
| Baselines and evaluation harness | TBD |

## First milestone

A thin end-to-end slice: 3 models, about 100 utterances from each of 2 datasets, through every stage
down to a leave-one-model-out prediction.
