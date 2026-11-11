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

## Response matrix schema

Stored as Parquet at `data/responses/{dataset}_{split}.parquet`.

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
  data/        dataset loaders and subsampling
  asr/         one runner per toolkit (HF, ESPnet, NeMo)
  normalize.py shared text normalizer
  score.py     error counting into the response matrix
  irt/         baselines, 1D IRT, MIRT
  eval/        leave-one-model-out harness
configs/       model pool and dataset configs
data/          cached hypotheses and response matrices (git-ignored)
```

## Work split

| Area | Owner |
|---|---|
| Data loading, subsampling, normalization | TBD |
| ASR inference runners | TBD |
| IRT models | TBD |
| Baselines and evaluation harness | TBD |

## First milestone

A thin end-to-end slice: 3 models, about 100 utterances from each of 2 datasets, through every stage
down to a leave-one-model-out prediction.
