from datasets import Audio, load_dataset

CORPORA={
    "librispeech": ("openslr/librispeech_asr", "all", "id", {"dev-clean": "validation.clean", "dev-other": "validation.other", "test-clean": "test.clean", "test-other": "test.other"}),
    "tedlium2": ("distil-whisper/tedlium-dev-test", None, "id", {"dev": "validation", "test": "test"}),
    "ami_ihm": ("edinburghcstr/ami", "ihm", "audio_id", {"dev": "validation", "test": "test"}),
    "ami_sdm": ("edinburghcstr/ami", "sdm", "audio_id", {"dev": "validation", "test": "test"}),
}


def stream_split(corpus, split, audio=False):
    path, config, _, splits=CORPORA[corpus]
    ds=load_dataset(path, config, split=splits[split], streaming=True)
    return ds if audio else ds.cast_column("audio", Audio(decode=False))
