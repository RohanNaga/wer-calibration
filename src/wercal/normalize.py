from whisper_normalizer.english import EnglishTextNormalizer

_normalizer=EnglishTextNormalizer()


def normalize(text):
    return " ".join(_normalizer(text.replace("<unk>", " ")).split())
