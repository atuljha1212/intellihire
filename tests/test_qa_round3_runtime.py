from intellihire.media import transcriber as T
from intellihire.media.transcriber import Segment, drop_prompt_echo


def test_prompt_echo_dropped():
    segs = [Segment(0, 2, "Umm, let me think, like, hmm... Okay, so, uh, I mean, you know."),
            Segment(2, 3, "Okay, so, uh, I mean"),
            Segment(3, 4, "um uh like"),
            Segment(4, 8, "I built a payments service in Python.")]
    out = drop_prompt_echo(segs)
    assert [s.text for s in out] == ["I built a payments service in Python."]


def test_transcribe_silent_audio_returns_empty(monkeypatch):
    fake = T.Transcript(text="", segments=[Segment(0, 3, "Umm, let me think, like, hmm... Okay, so, uh, I mean, you know.")],
                        engine="whisper-local:x")
    monkeypatch.setattr(T, "_transcribe_local", lambda p: fake)
    tr = T.transcribe("x.wav", duration_sec=3)
    assert tr.text == "" and tr.segments == []


def test_encoder_tries_local_first(monkeypatch):
    import sentence_transformers
    from intellihire.embeddings.encoder import SentenceTransformerEncoder
    calls = []

    class Fake:
        def __init__(self, name, **kw):
            calls.append(kw.get("local_files_only", False))
            if kw.get("local_files_only"):
                raise OSError("not cached")

        def get_sentence_embedding_dimension(self):
            return 384

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Fake)
    SentenceTransformerEncoder("m", "c")
    assert calls == [True, False]


def test_pypdf_importable():
    import pypdf  # noqa: F401
