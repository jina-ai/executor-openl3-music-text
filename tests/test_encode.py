import pytest as pytest
from docarray import DocumentArray, Document

from executor import OpenL3MusicText


@pytest.fixture()
def audio_docs() -> DocumentArray:
    return DocumentArray(
        [
            Document(
                uri="https://p.scdn.co/mp3-preview/"
                "4d26180e6961fd46866cd9106936ea55dfcbaa75?cid=774b29d4f13844c495f206cafdad9c86"
            ),
            Document(
                uri="https://p.scdn.co/mp3-preview/"
                "d012e536916c927bd6c8ced0dae75ee3b7715983?cid=774b29d4f13844c495f206cafdad9c86"
            ),
            Document(
                uri="https://p.scdn.co/mp3-preview/"
                "a1c11bb1cb231031eb20e5951a8bfb30503224e9?cid=774b29d4f13844c495f206cafdad9c86"
            ),
        ]
    )


@pytest.fixture()
def text_docs() -> DocumentArray:
    return DocumentArray(
        [
            Document(
                text="electronic alternative"
            ),
            Document(
                text="hip hop 80s"
            ),
        ]
    )


def test_encode_audio(audio_docs: DocumentArray):
    encoder = OpenL3MusicText(hop_size_in_sec=20, trim_to_seconds=30)

    encoder.encode(audio_docs, {})

    for doc in audio_docs['@c']:
        assert doc.embedding is not None
        assert doc.embedding.size == 512


def test_encode_text(text_docs: DocumentArray):
    encoder = OpenL3MusicText()

    encoder.encode(text_docs, {})

    for doc in text_docs:
        assert doc.embedding is not None
        assert doc.embedding.size == 512


def test_fails_if_both_text_and_audio_are_set():
    docs = DocumentArray([Document(text='hip hop', uri='https://someuri.com')])
    encoder = OpenL3MusicText()

    with pytest.raises(ValueError):
        encoder.encode(docs, {})
