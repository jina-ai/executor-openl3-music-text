import glob
import os
from typing import List

import numpy as np
import pytest as pytest
from docarray import DocumentArray, Document

from executor import BiModalMusicTextEncoder


@pytest.fixture()
def mp3_files() -> List[str]:
    resource_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')
    return list(glob.glob(os.path.join(resource_path, '*.mp3')))


@pytest.fixture()
def audio_docs(mp3_files: List[str]) -> DocumentArray:
    return DocumentArray(
        [
            Document(
                blob=open(p, 'rb').read()
            ) for p in mp3_files
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


@pytest.fixture()
def embedded_docs() -> DocumentArray:
    return DocumentArray(
        [
            Document(
                embedding=np.ones(512)
            ),
            Document(
                embeddings=np.ones(512)
            ),
        ]
    )


def test_encode_audio(audio_docs: DocumentArray):
    encoder = BiModalMusicTextEncoder(hop_size_in_sec=20, trim_to_seconds=30)

    audio_docs = encoder.encode(audio_docs, {})

    for doc in audio_docs:
        assert doc.embedding is not None
        assert doc.embedding.size == 512


def test_encode_text(text_docs: DocumentArray):
    encoder = BiModalMusicTextEncoder()

    text_docs = encoder.encode(text_docs, {})

    for doc in text_docs:
        assert doc.embedding is not None
        assert doc.embedding.size == 512


def test_passes_already_encoded(embedded_docs: DocumentArray):
    encoder = BiModalMusicTextEncoder()

    embedded_docs = encoder.encode(embedded_docs, {})

    for doc in embedded_docs:
        assert doc.embedding is not None
        assert doc.embedding.size == 512
