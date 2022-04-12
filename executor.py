import array
import os
import tempfile
import urllib
import warnings
from typing import Dict, Optional, Tuple, Union, Sequence
from urllib.parse import urlparse

import numpy as np
import torch
from docarray import Document
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from openl3 import preprocess_audio
from openl3.models import load_audio_embedding_model
from pydub import AudioSegment
from pydub.utils import get_array_type
from transformers import CLIPModel, CLIPTokenizer

warnings.filterwarnings("ignore")


def _load_mp3(path_to_mp3: str) -> Tuple[np.ndarray, int]:
    sound = AudioSegment.from_mp3(file=path_to_mp3)
    left, right = sound.split_to_mono()

    bit_depth = left.sample_width * 8
    array_type = get_array_type(bit_depth)

    left = np.array(array.array(array_type, left._data))
    right = np.array(array.array(array_type, right._data))

    mean = np.mean([left, right], axis=0)
    normalized = mean / np.max(np.abs(mean))

    return normalized, sound.frame_rate


def _remove_first_and_last(embeddings, ts_list):
    """
    Internal analysis has shown that the first an last embedding is usually poorly projected.
    Likely due to padding.
    """
    return [e[1:-1] for e in embeddings], [t[1:-1] for t in ts_list]


def _download_mp3(doc: Document) -> str:
    def download_file(
        url: str,
        output_file: Union[str, os.PathLike],
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Download a file via HTTP[S] to a specified directory

        :param url: URL of the file to be downloaded
        :param output_file: Destination path for the downloaded file
        :param headers: Optional headers to add to request, e.g. {"Authorization": "Bearer <access_token>" }
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if headers:
            headers_list = [(k, v) for k, v in headers.items()]
            opener = urllib.request.build_opener()
            opener.addheaders = headers_list
            urllib.request.install_opener(opener)

        urllib.request.urlretrieve(url, output_file)

    uri = doc.uri
    temp_save_path = os.path.join(tempfile.tempdir, _track_hash_from_uri(uri) + ".mp3")
    if os.path.isfile(temp_save_path):
        return temp_save_path
    else:
        download_file(uri, output_file=temp_save_path)
        return temp_save_path


def _track_hash_from_uri(uri: str) -> str:
    """
    Example URI pattern:
    https://p.scdn.co/mp3-preview/4d26180e6961fd46866cd9106936ea55dfcbaa75?cid=774b29d4f13844c495f206cafdad9c86
    """
    return uri.split("/")[-1].split("?")[0]


class MP3AudioLoader:
    """
    Loads MP3 file into numpy array and stores it under the `.tensor` attribute.
    If the uri is a valid url the loader first downloads the file, loads the audio and deletes the file again.
    """

    __MP3_SR = 44100

    def __init__(self, trim_to_seconds: Optional[int] = None):
        self._trim_to_seconds = trim_to_seconds
        self.log = JinaLogger("MP3AudioLoader")

    def load(self, docs: DocumentArray):
        docs = DocumentArray(list(filter(lambda doc_: bool(doc_.uri), docs)))
        for doc in docs:
            if MP3AudioLoader.is_valid_url(doc.uri):
                path_to_mp3 = _download_mp3(doc)
            else:
                path_to_mp3 = doc.uri

            audio, sr = _load_mp3(path_to_mp3)
            assert MP3AudioLoader.__MP3_SR == sr, "Conflicting frame rates detected."
            if self._trim_to_seconds is not None:
                audio = self.trim_to_seconds(audio)

            doc.tensor = audio
            doc.tags["sr"] = int(sr)
            os.remove(path_to_mp3)

    def trim_to_seconds(self, audio: np.ndarray) -> np.ndarray:
        trim_length = self._trim_to_seconds * MP3AudioLoader.__MP3_SR
        audio = audio[:trim_length]
        if audio.size < trim_length:
            audio = np.concatenate(
                [
                    audio,
                    np.zeros(trim_length - audio.size),
                ]
            )
        return audio

    @staticmethod
    def is_valid_url(url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False


class OpenL3MusicText(Executor):
    """Works only on mp3 codec."""

    def __init__(
        self,
        traversal_paths: str = "@r",
        effective_sample_rate: int = 44100,
        hop_size_in_sec: int = 1,
        batch_size: int = 32,
        trim_to_seconds: Optional[int] = None,
        pretrained_model_name_or_path: str = "openai/clip-vit-base-patch32",
        base_tokenizer_model: Optional[str] = None,
        max_length: int = 77,
        *args,
        **kwargs,
    ):
        """
        :param traversal_paths: Default traversal paths for encoding, used if
            the traversal path is not passed as a parameter with the request.
        :param effective_sample_rate: The sample rate of the audio blobs.
        :param hop_size_in_sec: The embedding are computed for 1 seconds chunks. This parameter controls
            how stride of this sliding window.
        :param batch_size: Default batch size for encoding, used if the
            batch size is not passed as a parameter with the request.
        :param trim_to_seconds: If set, the audio will be either cut or padded with zeros
            to the desired length in seconds
        :param pretrained_model_name_or_path: Can be either:
            - A string, the model id of a pretrained CLIP model hosted
                inside a model repo on huggingface.co, e.g., 'openai/clip-vit-base-patch32'
            - A path to a directory containing model weights saved, e.g., ./my_model_directory/
        :param base_tokenizer_model: Base tokenizer model.
            Defaults to ``pretrained_model_name_or_path`` if None
        :param max_length: Max length argument for the tokenizer.
            All CLIP models use 77 as the max length
        """
        super().__init__(*args, **kwargs)
        self.traversal_paths = traversal_paths
        self.batch_size = batch_size
        self.hop_size_in_seconds = hop_size_in_sec
        self.sr = effective_sample_rate

        self._mp3_loader = MP3AudioLoader(trim_to_seconds=trim_to_seconds)

        self.audio_model = load_audio_embedding_model(
            input_repr="mel256",
            content_type="music",
            embedding_size=512,
            frontend="kapre",
        )

        self.log = JinaLogger("OpenL3MusicText")

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.base_tokenizer_model = (
            base_tokenizer_model or pretrained_model_name_or_path
        )
        self.max_length = max_length

        self.device = 'cpu'
        self.tokenizer = CLIPTokenizer.from_pretrained(self.base_tokenizer_model)
        self.text_model = CLIPModel.from_pretrained(self.pretrained_model_name_or_path)
        self.text_model.eval().to(self.device)

    @requests
    def encode(self, docs: DocumentArray, parameters: Dict, **kwargs):
        audio_docs = DocumentArray(list(filter(lambda doc_: bool(doc_.uri), docs)))
        text_docs = DocumentArray(list(filter(lambda doc_: bool(doc_.text), docs)))

        OpenL3MusicText._assert_doc_has_either_text_or_audio_uri(audio_docs, text_docs)

        self.log.info(
            f"Received batch of size={len(docs)} "
            f"containing {len(audio_docs)} docs with an audio uri "
            f"and {len(text_docs)} with a text attribute"
        )

        for docs_batch in DocumentArray(
            audio_docs[parameters.get("traversal_paths", self.traversal_paths)],
        ).batch(batch_size=parameters.get("batch_size", self.batch_size)):
            self._load_audio(audio_docs)

            embeddings_list, ts_list = self._compute_audio_embeddings(docs_batch)
            embeddings_list, ts_list = _remove_first_and_last(embeddings_list, ts_list)

            for doc, embeddings_this_doc, ts_this_doc in zip(
                docs_batch, embeddings_list, ts_list
            ):
                chunks = DocumentArray()
                for emb, ts in zip(embeddings_this_doc, ts_this_doc):
                    document = Document()
                    document.tags = doc.tags.copy()
                    document.uri = doc.uri
                    document.tags["location"] = int(ts)
                    document.embedding = emb
                    chunks.append(document)
                doc.pop("tensor")
                doc.chunks = chunks

        for docs_batch in DocumentArray(
            text_docs[parameters.get("traversal_paths", self.traversal_paths)],
        ).batch(batch_size=parameters.get("batch_size", self.batch_size)):
            text_batch = docs_batch.texts

            with torch.inference_mode():
                input_tokens = self._generate_input_tokens(text_batch)
                embeddings = self.text_model.get_text_features(**input_tokens).cpu().numpy()
                for doc, embedding in zip(docs_batch, embeddings):
                    doc.embedding = embedding

    def _load_audio(self, docs: DocumentArray):
        self._mp3_loader.load(docs)

    def _generate_input_tokens(self, texts: Sequence[str]):
        input_tokens = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        input_tokens = {k: v.to(self.device) for k, v in input_tokens.items()}
        return input_tokens

    def _compute_audio_embeddings(self, docs: DocumentArray):
        # Collect all audio arrays in a single array
        batch = []
        audio_list = docs.tensors
        sr_list = [self.sr] * len(audio_list)
        for x, sr in zip(audio_list, sr_list):
            x = preprocess_audio(x, sr, hop_size=self.hop_size_in_seconds, center=True)
            batch.append(x)

        file_batch_size_list = [x.shape[0] for x in batch]
        batch = np.vstack(batch)
        # Compute embeddings
        batch_embedding = self.audio_model.predict(
            batch, verbose=0, batch_size=batch.shape[0]
        )

        embedding_list = []
        start_idx = 0
        for file_batch_size in file_batch_size_list:
            end_idx = start_idx + file_batch_size
            embedding_list.append(batch_embedding[start_idx:end_idx, ...])
            start_idx = end_idx

        ts_list = [
            np.arange(z.shape[0]) * self.hop_size_in_seconds for z in embedding_list
        ]

        return embedding_list, ts_list

    @staticmethod
    def _assert_doc_has_either_text_or_audio_uri(audio_docs: DocumentArray, text_docs: DocumentArray):
        all_ids_audio = [d.id for d in audio_docs]
        all_ids_text = [d.id for d in text_docs]

        intersecting_docs = set(all_ids_text).intersection(all_ids_audio)
        if intersecting_docs != set():
            raise ValueError(f'Documents with ids={intersecting_docs} have both the text and uri attribute set.'
                             f'Embeddings would overwrite each other.')
