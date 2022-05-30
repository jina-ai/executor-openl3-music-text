import array
import io
import warnings
from typing import Dict, Optional, Sequence

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


def _load_audio(doc: Document) -> Document:
    sound = AudioSegment.from_file(io.BytesIO(doc.blob))
    left, right = sound.split_to_mono()

    bit_depth = left.sample_width * 8
    array_type = get_array_type(bit_depth)

    left = np.array(array.array(array_type, left._data))
    right = np.array(array.array(array_type, right._data))

    mean = np.mean([left, right], axis=0)
    normalized = mean / np.max(np.abs(mean))
    doc.tensor = normalized
    doc.tags['sr'] = sound.frame_rate
    return doc


def _remove_first_and_last(embeddings, ts_list):
    """
    Internal analysis has shown that the first an last embedding is usually poorly projected.
    Likely due to padding.
    """
    return [e[1:-1] for e in embeddings], [t[1:-1] for t in ts_list]


class BiModalMusicTextEncoder(Executor):
    """Works only on mp3 codec."""

    def __init__(
        self,
        traversal_paths: str = "@r",
        effective_sample_rate: int = 44100,
        hop_size_in_sec: int = 5,
        batch_size: int = 32,
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
        need_to_compute_audio_doc_ids = [
            doc.id for doc in list(filter(lambda doc_: bool(doc_.blob) and doc_.embedding is None, docs))
        ]
        already_embedded = [
            doc.id for doc in list(filter(lambda doc_: doc_.embedding is not None, docs))

        ]
        need_to_compute_text_doc_ids = [
            doc.id for doc in list(filter(lambda doc_: bool(doc_.text) and doc_.embedding is None, docs))
        ]

        self.log.info(
            f"Received batch of size={len(docs)} "
            f"containing {len(need_to_compute_audio_doc_ids)} docs with an audio blobs "
            f"and {len(need_to_compute_text_doc_ids)} with a text attribute"
        )

        audio_results = []
        if len(need_to_compute_audio_doc_ids) > 0:
            for docs_batch in DocumentArray(
                docs[need_to_compute_audio_doc_ids][parameters.get("traversal_paths", self.traversal_paths)],
            ).batch(batch_size=parameters.get("batch_size", self.batch_size)):
                docs_batch.apply(_load_audio)
                embeddings_list, ts_list = self._compute_audio_embeddings(docs_batch)
                embeddings_list, ts_list = _remove_first_and_last(embeddings_list, ts_list)

                for doc, embeddings_this_doc, ts_this_doc in zip(
                    docs_batch, embeddings_list, ts_list
                ):
                    for emb, ts in zip(embeddings_this_doc, ts_this_doc):
                        document = Document()
                        document.tags = doc.tags.copy()
                        document.tags["location"] = int(ts)
                        document.embedding = emb
                        audio_results.append(document)

        text_results = []
        if len(need_to_compute_text_doc_ids) > 0:
            for docs_batch in DocumentArray(
                docs[need_to_compute_text_doc_ids][parameters.get("traversal_paths", self.traversal_paths)],
            ).batch(batch_size=parameters.get("batch_size", self.batch_size)):
                text_batch = docs_batch.texts

                with torch.inference_mode():
                    input_tokens = self._generate_input_tokens(text_batch)
                    embeddings = self.text_model.get_text_features(**input_tokens).cpu().numpy()
                    for doc, embedding in zip(docs_batch, embeddings):
                        document = Document()
                        document.tags = doc.tags.copy()
                        document.embedding = embedding
                        text_results.append(document)

        # text embeddings have been modified in place and can be added to the result directly
        # for audio docs, we create multiple embeddings per doc and therefore need to replace them and
        # add audio docs that had an embedding already
        result = DocumentArray(
            text_results +
            audio_results
        )
        if len(already_embedded) > 0:
            result.extend(docs[already_embedded])
        self.log.info(f'Returning result (size={len(result)})')
        return result

    def _load_audio(self, docs: DocumentArray):
        docs.apply(_load_audio)

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
