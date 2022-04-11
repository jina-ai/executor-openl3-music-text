from jina import Executor, DocumentArray, requests


class OpenL3MusicText(Executor):
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        pass
