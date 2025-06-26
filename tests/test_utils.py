import os
import sys
import types

# Ensure project root is in the Python path so src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Provide a dummy langchain_community.embeddings module to avoid import errors
dummy_module = types.ModuleType("langchain_community.embeddings")
class DummyHF:
    def __init__(self, *args, **kwargs):
        pass
    def embed_query(self, text):
        return [0.0]

dummy_module.HuggingFaceEmbeddings = DummyHF
sys.modules["langchain_community.embeddings"] = dummy_module

from src.utils import get_embedding_model

def test_get_embedding_model_fallback(monkeypatch):
    call_count = {"count": 0}

    class DummyEmbedding:
        def embed_query(self, text):
            return [0.0]

    def fake_constructor(*args, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise Exception("primary fail")
        return DummyEmbedding()

    monkeypatch.setattr("src.utils.HuggingFaceEmbeddings", fake_constructor)

    embeddings = get_embedding_model()

    assert isinstance(embeddings, DummyEmbedding)
    assert call_count["count"] == 2
