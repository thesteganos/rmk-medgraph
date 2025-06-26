import os
import sys
import types

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Create dummy modules for dependencies not installed
# Dummy langchain.tools
dummy_langchain_tools = types.ModuleType("langchain.tools")
class DummyTool:
    def __init__(self, *args, **kwargs):
        self.name = kwargs.get("name")
        self.func = kwargs.get("func")
        self.description = kwargs.get("description")

dummy_langchain_tools.Tool = DummyTool
sys.modules.setdefault("langchain.tools", dummy_langchain_tools)

# Dummy Bio.Entrez
dummy_bio = types.ModuleType("Bio")
class DummyHandle:
    def __init__(self, data=None):
        self.data = data
    def read(self):
        return self.data
    def close(self):
        pass

class DummyEntrez:
    email = None
    api_key = None
    def esearch(self, *args, **kwargs):
        return DummyHandle()
    def efetch(self, *args, **kwargs):
        return DummyHandle("abstract text")
    def read(self, handle):
        return {"IdList": ["1"]}

dummy_bio.Entrez = DummyEntrez()
sys.modules.setdefault("Bio", dummy_bio)

from tools import search_pubmed


def test_search_pubmed_success(monkeypatch):
    class SearchHandle(DummyHandle):
        pass
    class FetchHandle(DummyHandle):
        pass
    def fake_esearch(db=None, term=None, retmax=None, sort=None):
        return SearchHandle()
    def fake_read(handle):
        return {"IdList": ["123"]}
    def fake_efetch(db=None, id=None, rettype=None, retmode=None):
        return FetchHandle("abstract A")

    monkeypatch.setattr(dummy_bio.Entrez, "esearch", fake_esearch)
    monkeypatch.setattr(dummy_bio.Entrez, "read", fake_read)
    monkeypatch.setattr(dummy_bio.Entrez, "efetch", fake_efetch)

    result = search_pubmed("test query")
    assert result == "abstract A"


def test_search_pubmed_no_results(monkeypatch):
    def fake_esearch(db=None, term=None, retmax=None, sort=None):
        return DummyHandle()
    def fake_read(handle):
        return {"IdList": []}

    monkeypatch.setattr(dummy_bio.Entrez, "esearch", fake_esearch)
    monkeypatch.setattr(dummy_bio.Entrez, "read", fake_read)

    result = search_pubmed("nothing")
    assert "No relevant articles found" in result


def test_search_pubmed_exception(monkeypatch):
    def fake_esearch(*args, **kwargs):
        raise Exception("network error")

    monkeypatch.setattr(dummy_bio.Entrez, "esearch", fake_esearch)

    result = search_pubmed("test")
    assert "An error occurred" in result
