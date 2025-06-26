import os
import sys
import types

# Ensure the repository root is in the Python path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Create dummy modules required for importing graph
# dotenv
dummy_dotenv = types.ModuleType("dotenv")

def load_dotenv():
    pass

dummy_dotenv.load_dotenv = load_dotenv
sys.modules.setdefault("dotenv", dummy_dotenv)

# neo4j
dummy_neo4j = types.ModuleType("neo4j")
class DummyGraphDatabase:
    def driver(self, *args, **kwargs):
        return None

dummy_neo4j.GraphDatabase = DummyGraphDatabase()
sys.modules.setdefault("neo4j", dummy_neo4j)

# langchain_google_genai
dummy_lc_genai = types.ModuleType("langchain_google_genai")
class DummyLLM:
    def __init__(self, *args, **kwargs):
        pass
    def invoke(self, *args, **kwargs):
        return types.SimpleNamespace(content="response")

dummy_lc_genai.ChatGoogleGenerativeAI = DummyLLM
sys.modules.setdefault("langchain_google_genai", dummy_lc_genai)

# langchain_core.prompts
dummy_prompts = types.ModuleType("langchain_core.prompts")
class DummyChatPromptTemplate:
    @classmethod
    def from_template(cls, template):
        return cls()
    def __or__(self, other):
        return other

dummy_prompts.ChatPromptTemplate = DummyChatPromptTemplate
sys.modules.setdefault("langchain_core.prompts", dummy_prompts)

# langchain_core.output_parsers
dummy_parsers = types.ModuleType("langchain_core.output_parsers")
class DummyStrOutputParser:
    def __or__(self, other):
        return other

sys.modules.setdefault("langchain_core.output_parsers", dummy_parsers)
dummy_parsers.StrOutputParser = DummyStrOutputParser

# langgraph.graph
dummy_langgraph = types.ModuleType("langgraph.graph")
class DummyStateGraph:
    def __init__(self, *args, **kwargs):
        pass
    def add_node(self, *args, **kwargs):
        pass
    def add_edge(self, *args, **kwargs):
        pass
    def set_entry_point(self, *args, **kwargs):
        pass
    def add_conditional_edges(self, *args, **kwargs):
        pass
    def compile(self):
        return "compiled"

END = "END"

dummy_langgraph.StateGraph = DummyStateGraph
dummy_langgraph.END = END
sys.modules.setdefault("langgraph.graph", dummy_langgraph)

# google.ai.generativelanguage_v1beta.types
google_pkg = types.ModuleType("google")
ai_pkg = types.ModuleType("google.ai")
gen_pkg = types.ModuleType("google.ai.generativelanguage_v1beta")
types_pkg = types.ModuleType("google.ai.generativelanguage_v1beta.types")
class DummyTool:
    def __init__(self, *args, **kwargs):
        pass

types_pkg.Tool = DummyTool
sys.modules.setdefault("google", google_pkg)
sys.modules.setdefault("google.ai", ai_pkg)
sys.modules.setdefault("google.ai.generativelanguage_v1beta", gen_pkg)
sys.modules.setdefault("google.ai.generativelanguage_v1beta.types", types_pkg)

# modules for utils and tools to avoid missing dependencies
# For get_embedding_model in utils, create dummy to return object with embed_query method

class DummyEmbedding:
    def embed_query(self, text):
        return [0.0]

dummy_utils = types.ModuleType("utils")

def get_embedding_model():
    return DummyEmbedding()

dummy_utils.get_embedding_model = get_embedding_model
sys.modules.setdefault("src.utils", dummy_utils)

# For tools.pubmed_tool if imported

class DummyPubmedTool:
    pass

dummy_src_tools = types.ModuleType("src.tools")
dummy_src_tools.pubmed_tool = DummyPubmedTool()
sys.modules.setdefault("src.tools", dummy_src_tools)

from src.graph import WeightManagementGraph


def test_decide_after_safety():
    g = WeightManagementGraph.__new__(WeightManagementGraph)
    assert g.decide_after_safety({"query_type": "unsafe"}) == "canned_safety_response"
    assert g.decide_after_safety({"query_type": "safe"}) == "classify_query"


def test_decide_after_classification():
    g = WeightManagementGraph.__new__(WeightManagementGraph)
    assert g.decide_after_classification({"query_type": "foundational"}) == "foundational"


def test_decide_after_protocol_rag():
    g = WeightManagementGraph.__new__(WeightManagementGraph)
    assert g.decide_after_protocol_rag({"final_answer": "KNOWLEDGE_GAP"}) == "log_and_reroute"
    assert g.decide_after_protocol_rag({"final_answer": "ok"}) == "add_disclaimer"
