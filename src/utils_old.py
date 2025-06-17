# src/utils.py

from langchain_community.embeddings import HuggingFaceEmbeddings
import sys

def get_embedding_model():
    """
    Initializes and returns the HuggingFace sentence transformer embedding model.
    """
    try:
        model_name = "all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
        )
    except Exception as e:
        print(f"Error: Failed to load HuggingFace embedding model '{model_name}': {e}", file=sys.stderr)
        raise